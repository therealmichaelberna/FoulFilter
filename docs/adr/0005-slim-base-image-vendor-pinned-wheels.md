# ADR-0005: Slim base image with vendor-pinned PyTorch wheels

Date: 2026-08-23
Status: Accepted

## Context

Full SDK images (`rocm/pytorch`, `nvidia/cuda`) exist for *developing* GPU
software — compilers, debuggers, dev headers, multiple library generations —
and weigh 50-100 GB. FoulFilter only performs inference: HF Transformers
Whisper for transcription and WhisperX's wav2vec2 aligner, both plain PyTorch
workloads. The host already provides the kernel driver; the container only
needs the runtime libraries that ship inside PyTorch's own wheel packages.

## Decision

Build from `python:3.11-slim` and install the official PyTorch wheels from the
appropriate vendor index, which bundle the needed runtime libraries inside
site-packages:

```dockerfile
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128   # or rocm7.2, cpu
RUN pip install torch torchaudio --index-url ${TORCH_INDEX_URL}
```

The wheel index is an `ARG` (`TORCH_INDEX_URL`, default CUDA) passed through
`docker-compose.yml` from `.env`, so switching vendors or bumping versions is a
one-line change plus rebuild. A constraints file generated from `pip freeze`
pins the exact vendor-specific builds so transitive dependencies (e.g.
`pyannote.audio → torch`) cannot silently swap in the wrong wheel from PyPI.

Alternatives rejected:

- **Bind-mounting host `/opt/rocm` or `/usr/local/cuda`** — couples container
  correctness to the host userspace version matching PyTorch's expectations;
  non-portable.
- **Full SDK base images** — 50-100 GB for inference-only workloads.

## Consequences

- Image drops to roughly 10 GB; rebuilds churn far less cache.
- Switching GPU vendor = edit one variable in `.env` and rebuild.
- Wheel releases can trail brand-new driver point releases slightly; pinning
  means we choose when to move.
- Triton JIT-compiles ROCm/CUDA kernels on first use and needs a C compiler
  (`gcc` + `libc6-dev`) plus `TRITON_CACHE_DIR` on a persistent volume so
  kernels survive restarts.
- CI stays CPU-only: tests never import torch (lazy imports), so nothing
  changes there.
