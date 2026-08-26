# FoulFilter

<img src="logo.png" width="300">

Automated profanity removal for audio/video files — built for long-form media
(audiobooks, podcasts, movies) on AMD ROCm and NVIDIA CUDA GPUs. Upload files
in the browser, watch progress live, download censored results individually or
as a ZIP.

## How it works

1. **GPU transcription** — Hugging Face Whisper transcribes the file once on the GPU.
2. **Matching** — the transcript is scanned against your Bad Words List (`data/bad_words.txt`).
3. **Alignment** — WhisperX forced alignment timestamps every word (GPU by default, CPU fallback).
4. **Smart Cut** *(optional)* — an LLM widens cuts into whole idioms ("go to hell") for jump-cut removals, or rejects false positives ("hoe" the garden tool).
5. **Edit** — FFmpeg renders `silence`, `bleep`, or `remove` (cut) per hit.

WhisperX's CTranslate2 transcription backend does not run on RDNA2 ROCm, so only
its aligner is used — see `docs/adr/0001`. Jobs run strictly one at a time.

## Prerequisites

- **Linux** (Ubuntu 22.04+ recommended)
- **Docker** with Compose v2
- **AMD ROCm GPU** (RX 6000/7000 series) or **NVIDIA GPU** (with CUDA drivers)

## Quick start

```bash
cp .env.example .env   # edit to taste — see .env.example for all options
docker compose up -d --build
```

Open **http://localhost:8000**, drop files, pick a censor method, upload.

### Bad words customization

The image ships a starter list at `/data/bad_words.txt` with a few silly
placeholder words. Edit it inside the container:

```bash
docker compose exec foulfilter nano /data/bad_words.txt
```

One word or phrase per line. Lines starting with `#` are ignored. Matching is
case- and punctuation-insensitive; phrases up to 3 words are supported.
See `data/bad_words.example.txt` for the format.

## GPU setup

### AMD ROCm

The default configuration targets AMD GPUs. After `docker compose up -d --build`:

```bash
docker compose exec foulfilter python /scripts/check_gpu.py        # torch-level
docker compose exec foulfilter python /scripts/checkgpu.py --full  # loads Whisper too
```

If the GPU check fails, verify your host has `/dev/kfd` and `/dev/dri` available
and your user is in the `video` and `render` groups:

```bash
ls -la /dev/kfd /dev/dri
groups   # should include video and render
```

For RDNA2 GPUs (RX 6700/6800/6900 series), the `HSA_OVERRIDE_GFX_VERSION`
environment variable is typically needed. Set it in your `.env`:

```
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### NVIDIA

Change the PyTorch wheel index in `.env` and adjust device mappings:

```bash
# .env
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
```

In `docker-compose.yml`, replace the `devices` and `group_add` blocks with the
NVIDIA runtime:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Remove `HSA_OVERRIDE_GFX_VERSION` from `.env` if present.

## CLI (inside the container)

```bash
docker compose exec foulfilter \
  python find_and_remove.py /data/uploads/book.m4b /data/bad_words.txt --output /data/outputs/book_clean.m4b
```

Flags: `--bleep`, `--delete` (cut), `--censor_method`, `--rescan` (second detection
pass with shifted chunk boundaries — catches words garbled at chunk edges),
`--no_edit` (report hits only), `--debug`.

## Configuration (`.env`)

All configuration is in `.env` — copy `.env.example` and edit. See
`.env.example` for the full list with inline documentation. Key variables:

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `base` | HF model id or bare size (`medium`, `large-v3`, …) |
| `CENSOR_METHOD` | `silence` | `silence`, `bleep`, `remove` |
| `AI_ENHANCE` | `False` | Enable Smart Cut |
| `AI_MODE` | `google` | `google` or `local` (OpenAI-compatible server) |
| `UNLOAD_MODELS_AFTER_JOB` | `True` | Free GPU memory between jobs |
| `TORCH_INDEX_URL` | `…/whl/cu128` | PyTorch wheel index (ROCm, CUDA, or CPU) |

## Disk footprint

The image is built from `python:3.11-slim` plus PyTorch wheels that bundle
their own GPU runtime (~10 GB total). Upgrade by changing `TORCH_INDEX_URL` in
`.env` and rebuilding. Reclaim space after builds:

```bash
docker system prune -f     # stopped containers, dangling images
docker builder prune -f    # build cache — often tens of GB
```

## Verifying the GPU

```bash
docker compose exec foulfilter python /scripts/check_gpu.py        # torch-level
docker compose exec foulfilter python /scripts/check_gpu.py --full # loads Whisper too
```

## Measuring misses

`scripts/eval_misses.py` splices swear clips into clean audio at known
timestamps (deliberately clustered at chunk boundaries), runs the pipeline, and
scores recall / precision / boundary error:

```bash
docker compose exec foulfilter python /scripts/eval_misses.py \
  --audio /data/uploads/impractical_test.mp3 --clips-dir /data/swear_clips
```

## Development

Unit tests avoid loading models:

```bash
pytest src -q
```

Domain language lives in [`CONTEXT.md`](CONTEXT.md); architecture decisions in
[`docs/adr/`](docs/adr).
