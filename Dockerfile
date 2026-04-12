# Slim inference image: PyTorch wheels bundle their own ROCm runtime, so the
# giant rocm/pytorch SDK image (~100 GB) is not needed. The host only supplies
# the kernel driver via /dev/kfd + /dev/dri. See docs/adr/0005.
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

USER root
# ffmpeg for audio surgery, libmagic1 for python-magic
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first: biggest layer, changes only when the wheel index is bumped
RUN pip install --no-cache-dir torch torchaudio --index-url ${TORCH_INDEX_URL} \
 && pip freeze | grep -E '^(torch|pytorch-triton)' > /tmp/torch-constraints.txt

# Dependencies next so code edits don't bust this layer's cache. The
# constraints file pins the exact +rocm builds just installed - without it,
# pyannote.audio's transitive "torch" dep lets pip swap in the CUDA wheel
# from PyPI and the GPU silently disappears.
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -c /tmp/torch-constraints.txt

COPY src/ /app/

# Triton JIT-compiles ROCm kernels on first use (Whisper attention etc.) and
# needs a C compiler + glibc headers (libc6-dev: gcc's stdint.h #include_next's
# into them). Kept AFTER the heavy pip layers so bumping never re-downloads torch.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# GPU override: RDNA2 (RX 6700/6800/6900) needs gfx1030; remove for NVIDIA.
# Caches land on /data so rebuilds don't re-download or re-JIT kernels.
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    HF_HOME=/data/hf-cache \
    MIOPEN_USER_DB_PATH=/data/miopen \
    TRITON_CACHE_DIR=/data/triton-cache

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
