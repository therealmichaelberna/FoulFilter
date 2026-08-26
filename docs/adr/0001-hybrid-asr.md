# ADR-0001: Hybrid ASR — GPU transcription, GPU alignment

WhisperX cannot run its transcription path on AMD ROCm because it depends on
CTranslate2 (C++/HIP), which has no usable build for RDNA2 consumer GPUs —
forking WhisperX would not fix this since the wall is in the native backend, not
the Python package. However, the alignment half of WhisperX
(`whisperx.align`, a wav2vec2 forced aligner) is plain PyTorch and runs fine on
either CPU or GPU. We therefore transcribe each file exactly once with a
Hugging Face Whisper pipeline on the GPU, then run `whisperx.align` over the
entire transcript in padded batches (ALIGN_DEVICE defaults to cuda when a GPU is
present, cpu otherwise). The first-generation design (full HF Whisper scan for
detection + full WhisperX re-transcription on CPU) doubled inference cost and
was removed.

Alignment covers every Segment whenever any Candidate exists — not just flagged
neighborhoods — because uniform word-level precision is worth more than the
saved minutes now that the aligner shares the GPU, and it keeps detection
recall improvements (Rescan Pass) usable without re-aligning.

## Considered Options

- **Fork/patch WhisperX + CTranslate2 for ROCm**: rejected — CTranslate2 v4.7
  (2026-02) ships experimental ROCm wheels only for officially supported archs;
  RDNA2 consumer support requires unsupported source builds against HIP.
- **Align only windows around flagged Candidates**: rejected — saves little
  with a GPU-resident aligner and complicates hit confirmation; forced
  alignment cannot recover words missing from transcript text anyway, so recall
  belongs to the Rescan Pass, not to wider alignment.
- **GPU word-level timestamps only** (`return_timestamps="word"`, drop
  WhisperX): rejected for now — looser boundaries than forced alignment.

## Consequences

- One full-file inference per Job instead of two; no second model load per job
  (engines are process-wide singletons).
- The WhisperX pip dependency stays, but only its aligner is imported at
  runtime; CTranslate2 is never initialized.
