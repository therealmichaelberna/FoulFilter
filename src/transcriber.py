"""transcriber.py - GPU Whisper transcription via Hugging Face pipeline.

Runs on ROCm (torch.cuda) or CPU. Heavy imports are deferred to __init__ so
unit tests can import this module cheaply.
"""

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger("FoulFilter.transcriber")

_transcriber = None

RESCAN_OFFSET_S = 4.0


def normalize_model_name(name: str) -> str:
    name = (name or "base").strip()
    if "/" in name:
        return name
    return f"openai/whisper-{name}"


class Transcriber:
    """Full-file transcription returning segment-level text + timestamps."""

    def __init__(self, model_size=None, language=None):
        import torch
        from transformers import pipeline

        self.language = language or os.getenv("WHISPER_LANGUAGE") or None
        model = normalize_model_name(
            model_size or os.getenv("WHISPER_MODEL") or "base"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        engine_kwargs = {}
        if self.device == "cuda":
            multi_gpu = (os.getenv("WHISPER_MULTI_GPU") or "False").strip().lower() in ("1", "true", "yes")
            if multi_gpu and torch.cuda.device_count() > 1:
                # Spread layers across every card (pooled VRAM). Off by default:
                # RDNA2 cross-card sharding has shown GPU memory faults here.
                engine_kwargs["device_map"] = "auto"
            else:
                engine_kwargs["device"] = "cuda"
        attn = os.getenv("WHISPER_ATTN")
        if attn:
            # e.g. WHISPER_ATTN=eager sidesteps Triton SDPA kernels entirely
            engine_kwargs["model_kwargs"] = {"attn_implementation": attn}
        logger.info("Loading transcription model %s on %s...", model, self.device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            torch_dtype=torch.float16 if self.device == "cuda" else None,
            **engine_kwargs,
        )
        logger.info("Transcription model ready.")

    def transcribe(self, audio_path):
        """Return a list of segments: [{start, end, text}] in seconds."""
        generate_kwargs = {"task": "transcribe"}
        if self.language:
            generate_kwargs["language"] = self.language
        result = self.pipe(
            audio_path,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        return _segments_from_result(result)

    def transcribe_shifted(self, audio_path, offset=RESCAN_OFFSET_S):
        """Rescan Pass: transcribe with every chunk boundary shifted.

        Pads the start of the audio with `offset` seconds of silence before
        inference, then subtracts the offset from all timestamps. A word that
        was garbled when it straddled a boundary in the first pass lands
        cleanly inside a chunk here.
        """
        padded = _pad_start(audio_path, offset)
        try:
            segments = self.transcribe(padded)
        finally:
            if os.path.exists(padded):
                os.remove(padded)
        shifted = []
        for seg in segments:
            start, end = seg["start"] - offset, seg["end"] - offset
            if end <= 0:
                continue
            shifted.append(
                {
                    "start": round(max(0.0, start), 3),
                    "end": round(end, 3),
                    "text": seg["text"],
                }
            )
        return shifted


def _pad_start(audio_path, offset):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    delay_ms = int(offset * 1000)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", audio_path,
            "-af", f"adelay={delay_ms}|{delay_ms}",
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", tmp.name,
        ],
        check=True,
    )
    return tmp.name


def _segments_from_result(result):
    segments = []
    for chunk in result.get("chunks", []):
        ts = chunk.get("timestamp") or [None, None]
        start, end = ts[0], ts[1]
        if start is None or end is None or end <= start:
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        segments.append({"start": float(start), "end": float(end), "text": text})
    logger.info("Transcribed %d segments from %d chars",
                len(segments), sum(len(s["text"]) for s in segments))
    return segments


def get_transcriber():
    """Process-wide singleton so the model stays loaded across jobs."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def release_transcriber():
    """Drop the singleton and hand its VRAM back (UNLOAD_MODELS_AFTER_JOB)."""
    global _transcriber
    if _transcriber is None:
        return
    _transcriber = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Transcription model released; VRAM cleared.")
    except Exception:
        pass
