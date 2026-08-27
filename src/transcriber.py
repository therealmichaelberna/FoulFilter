"""transcriber.py - GPU Whisper transcription via Hugging Face pipeline.

Runs on ROCm (torch.cuda) or CPU. Heavy imports are deferred to __init__ so
unit tests can import this module cheaply.

Multi-GPU: when WHISPER_MULTI_GPU=True, one Transcriber is loaded per GPU
and audio is split across them for parallel transcription.
"""

import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("FoulFilter.transcriber")

_transcriber = None  # single-GPU singleton
_transcribers = []   # N-GPU list

RESCAN_OFFSET_S = 4.0


def normalize_model_name(name: str) -> str:
    name = (name or "base").strip()
    if "/" in name:
        return name
    return f"openai/whisper-{name}"


def _gpu_count():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def _is_multi_gpu_enabled():
    return (
        _gpu_count() > 1
        and (os.getenv("WHISPER_MULTI_GPU") or "False").strip().lower()
        in ("1", "true", "yes")
    )


class Transcriber:
    """Full-file transcription returning segment-level text + timestamps."""

    def __init__(self, model_size=None, language=None, device_id=None):
        import torch
        from transformers import pipeline

        self.language = language or os.getenv("WHISPER_LANGUAGE") or None
        model = normalize_model_name(
            model_size or os.getenv("WHISPER_MODEL") or "base"
        )

        if device_id is not None and torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
            engine_kwargs = {"device": f"cuda:{device_id}"}
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            engine_kwargs = {}
            if self.device == "cuda":
                engine_kwargs["device"] = "cuda"

        attn = os.getenv("WHISPER_ATTN")
        if attn:
            engine_kwargs["model_kwargs"] = {"attn_implementation": attn}
        logger.info("Loading transcription model %s on %s...", model, self.device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            torch_dtype=torch.float16 if "cuda" in self.device else None,
            **engine_kwargs,
        )
        logger.info("Transcription model ready on %s.", self.device)

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


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def _get_audio_duration(path):
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _split_audio(path, n_parts, scratch_dir):
    """Split audio into n_parts equal chunks. Returns list of paths."""
    duration = _get_audio_duration(path)
    chunk_len = duration / n_parts
    paths = []
    for i in range(n_parts):
        start = i * chunk_len
        out = os.path.join(scratch_dir, f"split_{i}.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-t", f"{chunk_len:.3f}",
                "-i", path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-acodec", "pcm_s16le", out,
            ],
            check=True,
        )
        paths.append((out, start))
    return paths


def _transcribe_chunk(args):
    """Transcribe a single audio chunk. Returns (segments, time_offset)."""
    transcriber, audio_path, time_offset = args
    segments = transcriber.transcribe(audio_path)
    for seg in segments:
        seg["start"] = round(seg["start"] + time_offset, 3)
        seg["end"] = round(seg["end"] + time_offset, 3)
    return segments


def transcribe_parallel(audio_path, transcribers, scratch_dir):
    """Split audio across N GPUs, transcribe in parallel, merge segments."""
    n = len(transcribers)
    if n <= 1:
        return transcribers[0].transcribe(audio_path)

    chunks = _split_audio(audio_path, n, scratch_dir)
    logger.info("Split audio into %d chunks for parallel transcription", n)

    tasks = [
        (transcribers[i], path, offset)
        for i, (path, offset) in enumerate(chunks)
    ]

    all_segments = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        for segs in pool.map(_transcribe_chunk, tasks):
            all_segments.extend(segs)

    all_segments.sort(key=lambda s: s["start"])
    logger.info("Parallel transcription produced %d segments", len(all_segments))

    # Clean up split files
    for path, _ in chunks:
        try:
            os.remove(path)
        except OSError:
            pass

    return all_segments


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

def get_transcriber():
    """Process-wide singleton so the model stays loaded across jobs."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def get_transcribers():
    """Return one Transcriber per GPU (multi-GPU) or a single Transcriber."""
    global _transcribers
    if _transcribers:
        return _transcribers
    if _is_multi_gpu_enabled():
        count = _gpu_count()
        _transcribers = [Transcriber(device_id=i) for i in range(count)]
        logger.info("Loaded %d transcription models across %d GPUs", count, count)
    else:
        _transcribers = [get_transcriber()]
    return _transcribers


def release_transcriber():
    """Drop the singleton and hand its VRAM back (UNLOAD_MODELS_AFTER_JOB)."""
    global _transcriber, _transcribers
    _transcriber = None
    _transcribers = []
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Transcription model released; VRAM cleared.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
