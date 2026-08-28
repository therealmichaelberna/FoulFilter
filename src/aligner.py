"""aligner.py - WhisperX forced alignment over the whole transcript.

Only whisperx.align is used (pure PyTorch wav2vec2); the CTranslate2
transcription path of WhisperX is never touched (see ADR-0001). The aligner
defaults to the GPU when one is available and falls back to CPU. Segments are
aligned in contiguous padded batches so memory stays bounded and progress can
be reported. Heavy imports are deferred to __init__.

Multi-GPU: when WHISPER_MULTI_GPU=True, one Aligner is loaded per GPU and
batches are distributed across them for parallel alignment.
"""

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("FoulFilter.aligner")

_aligner = None
_aligners = []

BATCH_SEGMENTS = 50
BATCH_MAX_SPAN = 60  # max audio seconds per alignment batch to bound VRAM
BATCH_PAD = 0.5

# Alignment is best-effort: if a window still OOMs we split it and retry
# smaller; if it ultimately cannot fit, we return no words for that span so
# the pipeline falls back to segment-level timestamps rather than crashing.
MAX_ALIGN_RETRIES = 3


def _is_oom(exc):
    import torch

    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(exc).lower()
        or "HIPCachingAllocator" in str(exc)
        or "CUDA out of memory" in str(exc)
    )


def build_batches(segments, max_segments=BATCH_SEGMENTS, max_span=BATCH_MAX_SPAN):
    """Group segments into batches bounded by count AND continuous audio span.

    A batch must not span more than `max_span` seconds (start of first to end
    of last) so the wav2vec2 encoder never processes a window large enough to
    OOM the GPU. Gaps between segments are included in the span because the
    whole window is aligned together.
    """
    if not segments:
        return []
    batches = []
    cur = [segments[0]]
    for seg in segments[1:]:
        span = (seg["end"] - cur[0]["start"])
        if len(cur) >= max_segments or span > max_span:
            batches.append(cur)
            cur = [seg]
        else:
            cur.append(seg)
    batches.append(cur)
    return batches


class Aligner:
    def __init__(self, language="en", device=None, device_id=None):
        import torch
        import whisperx

        if device_id is not None and torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
        elif device:
            self.device = device
        else:
            self.device = os.getenv("ALIGN_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        logger.info("Loading WhisperX alignment model on %s...", self.device)
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )
        logger.info("Alignment model ready on %s.", self.device)

    def align_transcript(self, source_audio, segments, progress=None):
        """Align every segment best-effort; returns absolute-time words.

        segments: [{start, end, text}] covering the transcript.
        progress(done, total) is called after each batch.

        If a batch cannot be aligned (out of memory), it is skipped and its
        segments are left unaligned so the pipeline can use estimates.
        """
        words = []
        batches = build_batches(segments)
        for idx, batch in enumerate(batches):
            try:
                words.extend(self._align_batch_resilient(source_audio, batch))
            except Exception as exc:  # noqa: BLE001 - never crash a job on alignment
                logger.warning("Alignment batch failed (%s); using estimates", exc)
            if progress:
                progress(idx + 1, len(batches))
        words.sort(key=lambda w: w["start"])
        return words

    def _align_batch_resilient(self, source_audio, batch):
        """Align one batch, retrying with halved audio spans on OOM.

        Returns a flat list of words (possibly fewer than expected if some
        sub-windows could not be aligned). Never raises on OOM.
        """
        span_start = max(0.0, batch[0]["start"] - BATCH_PAD)
        span_end = batch[-1]["end"] + BATCH_PAD
        try:
            return self._align_span(
                source_audio, span_start, span_end - span_start, batch
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise
            import torch
            torch.cuda.empty_cache()
            logger.warning(
                "Alignment OOM on %ds window; retrying smaller",
                span_end - span_start,
            )
            return self._align_split_recurse(source_audio, batch, 0)

    def _align_split_recurse(self, source_audio, batch, depth):
        """Split the span in half and align each half; handled OOM gracefully."""
        if depth >= MAX_ALIGN_RETRIES:
            return []
        if len(batch) <= 1:
            return []
        mid = len(batch) // 2
        first, second = batch[:mid], batch[mid:]
        out = []
        for half in (first, second):
            if not half:
                continue
            s0 = max(0.0, half[0]["start"] - BATCH_PAD)
            s1 = half[-1]["end"] + BATCH_PAD
            try:
                out.extend(self._align_span(source_audio, s0, s1 - s0, half))
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    raise
                import torch
                torch.cuda.empty_cache()
                logger.warning("Alignment OOM on sub-window; retrying smaller")
                out.extend(self._align_split_recurse(source_audio, half, depth + 1))
        return out

    def _align_span(self, source_audio, offset, duration, segments):
        """Align a cropped span of audio; returns rebased absolute words."""
        import whisperx

        wav_path = crop_audio(source_audio, offset, duration)
        try:
            audio = whisperx.load_audio(wav_path)
            window_len = float(audio.shape[-1]) / 16000.0

            rel_segments = []
            for seg in segments:
                rs = max(0.0, seg["start"] - offset)
                re_ = min(window_len, seg["end"] - offset)
                if re_ - rs < 0.05:
                    continue
                rel_segments.append(
                    {"start": rs, "end": re_, "text": seg["text"]}
                )
            if not rel_segments:
                return []

            result = whisperx.align(
                rel_segments,
                self.model_a,
                self.metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            out = []
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    start, end = w.get("start"), w.get("end")
                    if start is None or end is None or end <= start:
                        continue
                    out.append(
                        {
                            "word": w["word"].lower().strip(),
                            "start": round(start + offset, 3),
                            "end": round(end + offset, 3),
                        }
                    )
            return out
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def _align_batch_task(args):
    """Align a single batch on a specific Aligner. Returns list of words."""
    aligner, source_audio, batch = args
    try:
        return aligner._align_batch_resilient(source_audio, batch)
    except Exception as exc:  # noqa: BLE001 - never crash a job on alignment
        logger.warning("Alignment batch failed (%s); using estimates", exc)
        return []


def align_transcript_parallel(source_audio, segments, aligners, progress=None):
    """Distribute alignment batches across N aligners in parallel."""
    if len(aligners) <= 1:
        return aligners[0].align_transcript(source_audio, segments, progress)

    batches = build_batches(segments)
    total_batches = len(batches)
    logger.info("Parallel alignment: %d batches across %d GPUs", total_batches, len(aligners))

    # Round-robin assign batches to aligners
    tasks = [
        (aligners[i % len(aligners)], source_audio, batch)
        for i, batch in enumerate(batches)
    ]

    all_words = []
    done = [0]

    def _run_and_track(args):
        result = _align_batch_task(args)
        done[0] += 1
        if progress:
            progress(done[0], total_batches)
        return result

    with ThreadPoolExecutor(max_workers=len(aligners)) as pool:
        for words in pool.map(_run_and_track, tasks):
            all_words.extend(words)

    all_words.sort(key=lambda w: w["start"])
    logger.info("Parallel alignment produced %d words", len(all_words))
    return all_words


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

def get_aligner():
    global _aligner
    if _aligner is None:
        _aligner = Aligner()
    return _aligner


def get_aligners():
    """Return one Aligner per GPU (multi-GPU) or a single Aligner."""
    global _aligners
    if _aligners:
        return _aligners
    from transcriber import _gpu_count, _is_multi_gpu_enabled
    if _is_multi_gpu_enabled():
        count = _gpu_count()
        _aligners = [Aligner(device_id=i) for i in range(count)]
        logger.info("Loaded %d alignment models across %d GPUs", count, count)
    else:
        _aligners = [get_aligner()]
    return _aligners


def release_aligner():
    """Drop the singleton and hand its VRAM back (UNLOAD_MODELS_AFTER_JOB)."""
    global _aligner, _aligners
    _aligner = None
    _aligners = []
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Alignment model released; VRAM cleared.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def crop_audio(source_path, offset, duration):
    """Extract a mono 16 kHz WAV snippet [offset, offset+duration]."""
    out_path = source_path + f".span_{offset:.3f}_{duration:.3f}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}",
            "-i", source_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", out_path,
        ],
        check=True,
    )
    return out_path
