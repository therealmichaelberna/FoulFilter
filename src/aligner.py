"""aligner.py - WhisperX forced alignment over the whole transcript.

Only whisperx.align is used (pure PyTorch wav2vec2); the CTranslate2
transcription path of WhisperX is never touched (see ADR-0001). The aligner
defaults to the GPU when one is available and falls back to CPU. Segments are
aligned in contiguous padded batches so memory stays bounded and progress can
be reported. Heavy imports are deferred to __init__.
"""

import logging
import os
import subprocess

logger = logging.getLogger("FoulFilter.aligner")

_aligner = None

BATCH_SEGMENTS = 50
BATCH_PAD = 0.5


class Aligner:
    def __init__(self, language="en", device=None):
        import torch
        import whisperx

        if device:
            self.device = device
        else:
            self.device = os.getenv("ALIGN_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        logger.info("Loading WhisperX alignment model on %s...", self.device)
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )
        logger.info("Alignment model ready.")

    def align_transcript(self, source_audio, segments, progress=None):
        """Align every segment; returns absolute-time words.

        segments: [{start, end, text}] covering the transcript.
        progress(done, total) is called after each batch.
        """
        words = []
        batches = [
            segments[i : i + BATCH_SEGMENTS]
            for i in range(0, len(segments), BATCH_SEGMENTS)
        ]
        for idx, batch in enumerate(batches):
            span_start = max(0.0, batch[0]["start"] - BATCH_PAD)
            span_end = batch[-1]["end"] + BATCH_PAD
            words.extend(
                self._align_span(
                    source_audio, span_start, span_end - span_start, batch
                )
            )
            if progress:
                progress(idx + 1, len(batches))
        words.sort(key=lambda w: w["start"])
        return words

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


def get_aligner():
    global _aligner
    if _aligner is None:
        _aligner = Aligner()
    return _aligner


def release_aligner():
    """Drop the singleton and hand its VRAM back (UNLOAD_MODELS_AFTER_JOB)."""
    global _aligner
    if _aligner is None:
        return
    _aligner = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Alignment model released; VRAM cleared.")
    except Exception:
        pass
