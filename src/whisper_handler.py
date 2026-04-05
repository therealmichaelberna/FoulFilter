import os
import logging
import whisperx
import torch

logger = logging.getLogger("FoulFilter.Whisper")


class WhisperEngine:
    def __init__(self, model_size=None, language="en"):
        if model_size is None:
            model_size = os.getenv("WHISPER_MODEL", "base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        if self.device == "cpu":
            logger.warning("Loaded CPU instead of CUDA, may be slower than using CUDA.")
        self.language = language

        logger.info(f"Loading Whisper {model_size} on {self.device}...")
        self.model = whisperx.load_model(
            model_size,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
        )

        # Pre-load alignment model once
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code=self.language, device=self.device
        )

    def process_segment(self, audio_path):
        audio = whisperx.load_audio(audio_path)

        # Transcribe
        result = self.model.transcribe(audio, batch_size=16, language=self.language)

        # Align
        result = whisperx.align(
            result["segments"],
            self.model_a,
            self.metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        word_list = []
        for segment in result["segments"]:
            for word in segment.get("words", []):
                if "start" in word and "end" in word:
                    word_list.append(
                        {
                            "word": word["word"].lower().strip(),
                            "start": float(word["start"]),
                            "end": float(word["end"]),
                        }
                    )
        return word_list
