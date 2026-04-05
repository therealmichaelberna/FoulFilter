from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import wave
import os

SetLogLevel(-1)


def process_vosk(audio_path):
    # Ensure the model is available in your Docker volume
    model_path = "/app/models/vosk-model-small-en-us"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at {model_path}")

    model = Model(model_path)

    word_list = []
    with wave.open(audio_path, "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if "result" in res:
                    word_list.extend(res["result"])

        final = json.loads(rec.FinalResult())
        if "result" in final:
            word_list.extend(final["result"])

    # Normalize Vosk keys to match WhisperX output
    return [
        {"word": w["word"].lower(), "start": w["start"], "end": w["end"]}
        for w in word_list
    ]
