"""GPU sanity check for the FoulFilter image. Run inside the container:

    python /scripts/check_gpu.py          # torch-level checks
    python /scripts/check_gpu.py --full   # also load Whisper and transcribe once
"""
import sys


def main():
    import torch

    print(f"torch       {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    print(f"GPU runtime {hip or 'CUDA (no HIP)'}")
    print(f"cuda avail  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: no GPU visible. Check /dev/kfd + /dev/dri (AMD) or")
        print("      nvidia runtime + /dev/nvidia* (NVIDIA) in docker-compose.")
        return 1

    print(f"device      {torch.cuda.get_device_name(0)}")
    x = torch.randn(1024, 1024, device="cuda")
    assert torch.isfinite(x @ x).all(), "GPU matmul produced NaN/inf"
    print("matmul      ok")

    if "--full" in sys.argv:
        import tempfile

        import numpy as np
        import soundfile as sf

        sys.path.insert(0, "/app")
        import transcriber

        t = transcriber.get_transcriber()
        print(f"whisper     {t.pipe.model.name_or_path} loaded on {t.device}")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, np.zeros((16000,), dtype="float32"), 16000)
        segments = t.transcribe(tmp.name)
        print(f"inference   ok ({len(segments)} segments on silence)")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
