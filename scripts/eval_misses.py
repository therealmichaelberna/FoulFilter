"""eval_misses.py - measure detection misses against synthetic ground truth.

Splices swear clips into a clean audio file at exact known timestamps
(deliberately clustered around Whisper's 30s chunk boundaries), runs the
FoulFilter pipeline, and scores:

  recall          found hits / planted swears
  precision       true-positive hits / all reported hits
  boundary error  mean |predicted_start - true_start|, same for ends

Run inside the container (models live there):
  python /scripts/eval_misses.py --audio /data/uploads/clean.mp3 \
      --clips-dir /data/swear_clips --out /data/eval_report.json

Clips-dir should contain short recordings of the words to plant (one word per
file, wav/mp3/m4a).
"""

import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, "/app")

CHUNK_S = 30.0
BOUNDARY_JITTER = 1.5
EDGE_SKIP = 5.0


def duration_of(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path]
    )
    return float(json.loads(out)["format"]["duration"])


def build_truth(duration, clip_paths, seed=7):
    """Choose plant positions hugging chunk boundaries."""
    rng = random.Random(seed)
    truth = []
    t = CHUNK_S
    idx = 0
    while t < duration - EDGE_SKIP:
        jitter = rng.uniform(-BOUNDARY_JITTER, BOUNDARY_JITTER)
        pos = max(EDGE_SKIP, t + jitter)
        clip = clip_paths[idx % len(clip_paths)]
        truth.append({"position": round(pos, 3), "clip": clip})
        idx += 1
        t += CHUNK_S * rng.uniform(0.9, 1.4)  # not perfectly periodic
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Clean source audio")
    parser.add_argument("--clips-dir", required=True,
                        help="Directory of short swear-word recordings")
    parser.add_argument("--bad-words", default="/app/bad_words.txt")
    parser.add_argument("--out", default="/data/eval_report.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    clips = [
        os.path.join(args.clips_dir, f)
        for f in sorted(os.listdir(args.clips_dir))
        if f.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac"))
    ]
    if not clips:
        sys.exit(f"No audio clips found in {args.clips_dir}")

    duration = duration_of(args.audio)
    truth = build_truth(duration, clips, seed=args.seed)
    print(f"Planting {len(truth)} swear(s) into {duration:.1f}s of audio...")

    # Mix: pad each clip with silence up to its position, then amix everything.
    work_dir = os.path.join("/data", "scratch", "eval_misses")
    os.makedirs(work_dir, exist_ok=True)
    delayed = []
    for i, plant in enumerate(truth):
        out = os.path.join(work_dir, f"plant_{i}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", plant["clip"],
             "-af", f"adelay={int(plant['position'] * 1000)}|{int(plant['position'] * 1000)}",
             "-ar", "44100", "-ac", "2", out],
            check=True,
        )
        delayed.append(out)

    mixed = os.path.join(work_dir, "planted_input.wav")
    inputs = []
    for d in [args.audio] + delayed:
        inputs += ["-i", d]
    labels = "".join(f"[{i}:a]" for i in range(len(delayed) + 1))
    graph = (
        f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[base];"
        f"{labels}amix=inputs={len(delayed) + 1}:duration=first:normalize=0[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error"] + inputs +
        ["-filter_complex", graph, "-map", "[out]", mixed],
        check=True,
    )

    from pipeline import run_job

    summary = run_job(
        input_path=mixed,
        output_path=os.path.join(work_dir, "planted_output.wav"),
        bad_words_list_path=args.bad_words,
        transcript_dir=os.path.join("/data", "transcripts"),
        scratch_dir=os.path.join(work_dir, "job_scratch"),
        censor_method="silence",
    )

    hits = summary["hits"]
    matched, boundary_errs = [], []
    used = set()
    for plant in truth:
        best = None
        for j, hit in enumerate(hits):
            if j in used:
                continue
            if hit["start"] - 0.25 <= plant["position"] <= hit["end"] + 0.25:
                best = j
                break
        if best is None:
            continue
        used.add(best)
        matched.append(plant)
        boundary_errs.append(abs(hits[best]["start"] - plant["position"]))

    report = {
        "planted": len(truth),
        "detected": len(hits),
        "true_positives": len(matched),
        "missed": len(truth) - len(matched),
        "false_positives": len(hits) - len(matched),
        "recall": round(len(matched) / max(1, len(truth)), 3),
        "precision": round(len(matched) / max(1, len(hits)), 3),
        "mean_start_error_s": round(
            sum(boundary_errs) / max(1, len(boundary_errs)), 3
        ),
        "truth": truth,
        "hits": hits,
        "seed": args.seed,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("truth", "hits")}, indent=2))
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
