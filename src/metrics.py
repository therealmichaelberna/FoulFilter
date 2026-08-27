"""metrics.py - Historical job timing for ETA estimates.

Stores the last N completed jobs (clip duration, stage timings) in a JSON
file so the frontend can predict remaining time for in-progress jobs based
on similar past jobs.
"""

import fcntl
import json
import logging
import os

logger = logging.getLogger("FoulFilter.metrics")

MAX_HISTORY = 20
METRICS_PATH = os.getenv("METRICS_PATH") or "/data/metrics.json"

# Map pipeline stage names to their approximate progress midpoints.
# Used when historical data is sparse to interpolate partial stage timings.
_STAGE_MIDPOINTS = {
    "transcribing": 0.225,
    "matching": 0.50,
    "aligning": 0.65,
    "refining": 0.825,
    "editing": 0.90,
    "completed": 1.00,
}


def _load(path=None):
    path = path or METRICS_PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save(metrics, path=None):
    path = path or METRICS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(metrics, f)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    os.replace(tmp, path)


def record_job(clip_seconds, stage_seconds, path=None):
    """Record timing for a completed job.

    clip_seconds: total audio duration in seconds.
    stage_seconds: dict mapping stage name -> wall-clock seconds spent in that stage.
    """
    if not stage_seconds:
        return
    metrics = _load(path)
    metrics.append({
        "clip_seconds": clip_seconds,
        "stages": stage_seconds,
    })
    metrics = metrics[-MAX_HISTORY:]
    _save(metrics, path)
    logger.info(
        "Recorded metrics: clip=%.0fs, stages=%s",
        clip_seconds,
        {k: round(v, 1) for k, v in stage_seconds.items()},
    )


def get_metrics(path=None):
    return _load(path)


def estimate_remaining(clip_seconds, current_stage, progress_pct, history):
    """Estimate remaining seconds using stage-weighted historical data.

    Falls back to simple linear extrapolation when no history exists.
    """
    if not history or progress_pct <= 0 or progress_pct >= 100:
        return None

    # --- find similar clips (within 50% of current duration) ---
    lo, hi = clip_seconds * 0.5, clip_seconds * 1.5
    similar = [m for m in history if lo <= m["clip_seconds"] <= hi]

    if similar:
        return _estimate_from_history(
            similar, clip_seconds, current_stage, progress_pct
        )

    # --- no similar clips: linear extrapolation ---
    elapsed_est = clip_seconds * (progress_pct / 100.0)
    if elapsed_est <= 0:
        return None
    remaining = elapsed_est * (100.0 - progress_pct) / progress_pct
    return max(0.0, remaining)


def _estimate_from_history(similar, clip_seconds, current_stage, progress_pct):
    """Use historical stage timings to predict remaining time.

    For each past job:
      - If the job reached the current stage, compute how much time was spent
        in that stage and what fraction of the stage is likely done.
      - Add the full time of all stages after the current one.
    Average across similar jobs.
    """
    progress_frac = progress_pct / 100.0
    stage_order = ["transcribing", "matching", "aligning", "refining", "editing"]

    try:
        cur_idx = stage_order.index(current_stage)
    except ValueError:
        # Unknown stage — fall back to simple extrapolation
        return _simple_extrapolation(clip_seconds, progress_frac)

    ratios = []
    for m in similar:
        stages = m["stages"]
        total = sum(stages.values())
        if total <= 0:
            continue

        # Time remaining = partially-spent current stage + all later stages
        remaining_secs = 0.0

        cur_stage_time = stages.get(current_stage, 0)
        if cur_stage_time > 0:
            # Fraction of this stage elapsed = how far progress is into this
            # stage's portion of the progress bar.
            cur_mid = _STAGE_MIDPOINTS.get(current_stage, progress_frac)
            prev_mid = (
                _STAGE_MIDPOINTS[stage_order[cur_idx - 1]]
                if cur_idx > 0
                else 0.0
            )
            stage_range = max(cur_mid - prev_mid, 0.01)
            stage_frac_done = min(
                1.0, max(0.0, (progress_frac - prev_mid) / stage_range)
            )
            remaining_secs += cur_stage_time * (1.0 - stage_frac_done)

        for s in stage_order[cur_idx + 1:]:
            remaining_secs += stages.get(s, 0)

        if remaining_secs > 0:
            # Scale by how this clip's total time compares to the average
            avg_total = total
            ratios.append(remaining_secs / max(0.01, avg_total))

    if not ratios:
        return _simple_extrapolation(clip_seconds, progress_frac)

    avg_ratio = sum(ratios) / len(ratios)
    return max(0.0, clip_seconds * avg_ratio)


def _simple_extrapolation(clip_seconds, progress_frac):
    if progress_frac <= 0:
        return None
    elapsed_est = clip_seconds * progress_frac
    remaining = elapsed_est * (1.0 - progress_frac) / progress_frac
    return max(0.0, remaining)
