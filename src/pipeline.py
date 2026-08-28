"""pipeline.py - FoulFilter orchestration.

transcribe (GPU) -> optional offset Rescan Pass -> match Candidates ->
align whole Transcript (GPU/CPU) -> confirm Hits -> scoped Smart Cut ->
render Censor Method.

Pure orchestration: engines are injected singletons so unit tests can stub
them and CI never loads models.
"""

import hashlib
import json
import logging
import os
import shutil

import detector
import word_remover
from analyze_file_type import detect_media_type

logger = logging.getLogger("FoulFilter.pipeline")

PRE_PADDING = 0.15
POST_PADDING = 0.25


class JobCancelled(Exception):
    pass


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transcript_path_for(transcript_dir, digest, base_name):
    return os.path.join(transcript_dir, f"{digest}_{base_name}.json")


def find_cached_transcript(transcript_dir, digest):
    if not os.path.isdir(transcript_dir):
        return None
    for name in os.listdir(transcript_dir):
        if name.startswith(digest) and name.endswith(".json"):
            path = os.path.join(transcript_dir, name)
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("file_hash") == digest:
                    return data
            except (OSError, ValueError):
                continue
    return None


def load_bad_words(path):
    with open(path) as f:
        words = f.readlines()
    entries = [w.strip() for w in words]
    if not any(e and not e.startswith("#") for e in entries):
        raise ValueError(f"Bad Words List is empty: {path}")
    return entries


def merge_hits(hits):
    """Pad Hits, drop inversions, merge overlaps into final cut windows."""
    padded = []
    for h in hits:
        if h["end"] <= h["start"]:
            continue
        start = max(0.0, h["start"] - PRE_PADDING)
        end = h["end"] + POST_PADDING
        if end <= start:
            continue
        padded.append({**h, "start": round(start, 3), "end": round(end, 3)})
    padded.sort(key=lambda x: x["start"])

    merged = []
    for hit in padded:
        if merged and hit["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], hit["end"])
            merged[-1]["phrase"] += f"+{hit['phrase']}"
        else:
            merged.append(hit)
    return merged


def _run_job_impl(
    input_path,
    output_path,
    bad_words_list_path,
    transcript_dir,
    scratch_dir,
    censor_method="silence",
    debug=False,
    rescan=False,
    render=True,
    progress=None,
    transcriber=None,
    aligner=None,
    smart_cut=None,
):
    """Process one file end to end. Returns a summary dict.

    progress(stage, percent, detail) is invoked at checkpoints; raising from
    inside it cancels the job via JobCancelled.
    smart_cut(context_words, phrase, center_index, allow_widening)
        -> None|False|{cut_start, cut_end}
    """
    report = lambda *a: progress(*a) if progress else None  # noqa: E731
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    media_type = detect_media_type(input_path)
    if media_type == "unknown":
        raise ValueError(f"Unrecognized file type: {input_path}")

    # Smart Cut widening only ever makes sense for jump-cut removals on audio
    # (ADR-0004): silence/bleep gaps and video edits stay surgical.
    allow_widening = censor_method == "remove" and media_type == "audio"

    # Video: analyze the extracted audio track.
    analysis_source = input_path
    if media_type == "video":
        report("preparing", 2, "Extracting audio track")
        analysis_source = os.path.join(scratch_dir, "extracted_audio.m4a")
        _extract_audio(input_path, analysis_source)

    bad_words = load_bad_words(bad_words_list_path)

    # --- Transcript: load cached or transcribe on GPU ---
    digest = file_hash(input_path)
    base_name = _safe(os.path.splitext(os.path.basename(input_path))[0][:80])
    cached = find_cached_transcript(transcript_dir, digest)
    if cached and not rescan:
        report("transcribing", 45, "Reusing persisted transcript")
        transcript = cached
    else:
        if transcriber is None:
            from transcriber import get_transcribers
            transcribers = get_transcribers()
        else:
            transcribers = [transcriber]
        gpu_label = f"{len(transcribers)} GPU(s)" if len(transcribers) > 1 else "GPU"
        report("transcribing", 5, f"{gpu_label} transcription started")
        from transcriber import transcribe_parallel
        segments = transcribe_parallel(analysis_source, transcribers, scratch_dir)
        transcript = {
            "version": 3,
            "file_hash": digest,
            "segments": segments,
        }
        report("transcribing", 40, f"{len(segments)} segments transcribed")

        if rescan:
            report("transcribing", 42, "Rescan pass (offset boundaries)")
            shifted = transcribers[0].transcribe_shifted(analysis_source)
            transcript["segments"] = _union_segments(
                transcript["segments"], shifted
            )
            logger.info(
                "Rescan grew transcript to %d segments",
                len(transcript["segments"]),
            )

        # Checkpoint: persist segments immediately. Alignment is expensive and
        # may be skipped or fail; saving now means a later Resume reuses the
        # transcription and only redoes matching/alignment, not the GPU pass.
        _persist_transcript(
            transcript, transcript_dir, digest, base_name, logger
        )

    # --- Match candidates against segment text ---
    report("matching", 50, "Matching Bad Words List")
    candidates = detector.find_candidates(transcript["segments"], bad_words)
    logger.info("%d candidate(s)", len(candidates))

    # --- Align the whole transcript whenever anything was flagged ---
    words = list(transcript.get("words") or [])
    if candidates and not words:
        if aligner is None:
            from aligner import get_aligners
            aligners = get_aligners()
        else:
            aligners = [aligner]

        def align_progress(done, total):
            pct = 55 + int(20 * done / max(1, total))
            report("aligning", pct, f"Alignment batch {done}/{total}")

        report("aligning", 55, "Forced alignment started")
        from aligner import align_transcript_parallel
        words = align_transcript_parallel(
            analysis_source, transcript["segments"], aligners, align_progress
        )

    # Persist refined transcript for Resume (segments + aligned words).
    transcript["words"] = words
    _persist_transcript(transcript, transcript_dir, digest, base_name, logger)

    # --- Confirm hits (precise times when aligned, estimates otherwise) ---
    # With best-effort alignment, some candidates may lack precise word
    # timestamps. Match each candidate occurrence to aligned hits by
    # (phrase, approximate position); fall back to segment estimates for any
    # occurrence without an aligned match, so nothing flagged is dropped.
    hits = detector.find_hits(words, bad_words) if words else []

    def _hit_covers(cand, hit):
        # A hit covers a candidate if same phrase and spans overlap loosely.
        return hit["phrase"] == cand["phrase"] and not (
            hit["end"] < cand["approx_start"]
            or hit["start"] > cand["approx_end"]
        )

    covered = [False] * len(candidates)
    for hidx, hit in enumerate(hits):
        for cidx, cand in enumerate(candidates):
            if not covered[cidx] and _hit_covers(cand, hit):
                covered[cidx] = True
                break
    for cidx, cand in enumerate(candidates):
        if covered[cidx]:
            continue
        logger.warning(
            "No aligned timestamps for '%s'; using segment estimate", cand["phrase"]
        )
        hits.append(
            {
                "phrase": cand["phrase"],
                "start": cand["approx_start"],
                "end": cand["approx_end"],
            }
        )
    hits = merge_hits(hits)
    logger.info("%d hit(s) after merge", len(hits))

    # --- Smart Cut refinement ---
    if smart_cut and words and hits:
        for i, hit in enumerate(hits):
            report(
                "refining",
                78 + int(10 * i / max(1, len(hits))),
                f"Smart Cut {i + 1}/{len(hits)}",
            )
            context, center = _context_window(words, hit)
            try:
                result = smart_cut(
                    context,
                    hit["phrase"],
                    center_index=center,
                    allow_widening=allow_widening,
                )
            except Exception as exc:  # noqa: BLE001 - AI failures must not kill jobs
                logger.warning("Smart Cut failed: %s", exc)
                continue
            if result is False:
                hit["skip"] = True
            elif result:
                hit["start"] = max(0.0, result["cut_start"] - PRE_PADDING)
                hit["end"] = result["cut_end"] + POST_PADDING
        kept = [h for h in hits if not h.pop("skip", False)]
        if len(kept) != len(hits):
            logger.info("Smart Cut rejected %d hit(s)", len(hits) - len(kept))
            hits = merge_hits(kept)

    if debug:
        txt = os.path.join(scratch_dir, "transcript.txt")
        with open(txt, "w") as f:
            tokens = [w["word"] for w in words] if words else [
                t
                for s in transcript["segments"]
                for t in detector.tokenize(s["text"])
            ]
            for i, tok in enumerate(tokens):
                f.write(tok + ("\n" if (i + 1) % 20 == 0 else " "))
        report("editing", 90, "Debug transcript written")

    # --- Render ---
    if not render:
        report("completed", 100, "Analysis complete (no edit requested)")
        return {
            "hits": [
                {"phrase": h["phrase"], "start": h["start"], "end": h["end"]}
                for h in hits
            ],
            "transcript_words": len(words),
            "cached_transcript": bool(cached),
            "rescanned": bool(rescan),
        }

    report("editing", 88, f"Rendering {censor_method} edit")
    if not hits:
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            shutil.copyfile(input_path, output_path)
    elif media_type == "video":
        effective = "silence" if censor_method == "remove" else censor_method
        word_remover.censor_video(input_path, hits, effective, output_path)
    else:
        word_remover.censor_audio(input_path, hits, censor_method, output_path)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Output missing after render: {output_path}")

    report("completed", 100, "Finished")
    return {
        "hits": [
            {"phrase": h["phrase"], "start": h["start"], "end": h["end"]}
            for h in hits
        ],
        "transcript_words": len(words),
        "cached_transcript": bool(cached),
        "rescanned": bool(rescan),
    }


def run_job(*args, **kwargs):
    """Public entry: guarantees GPU models are dropped after the job when
    UNLOAD_MODELS_AFTER_JOB is enabled, so llama-server / other apps can
    reclaim VRAM between jobs."""
    try:
        return _run_job_impl(*args, **kwargs)
    finally:
        _release_models_if_configured()


def _release_models_if_configured():
    flag = (os.getenv("UNLOAD_MODELS_AFTER_JOB") or "True").strip().lower()
    if flag not in ("1", "true", "yes"):
        return
    try:
        from aligner import release_aligner
        from transcriber import release_transcriber

        release_transcriber()
        release_aligner()
        logger.info("GPU models unloaded after job.")
    except Exception:
        logger.debug("Model release skipped", exc_info=True)


def _extract_audio(video_path, out_path):
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
            "-vn", "-acodec", "aac", out_path,
        ],
        check=True,
    )


def _persist_transcript(transcript, transcript_dir, digest, base_name, logger=None):
    """Write/overwrite the persisted transcript for Resume.

    Called right after transcription (segments-only checkpoint) and again
    after alignment (segments + words). Multiple calls are safe: the final
    aligned version wins, and the file hash guards against partial writes
    being reused (find_cached_transcript verifies file_hash).
    """
    path = transcript_path_for(transcript_dir, digest, base_name)
    os.makedirs(transcript_dir, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(transcript, f)
        os.replace(tmp, path)
    except OSError as exc:
        if logger:
            logger.warning("Could not persist transcript to %s: %s", path, exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _union_segments(primary, extra):
    """Union two segment lists without duplicating identical spans."""
    merged = sorted(primary + extra, key=lambda s: s["start"])
    out = []
    for seg in merged:
        if out and seg["text"] == out[-1]["text"] and (
            seg["start"] < out[-1]["end"]
        ):
            continue
        out.append(seg)
    return out


def _context_window(words, hit, size=11):
    """Words surrounding a Hit plus the target's index inside that window."""
    center = min(
        range(len(words)),
        key=lambda i: abs(words[i]["start"] - hit["start"]),
    )
    lo = max(0, center - size)
    hi = min(len(words), center + size + 1)
    return words[lo:hi], center - lo
