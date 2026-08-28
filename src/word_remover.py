"""word_remover.py - ffmpeg-based Censor Method rendering.

silence: zero-volume over each Hit window
bleep:   1 kHz sine mixed over muted Hit windows
remove:  audio cut out (audio files only; video falls back to silence)
"""

import logging
import os
import subprocess

import ffmpeg

logger = logging.getLogger("FoulFilter.word_remover")


def _probe_sample_rate(path):
    probe = ffmpeg.probe(path)
    stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    if not stream:
        raise ValueError(f"No audio stream found in {path}")
    return int(stream["sample_rate"])


def _probe_audio_stream(path):
    """Return the first audio stream's format dict from ffprobe."""
    probe = ffmpeg.probe(path)
    stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    if not stream:
        raise ValueError(f"No audio stream found in {path}")
    return stream


def _audio_encoder_args(path):
    """ffmpeg args that preserve the source audio's codec/bitrate/layout.

    Sub-second censor edits (volume/atrim/concat/sine) force a re-encode; true
    -c:a copy is impossible for mid-packet cuts (see ADR-0006). To avoid the
    drastic size shrink of ffmpeg's default encoder settings, we re-encode with
    the source's codec, sample rate, channels and bitrate so the output
    size/quality closely matches the input.
    """
    stream = _probe_audio_stream(path)
    args = []

    codec = stream.get("codec_name")
    if codec:
        codec_map = {
            "mp3": "libmp3lame",
            "aac": "aac",
            "flac": "flac",
            "opus": "libopus",
            "vorbis": "libvorbis",
            "pcm_s16le": "pcm_s16le",
            "pcm_s24le": "pcm_s24le",
            "pcm_s32le": "pcm_s32le",
        }
        args += ["-c:a", codec_map.get(codec, codec)]

    bit_rate = stream.get("bit_rate")
    if bit_rate:
        args += ["-b:a", str(bit_rate)]

    sample_rate = stream.get("sample_rate")
    if sample_rate:
        args += ["-ar", str(sample_rate)]

    channels = stream.get("channels")
    if channels:
        args += ["-ac", str(channels)]

    return args


def _volume_chain(hits):
    """volume=0 filters gated by between(t,start,end), comma-joined."""
    return ",".join(
        f"volume=enable='between(t,{h['start']},{h['end']})':volume=0"
        for h in hits
    )


def build_silence_filter(hits):
    if not hits:
        raise ValueError("No hits provided")
    return _volume_chain(hits)


def build_bleep_filter(hits, sample_rate=44100):
    """Mute the original over hit windows and mix delayed 1 kHz sines on top.

    Everything is normalized to 44.1 kHz stereo so amix sees uniform streams;
    normalize=0 keeps the untouched audio at full volume.
    """
    if not hits:
        raise ValueError("No hits provided")

    graph = (
        "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,"
        + _volume_chain(hits)
        + "[base]"
    )
    mix_labels = ["[base]"]
    for i, h in enumerate(hits):
        duration = max(0.05, h["end"] - h["start"])
        delay_ms = int(h["start"] * 1000)
        graph += (
            f";sine=frequency=1000:duration={duration:.3f}[s{i}]"
            f";[s{i}]adelay={delay_ms}|{delay_ms}[b{i}]"
        )
        mix_labels.append(f"[b{i}]")
    graph += (
        f";{''.join(mix_labels)}"
        f"amix=inputs={len(mix_labels)}:duration=first:normalize=0[out]"
    )
    return graph


def _run(cmd):
    logger.debug("ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _probe_duration(path):
    probe = ffmpeg.probe(path)
    return float(probe["format"]["duration"])


def censor_audio(input_path, hits, method="silence", output_path=None):
    """Render a censored copy of an audio file; returns output_path."""
    output_path = output_path or _default_out(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if method == "remove":
        graph = build_remove_filter(hits, total_duration=_probe_duration(input_path))
        _run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", input_path,
                "-filter_complex", graph,
                "-map", "[out]",
                *_audio_encoder_args(input_path),
                output_path,
            ]
        )
    elif method == "bleep":
        _run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", input_path,
                "-filter_complex", build_bleep_filter(hits),
                "-map", "[out]",
                *_audio_encoder_args(input_path),
                output_path,
            ]
        )
    else:
        _run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", input_path,
                "-filter_complex", f"[0:a]{build_silence_filter(hits)}[aout]",
                "-map", "[aout]",
                *_audio_encoder_args(input_path),
                output_path,
            ]
        )
    return output_path


def censor_video(input_path, hits, method="silence", output_path=None):
    """Censor a video's audio track while keeping video frames untouched."""
    output_path = output_path or _default_out(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if method == "bleep":
        # Bleep needs generated audio sources: render a censored track first.
        tmp_audio = output_path + ".bleep_track.m4a"
        try:
            _run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", input_path, "-vn",
                    "-filter_complex", build_bleep_filter(hits),
                    "-map", "[out]",
                    *_audio_encoder_args(input_path),
                    tmp_audio,
                ]
            )
            from video_edit import replace_audio_in_video

            replace_audio_in_video(input_path, tmp_audio, output_path)
        finally:
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)
        return output_path

    # silence / remove-fallback: filter audio, stream-copy video, keep audio codec
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_path,
            "-filter_complex",
            f"[0:a]{build_silence_filter(hits)}[aout]",
            "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
            *_audio_encoder_args(input_path),
            output_path,
        ]
    )
    return output_path


def build_remove_filter(hits, total_duration=None):
    """Concat every span outside Hits; removes them entirely."""
    if not hits:
        raise ValueError("No hits provided")
    ordered = sorted(hits, key=lambda h: h["start"])

    parts = []
    concat_refs = ""
    cursor = 0.0
    n = 0
    for h in ordered:
        if h["start"] > cursor:
            parts.append(
                f"[0:a]atrim=start={cursor:.3f}:end={h['start']:.3f},"
                f"asetpts=PTS-STARTPTS[clip{n}]"
            )
            concat_refs += f"[clip{n}]"
            n += 1
        cursor = max(cursor, h["end"])

    has_tail = total_duration is None or cursor < total_duration - 0.001
    if has_tail:
        parts.append(
            f"[0:a]atrim=start={cursor:.3f},asetpts=PTS-STARTPTS[clip{n}]"
        )
        concat_refs += f"[clip{n}]"
        n += 1

    if n == 0:
        raise ValueError("Hits cover the entire file; nothing would remain")

    graph = ";".join(parts) + f";{concat_refs}concat=n={n}:v=0:a=1[out]"
    return graph


def _default_out(input_path):
    base, ext = os.path.splitext(input_path)
    return f"{base}_CENSORED{ext}"
