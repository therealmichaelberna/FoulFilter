"""Unit tests for pure pipeline logic - no models, no GPU, no ffmpeg runs."""

import pytest

import detector
from ai_helper import apply_smart_cut_indices
from main import sanitize_filename
from word_remover import build_bleep_filter, build_remove_filter, build_silence_filter


# --- detector ---------------------------------------------------------------

def test_single_word_match_ignores_case_and_punctuation():
    segments = [{"start": 0.0, "end": 2.0, "text": "Well, Damn. that's bad"}]
    cands = detector.find_candidates(segments, ["damn"])
    assert len(cands) == 1
    assert cands[0]["phrase"] == "damn"
    assert 0 < cands[0]["approx_start"] < cands[0]["approx_end"] < 2.0


def test_multiword_phrase_matches_across_tokens():
    segments = [{"start": 10.0, "end": 14.0, "text": "he can go to hell for all I care"}]
    cands = detector.find_candidates(segments, ["go to hell"])
    assert len(cands) == 1
    assert cands[0]["phrase"] == "go to hell"


def test_apostrophes_survive_tokenization():
    segments = [{"start": 0.0, "end": 1.0, "text": "you son of a bitch"}]
    assert len(detector.find_candidates(segments, ["bitch"])) == 1


def test_no_false_positive_substrings():
    segments = [{"start": 0.0, "end": 1.0, "text": "classify the class"}]
    assert detector.find_candidates(segments, ["ass"]) == []


def test_find_hits_returns_word_spans():
    words = [
        {"word": "go", "start": 1.0, "end": 1.2},
        {"word": "to", "start": 1.2, "end": 1.4},
        {"word": "hell", "start": 1.4, "end": 1.9},
        {"word": "friend", "start": 1.9, "end": 2.4},
    ]
    hits = detector.find_hits(words, ["hell"])
    assert hits == [{"phrase": "hell", "word_index": 2, "start": 1.4, "end": 1.9}]


def test_comments_and_blank_lines_are_skipped():
    entries = detector.normalize_entries(["damn", "", "# comment", "hell"])
    assert entries == {("damn",), ("hell",)}


# --- merge_hits (padding + overlap merge) ------------------------------------

def _hit(start, end, phrase="x"):
    return {"phrase": phrase, "start": start, "end": end}


def test_merge_pads_and_merges_overlaps():
    from pipeline import merge_hits

    merged = merge_hits([_hit(1.0, 1.5), _hit(1.55, 2.0)])
    # padding: -0.15/+0.25 makes these two windows touch
    assert len(merged) == 1
    assert merged[0]["start"] == pytest.approx(0.85)
    assert merged[0]["end"] == pytest.approx(2.25)
    assert "+" in merged[0]["phrase"]


def test_merge_drops_inverted_windows_and_keeps_order():
    from pipeline import merge_hits

    merged = merge_hits([_hit(5.0, 5.2), _hit(9.0, 9.0), _hit(3.0, 3.6)])
    assert [h["start"] for h in merged] == [pytest.approx(2.85), pytest.approx(4.85)]
    assert all(h["end"] > h["start"] for h in merged)


def test_merge_clamps_negative_start():
    from pipeline import merge_hits

    merged = merge_hits([_hit(0.02, 0.4)])
    assert merged[0]["start"] == 0.0


# --- ffmpeg filtergraph builders ---------------------------------------------

def test_silence_filter_chains_volume_gates():
    graph = build_silence_filter([_hit(1.0, 2.0), _hit(4.0, 5.0)])
    assert graph == (
        "volume=enable='between(t,1.0,2.0)':volume=0,"
        "volume=enable='between(t,4.0,5.0)':volume=0"
    )


def test_silence_render_is_labeled_single_stream(tmp_path):
    # Regression: an unlabeled filter_complex output got auto-mapped next to
    # -map 0:a and the wav muxer rejected the second stream.
    import subprocess

    src = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1", str(src)],
        check=True,
    )
    out = tmp_path / "out.wav"
    from word_remover import censor_audio

    censor_audio(str(src), [_hit(0.2, 0.5)], method="silence", output_path=str(out))
    assert out.exists() and out.stat().st_size > 1000


def test_bleep_graph_mixes_delayed_sines_over_muted_base():
    graph = build_bleep_filter([_hit(2.0, 2.7)])
    assert "[0:a]aformat=sample_rates=44100:channel_layouts=stereo," in graph
    assert "volume=enable='between(t,2.0,2.7)':volume=0[base]" in graph
    assert "sine=frequency=1000:duration=0.700[s0]" in graph
    assert "[s0]adelay=2000|2000[b0]" in graph
    assert "[base][b0]amix=inputs=2:duration=first:normalize=0[out]" in graph


def test_remove_filter_trims_between_hits():
    graph = build_remove_filter([_hit(1.0, 2.0), _hit(3.0, 4.0)])
    assert "atrim=start=0.000:end=1.000" in graph
    assert "atrim=start=2.000:end=3.000" in graph
    assert "atrim=start=4.000" in graph
    assert "concat=n=3:v=0:a=1[out]" in graph


def test_builders_reject_empty_hits():
    with pytest.raises(ValueError):
        build_silence_filter([])
    with pytest.raises(ValueError):
        build_bleep_filter([])
    with pytest.raises(ValueError):
        build_remove_filter([])


# --- smart cut index mapping ---------------------------------------------------

WINDOW = [
    {"word": "just", "start": 5.0, "end": 5.2},
    {"word": "tell", "start": 5.2, "end": 5.4},
    {"word": "him", "start": 5.4, "end": 5.5},
    {"word": "to", "start": 5.5, "end": 5.6},
    {"word": "go", "start": 5.6, "end": 5.8},
    {"word": "to", "start": 5.8, "end": 5.9},
    {"word": "hell", "start": 5.9, "end": 6.2},
]


def test_widening_maps_phrase_span():
    out = apply_smart_cut_indices(
        {"start_index": 1, "end_index": 6}, WINDOW, center_index=6,
        allow_widening=True,
    )
    assert out["cut_start"] == pytest.approx(5.2)
    assert out["cut_end"] == pytest.approx(6.2)


def test_surgical_mode_clamps_to_target():
    out = apply_smart_cut_indices(
        {"start_index": 1, "end_index": 6}, WINDOW, center_index=6,
        allow_widening=False,
    )
    # ADR-0004: silence/bleep/video never widen beyond the target word.
    assert out["cut_start"] == pytest.approx(5.9)
    assert out["cut_end"] == pytest.approx(6.2)


def test_minus_one_means_skip():
    out = apply_smart_cut_indices(
        {"start_index": -1, "end_index": -1}, WINDOW, center_index=3,
        allow_widening=True,
    )
    assert out is False


def test_out_of_bounds_indices_clamped():
    out = apply_smart_cut_indices(
        {"start_index": -5, "end_index": 99}, WINDOW, center_index=0,
        allow_widening=True,
    )
    assert out["cut_start"] == pytest.approx(WINDOW[0]["start"])
    assert out["cut_end"] == pytest.approx(WINDOW[-1]["end"])


# --- upload sanitization --------------------------------------------------------

def test_sanitize_filename_strips_paths_and_brackets():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename('book: "chapter [1]" .mp3') == 'book - chapter 1.mp3'
    assert sanitize_filename("") == "file"
    assert sanitize_filename("..") == "file"


def test_sanitize_filename_lowercases_extension_only():
    name = sanitize_filename("My Audio.File.MP3")
    assert name.endswith(".mp3")
