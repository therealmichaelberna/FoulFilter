"""detector.py - match Bad Words List entries against transcript tokens.

Matching is case- and punctuation-insensitive. Entries may be multi-word
phrases up to three words. Works on both segment text (Candidates with
approximate timestamps) and aligned word lists (Hits with precise ones).
"""

import re
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9']+")
MAX_PHRASE_WORDS = 3


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def normalize_entries(bad_words: Iterable[str]):
    """Turn raw list lines into a set of token-tuples (1..3 words)."""
    entries = set()
    for line in bad_words:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = tuple(tokenize(line))
        if 0 < len(toks) <= MAX_PHRASE_WORDS:
            entries.add(toks)
    return entries


def _match_tokens(tokens, entries):
    """Yield (start_idx, end_idx_exclusive, phrase) for every match."""
    matches = []
    n = len(tokens)
    max_n = min(MAX_PHRASE_WORDS, n)
    for size in range(max_n, 0, -1):  # longest phrases first
        for i in range(0, n - size + 1):
            gram = tuple(tokens[i : i + size])
            if len(gram) == 1 and not gram[0]:
                continue
            if gram in entries:
                matches.append((i, i + size, " ".join(gram)))
    return matches


def _dedupe(matches):
    """Drop matches fully contained in an earlier (longer/lefter) one."""
    kept = []
    for start, end, phrase in sorted(matches, key=lambda m: (m[0], -(m[1] - m[0]))):
        if any(start >= s and end <= e for s, e, _ in kept):
            continue
        kept.append((start, end, phrase))
    return kept


def find_candidates(segments, bad_words):
    """Locate Candidates in segment text.

    Returns [{phrase, seg_index, approx_start, approx_end}] where approximate
    times are linearly interpolated across the segment duration.
    """
    entries = normalize_entries(bad_words)
    candidates = []
    for idx, seg in enumerate(segments):
        text = seg.get("text", "")
        tokens = tokenize(text)
        spans = []
        pos = 0
        low = text.lower()
        for tok in tokens:  # char span of each token, for time interpolation
            found = low.find(tok, pos)
            spans.append((found, found + len(tok)))
            pos = found + len(tok)

        hits = _dedupe(_match_tokens(tokens, entries))
        dur = seg["end"] - seg["start"]
        total_chars = max(1, len(text))
        for start_i, end_i, phrase in hits:
            c_start = spans[start_i][0]
            c_end = spans[end_i - 1][1]
            approx_start = seg["start"] + dur * (c_start / total_chars)
            approx_end = seg["start"] + dur * (c_end / total_chars)
            candidates.append(
                {
                    "phrase": phrase,
                    "seg_index": idx,
                    "approx_start": round(approx_start, 3),
                    "approx_end": round(approx_end, 3),
                }
            )
    return candidates


def find_hits(words, bad_words):
    """Same matching over an aligned word list.

    Returns [{phrase, word_index, start, end}] with precise timestamps.
    """
    entries = normalize_entries(bad_words)
    tokens = [w["word"] for w in words]
    hits = []
    for start_i, end_i, phrase in _dedupe(_match_tokens(tokens, entries)):
        hits.append(
            {
                "phrase": phrase,
                "word_index": start_i,
                "start": words[start_i]["start"],
                "end": words[end_i - 1]["end"],
            }
        )
    return hits
