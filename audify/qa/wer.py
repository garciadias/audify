"""Word Error Rate (WER) helpers for the boundary-sampling fidelity check.

The cycle-3 fidelity detector (issue #38) compares a short STT transcript of an
episode's head/tail window against the corresponding opening/closing words of the
*script* that was fed to TTS. A high WER on either window indicates the chapter was
likely truncated.

Implemented with a small word-level Levenshtein distance — no extra dependency.
"""

from __future__ import annotations

import re

__all__ = ["normalize_words", "word_error_rate", "comparable_reference"]

# Keep apostrophes inside words ("don't"), drop everything else that is not a
# word character or whitespace, then split on whitespace.
_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)
_APOSTROPHE_EDGE_RE = re.compile(r"(^'+|'+$)")


def normalize_words(text: str) -> list[str]:
    """Lowercase *text*, strip punctuation, and split into comparable words.

    Apostrophes are preserved word-internally so contractions survive, but
    stripped from word edges. Returns an empty list for empty / whitespace text.
    """
    if not text:
        return []
    lowered = _PUNCT_RE.sub(" ", text.lower())
    words = []
    for token in lowered.split():
        token = _APOSTROPHE_EDGE_RE.sub("", token)
        if token:
            words.append(token)
    return words


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein edit distance between two token lists."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return the word error rate of *hypothesis* against *reference*.

    ``WER = edit_distance(reference_words, hypothesis_words) / len(reference_words)``,
    clamped to ``[0.0, 1.0]``.

    Edge cases:
      * empty reference **and** empty hypothesis → ``0.0`` (nothing to carry,
        nothing carried — a faithful match).
      * empty reference, non-empty hypothesis → ``1.0`` (total mismatch).
    """
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    distance = _levenshtein(ref_words, hyp_words)
    return min(distance / len(ref_words), 1.0)


def comparable_reference(script: str, hypothesis: str, *, side: str) -> str:
    """Slice the head/tail of *script* to a word span comparable to *hypothesis*.

    Boundary windows transcribe only a few seconds of audio, so the reference
    must be length-matched to the transcript — otherwise an intact window scores
    a high WER purely from length mismatch. Returns the first (``side="head"``)
    or last (``side="tail"``) ``len(hypothesis_words)`` words of the script,
    normalized. Returns ``""`` when either side is empty.
    """
    if side not in ("head", "tail"):
        raise ValueError(f"side must be 'head' or 'tail', got {side!r}")

    ref_words = normalize_words(script)
    n = len(normalize_words(hypothesis))
    if n == 0 or not ref_words:
        return ""

    span = ref_words[:n] if side == "head" else ref_words[-n:]
    return " ".join(span)
