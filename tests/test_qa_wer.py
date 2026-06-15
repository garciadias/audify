"""Unit tests for the cycle-3 WER helpers (issue #38)."""

from __future__ import annotations

import pytest

from audify.qa.wer import comparable_reference, normalize_words, word_error_rate


class TestNormalizeWords:
    def test_lowercases_and_splits(self):
        assert normalize_words("The Quick Brown Fox") == [
            "the",
            "quick",
            "brown",
            "fox",
        ]

    def test_strips_punctuation_but_keeps_contractions(self):
        assert normalize_words("Don't stop, now!") == ["don't", "stop", "now"]

    def test_strips_edge_apostrophes(self):
        assert normalize_words("'quoted' words") == ["quoted", "words"]

    def test_empty(self):
        assert normalize_words("") == []
        assert normalize_words("   ") == []


class TestWordErrorRate:
    def test_identical_is_zero(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_punctuation_and_case_ignored(self):
        assert word_error_rate("Hello, World!", "hello world") == 0.0

    def test_completely_wrong_is_one(self):
        assert word_error_rate("alpha beta gamma", "x y z") == 1.0

    def test_single_substitution(self):
        # one of three words wrong → 1/3
        assert word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)

    def test_deletion(self):
        # reference has 4 words, hypothesis drops one → 1/4
        assert word_error_rate("a b c d", "a b c") == pytest.approx(1 / 4)

    def test_both_empty_is_zero(self):
        assert word_error_rate("", "") == 0.0

    def test_empty_reference_nonempty_hyp_is_one(self):
        assert word_error_rate("", "spurious words") == 1.0

    def test_clamped_to_one(self):
        # many insertions cannot push WER above 1.0
        assert word_error_rate("a", "a b c d e f g") == 1.0


class TestComparableReference:
    def test_head_matches_hypothesis_length(self):
        script = "one two three four five six"
        ref = comparable_reference(script, "ONE two!", side="head")
        assert ref == "one two"

    def test_tail_matches_hypothesis_length(self):
        script = "one two three four five six"
        ref = comparable_reference(script, "five six", side="tail")
        assert ref == "five six"

    def test_empty_hypothesis_returns_empty(self):
        assert comparable_reference("a b c", "", side="head") == ""

    def test_empty_script_returns_empty(self):
        assert comparable_reference("", "a b", side="tail") == ""

    def test_invalid_side(self):
        with pytest.raises(ValueError, match="side must be"):
            comparable_reference("a b", "a", side="middle")

    def test_intact_tail_scores_low_wer(self):
        """A length-matched intact tail window scores ~0 WER."""
        script = "the chapter ends with these final closing words"
        tail_transcript = "these final closing words"
        ref = comparable_reference(script, tail_transcript, side="tail")
        assert word_error_rate(ref, tail_transcript) == 0.0

    def test_truncated_tail_scores_high_wer(self):
        """A truncated episode's tail window transcribes earlier content."""
        script = "the chapter ends with these final closing words"
        # Audio was cut short, so the "tail" window actually holds mid-chapter
        # speech that does not match the script's closing words.
        tail_transcript = "ends with these"
        ref = comparable_reference(script, tail_transcript, side="tail")
        assert word_error_rate(ref, tail_transcript) > 0.3
