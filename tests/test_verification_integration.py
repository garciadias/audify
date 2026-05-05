#!/usr/bin/env python3
"""
Unit tests for the verification integration module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from audify.verification_integration import (
    ChapterDurationChecker,
    AudiobookVerificationCheck,
    VerificationPrompts,
    check_chapter_during_synthesis,
    verify_complete_audiobook,
)


class TestChapterDurationChecker:
    """Test chapter duration checking."""

    def test_estimate_duration_basic(self):
        """Test duration estimation from word count."""
        # 1500 words at 75 wpm = 20 minutes = 1200 seconds
        duration = ChapterDurationChecker.estimate_duration(1500)
        assert duration == pytest.approx(1200, rel=0.1)

    def test_estimate_duration_short(self):
        """Test duration estimation for short chapters."""
        # 10 words should still give at least 0.5 minutes
        duration = ChapterDurationChecker.estimate_duration(10)
        assert duration >= 30

    def test_get_actual_duration(self):
        """Test actual duration retrieval."""
        mock_path = Path("/test/audio.mp3")
        with patch("audify.verification_integration.AudioProcessor.get_duration") as mock_get:
            mock_get.return_value = 1200.0
            duration = ChapterDurationChecker.get_actual_duration(mock_path)
            assert duration == 1200.0

    def test_check_chapter_good_ratio(self):
        """Test chapter check with good duration ratio."""
        mock_path = Path("/test/episode.mp3")
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(ChapterDurationChecker, "get_actual_duration", return_value=1000):
                is_ok, message, ratio = ChapterDurationChecker.check_chapter(
                    chapter_number=1,
                    chapter_title="Test Chapter",
                    audio_path=mock_path,
                    script_word_count=1200,  # expect ~960 seconds
                    threshold=0.7,
                )
        
        assert is_ok is True
        assert ratio > 0.7
        assert "✅" in message

    def test_check_chapter_short_ratio(self):
        """Test chapter check with short duration ratio."""
        mock_path = Path("/test/episode.mp3")
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(ChapterDurationChecker, "get_actual_duration", return_value=500):
                is_ok, message, ratio = ChapterDurationChecker.check_chapter(
                    chapter_number=1,
                    chapter_title="Test Chapter",
                    audio_path=mock_path,
                    script_word_count=2000,  # expect ~1600 seconds
                    threshold=0.7,
                )
        
        assert is_ok is False
        assert ratio < 0.7
        assert "⚠️" in message
        assert "SHORTER" in message

    def test_check_chapter_file_not_found(self):
        """Test chapter check when audio file doesn't exist."""
        mock_path = Path("/nonexistent/episode.mp3")
        
        is_ok, message, ratio = ChapterDurationChecker.check_chapter(
            chapter_number=1,
            chapter_title="Test Chapter",
            audio_path=mock_path,
            script_word_count=1000,
            threshold=0.7,
        )
        
        assert is_ok is False
        assert ratio == 0.0
        assert "not found" in message.lower()


class TestAudiobookVerificationCheck:
    """Test audiobook verification checking."""

    @patch("audify.verification_integration.AudiobookVerifier")
    def test_verify_audiobook_passed(self, mock_verifier_class):
        """Test audiobook verification when passed."""
        # Mock the verifier and report
        mock_verifier = MagicMock()
        mock_verifier_class.return_value = mock_verifier
        
        mock_report = MagicMock()
        mock_report.has_missing_chapters.return_value = False
        mock_report.has_extra_chapters.return_value = False
        mock_report.has_order_issues.return_value = False
        mock_report.duration_hint = MagicMock(ratio=0.95)
        
        mock_verifier.verify.return_value = mock_report
        mock_verifier.generate_report.return_value = {
            "summary": {"matched": 6},
        }
        
        passed, report = AudiobookVerificationCheck.verify_audiobook(
            Path("/test/source.epub"),
            Path("/test/audiobook.m4b"),
        )
        
        assert passed is True
        assert report["verification_passed"] is True
        assert len(report.get("issues", [])) == 0

    @patch("audify.verification_integration.AudiobookVerifier")
    def test_verify_audiobook_missing_chapters(self, mock_verifier_class):
        """Test audiobook verification with missing chapters."""
        mock_verifier = MagicMock()
        mock_verifier_class.return_value = mock_verifier
        
        mock_missing = MagicMock()
        mock_missing.title = "Missing Chapter"
        
        mock_report = MagicMock()
        mock_report.has_missing_chapters.return_value = True
        mock_report.missing = [mock_missing]
        mock_report.has_extra_chapters.return_value = False
        mock_report.has_order_issues.return_value = False
        mock_report.duration_hint = MagicMock(ratio=0.95)
        
        mock_verifier.verify.return_value = mock_report
        mock_verifier.generate_report.return_value = {}
        
        passed, report = AudiobookVerificationCheck.verify_audiobook(
            Path("/test/source.epub"),
            Path("/test/audiobook.m4b"),
        )
        
        assert passed is False
        assert report["verification_passed"] is False
        assert len(report["issues"]) > 0
        assert "Missing" in report["issues"][0]

    @patch("audify.verification_integration.AudiobookVerifier")
    def test_verify_audiobook_short_duration(self, mock_verifier_class):
        """Test audiobook verification with short duration."""
        mock_verifier = MagicMock()
        mock_verifier_class.return_value = mock_verifier
        
        mock_report = MagicMock()
        mock_report.has_missing_chapters.return_value = False
        mock_report.has_extra_chapters.return_value = False
        mock_report.has_order_issues.return_value = False
        mock_report.duration_hint = MagicMock(
            ratio=0.5,
            actual_duration_s=5000,
            expected_duration_s=10000,
        )
        
        mock_verifier.verify.return_value = mock_report
        mock_verifier.generate_report.return_value = {}
        
        passed, report = AudiobookVerificationCheck.verify_audiobook(
            Path("/test/source.epub"),
            Path("/test/audiobook.m4b"),
            duration_ratio_threshold=0.6,
        )
        
        assert passed is False
        assert report["verification_passed"] is False
        assert any("duration" in issue.lower() for issue in report["issues"])


class TestVerificationPrompts:
    """Test user prompts for verification issues."""

    def test_prompt_short_chapter_above_threshold(self):
        """Test no prompt needed when ratio is above threshold."""
        result = VerificationPrompts.prompt_short_chapter(
            chapter_number=1,
            chapter_title="Test",
            ratio=0.8,
            threshold=0.7,
        )
        assert result is True

    @patch("builtins.input", return_value="y")
    def test_prompt_short_chapter_user_accepts(self, mock_input):
        """Test user accepting short chapter."""
        result = VerificationPrompts.prompt_short_chapter(
            chapter_number=1,
            chapter_title="Test",
            ratio=0.5,
            threshold=0.7,
        )
        assert result is True
        assert mock_input.called

    @patch("builtins.input", return_value="n")
    def test_prompt_short_chapter_user_rejects(self, mock_input):
        """Test user rejecting short chapter."""
        result = VerificationPrompts.prompt_short_chapter(
            chapter_number=1,
            chapter_title="Test",
            ratio=0.5,
            threshold=0.7,
        )
        assert result is False


class TestCheckChapterDuringSynthesis:
    """Test per-chapter verification during synthesis."""

    @patch("audify.verification_integration.ChapterDurationChecker.check_chapter")
    def test_check_chapter_ok(self, mock_check):
        """Test chapter check when OK."""
        mock_check.return_value = (True, "OK", 0.8)
        
        result = check_chapter_during_synthesis(
            chapter_number=1,
            chapter_title="Test",
            audio_path=Path("/test/audio.mp3"),
            script_word_count=1000,
            confirm=True,
        )
        
        assert result is True

    @patch("audify.verification_integration.ChapterDurationChecker.check_chapter")
    def test_check_chapter_short_confirm_mode(self, mock_check):
        """Test short chapter in confirm mode."""
        mock_check.return_value = (False, "SHORT", 0.5)
        
        result = check_chapter_during_synthesis(
            chapter_number=1,
            chapter_title="Test",
            audio_path=Path("/test/audio.mp3"),
            script_word_count=1000,
            confirm=True,
        )
        
        assert result is True

    @patch("audify.verification_integration.ChapterDurationChecker.check_chapter")
    @patch("audify.verification_integration.VerificationPrompts.prompt_short_chapter", return_value=True)
    def test_check_chapter_short_user_accepts(self, mock_prompt, mock_check):
        """Test short chapter when user accepts."""
        mock_check.return_value = (False, "SHORT", 0.5)
        
        result = check_chapter_during_synthesis(
            chapter_number=1,
            chapter_title="Test",
            audio_path=Path("/test/audio.mp3"),
            script_word_count=1000,
            confirm=False,
        )
        
        assert result is True
        assert mock_prompt.called


class TestVerifyCompleteAudiobook:
    """Test full audiobook verification."""

    def test_verify_audiobook_no_source(self):
        """Test verification skipped when no source provided."""
        result = verify_complete_audiobook(
            source_path=None,
            audiobook_path=Path("/test/audiobook.m4b"),
        )
        
        assert result is True

    def test_verify_audiobook_source_not_found(self):
        """Test verification skipped when source doesn't exist."""
        result = verify_complete_audiobook(
            source_path=Path("/nonexistent/source.epub"),
            audiobook_path=Path("/test/audiobook.m4b"),
        )
        
        assert result is True

    @patch("audify.verification_integration.AudiobookVerificationCheck.verify_audiobook")
    def test_verify_audiobook_passed(self, mock_verify):
        """Test verification passed."""
        mock_verify.return_value = (True, {"verification_passed": True})
        
        with patch("pathlib.Path.exists", return_value=True):
            result = verify_complete_audiobook(
                source_path=Path("/test/source.epub"),
                audiobook_path=Path("/test/audiobook.m4b"),
            )
        
        assert result is True

    @patch("audify.verification_integration.AudiobookVerificationCheck.verify_audiobook")
    def test_verify_audiobook_confirm_mode(self, mock_verify):
        """Test verification failed but accepted in confirm mode."""
        mock_verify.return_value = (False, {"verification_passed": False})
        
        with patch("pathlib.Path.exists", return_value=True):
            result = verify_complete_audiobook(
                source_path=Path("/test/source.epub"),
                audiobook_path=Path("/test/audiobook.m4b"),
                confirm=True,
            )
        
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
