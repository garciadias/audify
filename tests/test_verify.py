#!/usr/bin/env python3
"""
Unit tests for the verify module (audiobook verification/comparison).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from audify.verify import (
    AudiobookVerifier,
    Chapter,
    DurationHint,
    MissingChapter,
    VerifyReport,
    _parse_ffmetadata_from_bytes,
)


class TestChapterDataClass:
    """Test Chapter dataclass."""

    def test_chapter_creation(self):
        """Test creating a Chapter instance."""
        ch = Chapter(number=1, title="Chapter One", index=0)
        assert ch.number == 1
        assert ch.title == "Chapter One"
        assert ch.index == 0

    def test_chapter_equality(self):
        """Test Chapter equality."""
        ch1 = Chapter(number=1, title="Chapter One", index=0)
        ch2 = Chapter(number=1, title="Chapter One", index=0)
        ch3 = Chapter(number=2, title="Chapter Two", index=1)
        assert ch1 == ch2
        assert ch1 != ch3

    def test_chapter_hash(self):
        """Test Chapter is hashable."""
        ch1 = Chapter(number=1, title="Chapter One", index=0)
        ch2 = Chapter(
            number=1, title="Chapter One", index=1
        )  # Different index, same hash
        assert hash(ch1) == hash(ch2)

    def test_chapter_repr(self):
        """Test Chapter string representation."""
        ch = Chapter(number=1, title="Chapter One", index=0)
        assert "Chapter One" in repr(ch)


class TestParseFFMetadata:
    """Test FFMETADATA1 parsing from byte data."""

    def test_parse_ffmetadata_basic(self):
        """Test parsing basic FFMETADATA1 format."""
        metadata = b"""[CHAPTER]
TIMEBASE=1/1000
START=0
END=1000
title=Chapter 1
[CHAPTER]
TIMEBASE=1/1000
START=1000
END=2000
title=Chapter 2
"""
        chapters = _parse_ffmetadata_from_bytes(metadata)
        assert len(chapters) == 2
        assert chapters[0].title == "Chapter 1"
        assert chapters[1].title == "Chapter 2"

    def test_parse_ffmetadata_with_whitespace(self):
        """Test parsing FFMETADATA1 with varying whitespace."""
        metadata = b"""  [CHAPTER]
  TIMEBASE=1/1000
  START=0
  END=1000
  title=Chapter 1
"""
        chapters = _parse_ffmetadata_from_bytes(metadata)
        assert len(chapters) == 1
        assert chapters[0].title == "Chapter 1"

    def test_parse_ffmetadata_no_chapters(self):
        """Test parsing FFMETADATA1 with no chapters."""
        metadata = b"""[METADATA]
key=value
"""
        chapters = _parse_ffmetadata_from_bytes(metadata)
        assert len(chapters) == 0

    def test_parse_ffmetadata_missing_title(self):
        """Test parsing chapter without title (should generate default)."""
        metadata = b"""[CHAPTER]
TIMEBASE=1/1000
START=0
END=1000
"""
        chapters = _parse_ffmetadata_from_bytes(metadata)
        assert len(chapters) == 1
        assert chapters[0].title.startswith("Chapter")


class TestVerifyReportDataClass:
    """Test VerifyReport dataclass."""

    def test_report_creation(self):
        """Test creating a VerifyReport."""
        source_chapters = [Chapter(1, "Source Chapter 1", 0)]
        audiobook_chapters = [Chapter(1, "Source Chapter 1", 0)]

        report = VerifyReport(
            source_path=Path("/test/source.epub"),
            audiobook_path=Path("/test/audio.m4b"),
            source_chapters=source_chapters,
            audiobook_chapters=audiobook_chapters,
            source_type="epub",
            source_word_count=1000,
        )

        assert report.total_source == 1
        assert report.total_audiobook == 1
        assert not report.has_missing_chapters()
        assert not report.has_extra_chapters()
        assert not report.has_order_issues()

    def test_report_with_missing_chapters(self):
        """Test report with missing chapters."""
        report = VerifyReport(
            source_path=Path("/test/source.epub"),
            audiobook_path=Path("/test/audio.m4b"),
            source_chapters=[Chapter(1, "Ch1", 0), Chapter(2, "Ch2", 1)],
            audiobook_chapters=[Chapter(1, "Ch1", 0)],
            source_type="epub",
            source_word_count=1000,
            missing=[MissingChapter(2, "Ch2", 1)],
        )

        assert report.has_missing_chapters()
        assert not report.has_extra_chapters()


class TestAudiobookVerifierMocked:
    """Test AudiobookVerifier using mocked file I/O."""

    @patch("audify.verify.extract_epub_chapters")
    @patch("audify.verify._count_epub_words")
    @patch("audify.verify.extract_chapters_from_m4b")
    def test_verifier_matching_chapters(self, mock_audio, mock_count, mock_extract):
        """Test verifier with matching chapters."""
        source_chapters = [
            Chapter(1, "Chapter One", 0),
            Chapter(2, "Chapter Two", 1),
        ]
        mock_extract.return_value = source_chapters
        mock_count.return_value = 5000
        mock_audio.return_value = source_chapters

        with patch("pathlib.Path.exists", return_value=True):
            verifier = AudiobookVerifier(
                Path("/test/book.epub"),
                Path("/test/audio.m4b"),
            )

            report = verifier.verify()

            assert report.total_source == 2
            assert report.total_audiobook == 2
            assert report.matched == 2


class TestAudiobookVerifierIntegration:
    """Integration tests using real files (if available)."""

    def test_verifier_missing_files(self):
        """Test verifier raises error for missing files."""
        with pytest.raises(FileNotFoundError):
            AudiobookVerifier(
                Path("/nonexistent/source.epub"),
                Path("/nonexistent/audio.m4b"),
            )

    @patch("audify.verify.extract_epub_chapters")
    @patch("audify.verify.extract_chapters_from_m4b")
    @patch("audify.verify._count_epub_words")
    def test_verifier_file_detection(self, mock_count, mock_audio, mock_epub):
        """Test verifier detects file types."""
        mock_epub.return_value = []
        mock_audio.return_value = []
        mock_count.return_value = 1000

        with patch("pathlib.Path.exists", return_value=True):
            verifier = AudiobookVerifier(
                Path("/test/book.epub"),
                Path("/test/audio.m4b"),
            )

            assert verifier.source_type == "epub"


class TestChapterComparison:
    """Test chapter comparison logic."""

    @patch("audify.verify.extract_epub_chapters")
    @patch("audify.verify.extract_chapters_from_m4b")
    @patch("audify.verify._count_epub_words")
    def test_missing_chapters_detection(self, mock_count, mock_audio, mock_epub):
        """Test detection of missing chapters."""
        source_chapters = [
            Chapter(1, "Chapter One", 0),
            Chapter(2, "Chapter Two", 1),
            Chapter(3, "Chapter Three", 2),
        ]
        audiobook_chapters = [
            Chapter(1, "Chapter One", 0),
            Chapter(3, "Chapter Three", 1),
        ]

        mock_epub.return_value = source_chapters
        mock_audio.return_value = audiobook_chapters
        mock_count.return_value = 5000

        with patch("pathlib.Path.exists", return_value=True):
            verifier = AudiobookVerifier(
                Path("/test/book.epub"),
                Path("/test/audio.m4b"),
            )

            report = verifier.verify()

            assert report.total_source == 3
            assert report.total_audiobook == 2
            assert len(report.missing) == 1
            assert report.missing[0].title == "Chapter Two"


class TestDurationAnalysis:
    """Test duration estimation."""

    def test_duration_hint_dataclass(self):
        """Test DurationHint creation and ratio calculation."""
        hint = DurationHint(
            source_word_count=4500,
            expected_duration_s=3600,
            actual_duration_s=3600,
            ratio=1.0,
        )

        assert hint.source_word_count == 4500
        assert hint.expected_duration_s == 3600
        assert hint.actual_duration_s == 3600
        assert hint.ratio == 1.0


class TestJsonGeneration:
    """Test JSON report generation."""

    @patch("audify.verify.extract_epub_chapters")
    @patch("audify.verify.extract_chapters_from_m4b")
    @patch("audify.verify._count_epub_words")
    def test_generate_report_json(self, mock_count, mock_audio, mock_epub):
        """Test generating JSON report."""
        source_chapters = [Chapter(1, "Chapter One", 0)]
        audiobook_chapters = [Chapter(1, "Chapter One", 0)]

        mock_epub.return_value = source_chapters
        mock_audio.return_value = audiobook_chapters
        mock_count.return_value = 5000

        with patch("pathlib.Path.exists", return_value=True):
            verifier = AudiobookVerifier(
                Path("/test/book.epub"),
                Path("/test/audio.m4b"),
            )

            json_report = verifier.generate_report()

            assert "source" in json_report
            assert "audiobook" in json_report
            assert "summary" in json_report
            assert json_report["summary"]["source_chapters"] == 1
            assert json_report["summary"]["audiobook_chapters"] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unsupported_file_format(self):
        """Test error on unsupported file format."""
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ValueError, match="Unsupported source format"):
                AudiobookVerifier(
                    Path("/test/book.txt"),
                    Path("/test/audio.m4b"),
                )

    @patch("audify.verify.extract_epub_chapters")
    @patch("audify.verify.extract_chapters_from_m4b")
    @patch("audify.verify._count_epub_words")
    def test_empty_chapters(self, mock_count, mock_audio, mock_epub):
        """Test handling of empty chapter lists."""
        mock_epub.return_value = []
        mock_audio.return_value = []
        mock_count.return_value = 0

        with patch("pathlib.Path.exists", return_value=True):
            verifier = AudiobookVerifier(
                Path("/test/book.epub"),
                Path("/test/audio.m4b"),
            )

            report = verifier.verify()

            assert report.total_source == 0
            assert report.total_audiobook == 0
            assert report.overall_match_percentage == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
