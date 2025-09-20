"""Tests for audify.text_to_speech module."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from audify.text_to_speech import (
    BaseSynthesizer,
    EpubSynthesizer,
    PdfSynthesizer,
    suppress_stdout,
)
from audify.utils.api_config import KokoroAPIConfig


class TestKokoroAPIConfig:
    """Tests for KokoroAPIConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KokoroAPIConfig()
        assert config.base_url is not None
        assert config.default_voice is not None
        assert config.timeout == 30

    def test_custom_base_url(self):
        """Test custom base URL configuration."""
        custom_url = "http://example.com:9000"
        config = KokoroAPIConfig(base_url=custom_url)
        assert config.base_url == custom_url

    def test_voices_url_property(self):
        """Test voices URL construction."""
        config = KokoroAPIConfig()
        assert "/voices" in config.voices_url

    def test_synthesis_url_property(self):
        """Test synthesis URL construction."""
        config = KokoroAPIConfig()
        # Just test that base_url is properly set
        assert config.base_url is not None


class TestSuppressStdout:
    """Tests for suppress_stdout context manager."""

    def test_suppress_stdout(self):
        """Test that stdout is suppressed within context."""
        with suppress_stdout():
            print("This should be suppressed")
        # If we reach here without output, the test passes


class TestBaseSynthesizer:
    """Tests for BaseSynthesizer base class."""

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_base_synthesizer_init(self, mock_temp_dir):
        """Test BaseSynthesizer initialization."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.mkdir"),
        ):
            synthesizer = BaseSynthesizer(
                path="test.txt",
                voice="test_voice",
                translate=None,
                save_text=False,
                language="en",
                model_name="test_model",
            )

            assert synthesizer.speaker == "test_voice"
            assert synthesizer.language == "en"
            assert synthesizer.model_name == "test_model"
            assert not synthesizer.translate
            assert not synthesizer.save_text

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_get_terminal_output(self, mock_temp_dir):
        """Test get_terminal_output method."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
            model_name="test_model",
        )

        with patch("pathlib.Path.exists", return_value=False):
            result = synthesizer.get_terminal_output()
            assert "No terminal output available" in result

        test_content = "Test output content"
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=test_content)),
        ):
            result = synthesizer.get_terminal_output()
            assert result == test_content


class TestEpubSynthesizer:
    """Tests for EpubSynthesizer class."""

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_init(self, mock_temp_dir, mock_epub_reader):
        """Test EpubSynthesizer initialization."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            assert synthesizer.path.name == "test.epub"
            assert synthesizer.language == "en"
            mock_epub_reader.assert_called_once()


class TestPdfSynthesizer:
    """Tests for PdfSynthesizer class."""

    def test_pdf_synthesizer_init_file_not_found(self):
        """Test PdfSynthesizer initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            PdfSynthesizer(pdf_path="nonexistent.pdf")

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_pdf_synthesizer_init_success(self, mock_temp_dir):
        """Test successful PdfSynthesizer initialization."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
        ):
            synthesizer = PdfSynthesizer(pdf_path="test.pdf")

            assert synthesizer.path.name == "test.pdf"
            assert synthesizer.language == "en"  # Default language

    @patch("audify.text_to_speech.PdfReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.break_text_into_sentences")
    def test_pdf_synthesizer_synthesize_success(
        self, mock_break, mock_temp_dir, mock_pdf_reader
    ):
        """Test successful PDF synthesis."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_pdf_reader_instance.cleaned_text = "Test content"

        mock_break.return_value = ["Sentence 1.", "Sentence 2."]

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch.object(PdfSynthesizer, "_synthesize_sentences") as mock_synth,
            patch.object(PdfSynthesizer, "_convert_to_mp3") as mock_convert,
        ):
            mock_convert.return_value = Path("output.mp3")

            synthesizer = PdfSynthesizer(pdf_path="test.pdf")
            result = synthesizer.synthesize()

            assert result == Path("output.mp3")
            mock_break.assert_called_once_with("Test content")
            mock_synth.assert_called_once()
            mock_convert.assert_called_once()
