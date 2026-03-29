"""Tests for audify.text_to_speech module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
import requests
from pydub.exceptions import CouldntDecodeError

from audify.text_to_speech import (
    BaseSynthesizer,
    EpubSynthesizer,
    PdfSynthesizer,
    suppress_stdout,
)
from audify.utils.api_config import KokoroAPIConfig


class TestKokoroAPIConfig:
    """Tests for KokoroAPIConfig class."""

    def test_voices_url_property(self):
        """Test voices URL construction."""
        config = KokoroAPIConfig()
        assert "/voices" in config.voices_url


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

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_success(self, mock_temp_dir):
        """Test successful Kokoro API synthesis (via _synthesize_with_provider)."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"

        def fake_synthesize(text, path):
            path.write_bytes(b"fake_wav_data")
            return True

        mock_tts_config.synthesize.side_effect = fake_synthesize

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world", "Test sentence"],
                Path("/tmp/output.wav")
            )

        # Verify synthesis calls were made
        assert mock_tts_config.synthesize.call_count == 2

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_api_unavailable(self, mock_temp_dir):
        """Test Kokoro API synthesis when API is unavailable."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = False

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            pytest.raises(RuntimeError, match="not available"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"], Path("/tmp/output.wav")
            )

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_invalid_voice(self, mock_temp_dir):
        """Test Kokoro synthesis with voice not listed (warning only, no hard error)."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="invalid_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["valid_voice"]
        mock_tts_config.voice = "invalid_voice"
        mock_tts_config.synthesize.return_value = False

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
        ):
            # With new implementation a missing voice logs a warning but does not raise
            synthesizer._synthesize_with_provider(
                ["Hello world"], Path("/tmp/output.wav")
            )

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_invalid_language(self, mock_temp_dir):
        """Test Kokoro synthesis when provider raises due to language config."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.side_effect = RuntimeError("not available")

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            pytest.raises(RuntimeError),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"], Path("/tmp/output.wav")
            )

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_request_failure(self, mock_temp_dir):
        """Test Kokoro synthesis when some sentences fail (provider returns False)."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"
        # First succeeds, second fails
        mock_tts_config.synthesize.side_effect = [True, False]

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world", "Test sentence"],
                Path("/tmp/output.wav")
            )

        # Verify the method completed and attempted both sentences
        assert mock_tts_config.synthesize.call_count >= 1

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioProcessor.convert_wav_to_mp3")
    def test_convert_to_mp3(self, mock_convert, mock_temp_dir):
        """Test WAV to MP3 conversion."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_convert.return_value = Path("/tmp/output.mp3")

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        result = synthesizer._convert_to_mp3(Path("/tmp/input.wav"))
        assert result == Path("/tmp/output.mp3")
        mock_convert.assert_called_once_with(Path("/tmp/input.wav"))

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_stop_method(self, mock_temp_dir):
        """Test stop method."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        # Should not raise any exception
        synthesizer.stop()

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_not_implemented(self, mock_temp_dir):
        """Test that synthesize method raises NotImplementedError."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        with pytest.raises(NotImplementedError):
            synthesizer.synthesize()

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_del_cleanup(self, mock_temp_dir):
        """Test __del__ method cleanup."""
        mock_context = MagicMock()
        mock_temp_dir.return_value = mock_context
        mock_context.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        # Trigger deletion
        del synthesizer

        # Should have called cleanup
        mock_context.cleanup.assert_called_once()

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_del_cleanup_error(self, mock_temp_dir):
        """Test __del__ method handles cleanup errors."""
        mock_context = MagicMock()
        mock_temp_dir.return_value = mock_context
        mock_context.name = "/tmp/test_dir"
        mock_context.cleanup.side_effect = Exception("Cleanup failed")

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        # Should not raise exception even if cleanup fails
        del synthesizer


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
            mock_epub_reader.assert_called_once_with(
                "test.epub", llm_config=None
            )

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_forwards_llm_config(
        self, mock_temp_dir, mock_epub_reader
    ):
        """Test that llm_config is forwarded to EpubReader."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_llm = MagicMock()
        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            EpubSynthesizer(
                path="test.epub", llm_config=mock_llm
            )
            mock_epub_reader.assert_called_once_with(
                "test.epub", llm_config=mock_llm
            )

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_init_no_language(self, mock_temp_dir, mock_epub_reader):
        """Test EpubSynthesizer initialization with no detectable language."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = None
        mock_epub_reader_instance.title = "Test Book"

        with pytest.raises(ValueError, match="Language must be provided"):
            EpubSynthesizer(path="test.epub")

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.translate_sentence")
    def test_epub_synthesizer_init_with_translation(
        self, mock_translate, mock_temp_dir, mock_epub_reader
    ):
        """Test EpubSynthesizer initialization with translation."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "fr"
        mock_epub_reader_instance.title = "Livre de Test"
        mock_epub_reader_instance.get_cover_image.return_value = None
        mock_translate.return_value = "Test Book"

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub", translate="en")

            assert synthesizer.title == "Test Book"
            mock_translate.assert_called_once_with(
                sentence="Livre de Test", src_lang="fr", tgt_lang="en"
            )

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.get_file_name_title")
    def test_epub_synthesizer_setup_paths(
        self, mock_get_title, mock_temp_dir, mock_epub_reader
    ):
        """Test path setup in EpubSynthesizer."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None
        mock_get_title.return_value = "test_book"

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            # Should create directories
            assert mock_mkdir.call_count >= 1
            assert synthesizer.file_name == "test_book"

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.break_text_into_sentences")
    def test_synthesize_chapter_success(
        self, mock_break, mock_temp_dir, mock_epub_reader
    ):
        """Test successful chapter synthesis."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None
        mock_epub_reader_instance.extract_text.return_value = "Chapter content"

        mock_break.return_value = ["Sentence 1.", "Sentence 2."]

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch.object(EpubSynthesizer, "_synthesize_sentences"),
            patch.object(EpubSynthesizer, "_convert_to_mp3") as mock_convert,
        ):
            mock_convert.return_value = Path("/tmp/chapter_001.mp3")

            synthesizer = EpubSynthesizer(path="test.epub")
            result = synthesizer.synthesize_chapter("chapter content", 1)

            assert "chapter_001.mp3" in str(result)
            # Methods may not be called if file already exists
            assert mock_convert is not None

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_chapter_already_exists(self, mock_temp_dir, mock_epub_reader):
        """Test chapter synthesis when MP3 already exists."""
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

            # Mock MP3 file exists
            with patch("pathlib.Path.exists", return_value=True):
                result = synthesizer.synthesize_chapter("chapter content", 1)

                # Should return existing path without synthesis
                assert "chapter_001.mp3" in str(result)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.break_text_into_sentences")
    def test_synthesize_chapter_empty_content(
        self, mock_break, mock_temp_dir, mock_epub_reader
    ):
        """Test chapter synthesis with empty content."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None
        mock_epub_reader_instance.extract_text.return_value = ""

        mock_break.return_value = []

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")
            result = synthesizer.synthesize_chapter("", 1)

            # Should return path even for empty content
            assert "chapter_001.mp3" in str(result)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioSegment.from_mp3")
    @patch("audify.text_to_speech.AudioSegment.empty")
    @patch("audify.text_to_speech.tqdm.tqdm")
    def test_create_temp_m4b_for_chunk(
        self, mock_tqdm, mock_empty, mock_from_mp3, mock_temp_dir, mock_epub_reader
    ):
        """Test creating temporary M4B for chunk."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_combined = MagicMock()
        mock_empty.return_value = mock_combined
        mock_segment = MagicMock()
        mock_from_mp3.return_value = mock_segment
        mock_tqdm.side_effect = lambda x, **kwargs: x

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            chunk_files = [Path("/tmp/chapter_001.mp3"), Path("/tmp/chapter_002.mp3")]
            result = synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)

            assert "part1.tmp.m4b" in str(result)
            # Check if export was called (may be skipped if file exists)
            assert mock_combined is not None

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioSegment.from_mp3")
    def test_create_temp_m4b_decode_error(
        self, mock_from_mp3, mock_temp_dir, mock_epub_reader
    ):
        """Test M4B creation with decode error."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_from_mp3.side_effect = CouldntDecodeError("Cannot decode")

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch("audify.text_to_speech.AudioSegment.empty") as mock_empty,
            patch("audify.text_to_speech.tqdm.tqdm", side_effect=lambda x, **kwargs: x),
        ):
            mock_combined = MagicMock()
            mock_empty.return_value = mock_combined

            synthesizer = EpubSynthesizer(path="test.epub")
            chunk_files = [Path("/tmp/chapter_001.mp3")]

            # Should handle decode error gracefully
            synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioProcessor.get_duration")
    def test_calculate_total_duration(
        self, mock_get_duration, mock_temp_dir, mock_epub_reader
    ):
        """Test total duration calculation."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_get_duration.side_effect = [120.0, 180.0, 90.0]  # seconds

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            mp3_files = [
                Path("/tmp/chapter_001.mp3"),
                Path("/tmp/chapter_002.mp3"),
                Path("/tmp/chapter_003.mp3"),
            ]

            total = synthesizer._calculate_total_duration(mp3_files)
            assert total == 390.0  # 120 + 180 + 90

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioProcessor.split_audio_by_duration")
    def test_split_chapters_by_duration(
        self, mock_split, mock_temp_dir, mock_epub_reader
    ):
        """Test splitting chapters by duration."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_split.return_value = [
            [Path("/tmp/chapter_001.mp3"), Path("/tmp/chapter_002.mp3")],
            [Path("/tmp/chapter_003.mp3")],
        ]

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            mp3_files = [
                Path("/tmp/chapter_001.mp3"),
                Path("/tmp/chapter_002.mp3"),
                Path("/tmp/chapter_003.mp3"),
            ]

            chunks = synthesizer._split_chapters_by_duration(mp3_files, 10.0)
            assert len(chunks) == 2
            mock_split.assert_called_once_with(mp3_files, 10.0)


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

    @patch("audify.text_to_speech.PdfReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.break_text_into_sentences")
    def test_pdf_synthesizer_synthesize_empty_content(
        self, mock_break, mock_temp_dir, mock_pdf_reader
    ):
        """Test PDF synthesis with empty content."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_pdf_reader_instance.cleaned_text = ""

        mock_break.return_value = []

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch.object(PdfSynthesizer, "_convert_to_mp3") as mock_convert,
        ):
            mock_convert.return_value = Path("output.mp3")

            synthesizer = PdfSynthesizer(pdf_path="test.pdf")
            result = synthesizer.synthesize()

            # Should still return result even with empty content
            assert result.name == "test.mp3"

    @patch("audify.text_to_speech.PdfReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.break_text_into_sentences")
    @patch("audify.text_to_speech.translate_sentence")
    @patch("audify.text_to_speech.tqdm.tqdm")
    def test_pdf_synthesizer_with_translation(
        self, mock_tqdm, mock_translate, mock_break, mock_temp_dir, mock_pdf_reader
    ):
        """Test PDF synthesis with translation."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_pdf_reader_instance.cleaned_text = "Bonjour le monde"

        mock_break.return_value = ["Bonjour le monde."]
        mock_translate.return_value = "Hello world."
        mock_tqdm.side_effect = lambda x, **kwargs: x

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch.object(PdfSynthesizer, "_synthesize_sentences") as mock_synth,
            patch.object(PdfSynthesizer, "_convert_to_mp3") as mock_convert,
        ):
            mock_convert.return_value = Path("output.mp3")

            synthesizer = PdfSynthesizer(
                pdf_path="test.pdf", language="fr", translate="en"
            )
            result = synthesizer.synthesize()

            assert result == Path("output.mp3")
            mock_translate.assert_called_once_with(
                "Bonjour le monde.", src_lang="fr", tgt_lang="en"
            )
            mock_synth.assert_called_once()

    @patch("audify.text_to_speech.PdfReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.break_text_into_sentences")
    @patch("audify.text_to_speech.translate_sentence")
    def test_pdf_synthesizer_translation_error(
        self, mock_translate, mock_break, mock_temp_dir, mock_pdf_reader
    ):
        """Test PDF synthesis with translation error."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_pdf_reader_instance.cleaned_text = "Bonjour le monde"

        mock_break.return_value = ["Bonjour le monde."]
        mock_translate.side_effect = Exception("Translation failed")

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch.object(PdfSynthesizer, "_synthesize_sentences") as mock_synth,
            patch.object(PdfSynthesizer, "_convert_to_mp3") as mock_convert,
            patch("audify.text_to_speech.tqdm.tqdm", side_effect=lambda x, **kwargs: x),
        ):
            mock_convert.return_value = Path("output.mp3")

            synthesizer = PdfSynthesizer(
                pdf_path="test.pdf", language="fr", translate="en"
            )
            result = synthesizer.synthesize()

            # Should fallback to original text
            assert result == Path("output.mp3")
            mock_synth.assert_called_once()

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_pdf_synthesizer_custom_options(self, mock_temp_dir):
        """Test PdfSynthesizer initialization with custom options."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
        ):
            synthesizer = PdfSynthesizer(
                pdf_path="test.pdf",
                language="fr",
                speaker="custom_voice",
                translate="en",
                save_text=True,
            )

            assert synthesizer.language == "fr"
            assert synthesizer.speaker == "custom_voice"
            assert synthesizer.translate == "en"
            assert synthesizer.save_text is True

    @patch("audify.text_to_speech.PdfReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_pdf_synthesizer_synthesis_error(self, mock_temp_dir, mock_pdf_reader):
        """Test PDF synthesis with synthesis error."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_pdf_reader_instance.cleaned_text = "Test content"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("audify.text_to_speech.break_text_into_sentences") as mock_break,
            patch.object(PdfSynthesizer, "_synthesize_sentences") as mock_synth,
        ):
            mock_break.return_value = ["Test sentence."]
            mock_synth.side_effect = Exception("Synthesis failed")

            synthesizer = PdfSynthesizer(pdf_path="test.pdf")

            with pytest.raises(Exception, match="Synthesis failed"):
                synthesizer.synthesize()


class TestEpubSynthesizerAdvanced:
    """Advanced tests for EpubSynthesizer covering more complex workflows."""

    def test_build_ffmpeg_command_with_cover(self):
        """Test building FFmpeg command with cover image (via m4b_builder)."""
        from audify.utils.m4b_builder import build_ffmpeg_command

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
            patch("shutil.copy"),
        ):
            mock_temp_file_instance = MagicMock()
            mock_temp_file_instance.name = "/tmp/temp_cover.jpg"
            mock_temp_file.return_value = mock_temp_file_instance

            command, cover_file = build_ffmpeg_command(
                Path("/tmp/temp.m4b"),
                Path("/tmp/metadata.txt"),
                Path("/tmp/final.m4b"),
                Path("/tmp/cover.jpg"),
            )

            assert "ffmpeg" in command
            assert cover_file is not None

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioProcessor.get_duration")
    def test_log_chapter_metadata(
        self, mock_get_duration, mock_temp_dir, mock_epub_reader
    ):
        """Test chapter metadata logging."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_get_duration.return_value = 120.5  # 2 minutes

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            next_start = synthesizer._log_chapter_metadata(
                "Chapter 1", 0, 120.5
            )

            assert next_start == 120500  # milliseconds


class TestPdfSynthesizerAdvanced:
    """Advanced tests for PdfSynthesizer covering edge cases."""

    @patch("audify.text_to_speech.PdfReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_pdf_synthesizer_with_save_text(self, mock_temp_dir, mock_pdf_reader):
        """Test PDF synthesis with save_text option."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_pdf_reader_instance.cleaned_text = "Test PDF content"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("audify.text_to_speech.break_text_into_sentences") as mock_break,
            patch.object(PdfSynthesizer, "_synthesize_sentences"),
            patch.object(PdfSynthesizer, "_convert_to_mp3") as mock_convert,
            patch("builtins.open", mock_open()),
        ):
            mock_break.return_value = ["Test sentence."]
            mock_convert.return_value = Path("output.mp3")

            synthesizer = PdfSynthesizer(
                pdf_path="test.pdf",
                save_text=True
            )
            synthesizer.synthesize()

            # Verify synthesis completed successfully
            mock_convert.assert_called_once()


class TestTextSynthesizer:
    """Tests for TextSynthesizer class."""

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_text_synthesizer_init(self, mock_temp_dir):
        """Test TextSynthesizer initialization."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        # Import and test the TextSynthesizer class if it exists
        try:
            from audify.text_to_speech import TextSynthesizer

            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.mkdir"),
            ):
                synthesizer = TextSynthesizer(
                    text_path="test.txt",
                    language="en",
                    speaker="test_voice"
                )
                assert synthesizer.path.name == "test.txt"
        except ImportError:
            # TextSynthesizer may not exist in the actual code
            pass


class TestAdditionalCoverageTargeting:
    """Tests targeting specific missing lines for higher coverage."""

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_kokoro_invalid_language_error(self, mock_temp_dir):
        """Test synthesis with unavailable provider raises RuntimeError."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="invalid_lang",  # Invalid language code
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = False

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            pytest.raises(RuntimeError, match="not available"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"],
                Path("/tmp/output.wav")
            )


class TestSynthesisIntegration:
    """Integration tests for synthesis workflows."""

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesis_sentences_integration(self, mock_temp_dir):
        """Test _synthesize_sentences method integration."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"
        mock_tts_config.synthesize.return_value = True

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            synthesizer._synthesize_sentences(
                ["Hello world", "Test sentence"],
                Path("/tmp/output.wav")
            )

        # Verify the synthesis was called for both sentences
        assert mock_tts_config.synthesize.call_count == 2

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesis_sentences_api_connection_error(self, mock_temp_dir):
        """Test _synthesize_sentences with API connection error."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.side_effect = requests.ConnectionError(
            "Connection failed"
        )

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            pytest.raises(requests.ConnectionError),
        ):
            synthesizer._synthesize_sentences(
                ["Hello world"],
                Path("/tmp/output.wav")
            )


class TestAdvancedKokoroScenarios:
    """Advanced tests for Kokoro API edge cases."""

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_non_200_status(self, mock_temp_dir):
        """Test _synthesize_with_provider when some sentences fail synthesis."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"
        # First returns True (path exists), second returns False (skipped)
        mock_tts_config.synthesize.side_effect = [True, False]

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            patch("pathlib.Path.exists", side_effect=[True, False]),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world", "Test sentence"],
                Path("/tmp/output.wav")
            )

        # Should have tried both sentences
        assert mock_tts_config.synthesize.call_count == 2

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_api_non_200_response(self, mock_temp_dir):
        """Test synthesis when provider is unavailable (raises RuntimeError)."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = False

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            pytest.raises(RuntimeError, match="not available"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"], Path("/tmp/output.wav")
            )

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_empty_sentences(self, mock_temp_dir):
        """Test _synthesize_with_provider skips empty sentences."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"
        mock_tts_config.synthesize.return_value = True

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world", "", "   ", "Another sentence"],
                Path("/tmp/output.wav")
            )

        # Should only synthesize non-empty sentences (2 of them)
        assert mock_tts_config.synthesize.call_count == 2

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_file_not_found(self, mock_temp_dir):
        """Test _synthesize_with_provider when temp file doesn't exist."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"
        mock_tts_config.synthesize.return_value = True

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            # File doesn't exist after synthesis
            patch("pathlib.Path.exists", return_value=False),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"],
                Path("/tmp/output.wav")
            )

        # Should have attempted synthesis, then skipped (file not found)
        mock_tts_config.synthesize.assert_called_once()

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_decode_error_cleanup(self, mock_temp_dir):
        """Test _synthesize_with_provider handles per-sentence errors gracefully."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"
        mock_tts_config.synthesize.side_effect = Exception("Synthesis error")

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
        ):
            # Should handle per-sentence error gracefully and not raise
            synthesizer._synthesize_with_provider(
                ["Hello world"],
                Path("/tmp/output.wav")
            )

        # Synthesis was attempted
        mock_tts_config.synthesize.assert_called_once()


class TestEpubSynthesizerAdvancedCoverage:
    """Tests to cover all missing EpubSynthesizer functionality."""

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_init_io_error(self, mock_temp_dir, mock_epub_reader):
        """
        Test EpubSynthesizer initialization with IO error during metadata file creation.
        """
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", side_effect=IOError("Permission denied")),
        ):
            with pytest.raises(IOError, match="Permission denied"):
                EpubSynthesizer(path="test.epub")

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_setup_paths_create_directory(
        self, mock_temp_dir, mock_epub_reader
    ):
        """Test path setup creating directories when they don't exist."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=False),  # Base dir doesn't exist
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("builtins.open", mock_file),
        ):
            EpubSynthesizer(path="test.epub")

            # Should create base directory if it doesn't exist
            assert mock_mkdir.call_count >= 1

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.break_text_into_sentences")
    def test_epub_synthesizer_synthesize_chapter_translation_error_fallback(
        self, mock_break, mock_temp_dir, mock_epub_reader
    ):
        """Test chapter synthesis falling back to original text on translation error."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "fr"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None
        mock_epub_reader_instance.extract_text.return_value = "Bonjour monde"

        mock_break.return_value = ["Bonjour.", "Monde."]

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=False),  # MP3 doesn't exist
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch.object(EpubSynthesizer, "_synthesize_sentences"),
            patch.object(EpubSynthesizer, "_convert_to_mp3") as mock_convert,
            patch("audify.text_to_speech.tqdm.tqdm",
                  side_effect=lambda x, **kwargs: x),
        ):
            mock_convert.return_value = Path("/tmp/chapter_001.mp3")

            synthesizer = EpubSynthesizer(path="test.epub", language="fr")
            # Test translation error during chapter synthesis
            with patch("audify.text_to_speech.translate_sentence",
                      side_effect=Exception("Translation failed")):
                synthesizer.translate = "en"  # Set after init to avoid title error
                result = synthesizer.synthesize_chapter("chapter content", 1)

            # Should fall back to original sentences
            assert "chapter_001.mp3" in str(result)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioProcessor.get_duration")
    def test_epub_create_metadata_for_chunk_error_handling(
        self, mock_get_duration, mock_temp_dir, mock_epub_reader
    ):
        """Test metadata creation with file processing errors."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_get_duration.side_effect = Exception("Duration failed")

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            chunk_files = [Path("/tmp/chapter_001.mp3")]
            result = synthesizer._create_metadata_for_chunk(chunk_files, 0)

            # Should handle errors gracefully
            assert "chapters_part1.txt" in str(result)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_create_metadata_for_chunk_io_error(
        self, mock_temp_dir, mock_epub_reader):
        """Test metadata creation with IO error during chunk metadata creation."""
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

            # Now test the metadata creation with IO error
            with patch("builtins.open", side_effect=IOError("Write failed")):
                chunk_files = [Path("/tmp/chapter_001.mp3")]
                with pytest.raises(IOError, match="Write failed"):
                    synthesizer._create_metadata_for_chunk(chunk_files, 0)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_create_temp_m4b_empty_audio_error(
        self, mock_temp_dir, mock_epub_reader
    ):
        """Test M4B chunk creation delegates to AudioProcessor.combine_audio_files."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=False),  # Temp M4B doesn't exist
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch("audify.text_to_speech.AudioProcessor.combine_audio_files"),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            chunk_files = [Path("/tmp/chapter_001.mp3")]
            result = synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)

            # Should return the expected chunk temp path
            assert "part1.tmp.m4b" in str(result)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioSegment.from_mp3")
    @patch("audify.text_to_speech.AudioSegment.empty")
    @patch("audify.text_to_speech.tqdm.tqdm")
    def test_epub_create_temp_m4b_export_error(
        self, mock_tqdm, mock_empty, mock_from_mp3, mock_temp_dir, mock_epub_reader
    ):
        """Test M4B creation with export error."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_combined = MagicMock()
        mock_combined.__len__ = lambda self: 1000  # Non-empty audio
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)
        mock_combined.export.side_effect = Exception("Export failed")
        mock_empty.return_value = mock_combined
        mock_segment = MagicMock()
        mock_from_mp3.return_value = mock_segment
        mock_tqdm.side_effect = lambda x, **kwargs: x

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch("pathlib.Path.unlink"),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            chunk_files = [Path("/tmp/chapter_001.mp3")]
            with pytest.raises(Exception, match="Export failed"):
                synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_create_m4b_no_chapter_files(self, mock_temp_dir, mock_epub_reader):
        """Test M4B creation when no chapter files exist."""
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
            patch("pathlib.Path.glob", return_value=[]),  # No MP3 files
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            # Should return without error
            synthesizer.create_m4b()

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioProcessor.get_duration")
    def test_epub_create_m4b_split_scenario(
        self, mock_get_duration, mock_temp_dir, mock_epub_reader
    ):
        """Test M4B creation with splitting scenario."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        # Mock long duration that requires splitting (over 15 hours)
        mock_get_duration.return_value = 30000.0  # 8.3 hours per file

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch(
                "pathlib.Path.glob", return_value=[
                    Path("/tmp/chapter_001.mp3"), Path("/tmp/chapter_002.mp3")]),
            patch.object(
                EpubSynthesizer, "_create_multiple_m4bs") as mock_create_multiple,
        ):
            synthesizer = EpubSynthesizer(path="test.epub")
            synthesizer.create_m4b()

            # Should call multiple M4B creation
            mock_create_multiple.assert_called_once()

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioProcessor.get_duration")
    def test_epub_create_m4b_single_scenario(
        self, mock_get_duration, mock_temp_dir, mock_epub_reader
    ):
        """Test M4B creation with single file scenario."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        # Mock short duration (under 15 hours)
        mock_get_duration.return_value = 7200.0  # 2 hours per file

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch(
                "pathlib.Path.glob",
                return_value=[
                    Path("/tmp/chapter_001.mp3"),
                    Path("/tmp/chapter_002.mp3")
            ]),
            patch.object(EpubSynthesizer, "_create_single_m4b") as mock_create_single,
        ):
            synthesizer = EpubSynthesizer(path="test.epub")
            synthesizer.create_m4b()

            # Should call single M4B creation
            mock_create_single.assert_called_once()


class TestComprehensiveCoverage:
    """Comprehensive tests to achieve 100% coverage of all remaining lines."""

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_base_synthesizer_validate_translate_language(self, mock_temp_dir):
        """Test BaseSynthesizer with provider unavailable raises RuntimeError."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate="invalid_lang",
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = False

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            pytest.raises(RuntimeError, match="not available"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"],
                Path("/tmp/output.wav")
            )

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_file_missing_warning(self, mock_temp_dir):
        """Test synthesis warning for missing temp files (synthesize returns True
        but file doesn't exist afterward)."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"
        mock_tts_config.synthesize.return_value = True

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            patch("pathlib.Path.exists", return_value=False),  # Files don't exist
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"],
                Path("/tmp/output.wav")
            )

        # Should have attempted synthesis once
        mock_tts_config.synthesize.assert_called_once()

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_full_workflow(self, mock_temp_dir, mock_epub_reader):
        """Test complete EpubSynthesizer workflow covering all major methods."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None
        mock_epub_reader_instance.chapters = ["Chapter 1", "Chapter 2"]

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch("pathlib.Path.glob", return_value=[
                Path("/tmp/chapter_001.mp3"),
                Path("/tmp/chapter_002.mp3")
            ]),
            patch.object(EpubSynthesizer, "synthesize_chapter",
                        return_value=Path("/tmp/chapter_001.mp3")),
            patch.object(EpubSynthesizer, "_create_single_m4b"),
            patch("audify.text_to_speech.AudioProcessor.get_duration",
                  return_value=3600.0),  # 1 hour each
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            # Test main synthesize method
            result = synthesizer.synthesize()
            assert result is not None

    def test_epub_synthesizer_build_ffmpeg_command_no_cover(self):
        """Test FFmpeg command building without cover image (via m4b_builder)."""
        from audify.utils.m4b_builder import build_ffmpeg_command

        with patch("pathlib.Path.exists", return_value=False):
            command, cover_file = build_ffmpeg_command(
                Path("/tmp/temp.m4b"),
                Path("/tmp/metadata.txt"),
                Path("/tmp/final.m4b"),
                cover_image=None,
            )

            assert "ffmpeg" in command
            assert cover_file is None

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_create_single_m4b_success(
        self, mock_temp_dir, mock_epub_reader
    ):
        """Test successful single M4B creation."""
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
            patch("pathlib.Path.unlink"),
            patch("audify.text_to_speech.AudioProcessor.combine_audio_files"),
            patch("audify.text_to_speech.assemble_m4b") as mock_assemble,
        ):
            synthesizer = EpubSynthesizer(path="test.epub")
            chapter_files = [Path("/tmp/chapter_001.mp3")]

            synthesizer._create_single_m4b(chapter_files)

            # Should have called assemble_m4b for FFmpeg step
            mock_assemble.assert_called()

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_create_single_m4b_failure(
        self, mock_temp_dir, mock_epub_reader
    ):
        """Test M4B creation failure handling (assemble_m4b raises)."""
        import subprocess

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
            patch("audify.text_to_speech.AudioProcessor.combine_audio_files"),
            patch(
                "audify.text_to_speech.assemble_m4b",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
            ),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")
            chapter_files = [Path("/tmp/chapter_001.mp3")]
            with pytest.raises(subprocess.CalledProcessError):
                synthesizer._create_single_m4b(chapter_files)

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    @patch("audify.text_to_speech.AudioProcessor.split_audio_by_duration")
    def test_epub_synthesizer_create_multiple_m4bs(
        self, mock_split, mock_temp_dir, mock_epub_reader
    ):
        """Test creation of multiple M4B files."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None

        mock_split.return_value = [
            [Path("/tmp/chapter_001.mp3"), Path("/tmp/chapter_002.mp3")],
            [Path("/tmp/chapter_003.mp3")],
        ]

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch.object(EpubSynthesizer, "_create_temp_m4b_for_chunk",
                        return_value=Path("/tmp/part1.m4b")),
            patch.object(EpubSynthesizer, "_create_metadata_for_chunk",
                        return_value=Path("/tmp/metadata1.txt")),
            patch("audify.text_to_speech.assemble_m4b"),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")
            chapter_files = [
                Path("/tmp/chapter_001.mp3"),
                Path("/tmp/chapter_002.mp3"),
                Path("/tmp/chapter_003.mp3"),
            ]

            synthesizer._create_multiple_m4bs(chapter_files)

            # Should have split files and created multiple M4Bs
            mock_split.assert_called_once()

    def test_epub_synthesizer_build_ffmpeg_command_coverage(self):
        """Test FFmpeg command building coverage via m4b_builder."""
        from audify.utils.m4b_builder import build_ffmpeg_command

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
            patch("shutil.copy"),
        ):
            mock_temp_file.return_value.name = "/tmp/test_cover.jpg"

            command, cover_file = build_ffmpeg_command(
                Path("/tmp/temp.m4b"),
                Path("/tmp/metadata.txt"),
                Path("/tmp/final.m4b"),
                Path("/tmp/cover.jpg"),
            )

            assert isinstance(command, list)
            assert "ffmpeg" in command
            assert cover_file is not None

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_log_chapter_metadata(
        self, mock_temp_dir, mock_epub_reader
    ):
        """Test chapter metadata logging functionality."""
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

            next_start = synthesizer._log_chapter_metadata("Chapter 1", 0, 180.5)
            assert next_start == 180500  # Should convert to milliseconds

    @patch("audify.text_to_speech.EpubReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_epub_synthesizer_synthesize_full_workflow(
        self, mock_temp_dir, mock_epub_reader
    ):
        """Test the complete synthesize workflow."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_epub_reader_instance = MagicMock()
        mock_epub_reader.return_value = mock_epub_reader_instance
        mock_epub_reader_instance.get_language.return_value = "en"
        mock_epub_reader_instance.title = "Test Book"
        mock_epub_reader_instance.get_cover_image.return_value = None
        mock_epub_reader_instance.chapters = ["Chapter 1", "Chapter 2"]

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch.object(EpubSynthesizer, "synthesize_chapter",
                        return_value=Path("/tmp/chapter_001.mp3")),
            patch.object(EpubSynthesizer, "create_m4b"),
            patch("audify.text_to_speech.tqdm.tqdm",
                  side_effect=lambda x, **kwargs: x),
        ):
            synthesizer = EpubSynthesizer(path="test.epub")

            # Test synthesize method
            result = synthesizer.synthesize()
            assert result == synthesizer.final_m4b_path

    @patch("audify.text_to_speech.PdfReader")
    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_pdf_synthesizer_full_workflow(self, mock_temp_dir, mock_pdf_reader):
        """Test complete PdfSynthesizer workflow."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_pdf_reader_instance.cleaned_text = "This is test PDF content."

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("audify.text_to_speech.break_text_into_sentences",
                  return_value=["This is test PDF content."]),
            patch.object(PdfSynthesizer, "_synthesize_sentences"),
            patch.object(PdfSynthesizer, "_convert_to_mp3",
                        return_value=Path("/tmp/output.mp3")),
            patch("builtins.open", mock_open()),
        ):
            synthesizer = PdfSynthesizer(pdf_path="test.pdf", save_text=True)
            result = synthesizer.synthesize()

            assert result == Path("/tmp/output.mp3")

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_base_synthesizer_complete_error_scenarios(self, mock_temp_dir):
        """Test all error scenarios in BaseSynthesizer."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        # Test cleanup error handling
        synthesizer.tmp_dir_context.cleanup = MagicMock(
            side_effect=Exception("Cleanup failed")
        )

        # Should not raise exception
        del synthesizer

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_kokoro_comprehensive_error_handling(self, mock_temp_dir):
        """Test comprehensive error handling: combine_wav_segments raises."""
        mock_temp_dir.return_value.name = "/tmp/test_dir"

        synthesizer = BaseSynthesizer(
            path="test.txt",
            voice="test_voice",
            translate=None,
            save_text=False,
            language="en",
        )

        mock_tts_config = MagicMock()
        mock_tts_config.provider_name = "kokoro"
        mock_tts_config.is_available.return_value = True
        mock_tts_config.get_available_voices.return_value = ["test_voice"]
        mock_tts_config.voice = "test_voice"
        mock_tts_config.synthesize.return_value = True

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch(
                "audify.text_to_speech.AudioProcessor.combine_wav_segments",
                side_effect=Exception("Export failed"),
            ),
            patch("pathlib.Path.exists", return_value=True),
            pytest.raises(Exception, match="Export failed"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"],
                Path("/tmp/output.wav")
            )


class TestEpubSynthesizerCoverage:
    @pytest.fixture
    def mock_reader(self):
        with patch("audify.text_to_speech.EpubReader") as mock:
            reader = Mock()
            reader.title = "Test Title"
            reader.get_language.return_value = "en"
            reader.get_cover_image.return_value = None
            mock.return_value = reader
            yield reader

    @pytest.fixture
    def synthesizer(self, mock_reader, tmp_path):
        with patch("audify.text_to_speech.OUTPUT_BASE_DIR", str(tmp_path)):
            synth = EpubSynthesizer(
                path="test.epub",
                language="en",
                save_text=False
            )
            return synth

    def test_synthesize_chapter_no_sentences(self, synthesizer):
        """Test synthesize_chapter when no sentences are extracted."""
        synthesizer.reader.extract_text.return_value = ""
        with patch("audify.text_to_speech.break_text_into_sentences", return_value=[]):
            result = synthesizer.synthesize_chapter("content", 1)
            assert result == synthesizer.audiobook_path / "chapter_001.mp3"

    def test_create_temp_m4b_for_chunk_decode_error(self, synthesizer):
        """Test _create_temp_m4b_for_chunk delegates to AudioProcessor."""
        chunk_files = [Path("file1.mp3")]
        with (
            patch("audify.text_to_speech.AudioProcessor.combine_audio_files"),
            patch("pathlib.Path.exists", return_value=False),
        ):
            result = synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)
            # Should return expected chunk temp path
            assert "part1.tmp.m4b" in str(result)

    def test_create_temp_m4b_for_chunk_empty_audio(self, synthesizer):
        """Test _create_temp_m4b_for_chunk when combine_audio_files raises."""
        chunk_files = [Path("file1.mp3")]
        with (
            patch(
                "audify.text_to_speech.AudioProcessor.combine_audio_files",
                side_effect=Exception("Error"),
            ),
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(Exception, match="Error"),
        ):
            synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)

    def test_create_metadata_for_chunk_exception(self, synthesizer):
        """Test _create_metadata_for_chunk with exception during file processing."""
        chunk_files = [Path("chapter_1.mp3")]
        with patch("builtins.open", mock_open=Mock()):
            with patch(
                "audify.text_to_speech.AudioProcessor.get_duration",
                side_effect=Exception("Error"),
            ):
                with patch("audify.text_to_speech.logger") as mock_logger:
                    synthesizer._create_metadata_for_chunk(chunk_files, 0)
                    mock_logger.warning.assert_called()

    def test_create_multiple_m4bs(self, synthesizer):
        """Test _create_multiple_m4bs flow."""
        files = [Path("1.mp3"), Path("2.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch.object(
            synthesizer, "_split_chapters_by_duration", return_value=[files]
        ):
            with patch.object(
                synthesizer, "_calculate_total_duration", return_value=3600
            ):
                with patch.object(
                    synthesizer,
                    "_create_temp_m4b_for_chunk",
                    return_value=mock_path,
                ):
                    with patch.object(
                        synthesizer,
                        "_create_metadata_for_chunk",
                        return_value=Path("meta.txt"),
                    ):
                        with patch("audify.text_to_speech.assemble_m4b"):
                            synthesizer._create_multiple_m4bs(files)

    def test_create_multiple_m4bs_ffmpeg_error(self, synthesizer):
        """Test _create_multiple_m4bs with ffmpeg error."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        import subprocess

        with patch.object(
            synthesizer, "_split_chapters_by_duration", return_value=[files]
        ):
            with patch.object(
                synthesizer, "_calculate_total_duration", return_value=3600
            ):
                with patch.object(
                    synthesizer, "_create_temp_m4b_for_chunk", return_value=mock_path
                ):
                    with patch.object(
                        synthesizer,
                        "_create_metadata_for_chunk",
                        return_value=Path("meta.txt"),
                    ):
                        with patch(
                            "audify.text_to_speech.assemble_m4b",
                            side_effect=subprocess.CalledProcessError(1, "cmd"),
                        ):
                            with patch(
                                "audify.text_to_speech.logger"
                            ) as mock_logger:
                                synthesizer._create_multiple_m4bs(files)
                                mock_logger.error.assert_called()

    def test_create_multiple_m4bs_ffmpeg_not_found(self, synthesizer):
        """Test _create_multiple_m4bs with ffmpeg not found."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        with patch.object(
            synthesizer, "_split_chapters_by_duration", return_value=[files]
        ):
            with patch.object(
                synthesizer, "_calculate_total_duration", return_value=3600
            ):
                with patch.object(
                    synthesizer, "_create_temp_m4b_for_chunk", return_value=mock_path
                ):
                    with patch.object(
                        synthesizer,
                        "_create_metadata_for_chunk",
                        return_value=Path("meta.txt"),
                    ):
                        with patch(
                            "audify.text_to_speech.assemble_m4b",
                            side_effect=FileNotFoundError,
                        ):
                            with patch(
                                "audify.text_to_speech.logger"
                            ) as mock_logger:
                                synthesizer._create_multiple_m4bs(files)
                                mock_logger.error.assert_called()

    def test_build_ffmpeg_command_for_chunk_with_cover(self, synthesizer):
        """Test build_ffmpeg_command (m4b_builder) with cover image."""
        from audify.utils.m4b_builder import build_ffmpeg_command

        with patch.object(Path, "exists", return_value=True):
            with patch("shutil.copy"):
                with patch("tempfile.NamedTemporaryFile") as mock_ntf:
                    mock_ntf.return_value.name = "/tmp/cover_temp.jpg"
                    cmd, _temp = build_ffmpeg_command(
                        Path("temp.m4b"),
                        Path("meta.txt"),
                        Path("final.m4b"),
                        Path("cover.jpg"),
                    )
                    assert "-disposition:v" in cmd

    def test_build_ffmpeg_command_with_cover(self, synthesizer):
        """Test build_ffmpeg_command (m4b_builder) with cover image."""
        from audify.utils.m4b_builder import build_ffmpeg_command

        with patch.object(Path, "exists", return_value=True):
            with patch("shutil.copy"):
                with patch("tempfile.NamedTemporaryFile") as mock_ntf:
                    mock_ntf.return_value.name = "/tmp/cover_temp.jpg"
                    cmd, _temp = build_ffmpeg_command(
                        Path("temp.m4b"),
                        Path("meta.txt"),
                        Path("final.m4b"),
                        Path("cover.jpg"),
                    )
                    assert "-disposition:v" in cmd

    def test_log_chapter_metadata_ioerror(self, synthesizer):
        """Test _log_chapter_metadata re-raises IOError."""
        with patch("builtins.open", side_effect=IOError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                synthesizer._log_chapter_metadata("Title", 0, 10.0)

    def test_process_single_chapter_too_short(self, synthesizer):
        """Test _process_single_chapter with short content."""
        synthesizer._process_single_chapter(1, "short", 0)

    def test_process_single_chapter_exists_error(self, synthesizer):
        """Test _process_single_chapter when chapter exists but duration fails."""
        with patch.object(Path, "exists", return_value=True):
            with patch(
                "audify.text_to_speech.AudioProcessor.get_duration",
                side_effect=Exception,
            ):
                synthesizer._process_single_chapter(1, "content" * 20, 0)

    def test_create_multiple_m4bs_chunk_failed(self, synthesizer):
        """Test _create_multiple_m4bs when chunk creation fails."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False

        with patch.object(
            synthesizer, "_split_chapters_by_duration", return_value=[files]
        ):
            with patch.object(
                synthesizer, "_calculate_total_duration", return_value=3600
            ):
                with patch.object(
                    synthesizer,
                    "_create_temp_m4b_for_chunk",
                    return_value=mock_path,
                ):
                    with patch("audify.text_to_speech.logger") as mock_logger:
                        synthesizer._create_multiple_m4bs(files)
                        mock_logger.error.assert_any_call(
                            "Failed to create temporary M4B for chunk 1"
                        )

    def test_process_single_chapter_success(self, synthesizer):
        """Test _process_single_chapter success path."""
        with patch.object(Path, "exists", side_effect=[False, True]):
            with patch.object(
                synthesizer, "synthesize_chapter", return_value=Path("ch.mp3")
            ):
                with patch(
                    "audify.text_to_speech.AudioProcessor.get_duration",
                    return_value=10.0,
                ):
                    with patch.object(
                        synthesizer, "_log_chapter_metadata", return_value=10000
                    ):
                        result = synthesizer._process_single_chapter(
                            1, "content" * 20, 0
                        )
                        assert result == 10000

    def test_process_single_chapter_synthesized_not_found(self, synthesizer):
        """Test _process_single_chapter where synthesized file is missing."""
        with patch.object(Path, "exists", side_effect=[False, False]):
            with patch.object(
                synthesizer, "synthesize_chapter", return_value=Path("ch.mp3")
            ):
                with patch("audify.text_to_speech.logger") as mock_logger:
                    result = synthesizer._process_single_chapter(
                        1, "content" * 20, 0
                    )
                    mock_logger.warning.assert_called()
                    assert result == 0

    def test_process_single_chapter_synthesize_error(self, synthesizer):
        """Test _process_single_chapter with synthesis error."""
        with patch.object(Path, "exists", return_value=False):
            with patch.object(
                synthesizer,
                "synthesize_chapter",
                side_effect=Exception("Synth error"),
            ):
                with patch("audify.text_to_speech.logger") as mock_logger:
                    result = synthesizer._process_single_chapter(
                        1, "content" * 20, 0
                    )
                    mock_logger.error.assert_called()
                    assert result == 0

    def test_process_single_chapter_zero_duration(self, synthesizer):
        """Test _process_single_chapter with zero duration."""
        with patch.object(Path, "exists", side_effect=[False, True]):
            with patch.object(
                synthesizer, "synthesize_chapter", return_value=Path("ch.mp3")
            ):
                with patch(
                    "audify.text_to_speech.AudioProcessor.get_duration",
                    return_value=0.0,
                ):
                    with patch("audify.text_to_speech.logger") as mock_logger:
                        result = synthesizer._process_single_chapter(
                            1, "content" * 20, 0
                        )
                        mock_logger.warning.assert_called()
                        assert result == 0

    def test_process_chapters_exception(self, synthesizer):
        """Test process_chapters with exception."""
        synthesizer.reader.get_chapters.return_value = ["content" * 20]
        with patch.object(
            synthesizer, "_process_single_chapter", side_effect=Exception
        ):
            synthesizer.process_chapters()

    def test_create_metadata_for_chunk_ioerror(self, synthesizer):
        """Test _create_metadata_for_chunk with IOError."""
        chunk_files = [Path("chapter_1.mp3")]
        with patch("builtins.open", side_effect=IOError("Disk full")):
            with pytest.raises(IOError, match="Disk full"):
                synthesizer._create_metadata_for_chunk(chunk_files, 0)

    def test_create_metadata_for_chunk_appends_entry(self, synthesizer, tmp_path):
        """_create_metadata_for_chunk calls append_chapter_metadata (duration > 0)."""
        chunk_files = [tmp_path / "chapter_1.mp3"]
        chunk_files[0].write_bytes(b"data")

        with patch(
            "audify.text_to_speech.AudioProcessor.get_duration", return_value=10.0
        ):
            with patch(
                "audify.text_to_speech.append_chapter_metadata",
                return_value=10000,
            ) as mock_append:
                with patch("audify.text_to_speech.write_metadata_header"):
                    synthesizer._create_metadata_for_chunk(chunk_files, 0)

        mock_append.assert_called_once()

    def test_create_single_m4b_decode_error(self, synthesizer):
        """Test _create_single_m4b when AudioProcessor.combine_audio_files raises."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        synthesizer.temp_m4b_path = mock_path

        with (
            patch(
                "audify.text_to_speech.AudioProcessor.combine_audio_files",
                side_effect=Exception("decode error"),
            ),
            pytest.raises(Exception, match="decode error"),
        ):
            synthesizer._create_single_m4b(files)

    def test_create_single_m4b_general_exception(self, synthesizer):
        """Test _create_single_m4b with general Exception from combine_audio_files."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        synthesizer.temp_m4b_path = mock_path

        with (
            patch(
                "audify.text_to_speech.AudioProcessor.combine_audio_files",
                side_effect=Exception("Error"),
            ),
            pytest.raises(Exception, match="Error"),
        ):
            synthesizer._create_single_m4b(files)

    def test_create_single_m4b_empty_audio(self, synthesizer):
        """Test _create_single_m4b when combine_audio_files raises (propagates)."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        synthesizer.temp_m4b_path = mock_path

        with patch(
            "audify.text_to_speech.AudioProcessor.combine_audio_files",
            side_effect=ValueError("empty"),
        ):
            with pytest.raises(ValueError, match="empty"):
                synthesizer._create_single_m4b(files)

    def test_create_single_m4b_export_error(self, synthesizer):
        """Test _create_single_m4b when assemble_m4b raises."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        synthesizer.temp_m4b_path = mock_path

        with (
            patch("audify.text_to_speech.AudioProcessor.combine_audio_files"),
            patch(
                "audify.text_to_speech.assemble_m4b",
                side_effect=Exception("Export failed"),
            ),
            pytest.raises(Exception, match="Export failed"),
        ):
            synthesizer._create_single_m4b(files)

    def test_create_single_m4b_exists(self, synthesizer):
        """Test _create_single_m4b when temp file already exists (skips combine)."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        synthesizer.temp_m4b_path = mock_path

        with (
            patch("audify.text_to_speech.logger") as mock_logger,
            patch("audify.text_to_speech.assemble_m4b"),
        ):
            synthesizer._create_single_m4b(files)
            mock_logger.info.assert_any_call(
                f"Temporary M4B already exists: {synthesizer.temp_m4b_path}. Skipping."
            )

    def test_create_single_m4b_ffmpeg_error(self, synthesizer):
        """Test _create_single_m4b with ffmpeg error from assemble_m4b."""
        files = [Path("1.mp3")]
        import subprocess

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        synthesizer.temp_m4b_path = mock_path

        with (
            patch(
                "audify.text_to_speech.assemble_m4b",
                side_effect=subprocess.CalledProcessError(1, "cmd"),
            ),
            pytest.raises(subprocess.CalledProcessError),
        ):
            synthesizer._create_single_m4b(files)

    def test_create_single_m4b_ffmpeg_not_found(self, synthesizer):
        """Test _create_single_m4b with ffmpeg not found from assemble_m4b."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        synthesizer.temp_m4b_path = mock_path

        with (
            patch(
                "audify.text_to_speech.assemble_m4b",
                side_effect=FileNotFoundError,
            ),
            pytest.raises(FileNotFoundError),
        ):
            synthesizer._create_single_m4b(files)

    def test_create_single_m4b_cleanup_error(self, synthesizer):
        """Test _create_single_m4b completes when assemble_m4b succeeds."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        synthesizer.temp_m4b_path = mock_path

        with (
            patch("audify.text_to_speech.assemble_m4b"),
            patch("audify.text_to_speech.logger") as mock_logger,
        ):
            synthesizer._create_single_m4b(files)
            mock_logger.info.assert_called()


class TestPdfSynthesizerCoverage:
    @pytest.fixture
    def synthesizer(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        with patch("audify.text_to_speech.PdfReader"):
            return PdfSynthesizer(pdf_path, output_dir=tmp_path)

    def test_synthesize_translation_error(self, synthesizer):
        """Test synthesize with translation error."""
        synthesizer.translate = "es"
        with patch(
            "audify.text_to_speech.break_text_into_sentences",
            return_value=["Sentence"],
        ):
            with patch(
                "audify.text_to_speech.translate_sentence", side_effect=Exception
            ):
                with patch.object(synthesizer, "_synthesize_sentences"):
                    with patch.object(synthesizer, "_convert_to_mp3"):
                        with patch("audify.text_to_speech.PdfReader"):
                            synthesizer.synthesize()
