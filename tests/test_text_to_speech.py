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
    TTSSynthesisError,
    suppress_stdout,
)
from audify.utils.api_config import KokoroAPIConfig

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
        mock_tts_config.max_text_length = 5000

        mock_tts_config.limit_unit = "chars"
        mock_tts_config.synthesize.return_value = False

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            pytest.raises(TTSSynthesisError, match="failure threshold"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"], Path("/tmp/output.wav")
            )

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

class TestEpubSynthesizer:
    """Tests for EpubSynthesizer class."""

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
                sentence="Livre de Test",
                src_lang="fr",
                tgt_lang="en",
                model=None,
                base_url=None,
            )

class TestPdfSynthesizer:
    """Tests for PdfSynthesizer class."""

    def test_pdf_synthesizer_init_file_not_found(self):
        """Test PdfSynthesizer initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            PdfSynthesizer(pdf_path="nonexistent.pdf")

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
        mock_tts_config.max_text_length = 15  # Force each sentence into its own batch

        mock_tts_config.limit_unit = "chars"
        mock_tts_config.synthesize.return_value = True

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            synthesizer._synthesize_sentences(
                ["Hello world", "Test sentence"], Path("/tmp/output.wav")
            )

        # Verify the synthesis was called for both sentences
        assert mock_tts_config.synthesize.call_count == 2

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
        mock_tts_config.max_text_length = 15  # Force each sentence into its own batch

        mock_tts_config.limit_unit = "chars"
        # First returns True (path exists), second returns False (skipped)
        mock_tts_config.synthesize.side_effect = [True, False]

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            patch("pathlib.Path.exists", side_effect=[True, False]),
            pytest.raises(TTSSynthesisError, match="failure threshold"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world", "Test sentence"], Path("/tmp/output.wav")
            )

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
        mock_tts_config.max_text_length = 5000

        mock_tts_config.limit_unit = "chars"
        mock_tts_config.synthesize.side_effect = Exception("Synthesis error")

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("audify.text_to_speech.AudioProcessor.combine_wav_segments"),
            pytest.raises(TTSSynthesisError, match="failure threshold"),
        ):
            synthesizer._synthesize_with_provider(
                ["Hello world"], Path("/tmp/output.wav")
            )

    @patch("audify.text_to_speech.tempfile.TemporaryDirectory")
    def test_synthesize_kokoro_all_invalid_segments_raises(self, mock_temp_dir):
        """Raise a clear error when synthesis yields no valid WAV segments."""
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
        mock_tts_config.max_text_length = 5000

        mock_tts_config.limit_unit = "chars"
        mock_tts_config.synthesize.return_value = True

        with (
            patch.object(synthesizer, "_get_tts_config", return_value=mock_tts_config),
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "audify.text_to_speech.AudioProcessor.combine_wav_segments",
                side_effect=ValueError(
                    "Combined WAV segments are empty; no valid segments found."
                ),
            ),
            pytest.raises(
                RuntimeError,
                match="No valid audio segments were synthesized",
            ),
        ):
            synthesizer._synthesize_with_provider(["Hello world"], Path("/tmp/out.wav"))


class TestEpubSynthesizerAdvancedCoverage:
    """Tests to cover all missing EpubSynthesizer functionality."""

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
            patch("audify.text_to_speech.track", side_effect=lambda x, **kwargs: x),
        ):
            mock_convert.return_value = Path("/tmp/chapter_001.mp3")

            synthesizer = EpubSynthesizer(path="test.epub", language="fr")
            # Test translation error during chapter synthesis
            with patch(
                "audify.text_to_speech.translate_sentence",
                side_effect=Exception("Translation failed"),
            ):
                synthesizer.translate = "en"  # Set after init to avoid title error
                result = synthesizer.synthesize_chapter("chapter content", 1)

            # Should fall back to original sentences
            assert "chapter_001.mp3" in str(result)

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

        # Mock long duration that requires splitting (over 6 hours)
        mock_get_duration.return_value = 30000.0  # 8.3 hours per file

        mock_file = mock_open()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_file),
            patch(
                "pathlib.Path.glob",
                return_value=[
                    Path("/tmp/chapter_001.mp3"),
                    Path("/tmp/chapter_002.mp3"),
                ],
            ),
            patch.object(
                EpubSynthesizer, "_create_multiple_m4bs"
            ) as mock_create_multiple,
        ):
            synthesizer = EpubSynthesizer(path="test.epub")
            synthesizer.create_m4b()

            # Should call multiple M4B creation
            mock_create_multiple.assert_called_once()


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
            synth = EpubSynthesizer(path="test.epub", language="en", save_text=False)
            return synth

    def test_synthesize_chapter_no_sentences(self, synthesizer):
        """Test synthesize_chapter when no sentences are extracted."""
        synthesizer.reader.extract_text.return_value = ""
        with patch("audify.text_to_speech.break_text_into_sentences", return_value=[]):
            result = synthesizer.synthesize_chapter("content", 1)
            assert result == synthesizer.audiobook_path / "chapter_001.mp3"

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
                    result = synthesizer._process_single_chapter(1, "content" * 20, 0)
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
                    result = synthesizer._process_single_chapter(1, "content" * 20, 0)
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

class TestPdfSynthesizerCoverage:
    @pytest.fixture
    def synthesizer(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        with patch("audify.text_to_speech.PdfReader"):
            return PdfSynthesizer(pdf_path, output_dir=tmp_path)
