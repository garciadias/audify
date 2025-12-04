
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

from audify.text_to_speech import (
    EpubSynthesizer,
    PdfSynthesizer,
    VoiceSamplesSynthesizer,
)


class TestTextToSpeechCoverage2:
    @pytest.fixture
    def synthesizer(self):
        with patch("audify.text_to_speech.EpubReader") as mock_reader, patch(
            "audify.text_to_speech.Path.exists", return_value=True
        ), patch("audify.text_to_speech.Path.mkdir"):
            mock_reader.return_value.get_language.return_value = "en"
            mock_reader.return_value.title = "Test Title"
            return EpubSynthesizer(
                Path("test.epub"), speaker="af_bella", translate=None, save_text=False
            )

    def test_create_temp_m4b_for_chunk_exception_cleanup(self, synthesizer):
        """Test cleanup in _create_temp_m4b_for_chunk when exception occurs."""
        chunk_files = [Path("1.mp3")]

        with patch("audify.text_to_speech.AudioSegment.empty") as mock_empty, \
             patch("audify.text_to_speech.AudioSegment.from_mp3") as mock_from_mp3:
            mock_audio = MagicMock()
            mock_audio.__len__.return_value = 1000
            mock_audio.__iadd__.return_value = mock_audio
            mock_empty.return_value = mock_audio
            mock_from_mp3.return_value = mock_audio
            # Simulate exception during export
            mock_audio.export.side_effect = Exception("Export failed")

            # Mock the temp path to verify unlink is called
            synthesizer.audiobook_path = Path("/tmp")
            with patch("pathlib.Path.unlink") as mock_unlink:
                with pytest.raises(Exception, match="Export failed"):
                    synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)

                # Verify unlink was called (line 436)
                assert mock_unlink.called

    def test_create_metadata_for_chunk_exception(self, synthesizer):
        """Test exception handling in _create_metadata_for_chunk."""
        chunk_files = [Path("chapter_1.mp3")]

        # Mock open to raise IOError
        with patch("builtins.open", side_effect=IOError("Write failed")):
            with pytest.raises(IOError, match="Write failed"):
                synthesizer._create_metadata_for_chunk(chunk_files, 0)

    def test_create_metadata_for_chunk_inner_exception(self, synthesizer):
        """Test inner exception handling in _create_metadata_for_chunk loop."""
        chunk_files = [Path("chapter_1.mp3")]

        # Mock open to succeed
        with patch("builtins.open", mock_open()):
            # Mock AudioProcessor.get_duration to raise exception
            with patch(
                "audify.text_to_speech.AudioProcessor.get_duration",
                side_effect=Exception("Duration failed"),
            ):
                with patch("audify.text_to_speech.logger") as mock_logger:
                    synthesizer._create_metadata_for_chunk(chunk_files, 0)
                    # Verify warning was logged (lines 464-471)
                    mock_logger.warning.assert_called_with(
                        "Could not process metadata for chapter_1.mp3: Duration failed"
                    )

    def test_create_single_m4b_export_exception_cleanup(self, synthesizer):
        """Test cleanup in _create_single_m4b when export fails."""
        files = [Path("1.mp3")]
        synthesizer.temp_m4b_path = MagicMock(spec=Path)
        synthesizer.temp_m4b_path.exists.return_value = False

        with patch(
            "audify.text_to_speech.AudioSegment.from_mp3"
        ) as mock_from_mp3, patch(
            "audify.text_to_speech.AudioSegment.empty"
        ) as mock_empty:
            mock_audio = MagicMock()
            mock_empty.return_value = mock_audio
            mock_from_mp3.return_value = mock_audio

            # Ensure combined audio is not empty
            mock_audio.__len__.return_value = 1000
            mock_audio.__iadd__.return_value = mock_audio

            mock_audio.export.side_effect = Exception("Export failed")

            with pytest.raises(Exception, match="Export failed"):
                synthesizer._create_single_m4b(files)

            # Verify unlink was called (line 588)
            synthesizer.temp_m4b_path.unlink.assert_called_with(missing_ok=True)

    def test_create_multiple_m4bs_cover_cleanup_exception(self, synthesizer):
        """Test cover cleanup exception in _create_multiple_m4bs."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        # Setup mocks for the happy path until cleanup
        with patch.object(
            synthesizer, "_split_chapters_by_duration", return_value=[files]
        ), patch.object(
            synthesizer, "_calculate_total_duration", return_value=3600
        ), patch.object(
            synthesizer, "_create_temp_m4b_for_chunk", return_value=mock_path
        ), patch.object(
            synthesizer, "_create_metadata_for_chunk", return_value=Path("meta.txt")
        ), patch(
            "subprocess.run"
        ):

            # Mock _build_ffmpeg_command_for_chunk to return a mock cover file
            mock_cover = MagicMock()
            mock_cover.name = "temp_cover.jpg"
            # Simulate exception during cleanup
            mock_cover.close.side_effect = Exception("Cleanup failed")

            with patch.object(
                synthesizer,
                "_build_ffmpeg_command_for_chunk",
                return_value=(["cmd"], mock_cover),
            ):
                with patch("audify.text_to_speech.logger") as mock_logger:
                    synthesizer._create_multiple_m4bs(files)
                    # Verify warning logged (lines 669-673)
                    mock_logger.warning.assert_called()
                    assert (
                        "Error cleaning up temporary cover file"
                        in mock_logger.warning.call_args[0][0]
                    )

    def test_process_single_chapter_existing_duration_error(self, synthesizer):
        """Test error getting duration for existing chapter."""
        synthesizer.audiobook_path = Path("/tmp")
        content = "a" * 200  # Long enough content

        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "audify.text_to_speech.AudioProcessor.get_duration",
                side_effect=Exception("Duration error"),
            ):
                with patch("audify.text_to_speech.logger") as mock_logger:
                    synthesizer._process_single_chapter(1, content, 0)
                    # Verify warning logged (lines 885-886)
                    mock_logger.warning.assert_called()
                    assert (
                        "Could not get duration for existing chapter 1"
                        in mock_logger.warning.call_args[0][0]
                    )

    def test_process_single_chapter_synthesized_not_found(self, synthesizer):
        """Test synthesized chapter not found in _process_single_chapter."""
        synthesizer.audiobook_path = Path("/tmp")
        content = "a" * 200  # Long enough content

        # First exists check returns False (not synthesized yet)
        # Second exists check (after synthesis) returns False (failed to create)
        with patch("pathlib.Path.exists", side_effect=[False, False]):
            with patch.object(
                synthesizer, "synthesize_chapter", return_value=Path("chapter_1.mp3")
            ):
                with patch("audify.text_to_speech.logger") as mock_logger:
                    synthesizer._process_single_chapter(1, content, 0)
                    # Verify warning logged (line 896)
                    mock_logger.warning.assert_called()
                    assert (
                        "Synthesized chapter 1 MP3 not found"
                        in mock_logger.warning.call_args[0][0]
                    )


class TestPdfSynthesizerCoverage:
    @pytest.fixture
    def pdf_synthesizer(self):
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.mkdir"
        ):
            return PdfSynthesizer(Path("test.pdf"))

    def test_synthesize_no_sentences(self, pdf_synthesizer):
        """Test synthesize when no sentences extracted."""
        with patch("audify.text_to_speech.PdfReader") as mock_reader:
            mock_reader.return_value.cleaned_text = ""
            with patch(
                "audify.text_to_speech.break_text_into_sentences", return_value=[]
            ):
                with patch("audify.text_to_speech.logger") as mock_logger:
                    pdf_synthesizer.synthesize()
                    # Verify warning logged (line 1073)
                    mock_logger.warning.assert_called_with(
                        "No text extracted from PDF. Cannot synthesize."
                    )

    def test_synthesize_translation_error(self, pdf_synthesizer):
        """Test translation error in synthesize."""
        pdf_synthesizer.translate = "es"
        pdf_synthesizer.language = "en"

        with patch("audify.text_to_speech.PdfReader") as mock_reader:
            mock_reader.return_value.cleaned_text = "Hello."
            with patch(
                "audify.text_to_speech.break_text_into_sentences",
                return_value=["Hello."],
            ):
                with patch(
                    "audify.text_to_speech.translate_sentence",
                    side_effect=Exception("Translation failed"),
                ):
                    with patch("audify.text_to_speech.logger") as mock_logger:
                        # Mock _synthesize_sentences and _convert_to_mp3
                        with patch.object(
                            pdf_synthesizer, "_synthesize_sentences"
                        ), patch.object(pdf_synthesizer, "_convert_to_mp3"):
                            pdf_synthesizer.synthesize()
                            # Verify error logged (lines 1093-1094)
                            mock_logger.error.assert_called()
                            assert (
                                "Error translating PDF content"
                                in mock_logger.error.call_args[0][0]
                            )


class TestVoiceSamplesSynthesizerCoverage:
    @pytest.fixture
    def voice_synthesizer(self):
        with patch("pathlib.Path.mkdir"), patch(
            "tempfile.mkdtemp", return_value="/tmp/voice_samples"
        ):
            return VoiceSamplesSynthesizer()

    def test_get_available_models_and_voices_request_exception(self, voice_synthesizer):
        """Test request exception in _get_available_models_and_voices."""
        with patch(
            "requests.get", side_effect=requests.RequestException("Network error")
        ):
            with patch("audify.text_to_speech.logger") as mock_logger:
                models, voices = voice_synthesizer._get_available_models_and_voices()
                assert models == []
                assert voices == []
                # Verify error logged (lines 1121-1123)
                mock_logger.error.assert_called()
                assert (
                    "Error fetching models and voices"
                    in mock_logger.error.call_args[0][0]
                )

    def test_create_sample_for_combination_output_not_found(self, voice_synthesizer):
        """Test output wav not found in _create_sample_for_combination."""
        with patch("audify.text_to_speech.BaseSynthesizer"), patch(
            "audify.text_to_speech.break_text_into_sentences"
        ), patch(
            "pathlib.Path.exists", return_value=False
        ):  # Output wav doesn't exist

            with patch("audify.text_to_speech.logger") as mock_logger:
                result = voice_synthesizer._create_sample_for_combination(
                    "model", "voice", 1
                )
                assert result is None
                # Verify warning logged (line 1149)
                mock_logger.warning.assert_called_with(
                    "Failed to create sample for model + voice"
                )

    def test_create_metadata_file_exception(self, voice_synthesizer):
        """Test exception in _create_metadata_file."""
        with patch("builtins.open", side_effect=Exception("Write failed")):
            with patch("audify.text_to_speech.logger") as mock_logger:
                with pytest.raises(Exception, match="Write failed"):
                    voice_synthesizer._create_metadata_file([])
                # Verify error logged (lines 1177-1179)
                mock_logger.error.assert_called()
                assert (
                    "Error creating metadata file" in mock_logger.error.call_args[0][0]
                )
