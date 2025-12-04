
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from pydub.exceptions import CouldntDecodeError

from audify.text_to_speech import (
    EpubSynthesizer,
    PdfSynthesizer,
    VoiceSamplesSynthesizer,
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
        """Test _create_temp_m4b_for_chunk with CouldntDecodeError."""
        chunk_files = [Path("file1.mp3")]
        with patch(
            "audify.text_to_speech.AudioSegment.from_mp3",
            side_effect=CouldntDecodeError,
        ):
            with patch("audify.text_to_speech.logger") as mock_logger:
                synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)
                mock_logger.error.assert_any_call(
                    "Could not decode chapter file: file1.mp3, skipping."
                )

    def test_create_temp_m4b_for_chunk_empty_audio(self, synthesizer):
        """Test _create_temp_m4b_for_chunk when combined audio is empty."""
        chunk_files = [Path("file1.mp3")]
        with patch("audify.text_to_speech.AudioSegment.from_mp3") as mock_audio:
            mock_audio.return_value = MagicMock(__len__=lambda x: 0)
            # We need __add__ to return empty audio too or just mock it
            # But simpler is to have from_mp3 raise exception so loop continues
            # and audio remains empty
            mock_audio.side_effect = Exception("Error")

            with patch("audify.text_to_speech.logger") as mock_logger:
                synthesizer._create_temp_m4b_for_chunk(chunk_files, 0)
                mock_logger.error.assert_any_call(
                    "Combined audio for chunk 1 is empty."
                )

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
                        with patch.object(
                            synthesizer,
                            "_build_ffmpeg_command_for_chunk",
                            return_value=(["cmd"], None),
                        ):
                            with patch("subprocess.run"):
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
                        with patch.object(
                            synthesizer,
                            "_build_ffmpeg_command_for_chunk",
                            return_value=(["cmd"], None),
                        ):
                            with patch(
                                "subprocess.run",
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
                        with patch.object(
                            synthesizer,
                            "_build_ffmpeg_command_for_chunk",
                            return_value=(["cmd"], None),
                        ):
                            with patch("subprocess.run", side_effect=FileNotFoundError):
                                with patch(
                                    "audify.text_to_speech.logger"
                                ) as mock_logger:
                                    synthesizer._create_multiple_m4bs(files)
                                    mock_logger.error.assert_called()

    def test_build_ffmpeg_command_for_chunk_with_cover(self, synthesizer):
        """Test _build_ffmpeg_command_for_chunk with cover image."""
        synthesizer.cover_image_path = Path("cover.jpg")
        with patch.object(Path, "exists", return_value=True):
            with patch("shutil.copy"):
                cmd, temp = synthesizer._build_ffmpeg_command_for_chunk(
                    Path("temp.m4b"), Path("meta.txt"), Path("final.m4b")
                )
                assert "-disposition:v" in cmd

    def test_build_ffmpeg_command_with_cover(self, synthesizer):
        """Test _build_ffmpeg_command with cover image."""
        synthesizer.cover_image_path = Path("cover.jpg")
        with patch.object(Path, "exists", return_value=True):
            with patch("shutil.copy"):
                cmd, temp = synthesizer._build_ffmpeg_command([])
                assert "-disposition:v" in cmd

    def test_log_chapter_metadata_ioerror(self, synthesizer):
        """Test _log_chapter_metadata with IOError."""
        with patch("builtins.open", side_effect=IOError):
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

    def test_create_single_m4b_decode_error(self, synthesizer):
        """Test _create_single_m4b with CouldntDecodeError."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        synthesizer.temp_m4b_path = mock_path

        with patch(
            "audify.text_to_speech.AudioSegment.from_mp3",
            side_effect=CouldntDecodeError,
        ):
            with patch("audify.text_to_speech.logger") as mock_logger:
                synthesizer._create_single_m4b(files)
                mock_logger.error.assert_any_call(
                    f"Could not decode chapter file: {files[0]}, skipping."
                )

    def test_create_single_m4b_general_exception(self, synthesizer):
        """Test _create_single_m4b with general Exception during processing."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        synthesizer.temp_m4b_path = mock_path

        with patch(
            "audify.text_to_speech.AudioSegment.from_mp3",
            side_effect=Exception("Error"),
        ):
            with patch("audify.text_to_speech.logger") as mock_logger:
                synthesizer._create_single_m4b(files)
                # Should log error and continue (resulting in empty audio)
                mock_logger.error.assert_called()

    def test_create_single_m4b_empty_audio(self, synthesizer):
        """Test _create_single_m4b resulting in empty audio."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        synthesizer.temp_m4b_path = mock_path

        with patch(
            "audify.text_to_speech.AudioSegment.from_mp3", side_effect=Exception
        ):
            with patch("audify.text_to_speech.logger") as mock_logger:
                synthesizer._create_single_m4b(files)
                mock_logger.error.assert_any_call(
                    "Combined audio is empty. Cannot create M4B."
                )

    def test_create_single_m4b_export_error(self, synthesizer):
        """Test _create_single_m4b export error."""
        files = [Path("1.mp3")]
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 1000
        mock_audio.__add__.return_value = mock_audio
        mock_audio.__iadd__.return_value = mock_audio
        mock_audio.export.side_effect = Exception("Export failed")

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        synthesizer.temp_m4b_path = mock_path

        with patch(
            "audify.text_to_speech.AudioSegment.from_mp3", return_value=mock_audio
        ):
            with patch(
                "audify.text_to_speech.AudioSegment.empty", return_value=mock_audio
            ):
                with pytest.raises(Exception, match="Export failed"):
                    synthesizer._create_single_m4b(files)

    def test_create_single_m4b_exists(self, synthesizer):
        """Test _create_single_m4b when temp file exists."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        synthesizer.temp_m4b_path = mock_path

        with patch("audify.text_to_speech.logger") as mock_logger:
            with patch.object(
                synthesizer,
                "_build_ffmpeg_command",
                return_value=(["cmd"], None),
            ):
                with patch("subprocess.run"):
                    synthesizer._create_single_m4b(files)
                    mock_logger.info.assert_any_call(
                        f"Temporary M4B file already exists: "
                        f"{synthesizer.temp_m4b_path}. Skipping combination."
                    )

    def test_create_single_m4b_ffmpeg_error(self, synthesizer):
        """Test _create_single_m4b with ffmpeg error."""
        files = [Path("1.mp3")]
        import subprocess

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        synthesizer.temp_m4b_path = mock_path

        with patch.object(
            synthesizer, "_build_ffmpeg_command", return_value=(["cmd"], None)
        ):
            with patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "cmd"),
            ):
                with pytest.raises(subprocess.CalledProcessError):
                    synthesizer._create_single_m4b(files)

    def test_create_single_m4b_ffmpeg_not_found(self, synthesizer):
        """Test _create_single_m4b with ffmpeg not found."""
        files = [Path("1.mp3")]
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        synthesizer.temp_m4b_path = mock_path

        with patch.object(
            synthesizer, "_build_ffmpeg_command", return_value=(["cmd"], None)
        ):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                with pytest.raises(FileNotFoundError):
                    synthesizer._create_single_m4b(files)

    def test_create_single_m4b_cleanup_error(self, synthesizer):
        """Test _create_single_m4b cleanup error."""
        files = [Path("1.mp3")]
        mock_cover = MagicMock()
        mock_cover.name = "cover.jpg"

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        synthesizer.temp_m4b_path = mock_path

        with patch.object(
            synthesizer,
            "_build_ffmpeg_command",
            return_value=(["cmd"], mock_cover),
        ):
            with patch("subprocess.run"):
                with patch(
                    "pathlib.Path.unlink", side_effect=Exception("Cleanup error")
                ):
                    with patch("audify.text_to_speech.logger") as mock_logger:
                        synthesizer._create_single_m4b(files)
                        mock_logger.warning.assert_called_with(
                            f"Error cleaning up temporary cover file "
                            f"{mock_cover.name}: Cleanup error"
                        )


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


class TestVoiceSamplesSynthesizerCoverage:
    @pytest.fixture
    def synthesizer(self, tmp_path):
        with patch("audify.text_to_speech.OUTPUT_BASE_DIR", tmp_path):
            return VoiceSamplesSynthesizer()

    def test_get_available_models_and_voices_error(self, synthesizer):
        """Test _get_available_models_and_voices with request error."""
        with patch("requests.get", side_effect=requests.RequestException):
            models, voices = synthesizer._get_available_models_and_voices()
            assert models == []
            assert voices == []

    def test_create_sample_for_combination_error(self, synthesizer):
        """Test _create_sample_for_combination with error."""
        with patch("audify.text_to_speech.BaseSynthesizer", side_effect=Exception):
            result = synthesizer._create_sample_for_combination("model", "voice", 1)
            assert result is None

    def test_create_m4b_from_samples_errors(self, synthesizer):
        """Test _create_m4b_from_samples with various errors."""
        # Empty samples
        synthesizer._create_m4b_from_samples([])

        # Error processing sample
        with patch(
            "audify.text_to_speech.AudioSegment.from_mp3", side_effect=Exception
        ):
            synthesizer._create_m4b_from_samples([Path("sample.mp3")])

    def test_append_chapter_metadata_error(self, synthesizer):
        """Test _append_chapter_metadata with error."""
        with patch("builtins.open", side_effect=Exception):
            synthesizer._append_chapter_metadata(0, 10.0, "Title")

    def test_finalize_m4b_filenotfound(self, synthesizer):
        """Test _finalize_m4b with FileNotFoundError."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                synthesizer._finalize_m4b()

    def test_synthesize_no_models(self, synthesizer):
        """Test synthesize with no models."""
        with patch.object(
            synthesizer, "_get_available_models_and_voices", return_value=([], [])
        ):
            synthesizer.synthesize()

    def test_synthesize_max_samples(self, synthesizer):
        """Test synthesize with max_samples."""
        synthesizer.max_samples = 1
        with patch.object(
            synthesizer,
            "_get_available_models_and_voices",
            return_value=(["m1", "m2"], ["v1"]),
        ):
            with patch.object(synthesizer, "_create_metadata_file"):
                with patch.object(
                    synthesizer, "_create_sample_for_combination", return_value=None
                ):
                    synthesizer.synthesize()
