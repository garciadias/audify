"""
Comprehensive tests for the AudioProcessor utility class.

Tests cover all methods including error handling, edge cases, and different
audio formats to achieve maximum code coverage.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydub.exceptions import CouldntDecodeError

from audify.utils.audio import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor class methods."""

    def test_get_duration_success(self):
        """Test get_duration with valid audio file."""
        with patch("pydub.AudioSegment.from_file") as mock_from_file:
            mock_audio = MagicMock()
            mock_audio.__len__.return_value = 5000  # 5 seconds in milliseconds
            mock_from_file.return_value = mock_audio

            duration = AudioProcessor.get_duration("test.mp3")

            assert duration == 5.0
            mock_from_file.assert_called_once_with("test.mp3")

    def test_get_duration_path_object(self):
        """Test get_duration with Path object."""
        with patch("pydub.AudioSegment.from_file") as mock_from_file:
            mock_audio = MagicMock()
            mock_audio.__len__.return_value = 3000  # 3 seconds
            mock_from_file.return_value = mock_audio

            file_path = Path("test.wav")
            duration = AudioProcessor.get_duration(file_path)

            assert duration == 3.0
            mock_from_file.assert_called_once_with("test.wav")

    def test_get_duration_couldnt_decode_error(self):
        """Test get_duration when file cannot be decoded."""
        with patch("pydub.AudioSegment.from_file") as mock_from_file:
            mock_from_file.side_effect = CouldntDecodeError("Invalid format")

            duration = AudioProcessor.get_duration("invalid.mp3")

            assert duration == 0.0

    def test_get_duration_generic_error(self):
        """Test get_duration with generic exception."""
        with patch("pydub.AudioSegment.from_file") as mock_from_file:
            mock_from_file.side_effect = Exception("File not found")

            duration = AudioProcessor.get_duration("missing.mp3")

            assert duration == 0.0

    def test_convert_wav_to_mp3_success(self, tmp_path):
        """Test successful WAV to MP3 conversion."""
        wav_path = tmp_path / "test.wav"
        mp3_path = tmp_path / "test.mp3"

        with patch("pydub.AudioSegment.from_wav") as mock_from_wav:
            mock_audio = MagicMock()
            mock_from_wav.return_value = mock_audio

            result = AudioProcessor.convert_wav_to_mp3(wav_path)

            assert result == mp3_path
            mock_from_wav.assert_called_once_with(str(wav_path))
            mock_audio.export.assert_called_once_with(
                str(mp3_path), format="mp3", bitrate="192k"
            )

    def test_convert_wav_to_mp3_custom_bitrate(self, tmp_path):
        """Test WAV to MP3 conversion with custom bitrate."""
        wav_path = tmp_path / "test.wav"

        with patch("pydub.AudioSegment.from_wav") as mock_from_wav:
            mock_audio = MagicMock()
            mock_from_wav.return_value = mock_audio

            AudioProcessor.convert_wav_to_mp3(
                wav_path, bitrate="320k", remove_original=False
            )

            mock_audio.export.assert_called_once_with(
                str(tmp_path / "test.mp3"), format="mp3", bitrate="320k"
            )

    def test_convert_wav_to_mp3_file_not_found(self, tmp_path):
        """Test convert_wav_to_mp3 when WAV file doesn't exist."""
        wav_path = tmp_path / "nonexistent.wav"

        with patch("pydub.AudioSegment.from_wav") as mock_from_wav:
            mock_from_wav.side_effect = FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                AudioProcessor.convert_wav_to_mp3(wav_path)

    def test_convert_wav_to_mp3_conversion_error(self, tmp_path):
        """Test convert_wav_to_mp3 with conversion error."""
        wav_path = tmp_path / "test.wav"

        with patch("pydub.AudioSegment.from_wav") as mock_from_wav:
            mock_from_wav.side_effect = Exception("Conversion failed")

            with pytest.raises(Exception, match="Conversion failed"):
                AudioProcessor.convert_wav_to_mp3(wav_path)

    def test_combine_audio_files_success(self, tmp_path):
        """Test successful combination of audio files."""
        file_paths = [tmp_path / "file1.mp3", tmp_path / "file2.wav"]
        output_path = tmp_path / "combined.wav"

        # Create a mock AudioSegment that behaves properly
        mock_combined = MagicMock()
        mock_combined.__len__.return_value = 1000  # Non-zero length
        mock_combined.__iadd__.return_value = mock_combined

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, patch(
            "pydub.AudioSegment.from_wav"
        ) as mock_from_wav, patch("pydub.AudioSegment.empty") as mock_empty:

            mock_audio1 = MagicMock()
            mock_audio2 = MagicMock()

            mock_from_mp3.return_value = mock_audio1
            mock_from_wav.return_value = mock_audio2
            mock_empty.return_value = mock_combined

            result = AudioProcessor.combine_audio_files(
                file_paths, output_path, show_progress=False
            )

            assert result == mock_combined
            assert mock_combined.__iadd__.call_count == 2
            mock_combined.export.assert_called_once()

    def test_combine_audio_files_with_progress(self, tmp_path):
        """Test combine_audio_files with progress bar."""
        file_paths = [tmp_path / "file1.mp3"]
        output_path = tmp_path / "combined.wav"

        mock_combined = MagicMock()
        mock_combined.__len__.return_value = 1000
        mock_combined.__iadd__.return_value = mock_combined

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, patch(
            "pydub.AudioSegment.empty"
        ) as mock_empty, patch("tqdm.tqdm") as mock_tqdm:

            mock_audio = MagicMock()

            mock_from_mp3.return_value = mock_audio
            mock_empty.return_value = mock_combined
            mock_tqdm.return_value = file_paths

            AudioProcessor.combine_audio_files(
                file_paths, output_path, show_progress=True, description="Testing"
            )

            mock_tqdm.assert_called_once_with(
                file_paths, desc="Testing", unit="file"
            )

    def test_combine_audio_files_empty_result(self, tmp_path):
        """Test combine_audio_files when result is empty."""
        file_paths = [tmp_path / "file1.mp3"]
        output_path = tmp_path / "combined.wav"

        mock_combined = MagicMock()
        mock_combined.__len__.return_value = 0  # Empty audio
        mock_combined.__iadd__.return_value = mock_combined

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, patch(
            "pydub.AudioSegment.empty"
        ) as mock_empty:

            mock_audio = MagicMock()

            mock_from_mp3.return_value = mock_audio
            mock_empty.return_value = mock_combined

            with pytest.raises(ValueError, match="Combined audio is empty"):
                AudioProcessor.combine_audio_files(
                    file_paths, output_path, show_progress=False
                )

    def test_combine_audio_files_no_valid_files(self, tmp_path):
        """Test combine_audio_files when no valid files are found."""
        file_paths = [tmp_path / "nonexistent.mp3"]
        output_path = tmp_path / "combined.wav"

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, patch(
            "pydub.AudioSegment.empty"
        ) as mock_empty:
            mock_from_mp3.side_effect = FileNotFoundError("File not found")
            mock_empty.return_value = MagicMock()

            with pytest.raises(
                ValueError, match="No valid audio files found to combine"
            ):
                AudioProcessor.combine_audio_files(
                    file_paths, output_path, show_progress=False
                )

    def test_combine_audio_files_mp3_output(self, tmp_path):
        """Test combine_audio_files with MP3 output format."""
        file_paths = [tmp_path / "file1.mp3"]
        output_path = tmp_path / "combined.mp3"

        mock_combined = MagicMock()
        mock_combined.__len__.return_value = 1000
        mock_combined.__iadd__.return_value = mock_combined

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, patch(
            "pydub.AudioSegment.empty"
        ) as mock_empty:

            mock_audio = MagicMock()

            mock_from_mp3.return_value = mock_audio
            mock_empty.return_value = mock_combined

            AudioProcessor.combine_audio_files(
                file_paths, output_path, output_format="mp3", show_progress=False
            )

            mock_combined.export.assert_called_once_with(
                str(output_path), format="mp3", bitrate="192k"
            )

    def test_combine_audio_files_mp4_output(self, tmp_path):
        """Test combine_audio_files with MP4 output format."""
        file_paths = [tmp_path / "file1.mp3"]
        output_path = tmp_path / "combined.mp4"

        mock_combined = MagicMock()
        mock_combined.__len__.return_value = 1000
        mock_combined.__iadd__.return_value = mock_combined

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, patch(
            "pydub.AudioSegment.empty"
        ) as mock_empty:

            mock_audio = MagicMock()

            mock_from_mp3.return_value = mock_audio
            mock_empty.return_value = mock_combined

            AudioProcessor.combine_audio_files(
                file_paths, output_path, output_format="mp4", show_progress=False
            )

            mock_combined.export.assert_called_once_with(
                str(output_path), format="mp4", codec="aac", bitrate="64k"
            )

    def test_combine_audio_files_decode_error(self, tmp_path):
        """Test combine_audio_files when one file has decode error."""
        file_paths = [tmp_path / "good.mp3", tmp_path / "bad.mp3"]
        output_path = tmp_path / "combined.wav"

        mock_combined = MagicMock()
        mock_combined.__len__.return_value = 1000
        mock_combined.__iadd__.return_value = mock_combined

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, patch(
            "pydub.AudioSegment.empty"
        ) as mock_empty:

            mock_audio = MagicMock()

            def side_effect(path):
                if "bad" in path:
                    raise CouldntDecodeError("Cannot decode")
                return mock_audio

            mock_from_mp3.side_effect = side_effect
            mock_empty.return_value = mock_combined

            result = AudioProcessor.combine_audio_files(
                file_paths, output_path, show_progress=False
            )

            assert result == mock_combined
            # Only one file should be successfully added
            assert mock_combined.__iadd__.call_count == 1

    def test_split_audio_by_duration_multiple_files(self):
        """Test successful audio splitting by duration with multiple files."""
        file_paths = [Path("file1.mp3"), Path("file2.mp3"), Path("file3.mp3")]

        with patch.object(AudioProcessor, "get_duration") as mock_get_duration:
            # Each file is 2 hours (7200 seconds)
            mock_get_duration.return_value = 7200.0

            chunks = AudioProcessor.split_audio_by_duration(
                file_paths, max_duration_hours=5.0
            )

            # With 5-hour limit, first two files fit together (4 hours < 5 hours)
            expected_chunks = [[file_paths[0], file_paths[1]], [file_paths[2]]]
            assert len(chunks) == 2
            assert chunks[0] == expected_chunks[0]
            assert chunks[1] == expected_chunks[1]

    def test_split_audio_by_duration_single_file_exceeds_limit(self):
        """Test splitting when a single file exceeds the duration limit."""
        file_paths = [Path("huge_file.mp3")]

        with patch.object(AudioProcessor, "get_duration") as mock_get_duration:
            # File is 20 hours, limit is 15 hours
            mock_get_duration.return_value = 72000.0  # 20 hours in seconds

            chunks = AudioProcessor.split_audio_by_duration(
                file_paths, max_duration_hours=15.0
            )

            # Single huge file should still create one chunk
            assert len(chunks) == 1
            assert chunks[0] == file_paths

    def test_split_audio_by_duration_with_errors(self):
        """Test splitting when some files have duration errors."""
        file_paths = [Path("good.mp3"), Path("bad.mp3"), Path("good2.mp3")]

        with patch.object(AudioProcessor, "get_duration") as mock_get_duration:
            # First and third files are good, second has error
            mock_get_duration.side_effect = [3600.0, Exception("Error"), 3600.0]

            chunks = AudioProcessor.split_audio_by_duration(
                file_paths, max_duration_hours=2.0
            )

            # All files should be included despite the error
            assert len(chunks) == 1
            assert chunks[0] == file_paths

    def test_split_audio_by_duration_empty_list(self):
        """Test splitting with empty file list."""
        file_paths = []

        chunks = AudioProcessor.split_audio_by_duration(file_paths)

        assert chunks == []

    def test_split_audio_by_duration_default_max_duration(self):
        """Test splitting with default max duration."""
        file_paths = [Path("file1.mp3")]

        with patch.object(AudioProcessor, "get_duration") as mock_get_duration:
            mock_get_duration.return_value = 3600.0  # 1 hour

            chunks = AudioProcessor.split_audio_by_duration(file_paths)

            # Should use default 15 hours, so all files fit in one chunk
            assert len(chunks) == 1
            assert chunks[0] == file_paths

    def test_create_temp_audio_file_success(self, tmp_path):
        """Test successful creation of temporary audio file."""
        file_paths = [Path("file1.mp3"), Path("file2.mp3")]
        output_prefix = "test_temp"

        with patch.object(AudioProcessor, "combine_audio_files") as mock_combine:
            mock_combine.return_value = MagicMock()

            result = AudioProcessor.create_temp_audio_file(
                file_paths, output_prefix, output_format="mp3", temp_dir=tmp_path
            )

            expected_path = tmp_path / f"{output_prefix}.tmp.mp3"
            assert result == expected_path
            mock_combine.assert_called_once_with(
                file_paths=file_paths,
                output_path=expected_path,
                output_format="mp3",
                description=f"Creating {output_prefix}",
            )

    def test_create_temp_audio_file_already_exists(self, tmp_path):
        """Test create_temp_audio_file when file already exists."""
        file_paths = [Path("file1.mp3")]
        output_prefix = "existing_temp"
        temp_file = tmp_path / f"{output_prefix}.tmp.mp4"
        temp_file.touch()  # Create the file

        result = AudioProcessor.create_temp_audio_file(
            file_paths, output_prefix, temp_dir=tmp_path
        )

        assert result == temp_file

    def test_create_temp_audio_file_default_temp_dir(self):
        """Test create_temp_audio_file with default temp directory."""
        file_paths = [Path("file1.mp3")]
        output_prefix = "test_temp"

        with patch.object(
            AudioProcessor, "combine_audio_files"
        ) as mock_combine, patch("tempfile.gettempdir") as mock_tempdir:
            mock_tempdir.return_value = "/tmp"
            mock_combine.return_value = MagicMock()

            result = AudioProcessor.create_temp_audio_file(file_paths, output_prefix)

            expected_path = Path("/tmp") / f"{output_prefix}.tmp.mp4"
            assert result == expected_path

    def test_create_temp_audio_file_default_format(self, tmp_path):
        """Test create_temp_audio_file with default format."""
        file_paths = [Path("file1.mp3")]
        output_prefix = "test_temp"

        with patch.object(AudioProcessor, "combine_audio_files") as mock_combine:
            mock_combine.return_value = MagicMock()

            result = AudioProcessor.create_temp_audio_file(
                file_paths, output_prefix, temp_dir=tmp_path
            )

            expected_path = tmp_path / f"{output_prefix}.tmp.mp4"
            assert result == expected_path
            assert mock_combine.call_args[1]["output_format"] == "mp4"

    def test_convert_wav_to_mp3_remove_original(self, tmp_path):
        """Test WAV to MP3 conversion with original file removal."""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()  # Create the file

        with patch("pydub.AudioSegment.from_wav") as mock_from_wav:
            mock_audio = MagicMock()
            mock_from_wav.return_value = mock_audio

            result = AudioProcessor.convert_wav_to_mp3(wav_path, remove_original=True)

            mock_audio.export.assert_called_once()
            # File should be removed (mocked by unlink)
            assert result == tmp_path / "test.mp3"

    def test_combine_audio_files_mixed_formats(self, tmp_path):
        """Test combine_audio_files with mixed audio formats."""
        file_paths = [
            tmp_path / "file1.mp3",
            tmp_path / "file2.wav",
            tmp_path / "file3.m4a"
        ]
        output_path = tmp_path / "combined.wav"

        mock_combined = MagicMock()
        mock_combined.__len__.return_value = 1000
        mock_combined.__iadd__.return_value = mock_combined

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, \
             patch("pydub.AudioSegment.from_wav") as mock_from_wav, \
             patch("pydub.AudioSegment.from_file") as mock_from_file, \
             patch("pydub.AudioSegment.empty") as mock_empty:

            mock_audio = MagicMock()
            mock_from_mp3.return_value = mock_audio
            mock_from_wav.return_value = mock_audio
            mock_from_file.return_value = mock_audio
            mock_empty.return_value = mock_combined

            AudioProcessor.combine_audio_files(
                file_paths, output_path, show_progress=False
            )

            # Verify different format handlers were called
            mock_from_mp3.assert_called_once()
            mock_from_wav.assert_called_once()
            mock_from_file.assert_called_once()
            assert mock_combined.__iadd__.call_count == 3

    def test_combine_audio_files_generic_error(self, tmp_path):
        """Test combine_audio_files when generic error occurs during processing."""
        file_paths = [tmp_path / "error.mp3", tmp_path / "good.mp3"]
        output_path = tmp_path / "combined.wav"

        mock_combined = MagicMock()
        mock_combined.__len__.return_value = 1000
        mock_combined.__iadd__.return_value = mock_combined

        with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, patch(
            "pydub.AudioSegment.empty"
        ) as mock_empty:

            mock_audio = MagicMock()

            def side_effect(path):
                if "error" in path:
                    raise RuntimeError("Generic processing error")
                return mock_audio

            mock_from_mp3.side_effect = side_effect
            mock_empty.return_value = mock_combined

            result = AudioProcessor.combine_audio_files(
                file_paths, output_path, show_progress=False
            )

            assert result == mock_combined
            # Only one file should be successfully added
            assert mock_combined.__iadd__.call_count == 1
