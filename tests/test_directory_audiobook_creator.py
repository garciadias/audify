#!/usr/bin/env python3
"""Tests for DirectoryAudiobookCreator class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from audify.audiobook_creator import DirectoryAudiobookCreator


class TestDirectoryAudiobookCreatorInitialization:
    """Test DirectoryAudiobookCreator initialization."""

    def test_init_with_non_directory_raises_error(self):
        """Test initialization with non-directory path raises ValueError."""
        with pytest.raises(ValueError, match="Path is not a directory"):
            DirectoryAudiobookCreator(directory_path="not_a_directory.txt")


class TestDirectoryAudiobookCreatorSynthesizeTitleAudio:
    """Test _synthesize_title_audio method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudioProcessor.convert_wav_to_mp3")
    def test_synthesize_title_audio(self, mock_convert, mock_base_synth):
        """Test synthesizing title audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use the temp directory as output_dir so paths are created there
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            expected_mp3_path = creator.episodes_path / "title_001.mp3"
            mock_convert.return_value = expected_mp3_path

            creator.tts_synthesizer = Mock()
            creator.tts_synthesizer._synthesize_sentences = Mock()

            result = creator._synthesize_title_audio("Test Title", 1)

            assert result == expected_mp3_path
            creator.tts_synthesizer._synthesize_sentences.assert_called_once()

class TestDirectoryAudiobookCreatorSynthesize:
    """Test synthesize method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_synthesize_with_no_files(self, mock_exists, mock_mkdir, mock_base_synth):
        """Test synthesize with no supported files returns early."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(directory_path=tmpdir, confirm=False)

            result = creator.synthesize()

            assert result == creator.audiobook_path
            assert len(creator.episode_paths) == 0

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_synthesize_with_confirm_cancelled(
        self, mock_exists, mock_mkdir, mock_base_synth
    ):
        """Test synthesize cancels when user says no."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.epub").touch()

            creator = DirectoryAudiobookCreator(directory_path=tmpdir, confirm=True)

            with patch("builtins.input", return_value="n"):
                result = creator.synthesize()

            assert result == creator.audiobook_path
            assert len(creator.episode_paths) == 0


class TestDirectoryAudiobookCreatorProcessSingleFile:
    """Test _process_single_file method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_process_single_file_unsupported_extension(self, mock_base_synth):
        """Test processing a file with unsupported extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            # Create a file with unsupported extension
            test_file = Path(tmpdir) / "test.docx"
            test_file.touch()

            result = creator._process_single_file(test_file, 1)

            # Should return None for unsupported format
            assert result is None


    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudiobookPdfCreator")
    def test_process_single_file_pdf(self, mock_pdf_creator, mock_base_synth):
        """Test processing a PDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            # Create a test pdf file
            test_file = Path(tmpdir) / "test_doc.pdf"
            test_file.touch()

            # Mock the PdfCreator
            mock_instance = Mock()
            mock_instance.audiobook_path = Path(tmpdir) / "output"
            mock_instance.episodes_path = Path(tmpdir) / "episodes"
            mock_instance.episodes_path.mkdir(parents=True, exist_ok=True)
            episode_path = Path(tmpdir) / "episodes" / "episode_001.mp3"
            episode_path.touch()
            mock_instance.synthesize.return_value = mock_instance.audiobook_path
            mock_pdf_creator.return_value = mock_instance

            with patch("pathlib.Path.glob", return_value=[episode_path]):
                creator._process_single_file(test_file, 1)

            # Verify chapter title was added
            assert "test doc" in creator.chapter_titles


    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_process_single_file_exception(self, mock_base_synth):
        """Test that exceptions in _process_single_file are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content")

            with patch.object(
                creator, "_process_text_file", side_effect=Exception("Test error")
            ):
                result = creator._process_single_file(test_file, 1)

                assert result is None


class TestDirectoryAudiobookCreatorProcessTextFile:
    """Test _process_text_file method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_process_text_file_exception(self, mock_base_synth):
        """Test text file processing handles exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            # File that doesn't exist
            test_file = Path(tmpdir) / "nonexistent.txt"

            result = creator._process_text_file(test_file, 1, "Test")

            assert result is None


class TestDirectoryAudiobookCreatorCreateM4b:
    """Test create_m4b method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_create_m4b_no_episodes(self, mock_base_synth):
        """Test create_m4b with no episode files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            # No episode files in the directory
            result = creator.create_m4b()

            # Should return early with no episodes
            assert result is None


    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("pathlib.Path.glob")
    def test_create_m4b_with_splitting(self, mock_glob, mock_base_synth):
        """Test create_m4b with duration >6 hours triggers splitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.chapter_titles = [
                "Chapter 1",
                "Chapter 2",
                "Chapter 3",
                "Chapter 4",
            ]

            # Mock episode files (4 episodes)
            episode_files = [
                creator.episodes_path / "episode_001.mp3",
                creator.episodes_path / "episode_002.mp3",
                creator.episodes_path / "episode_003.mp3",
                creator.episodes_path / "episode_004.mp3",
            ]
            mock_glob.return_value = episode_files

            # Mock chunks: split into 2 chunks of 2 episodes each
            chunk1 = episode_files[:2]
            chunk2 = episode_files[2:]

            with (
                patch.object(
                    creator, "_calculate_total_duration"
                ) as mock_calc_duration,
                patch.object(
                    creator,
                    "_split_episodes_by_duration",
                    return_value=[chunk1, chunk2],
                ),
                patch.object(creator, "_create_temp_m4b_for_chunk") as mock_create_temp,
                patch.object(creator, "_create_metadata_for_chunk") as mock_create_meta,
                patch("audify.audiobook_creator.assemble_m4b") as mock_assemble,
                patch(
                    "pathlib.Path.exists", return_value=True
                ),  # Make temp paths exist
            ):
                # Mock calculate_total_duration to return 7 hours for total,
                # 3.5 hours per chunk
                mock_calc_duration.side_effect = [
                    7 * 3600,  # total duration
                    3.5 * 3600,  # chunk1 duration
                    3.5 * 3600,  # chunk2 duration
                ]

                # Mock temporary M4B paths
                temp_path1 = creator.audiobook_path / "test_part1.tmp.m4b"
                temp_path2 = creator.audiobook_path / "test_part2.tmp.m4b"
                mock_create_temp.side_effect = [temp_path1, temp_path2]

                # Mock metadata paths
                meta_path1 = creator.audiobook_path / "chapters_part1.txt"
                meta_path2 = creator.audiobook_path / "chapters_part2.txt"
                mock_create_meta.side_effect = [meta_path1, meta_path2]

                creator.create_m4b()

                # Verify splitting was triggered - first call with all episodes
                mock_calc_duration.assert_any_call(episode_files)
                creator._split_episodes_by_duration.assert_called_once_with(
                    episode_files, max_hours=6.0
                )

                # Verify chunk processing
                assert mock_create_temp.call_count == 2
                mock_create_temp.assert_any_call(chunk1, 0)
                mock_create_temp.assert_any_call(chunk2, 1)

                assert mock_create_meta.call_count == 2
                mock_create_meta.assert_any_call(chunk1, 0)
                mock_create_meta.assert_any_call(chunk2, 1)

                # Verify assemble_m4b calls
                assert mock_assemble.call_count == 2
                expected_final_path1 = (
                    creator.audiobook_path / f"{creator.file_name}_part1.m4b"
                )
                expected_final_path2 = (
                    creator.audiobook_path / f"{creator.file_name}_part2.m4b"
                )
                mock_assemble.assert_any_call(
                    temp_path1,
                    meta_path1,
                    expected_final_path1,
                    None,
                )
                mock_assemble.assert_any_call(
                    temp_path2,
                    meta_path2,
                    expected_final_path2,
                    None,
                )


    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_split_episodes_by_duration(self, mock_base_synth):
        """Test _split_episodes_by_duration delegates to AudioProcessor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            episode_files = [
                creator.episodes_path / "episode_001.mp3",
                creator.episodes_path / "episode_002.mp3",
            ]

            with patch(
                "audify.audiobook_creator.AudioProcessor.split_audio_by_duration"
            ) as mock_split:
                mock_split.return_value = [[episode_files[0]], [episode_files[1]]]

                result = creator._split_episodes_by_duration(
                    episode_files, max_hours=5.0
                )

                mock_split.assert_called_once_with(episode_files, 5.0)
                assert result == [[episode_files[0]], [episode_files[1]]]

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_create_temp_m4b_for_chunk_exists(self, mock_base_synth):
        """Test _create_temp_m4b_for_chunk when temp file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.file_name = "test"

            chunk_files = [creator.episodes_path / "episode_001.mp3"]
            chunk_temp_path = creator.audiobook_path / "test_part1.tmp.m4b"

            with (
                patch.object(Path, "exists", return_value=True),
                patch("audify.audiobook_creator.logger.info") as mock_logger,
            ):
                result = creator._create_temp_m4b_for_chunk(chunk_files, 0)

                assert result == chunk_temp_path
                mock_logger.assert_called()

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_create_temp_m4b_for_chunk_create_new(self, mock_base_synth):
        """Test _create_temp_m4b_for_chunk creates new temp file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.file_name = "test"

            chunk_files = [creator.episodes_path / "episode_001.mp3"]
            chunk_temp_path = creator.audiobook_path / "test_part1.tmp.m4b"

            with (
                patch.object(Path, "exists", return_value=False),
                patch(
                    "audify.audiobook_creator.AudioProcessor.combine_audio_files"
                ) as mock_combine,
                patch("audify.audiobook_creator.logger.info") as mock_logger,
            ):
                result = creator._create_temp_m4b_for_chunk(chunk_files, 0)

                assert result == chunk_temp_path
                mock_combine.assert_called_once_with(
                    chunk_files,
                    chunk_temp_path,
                    output_format="mp4",
                    description="Combining Chunk 1",
                )
                mock_logger.assert_called()

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_create_metadata_for_chunk(self, mock_base_synth):
        """Test _create_metadata_for_chunk creates metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.chapter_titles = ["Chapter 1", "Chapter 2"]

            chunk_files = [
                creator.episodes_path / "episode_001.mp3",
                creator.episodes_path / "episode_002.mp3",
            ]

            with (
                patch(
                    "audify.audiobook_creator.write_metadata_header"
                ) as mock_write_header,
                patch(
                    "audify.audiobook_creator.AudioProcessor.get_duration"
                ) as mock_get_duration,
                patch(
                    "audify.audiobook_creator.append_chapter_metadata"
                ) as mock_append,
                patch("audify.audiobook_creator.logger.info") as mock_logger,
                patch("audify.audiobook_creator.logger.warning") as mock_warning,
            ):
                mock_get_duration.side_effect = [100.0, 200.0]  # durations in seconds
                mock_append.side_effect = [
                    100000,
                    300000,
                ]  # new start times after each chapter

                result = creator._create_metadata_for_chunk(chunk_files, 0)

                # Verify metadata file path
                expected_path = creator.audiobook_path / "chapters_part1.txt"
                assert result == expected_path

                # Verify header written
                mock_write_header.assert_called_once_with(expected_path)

                # Verify durations fetched
                assert mock_get_duration.call_count == 2
                mock_get_duration.assert_any_call(str(chunk_files[0]))
                mock_get_duration.assert_any_call(str(chunk_files[1]))

                # Verify metadata appended
                assert mock_append.call_count == 2
                mock_append.assert_any_call(expected_path, "Chapter 1", 0, 100.0)
                mock_append.assert_any_call(expected_path, "Chapter 2", 100000, 200.0)

                mock_logger.assert_called()
                mock_warning.assert_not_called()


class TestDirectoryAudiobookCreatorCreateSingleM4b:
    """Test _create_single_m4b method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudioSegment")
    @patch("audify.audiobook_creator.track")
    def test_create_single_m4b_empty_audio(
        self, mock_tqdm, mock_audio_segment, mock_base_synth
    ):
        """Test _create_single_m4b with empty combined audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.temp_m4b_path = creator.audiobook_path / "temp.m4b"
            creator.chapter_titles = []

            mock_tqdm.side_effect = lambda x, **kwargs: x

            # Mock empty audio
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=0)
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_audio_segment.empty.return_value = mock_combined

            from pydub.exceptions import CouldntDecodeError

            mock_audio_segment.from_mp3.side_effect = CouldntDecodeError("Error")

            episode_files = [Path(tmpdir) / "episode_001.mp3"]

            result = creator._create_single_m4b(episode_files)

            # Should return early with empty audio
            assert result is None

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("subprocess.run")
    @patch("audify.audiobook_creator.AudioSegment")
    @patch("audify.audiobook_creator.track")
    def test_create_single_m4b_existing_temp_file(
        self, mock_tqdm, mock_audio_segment, mock_subprocess, mock_base_synth
    ):
        """Test _create_single_m4b when temp file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            # Create temp M4B file that already exists
            creator.temp_m4b_path = creator.audiobook_path / "temp.m4b"
            creator.temp_m4b_path.touch()
            creator.final_m4b_path = creator.audiobook_path / "final.m4b"
            creator._initialize_metadata_file()
            creator.chapter_titles = []

            mock_tqdm.side_effect = lambda x, **kwargs: x

            mock_result = Mock()
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_subprocess.return_value = mock_result

            episode_files = [Path(tmpdir) / "episode_001.mp3"]

            creator._create_single_m4b(episode_files)

            # Should skip audio combination and go straight to FFmpeg
            mock_audio_segment.from_mp3.assert_not_called()
            mock_subprocess.assert_called_once()

class TestDirectoryAudiobookCreatorSynthesizeFull:
    """Test full synthesize method scenarios."""


    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_synthesize_with_failed_files(self, mock_base_synth):
        """Test synthesize when file processing fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir, confirm=False
            )

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content")

            with (
                patch.object(creator, "_process_single_file", return_value=None),
                patch.object(creator, "create_m4b") as mock_create_m4b,
            ):
                result = creator.synthesize()

                assert result == creator.audiobook_path
                assert len(creator.episode_paths) == 0
                # create_m4b should not be called if no episodes
                mock_create_m4b.assert_not_called()

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_synthesize_with_user_confirmation_yes(self, mock_base_synth):
        """Test synthesize with user confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir, confirm=True
            )

            # Create a test file
            test_file = Path(tmpdir) / "test.epub"
            test_file.touch()

            episode_path = creator.episodes_path / "episode_001.mp3"

            with (
                patch("builtins.input", return_value="y"),
                patch.object(
                    creator, "_process_single_file", return_value=episode_path
                ),
                patch.object(creator, "create_m4b"),
                patch("pathlib.Path.exists", return_value=True),
            ):
                creator.synthesize()

                assert len(creator.episode_paths) == 1


class TestDirectoryAudiobookCreatorSynthesizeTitleAudioExtended:
    """Extended tests for _synthesize_title_audio method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudioProcessor.convert_wav_to_mp3")
    def test_synthesize_title_audio_already_exists(self, mock_convert, mock_base_synth):
        """Test synthesizing title audio when it already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            # Create existing title audio file
            title_mp3_path = creator.episodes_path / "title_001.mp3"
            title_mp3_path.touch()

            creator.tts_synthesizer = Mock()

            result = creator._synthesize_title_audio("Test Title", 1)

            assert result == title_mp3_path
            # Synthesizer should not be called
            creator.tts_synthesizer._synthesize_sentences.assert_not_called()
