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

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_init_with_valid_directory(self, mock_exists, mock_mkdir, mock_base_synth):
        """Test initialization with valid directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(directory_path=tmpdir)

            assert creator.directory_path == Path(tmpdir)
            assert creator.language == "en"
            assert creator.voice == "af_bella"
            assert creator.title == Path(tmpdir).name
            assert creator.chapter_titles == []
            assert creator.episode_paths == []


class TestDirectoryAudiobookCreatorGetSupportedFiles:
    """Test _get_supported_files method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_supported_files(self, mock_exists, mock_mkdir, mock_base_synth):
        """Test getting supported files from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "test1.epub").touch()
            Path(tmpdir, "test2.pdf").touch()
            Path(tmpdir, "test3.txt").touch()
            Path(tmpdir, "test4.md").touch()
            Path(tmpdir, "test5.doc").touch()  # Unsupported
            Path(tmpdir, "test6.jpg").touch()  # Unsupported

            creator = DirectoryAudiobookCreator(directory_path=tmpdir)
            files = creator._get_supported_files()

            assert len(files) == 4
            assert all(f.suffix in [".epub", ".pdf", ".txt", ".md"] for f in files)


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


class TestDirectoryAudiobookCreatorCleanText:
    """Test _clean_text_for_audiobook method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_clean_text_removes_references(
        self, mock_exists, mock_mkdir, mock_base_synth
    ):
        """Test that clean_text removes citations and references."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(directory_path=tmpdir)

            text_with_refs = "This is text [1] with references (Smith 2020) and URLs http://example.com"
            cleaned = creator._clean_text_for_audiobook(text_with_refs)

            assert "[1]" not in cleaned
            assert "(Smith 2020)" not in cleaned
            assert "http://example.com" not in cleaned


class TestDirectoryAudiobookCreatorInitializeMetadata:
    """Test _initialize_metadata_file method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_initialize_metadata_file(self, mock_base_synth):
        """Test metadata file initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use the temp directory as output_dir so paths are created there
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator._initialize_metadata_file()

            assert creator.metadata_path.exists()
            content = creator.metadata_path.read_text()
            assert ";FFMETADATA1" in content
            assert "major_brand=M4A" in content


class TestDirectoryAudiobookCreatorLogEpisodeMetadata:
    """Test _log_episode_metadata method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_log_episode_metadata(self, mock_base_synth):
        """Test logging episode metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use the temp directory as output_dir so paths are created there
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator._initialize_metadata_file()

            end_time = creator._log_episode_metadata(
                episode_number=1,
                start_time_ms=0,
                duration_s=60.5,
                chapter_title="Test Chapter",
            )

            assert end_time == 60500
            content = creator.metadata_path.read_text()
            assert "[CHAPTER]" in content
            assert "title=Test Chapter" in content
            assert "START=0" in content
            assert "END=60500" in content


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
    @patch("audify.audiobook_creator.AudiobookEpubCreator")
    def test_process_single_file_epub(self, mock_epub_creator, mock_base_synth):
        """Test processing an EPUB file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            # Create a test epub file
            test_file = Path(tmpdir) / "test_book.epub"
            test_file.touch()

            # Mock the EpubCreator
            mock_instance = Mock()
            mock_instance.audiobook_path = Path(tmpdir) / "output"
            mock_instance.episodes_path = Path(tmpdir) / "episodes"
            mock_instance.episodes_path.mkdir(parents=True, exist_ok=True)
            episode_path = Path(tmpdir) / "episodes" / "episode_001.mp3"
            episode_path.touch()
            mock_instance.synthesize.return_value = mock_instance.audiobook_path
            mock_epub_creator.return_value = mock_instance

            with patch("pathlib.Path.glob", return_value=[episode_path]):
                creator._process_single_file(test_file, 1)

            # Verify chapter title was added
            assert "test book" in creator.chapter_titles

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
    def test_process_single_file_text(self, mock_base_synth):
        """Test processing a text file routes to _process_text_file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            # Create a test text file
            test_file = Path(tmpdir) / "test_article.txt"
            test_file.write_text("This is test content.")

            # Mock _process_text_file
            episode_path = Path(tmpdir) / "episode_001.mp3"
            with patch.object(
                creator, "_process_text_file", return_value=episode_path
            ) as mock_process:
                result = creator._process_single_file(test_file, 1)

                mock_process.assert_called_once()
                assert result == episode_path

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
    @patch("audify.audiobook_creator.LLMClient")
    @patch("audify.audiobook_creator.AudioProcessor.convert_wav_to_mp3")
    @patch("audify.audiobook_creator.AudioSegment")
    def test_process_text_file_success(
        self, mock_audio_segment, mock_convert, mock_llm_client, mock_base_synth
    ):
        """Test successful text file processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            # Create a test text file
            test_file = Path(tmpdir) / "test_article.txt"
            test_file.write_text("This is test content for the audiobook.")

            # Mock LLM client
            mock_llm_instance = Mock()
            mock_llm_instance.generate_script.return_value = "Generated script"
            mock_llm_client.return_value = mock_llm_instance

            # Mock audio operations
            content_mp3 = creator.episodes_path / "content_001.mp3"
            content_mp3.touch()
            mock_convert.return_value = content_mp3

            # Mock AudioSegment operations
            mock_combined = Mock()
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_segment.empty.return_value = mock_combined
            mock_audio_segment.from_mp3.return_value = Mock()
            mock_audio_segment.silent.return_value = Mock()

            # Mock synthesizer
            creator.tts_synthesizer = Mock()
            creator.tts_synthesizer._synthesize_sentences = Mock()

            # Mock title audio synthesis
            with patch.object(creator, "_synthesize_title_audio", return_value=None):
                result = creator._process_text_file(test_file, 1, "Test Article")

            # Verify result path
            assert result == creator.episodes_path / "episode_001.mp3"
            mock_llm_instance.generate_script.assert_called_once()

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.LLMClient")
    @patch("audify.audiobook_creator.AudioProcessor.convert_wav_to_mp3")
    @patch("audify.audiobook_creator.AudioSegment")
    def test_process_text_file_with_translation(
        self, mock_audio_segment, mock_convert, mock_llm_client, mock_base_synth
    ):
        """Test text file processing with translation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir, translate="es"
            )

            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("English content to translate.")

            mock_llm_instance = Mock()
            mock_llm_instance.generate_script.return_value = "Script"
            mock_llm_client.return_value = mock_llm_instance

            content_mp3 = creator.episodes_path / "content_001.mp3"
            content_mp3.touch()
            mock_convert.return_value = content_mp3

            mock_combined = Mock()
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_segment.empty.return_value = mock_combined
            mock_audio_segment.from_mp3.return_value = Mock()

            creator.tts_synthesizer = Mock()
            creator.tts_synthesizer._synthesize_sentences = Mock()

            with (
                patch.object(creator, "_synthesize_title_audio", return_value=None),
                patch(
                    "audify.audiobook_creator.translate_sentence",
                    return_value="Translated sentence",
                ),
            ):
                result = creator._process_text_file(test_file, 1, "Test")

            assert result is not None

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
    @patch("subprocess.run")
    @patch("audify.audiobook_creator.AudioSegment")
    @patch("audify.audiobook_creator.tqdm.tqdm")
    def test_create_m4b_success(
        self, mock_tqdm, mock_audio_segment, mock_subprocess, mock_base_synth
    ):
        """Test successful M4B creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.chapter_titles = ["Chapter 1", "Chapter 2"]

            # Create mock episode files
            ep1 = creator.episodes_path / "episode_001.mp3"
            ep2 = creator.episodes_path / "episode_002.mp3"
            ep1.touch()
            ep2.touch()

            # Mock tqdm to just pass through
            mock_tqdm.side_effect = lambda x, **kwargs: x

            # Mock audio segments
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=5000)
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_segment.empty.return_value = mock_combined

            mock_audio = Mock()
            mock_audio.__len__ = Mock(return_value=2500)
            mock_audio_segment.from_mp3.return_value = mock_audio

            # Mock subprocess
            mock_result = Mock()
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_subprocess.return_value = mock_result

            creator.create_m4b()

            # Verify FFmpeg was called
            mock_subprocess.assert_called_once()

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("subprocess.run")
    @patch("audify.audiobook_creator.AudioSegment")
    @patch("audify.audiobook_creator.tqdm.tqdm")
    def test_create_m4b_ffmpeg_error(
        self, mock_tqdm, mock_audio_segment, mock_subprocess, mock_base_synth
    ):
        """Test M4B creation with FFmpeg error."""
        import subprocess as sp

        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.chapter_titles = ["Chapter 1"]

            ep1 = creator.episodes_path / "episode_001.mp3"
            ep1.touch()

            mock_tqdm.side_effect = lambda x, **kwargs: x

            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=5000)
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_segment.empty.return_value = mock_combined
            mock_ep_audio = Mock(__len__=Mock(return_value=2500))
            mock_audio_segment.from_mp3.return_value = mock_ep_audio

            # Mock FFmpeg failure
            error = sp.CalledProcessError(1, "ffmpeg")
            error.stdout = ""
            error.stderr = "FFmpeg error"
            mock_subprocess.side_effect = error

            with pytest.raises(sp.CalledProcessError):
                creator.create_m4b()


class TestDirectoryAudiobookCreatorCreateSingleM4b:
    """Test _create_single_m4b method."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudioSegment")
    @patch("audify.audiobook_creator.tqdm.tqdm")
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
    @patch("audify.audiobook_creator.tqdm.tqdm")
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

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("subprocess.run")
    @patch("audify.audiobook_creator.AudioSegment")
    @patch("audify.audiobook_creator.tqdm.tqdm")
    def test_create_single_m4b_ffmpeg_not_found(
        self, mock_tqdm, mock_audio_segment, mock_subprocess, mock_base_synth
    ):
        """Test _create_single_m4b when FFmpeg is not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.temp_m4b_path = creator.audiobook_path / "temp.m4b"
            creator.final_m4b_path = creator.audiobook_path / "final.m4b"
            creator._initialize_metadata_file()
            creator.chapter_titles = ["Chapter 1"]

            mock_tqdm.side_effect = lambda x, **kwargs: x

            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=5000)
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_segment.empty.return_value = mock_combined
            mock_ep_audio = Mock(__len__=Mock(return_value=2500))
            mock_audio_segment.from_mp3.return_value = mock_ep_audio

            # Mock FFmpeg not found
            mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")

            episode_files = [Path(tmpdir) / "episode_001.mp3"]

            with pytest.raises(FileNotFoundError):
                creator._create_single_m4b(episode_files)


class TestDirectoryAudiobookCreatorSynthesizeFull:
    """Test full synthesize method scenarios."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_synthesize_with_files_success(self, mock_base_synth):
        """Test synthesize with files that process successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir, confirm=False
            )

            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content")

            # Create mock episode output
            episode_path = creator.episodes_path / "episode_001.mp3"

            with (
                patch.object(
                    creator, "_process_single_file", return_value=episode_path
                ),
                patch.object(creator, "create_m4b") as mock_create_m4b,
                patch("pathlib.Path.exists", return_value=True),
            ):
                result = creator.synthesize()

                assert result == creator.audiobook_path
                assert len(creator.episode_paths) == 1
                mock_create_m4b.assert_called_once()

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

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_synthesize_title_audio_error(self, mock_base_synth):
        """Test synthesizing title audio handles errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            creator.tts_synthesizer = Mock()
            creator.tts_synthesizer._synthesize_sentences.side_effect = Exception(
                "TTS Error"
            )

            result = creator._synthesize_title_audio("Test Title", 1)

            assert result is None
