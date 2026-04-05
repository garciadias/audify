#!/usr/bin/env python3
"""Tests for error paths in audify.audiobook_creator module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from audify.audiobook_creator import (
    AudiobookCreator,
    DirectoryAudiobookCreator,
    LLMClient,
)


class TestAudiobookCreatorOutputDirCreation:
    """Test output directory creation when it doesn't exist."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_output_base_dir_created_when_missing(self, mock_base_synth):
        """Test that output_base_dir is created if it doesn't exist (line 168)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_output"
            assert not output_dir.exists()

            DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=str(output_dir)
            )

            assert output_dir.exists()


class TestAudiobookCreatorScriptSaveIOError:
    """Test IOError when saving scripts."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_save_script_io_error(self, mock_base_synth):
        """Test IOError when saving script files (lines 352-353)."""
        from audify.readers.ebook import EpubReader

        creator = AudiobookCreator.__new__(AudiobookCreator)
        creator.save_text = True
        creator.scripts_path = Path("/proc/nonexistent")
        creator.language = "en"
        creator.translate = None
        creator.confirm = False
        creator.llm_client = Mock()
        creator.llm_client.generate_script.return_value = "Generated script"
        creator._task_prompt = "test prompt"
        creator._task_llm_params = {}
        creator.chapter_titles = []
        # Use a real EpubReader spec so isinstance works
        creator.reader = Mock(spec=EpubReader)
        creator.reader.get_chapter_title.return_value = "Ch1"

        long_text = " ".join(["word"] * 300)
        result = creator.generate_audiobook_script(
            long_text, 1, language="en"
        )

        # Should return script despite IOError saving
        assert "Generated script" in result


class TestAudiobookCreatorMetadataIOError:
    """Test IOError in metadata operations."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_initialize_metadata_io_error(self, mock_base_synth):
        """Test IOError in _initialize_metadata_file (lines 516-518)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            # Override audiobook_path so metadata_path resolves to invalid location
            creator.audiobook_path = Path("/proc/nonexistent")

            with pytest.raises(
                (IOError, FileNotFoundError), match="No such file"
            ):
                creator._initialize_metadata_file()

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_log_episode_metadata_io_error(self, mock_base_synth):
        """Test IOError in _log_episode_metadata is re-raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            # Point to nonexistent path to trigger IOError
            creator.metadata_path = Path("/proc/nonexistent/metadata.txt")

            with pytest.raises((IOError, FileNotFoundError)):
                creator._log_episode_metadata(
                    episode_number=1,
                    start_time_ms=0,
                    duration_s=60.0,
                    chapter_title="Test",
                )


class TestDirectoryCreatorMetadataIOErrors:
    """Test IOError in DirectoryAudiobookCreator metadata methods."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_dir_initialize_metadata_io_error(self, mock_base_synth):
        """Test IOError in Dir._initialize_metadata_file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.audiobook_path = Path("/proc/nonexistent")

            with pytest.raises(
                (IOError, FileNotFoundError), match="No such file"
            ):
                creator._initialize_metadata_file()

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_dir_log_episode_metadata_io_error(self, mock_base_synth):
        """Test IOError in Dir._log_episode_metadata is re-raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.metadata_path = Path("/proc/nonexistent/metadata.txt")

            with pytest.raises((IOError, FileNotFoundError)):
                creator._log_episode_metadata(
                    episode_number=1,
                    start_time_ms=0,
                    duration_s=30.0,
                    chapter_title="Test",
                )


class TestDirectoryCreatorM4bExportFailure:
    """Test M4B export failure paths."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudioSegment")
    @patch("audify.audiobook_creator.tqdm.tqdm")
    def test_create_single_m4b_export_failure(
        self, mock_tqdm, mock_audio_segment, mock_base_synth
    ):
        """Test _create_single_m4b when export fails (lines 1224-1227)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )
            creator.temp_m4b_path = creator.audiobook_path / "temp.m4b"
            creator.final_m4b_path = creator.audiobook_path / "final.m4b"
            creator._initialize_metadata_file()
            creator.chapter_titles = ["Chapter 1"]

            mock_tqdm.side_effect = lambda x, **_kwargs: x

            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=5000)
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export.side_effect = Exception("Export failed")
            mock_audio_segment.empty.return_value = mock_combined

            mock_ep_audio = Mock(__len__=Mock(return_value=2500))
            mock_audio_segment.from_mp3.return_value = mock_ep_audio

            ep = Path(tmpdir) / "episode_001.mp3"
            ep.touch()

            with pytest.raises(Exception, match="Export failed"):
                creator._create_single_m4b([ep])


class TestDirectoryCreatorProcessSingleFileNoEpisodes:
    """Test _process_single_file when no episodes are generated."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudiobookEpubCreator")
    def test_no_episodes_generated(self, mock_epub_creator, mock_base_synth):
        """Test _process_single_file returns None when no episodes (lines 981-982)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            test_file = Path(tmpdir) / "test.epub"
            test_file.touch()

            mock_instance = Mock()
            mock_instance.audiobook_path = Path(tmpdir) / "output"
            mock_instance.episodes_path = Path(tmpdir) / "episodes"
            mock_instance.episodes_path.mkdir(parents=True, exist_ok=True)
            mock_instance.synthesize.return_value = mock_instance.audiobook_path
            mock_epub_creator.return_value = mock_instance

            # No episode files exist -> glob returns empty
            result = creator._process_single_file(test_file, 1)

            assert result is None


class TestDirectoryCreatorTitleAudioHandling:
    """Test title audio paths in _process_single_file."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudiobookEpubCreator")
    @patch("audify.audiobook_creator.AudioSegment")
    def test_process_file_with_title_audio(
        self, mock_audio_segment, mock_epub_creator, mock_base_synth
    ):
        """Test _process_single_file with title audio present (lines 992-998)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            test_file = Path(tmpdir) / "test.epub"
            test_file.touch()

            mock_instance = Mock()
            mock_instance.audiobook_path = Path(tmpdir) / "output"
            mock_instance.episodes_path = Path(tmpdir) / "episodes"
            mock_instance.episodes_path.mkdir(parents=True, exist_ok=True)

            # Create episode file
            ep = mock_instance.episodes_path / "episode_001.mp3"
            ep.touch()
            mock_instance.synthesize.return_value = mock_instance.audiobook_path
            mock_epub_creator.return_value = mock_instance

            # Create title audio file
            title_audio = creator.episodes_path / "title_001.mp3"
            title_audio.touch()

            mock_combined = Mock()
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_segment.empty.return_value = mock_combined
            mock_audio_segment.from_mp3.return_value = Mock()
            mock_audio_segment.silent.return_value = Mock()

            with patch.object(
                creator, "_synthesize_title_audio", return_value=title_audio
            ):
                creator._process_single_file(test_file, 1)

            assert "test" in creator.chapter_titles

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.AudiobookEpubCreator")
    @patch("audify.audiobook_creator.AudioSegment")
    def test_process_file_title_audio_load_error(
        self, mock_audio_segment, mock_epub_creator, mock_base_synth
    ):
        """Test _process_single_file when title audio load fails (lines 997-998)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            test_file = Path(tmpdir) / "test.epub"
            test_file.touch()

            mock_instance = Mock()
            mock_instance.audiobook_path = Path(tmpdir) / "output"
            mock_instance.episodes_path = Path(tmpdir) / "episodes"
            mock_instance.episodes_path.mkdir(parents=True, exist_ok=True)

            ep = mock_instance.episodes_path / "episode_001.mp3"
            ep.touch()
            mock_instance.synthesize.return_value = mock_instance.audiobook_path
            mock_epub_creator.return_value = mock_instance

            title_audio = creator.episodes_path / "title_001.mp3"
            title_audio.touch()

            mock_combined = Mock()
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_segment.empty.return_value = mock_combined
            # Title audio fails to load
            mock_audio_segment.from_mp3.side_effect = Exception("Cannot load")

            with patch.object(
                creator, "_synthesize_title_audio", return_value=title_audio
            ):
                # Should not raise, just log error
                creator._process_single_file(test_file, 1)


class TestDirectoryCreatorTextFileTranslationFallback:
    """Test translation exception fallback in _process_text_file."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.LLMClient")
    @patch("audify.audiobook_creator.AudioProcessor.convert_wav_to_mp3")
    @patch("audify.audiobook_creator.AudioSegment")
    def test_translation_exception_fallback(
        self, mock_audio_segment, mock_convert, mock_llm_client, mock_base_synth
    ):
        """Test translation falls back to original on exception (lines 1057-1064)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir, translate="es"
            )

            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Hello world. This is a test.")

            mock_llm_instance = Mock()
            mock_llm_instance.generate_script.return_value = "Script text."
            mock_llm_client.return_value = mock_llm_instance

            content_mp3 = creator.episodes_path / "content_001.mp3"
            content_mp3.touch()
            mock_convert.return_value = content_mp3

            mock_combined = Mock()
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
                    side_effect=Exception("Translation failed"),
                ),
            ):
                result = creator._process_text_file(test_file, 1, "Test")

            # Should still succeed using original text
            assert result is not None


class TestDirectoryCreatorTextFileTitleAudio:
    """Test title audio in _process_text_file."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("audify.audiobook_creator.LLMClient")
    @patch("audify.audiobook_creator.AudioProcessor.convert_wav_to_mp3")
    @patch("audify.audiobook_creator.AudioSegment")
    def test_text_file_with_title_audio(
        self, mock_audio_segment, mock_convert, mock_llm_client, mock_base_synth
    ):
        """Test _process_text_file with title audio (lines 1073-1075)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=tmpdir
            )

            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content for audiobook creation.")

            mock_llm_instance = Mock()
            mock_llm_instance.generate_script.return_value = "Script"
            mock_llm_client.return_value = mock_llm_instance

            content_mp3 = creator.episodes_path / "content_001.mp3"
            content_mp3.touch()
            mock_convert.return_value = content_mp3

            mock_combined = Mock()
            mock_combined.__iadd__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_segment.empty.return_value = mock_combined
            mock_audio_segment.from_mp3.return_value = Mock()
            mock_audio_segment.silent.return_value = Mock()

            creator.tts_synthesizer = Mock()
            creator.tts_synthesizer._synthesize_sentences = Mock()

            # Return a valid title audio path
            title_mp3 = creator.episodes_path / "title_001.mp3"
            title_mp3.touch()

            with patch.object(
                creator, "_synthesize_title_audio", return_value=title_mp3
            ):
                result = creator._process_text_file(test_file, 1, "Test")

            assert result is not None


class TestDirectoryCreatorCleanTextHTML:
    """Test HTML cleaning in _clean_text_for_audiobook."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_clean_text_with_html(self, mock_exists, mock_mkdir, mock_base_synth):
        """Test _clean_text_for_audiobook strips HTML tags (line 1098)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = DirectoryAudiobookCreator(directory_path=tmpdir)

            html_text = "<p>This is <b>bold</b> text</p>"
            cleaned = creator._clean_text_for_audiobook(html_text)

            assert "<p>" not in cleaned
            assert "<b>" not in cleaned
            assert "bold" in cleaned


class TestDirectoryCreatorSetupPathsExistingDir:
    """Test _setup_paths when output_base_dir doesn't exist yet."""

    @patch("audify.audiobook_creator.BaseSynthesizer")
    def test_setup_paths_creates_output_base_dir(self, mock_base_synth):
        """Test _setup_paths creates output_base_dir (lines 876-877)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "deep" / "nested" / "output"
            DirectoryAudiobookCreator(
                directory_path=tmpdir, output_dir=str(output_dir)
            )
            assert output_dir.exists()


class TestLLMClientAPIKeyError:
    """Test LLM client API key error message."""

    @patch("audify.audiobook_creator.CommercialAPIConfig")
    def test_api_key_error_message(self, mock_config):
        """Test API key error returns helpful message."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        client = LLMClient(model="api:deepseek/deepseek-chat")

        # Simulate API key error
        error = Exception("Invalid api key provided")
        mock_config_instance.generate.side_effect = error

        result = client.generate_audiobook_script("Chapter text", "en")
        assert "API key issue" in result or "Error" in result
