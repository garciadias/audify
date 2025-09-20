#!/usr/bin/env python3
"""
Simplified tests for audify.podcast_creator module focusing on testable components.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from audify.podcast_creator import LLMClient, PodcastCreator
from audify.readers.ebook import EpubReader


def create_mock_text_file():
    """Create a mock text file for testing saving cleaned text."""
    content = "This is a test text file with unsupported format."
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_path.write_text(content, encoding="utf-8")
    return temp_path


class TestLLMClient:
    """Test cases for LLMClient class."""

    def test_init_default_values(self):
        """Test LLMClient initialization with default values."""
        with patch("audify.podcast_creator.OllamaAPIConfig") as mock_config:
            LLMClient()
            mock_config.assert_called_once_with(
                base_url="http://localhost:11434",
                model="qwen3:30b"  # Updated to current default
            )

    def test_generate_podcast_script_success(self):
        """Test successful podcast script generation."""
        with patch("audify.podcast_creator.OllamaAPIConfig") as mock_config:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Generated podcast script content"
            mock_config_instance = Mock()
            mock_config_instance.create_llm.return_value = mock_llm
            mock_config.return_value = mock_config_instance

            client = LLMClient()

            with patch("audify.podcast_creator.clean_text") as mock_clean:
                mock_clean.return_value = "Cleaned script content"

                result = client.generate_podcast_script("test chapter", None)

                assert result == "Cleaned script content"
                mock_clean.assert_called_once_with(
                    "Generated podcast script content"
                )

    def test_generate_podcast_script_empty_response(self):
        """Test handling of empty LLM response."""
        with patch("audify.podcast_creator.OllamaAPIConfig") as mock_config:
            mock_llm = Mock()
            mock_llm.invoke.return_value = ""
            mock_config_instance = Mock()
            mock_config_instance.create_llm.return_value = mock_llm
            mock_config.return_value = mock_config_instance

            client = LLMClient()

            result = client.generate_podcast_script("test chapter", None)

            expected = "Error: Unable to generate podcast script for this content."
            assert result == expected

    def test_generate_podcast_script_with_language(self):
        """Test podcast script generation with language translation."""
        with patch("audify.podcast_creator.OllamaAPIConfig") as mock_config:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Generated script"
            mock_config_instance = Mock()
            mock_config_instance.create_llm.return_value = mock_llm
            mock_config.return_value = mock_config_instance

            client = LLMClient()

            with patch("audify.podcast_creator.translate_sentence") as mock_translate, \
                 patch("audify.podcast_creator.clean_text") as mock_clean:
                mock_translate.return_value = "Translated prompt"
                mock_clean.return_value = "Cleaned script"

                result = client.generate_podcast_script("test chapter", "es")

                # Should translate the prompt
                mock_translate.assert_called_once()
                assert result == "Cleaned script"

    def test_generate_podcast_script_reasoning_model(self):
        """Test script generation with reasoning model (contains 'think')."""
        with patch("audify.podcast_creator.OllamaAPIConfig") as mock_config:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "<think>reasoning steps</think>Final output"
            mock_config_instance = Mock()
            mock_config_instance.create_llm.return_value = mock_llm
            mock_config.return_value = mock_config_instance

            client = LLMClient()

            with patch("audify.podcast_creator.clean_text") as mock_clean:
                mock_clean.return_value = "Cleaned final output"

                result = client.generate_podcast_script("test chapter", None)

                mock_clean.assert_called_once_with("Final output")
                assert result == "Cleaned final output"

    def test_generate_podcast_script_connection_error(self):
        """Test handling of connection errors."""
        with patch("audify.podcast_creator.OllamaAPIConfig") as mock_config:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("Connection refused")
            mock_config_instance = Mock()
            mock_config_instance.base_url = "http://localhost:11434"
            mock_config_instance.create_llm.return_value = mock_llm
            mock_config.return_value = mock_config_instance

            client = LLMClient()

            result = client.generate_podcast_script("test chapter", None)

            assert "Could not connect to local LLM server" in result
            assert "http://localhost:11434" in result

    def test_generate_podcast_script_timeout_error(self):
        """Test handling of timeout errors."""
        with patch("audify.podcast_creator.OllamaAPIConfig") as mock_config:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("timeout exceeded")
            mock_config_instance = Mock()
            mock_config_instance.create_llm.return_value = mock_llm
            mock_config.return_value = mock_config_instance

            client = LLMClient()

            result = client.generate_podcast_script("test chapter", None)

            assert "Request to LLM timed out" in result

    def test_generate_podcast_script_generic_error(self):
        """Test handling of generic errors."""
        with patch("audify.podcast_creator.OllamaAPIConfig") as mock_config:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("Some other error")
            mock_config_instance = Mock()
            mock_config_instance.create_llm.return_value = mock_llm
            mock_config.return_value = mock_config_instance

            client = LLMClient()

            result = client.generate_podcast_script("test chapter", None)

            assert "Failed to generate podcast script " \
            "due to: Some other error" in result


class TestPodcastCreatorMethods:
    """Test individual methods of PodcastCreator without complex initialization."""

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_clean_text_for_podcast(self, mock_init):
        """Test _clean_text_for_podcast method."""
        creator = PodcastCreator.__new__(PodcastCreator)

        test_text = """
        This is test content with [1] numbered references.
        It also has some http://example.com URLs.
        Figure 1 shows something and Table 2 has data.
        Smith et al. (2020) found results.

        References:
        [1] Some reference here
        Smith, J. (2020). Paper title.

        This should remain.
        """

        with patch("builtins.print"):  # Mock print statements
            cleaned = creator._clean_text_for_podcast(test_text)

            # Check that references, citations, etc. are removed
            assert "[1]" not in cleaned
            assert "http://example.com" not in cleaned
            assert "Figure 1" not in cleaned
            assert "Table 2" not in cleaned
            assert "Smith et al. (2020)" not in cleaned

            # Check that main content remains
            assert "This is test content" in cleaned
            assert "This should remain" in cleaned

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_break_script_into_segments(self, mock_init):
        """Test _break_script_into_segments method."""
        creator = PodcastCreator.__new__(PodcastCreator)

        with patch("audify.podcast_creator.break_text_into_sentences") as mock_break:
            mock_break.return_value = [
                "Short sentence.",
                "Another short sentence.",
                ("This is a much longer sentence that should be its own segment "
                 "because it exceeds the character limit and would make a "
                 "combined segment too long for processing."),
                "Final sentence."
            ]

            result = creator._break_script_into_segments("Test script")

            # Should combine short sentences but keep long ones separate
            assert len(result) >= 2
            # Check that some sentences were combined
            combined_found = any("Short sentence. Another short sentence." in segment
                               for segment in result)
            assert combined_found

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_initialize_metadata_file(self, mock_init):
        """Test _initialize_metadata_file method."""
        creator = PodcastCreator.__new__(PodcastCreator)

        # Create temporary directory for testing
        temp_dir = Path(tempfile.mkdtemp())
        try:
            creator.podcast_path = temp_dir
            creator._initialize_metadata_file()

            # Check that metadata file was created with correct content
            metadata_path = temp_dir / "chapters.txt"
            assert metadata_path.exists()

            content = metadata_path.read_text()
            assert ";FFMETADATA1" in content
            assert "major_brand=M4A" in content
            assert "encoder=Lavf61.7.100" in content
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_calculate_total_duration(self, mock_init):
        """Test _calculate_total_duration method."""
        creator = PodcastCreator.__new__(PodcastCreator)

        mock_files = [Path("file1.mp3"), Path("file2.mp3")]

        with patch(
                "audify.podcast_creator.AudioProcessor.get_duration"
            ) as mock_duration:
            mock_duration.side_effect = [120.5, 95.2]  # durations in seconds

            result = creator._calculate_total_duration(mock_files)

            assert result == 215.7
            assert mock_duration.call_count == 2

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_calculate_total_duration_with_error(self, mock_init):
        """Test _calculate_total_duration handles errors gracefully."""
        creator = PodcastCreator.__new__(PodcastCreator)

        mock_files = [Path("file1.mp3"), Path("file2.mp3")]

        with patch(
                "audify.podcast_creator.AudioProcessor.get_duration"
            ) as mock_duration:
            mock_duration.side_effect = [120.5, Exception("File error")]

            result = creator._calculate_total_duration(mock_files)

            assert result == 120.5  # Only the successful file counted

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_log_episode_metadata(self, mock_init):
        """Test _log_episode_metadata method."""
        creator = PodcastCreator.__new__(PodcastCreator)

        # Create temporary metadata file
        temp_dir = Path(tempfile.mkdtemp())
        try:
            creator.metadata_path = temp_dir / "chapters.txt"
            creator.metadata_path.write_text(";FFMETADATA1\n")

            result = creator._log_episode_metadata(1, 0, 120.5, "Test Chapter")

            # Should return end time in milliseconds
            assert result == 120500

            # Check file content
            content = creator.metadata_path.read_text()
            assert "[CHAPTER]" in content
            assert "START=0" in content
            assert "END=120500" in content
            assert "title=Test Chapter" in content
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestPodcastCreatorInitialization:
    """Test the PodcastCreator initialization with different file types."""

    @patch('audify.readers.pdf.PdfReader')
    @patch('audify.readers.ebook.EpubReader')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.mkdir')
    def test_init_with_epub_file(self, mock_mkdir, mock_exists,
                                 mock_epub_reader, mock_pdf_reader):
        """Test initialization with an EPUB file."""
        # Setup mocks
        mock_epub_instance = Mock()
        mock_epub_instance.get_language.return_value = "en"
        mock_epub_instance.title = "Test Book"
        mock_epub_instance.get_cover_image.return_value = None
        mock_epub_reader.return_value = mock_epub_instance

        with patch('audify.podcast_creator.BaseSynthesizer.__init__'):
            # Create the PodcastCreator with EPUB file
            creator = PodcastCreator(
                path='test.epub',
                llm_model='test-model'
            )

            # Verify initialization
            assert creator.resolved_language == 'en'
            assert creator.title == "Test Book"

    @patch('audify.readers.pdf.PdfReader')
    @patch('audify.readers.ebook.EpubReader')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.mkdir')
    def test_init_with_pdf_file(self, mock_mkdir, mock_exists,
                                mock_epub_reader, mock_pdf_reader):
        """Test initialization with a PDF file."""
        # Setup mocks
        mock_pdf_instance = Mock()
        mock_pdf_reader.return_value = mock_pdf_instance

        with patch('audify.podcast_creator.BaseSynthesizer.__init__'):
            # Create the PodcastCreator with PDF file
            creator = PodcastCreator(
                path='test.pdf',
                llm_model='test-model'
            )

            # Verify initialization
            assert creator.resolved_language == 'en'  # Default for PDF
            assert creator.title == "test"  # filename without extension

    @patch('audify.readers.ebook.EpubReader')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.mkdir')
    def test_init_with_optional_parameters(self, mock_mkdir, mock_exists,
                                          mock_epub_reader):
        """Test initialization with all optional parameters."""
        # Setup mocks
        mock_epub_instance = Mock()
        mock_epub_instance.get_language.return_value = "en"
        mock_epub_instance.title = "Test Book"
        mock_epub_instance.get_cover_image.return_value = None
        mock_epub_reader.return_value = mock_epub_instance

        with patch('audify.podcast_creator.BaseSynthesizer.__init__'):
            # Create the PodcastCreator with all parameters
            creator = PodcastCreator(
                path='test.epub',
                language='es',
                voice='custom-voice',
                translate='fr',
                max_chapters=5,
                confirm=False,
                llm_model='test-model'
            )

            # Verify all attributes are set
            assert creator.resolved_language == 'es'  # Override detected
            assert creator.max_chapters == 5
            assert not creator.confirm


class TestPodcastCreatorSetupPaths:
    """Test the _setup_paths method."""

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=False)
    def test_setup_paths_creates_directories(self, mock_exists, mock_mkdir):
        """Test that _setup_paths creates all necessary directories."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.output_base_dir = Path('/fake/output')

        creator._setup_paths(Path('test_file'))

        # Verify that mkdir was called to create directories
        assert mock_mkdir.call_count >= 1

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    def test_setup_paths_skips_existing_directories(self, mock_exists, mock_mkdir):
        """Test that _setup_paths doesn't create existing directories."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.output_base_dir = Path('/fake/output')

        creator._setup_paths(Path('test_file'))

        # Should still create the podcast-specific directories
        assert mock_mkdir.call_count >= 1


class TestPodcastCreatorSynthesizeEpisode:
    """Test the synthesize_episode method."""

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_synthesize_episode_existing_mp3(self, mock_exists, mock_init):
        """Test synthesize_episode when MP3 already exists."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.episodes_path = Path('/fake/episodes')

        # Mock exists to return True for MP3 file
        mock_exists.return_value = True

        result = creator.synthesize_episode("Script text", 1)
        expected_path = creator.episodes_path / "episode_001.mp3"
        assert result == expected_path

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_synthesize_episode_empty_script(self, mock_exists, mock_init):
        """Test synthesize_episode with empty script."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.episodes_path = Path('/fake/episodes')

        # Mock exists to return False for MP3 file
        mock_exists.return_value = False

        result = creator.synthesize_episode("", 1)
        expected_path = creator.episodes_path / "episode_001.mp3"
        assert result == expected_path

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_synthesize_episode_no_sentences(self, mock_exists, mock_init):
        """Test synthesize_episode when no sentences are extracted."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.episodes_path = Path('/fake/episodes')

        # Mock exists to return False for MP3 file
        mock_exists.return_value = False

        with patch.object(creator, '_break_script_into_segments', return_value=[]):
            result = creator.synthesize_episode("Script", 1)
            expected_path = creator.episodes_path / "episode_001.mp3"
            assert result == expected_path

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_synthesize_episode_basic(self, mock_exists, mock_init):
        """Test basic episode synthesis."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.episodes_path = Path('/fake/episodes')
        creator.translate = None

        mp3_path = creator.episodes_path / "episode_001.mp3"
        wav_path = creator.episodes_path / "episode_001.wav"
        sentences = ["First sentence.", "Second sentence."]

        # Mock exists to return False for MP3 file
        mock_exists.return_value = False

        with patch.object(
            creator, '_break_script_into_segments', return_value=sentences), \
             patch.object(creator, '_synthesize_sentences') as mock_synth, \
             patch.object(
                 creator, '_convert_to_mp3', return_value=mp3_path) as mock_convert:

            result = creator.synthesize_episode("Script text", 1)

            assert result == mp3_path
            mock_synth.assert_called_once_with(sentences, wav_path)
            mock_convert.assert_called_once_with(wav_path)

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_synthesize_episode_with_translation(self, mock_exists, mock_init):
        """Test episode synthesis with translation."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.episodes_path = Path('/fake/episodes')
        creator.translate = 'es'
        creator.language = 'en'

        mp3_path = creator.episodes_path / "episode_001.mp3"
        wav_path = creator.episodes_path / "episode_001.wav"
        sentences = ["First sentence.", "Second sentence."]

        # Mock exists to return False for MP3 file
        mock_exists.return_value = False

        def mock_translate(sentence, src_lang, tgt_lang):
            if sentence == "First sentence.":
                return "Primera oración."
            elif sentence == "Second sentence.":
                return "Segunda oración."
            return sentence

        with patch.object(
            creator, '_break_script_into_segments', return_value=sentences), \
             patch(
                 'audify.podcast_creator.translate_sentence',
                 side_effect=mock_translate), \
             patch('tqdm.tqdm', side_effect=lambda x, **kwargs: x), \
             patch.object(creator, '_synthesize_sentences') as mock_synth, \
             patch.object(creator, '_convert_to_mp3', return_value=mp3_path):

            result = creator.synthesize_episode("Script text", 1)

            assert result == mp3_path
            # Should call with translated sentences
            expected_translated = ["Primera oración.", "Segunda oración."]
            mock_synth.assert_called_once_with(expected_translated, wav_path)

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_synthesize_episode_translation_error(self, mock_exists, mock_init):
        """Test episode synthesis with translation error."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.episodes_path = Path('/fake/episodes')
        creator.translate = 'es'
        creator.language = 'en'

        mp3_path = creator.episodes_path / "episode_001.mp3"
        wav_path = creator.episodes_path / "episode_001.wav"
        sentences = ["First sentence.", "Second sentence."]

        # Mock exists to return False for MP3 file
        mock_exists.return_value = False

        with patch.object(
                creator, '_break_script_into_segments', return_value=sentences), \
             patch(
                 'audify.podcast_creator.translate_sentence',
                 side_effect=Exception("Translation error")), \
             patch('tqdm.tqdm', side_effect=lambda x, **kwargs: x), \
             patch.object(creator, '_synthesize_sentences') as mock_synth, \
             patch.object(creator, '_convert_to_mp3', return_value=mp3_path):

            result = creator.synthesize_episode("Script text", 1)

            # Should use original sentences when translation fails
            assert result == mp3_path
            mock_synth.assert_called_once_with(sentences, wav_path)


class TestPodcastCreatorCreateSeries:
    """Test the create_podcast_series method."""

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_create_podcast_series_epub_with_confirmation(self, mock_exists, mock_init):
        """Test create_podcast_series with EPUB reader and user confirmation."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.confirm = True
        creator.max_chapters = None
        creator.translate = None
        creator.resolved_language = 'en'

        # Mock EPUB reader
        mock_reader = Mock(spec=EpubReader)
        mock_reader.get_chapters.return_value = [
            "Chapter 1 content", "Chapter 2 content"
        ]
        creator.reader = mock_reader

        # Mock Path.exists to return True for episodes
        mock_exists.return_value = True

        with patch('builtins.input', return_value='y'), \
             patch('tqdm.tqdm', side_effect=lambda x, **kwargs: x), \
             patch.object(
                 creator, 'generate_podcast_script', return_value="Script content"), \
             patch.object(creator, 'synthesize_episode') as mock_synth, \
             patch.object(creator, 'create_m4b') as mock_m4b:

            # Mock episode paths
            episode1_path = Path('/fake/episode_001.mp3')
            episode2_path = Path('/fake/episode_002.mp3')
            mock_synth.side_effect = [episode1_path, episode2_path]

            result = creator.create_podcast_series()

            assert len(result) == 2
            assert result == [episode1_path, episode2_path]
            mock_m4b.assert_called_once()

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_create_podcast_series_pdf_no_confirmation(self, mock_exists, mock_init):
        """Test create_podcast_series with PDF reader and no confirmation."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.confirm = False
        creator.max_chapters = None
        creator.translate = None
        creator.resolved_language = 'en'

        # Mock PDF reader
        mock_reader = Mock()
        mock_reader.cleaned_text = "PDF content for episode"
        creator.reader = mock_reader

        # Mock Path.exists to return True for episodes
        mock_exists.return_value = True

        with patch('tqdm.tqdm', side_effect=lambda x, **kwargs: x), \
             patch.object(
                 creator, 'generate_podcast_script', return_value="Script content"), \
             patch.object(creator, 'synthesize_episode') as mock_synth, \
             patch.object(creator, 'create_m4b') as mock_m4b:

            episode_path = Path('/fake/episode_001.mp3')
            mock_synth.return_value = episode_path

            result = creator.create_podcast_series()

            assert len(result) == 1
            assert result == [episode_path]
            mock_m4b.assert_called_once()

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_create_podcast_series_user_cancels(self, mock_init):
        """Test create_podcast_series when user cancels."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.confirm = True
        creator.max_chapters = None

        mock_reader = Mock(spec=EpubReader)
        mock_reader.get_chapters.return_value = ["Chapter 1 content"]
        creator.reader = mock_reader

        with patch('builtins.input', return_value='n'):
            result = creator.create_podcast_series()
            assert result == []

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_create_podcast_series_with_max_chapters(self, mock_exists, mock_init):
        """Test create_podcast_series with max_chapters limit."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.confirm = True  # Force script generation instead of loading
        creator.max_chapters = 2
        creator.translate = None
        creator.resolved_language = 'en'
        creator.chapter_titles = []
        creator.scripts_path = Path('/fake/scripts')
        creator.episodes_path = Path('/fake/episodes')

        mock_reader = Mock(spec=EpubReader)
        mock_reader.get_chapters.return_value = ["Ch1", "Ch2", "Ch3", "Ch4"]
        creator.reader = mock_reader

        with patch('tqdm.tqdm', side_effect=lambda x, **kwargs: x), \
             patch.object(creator, 'generate_podcast_script', return_value="Script"), \
             patch.object(creator, 'synthesize_episode') as mock_synth, \
             patch.object(creator, 'create_m4b') as _, \
             patch('builtins.input', return_value='y'):

            episode1_path = Path('/fake/episode_001.mp3')
            episode2_path = Path('/fake/episode_002.mp3')
            mock_synth.side_effect = [episode1_path, episode2_path]

            # Mock Path.exists to return True for episodes
            mock_exists.return_value = True

            result = creator.create_podcast_series()

            # Should only process first 2 chapters due to max_chapters=2
            assert len(result) == 2
            assert mock_synth.call_count == 2

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_create_podcast_series_with_error(self, mock_exists, mock_init):
        """Test create_podcast_series with episode generation error."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.confirm = False
        creator.max_chapters = None
        creator.translate = None
        creator.resolved_language = 'en'
        creator.chapter_titles = []
        creator.scripts_path = Path('/fake/scripts')
        creator.episodes_path = Path('/fake/episodes')

        mock_reader = Mock(spec=EpubReader)
        mock_reader.get_chapters.return_value = ["Chapter 1", "Chapter 2"]
        creator.reader = mock_reader

        with patch('tqdm.tqdm', side_effect=lambda x, **kwargs: x), \
             patch.object(
                 creator, 'generate_podcast_script',
                 side_effect=[Exception("Script error"), "Script content"]), \
             patch.object(creator, 'synthesize_episode') as mock_synth, \
             patch.object(creator, 'create_m4b') as mock_m4b:

            episode_path = Path('/fake/episode_002.mp3')
            mock_synth.return_value = episode_path

            # Mock Path.exists to return True for episodes
            mock_exists.return_value = True

            result = creator.create_podcast_series()

            # Should only have 1 episode due to error in first episode
            assert len(result) == 1
            assert result == [episode_path]
            mock_m4b.assert_called_once()

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.exists')
    def test_create_podcast_series_no_episodes_created(self, mock_exists, mock_init):
        """Test create_podcast_series when no episodes are successfully created."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.confirm = False
        creator.max_chapters = None
        creator.translate = None
        creator.resolved_language = 'en'
        creator.chapter_titles = []
        creator.scripts_path = Path('/fake/scripts')
        creator.episodes_path = Path('/fake/episodes')

        mock_reader = Mock(spec=EpubReader)
        mock_reader.get_chapters.return_value = ["Chapter 1"]
        creator.reader = mock_reader

        with patch('tqdm.tqdm', side_effect=lambda x, **kwargs: x), \
             patch.object(creator, 'generate_podcast_script', return_value="Script"), \
             patch.object(creator, 'synthesize_episode') as mock_synth, \
             patch.object(creator, 'create_m4b') as mock_m4b:

            episode_path = Path('/fake/episode_001.mp3')
            mock_synth.return_value = episode_path

            # Mock episode doesn't exist
            mock_exists.return_value = False

            result = creator.create_podcast_series()

            assert result == []
            # Should not call create_m4b if no episodes created
            mock_m4b.assert_not_called()


class TestPodcastCreatorGenerateScript:
    """Test the generate_podcast_script method."""

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_generate_podcast_script_basic(self, mock_init):
        """Test basic podcast script generation."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.scripts_path = Path('/fake/scripts')
        creator.confirm = True
        creator.save_text = False
        creator.translate = None
        creator.title = "Test Book"
        creator.chapter_titles = []

        # Mock the reader and llm_client
        mock_reader = Mock()
        mock_reader.get_chapter_title.return_value = "Chapter Title"
        mock_path = Mock()
        mock_path.stem = "Chapter Title"
        mock_reader.path = mock_path
        creator.reader = mock_reader

        mock_llm_client = Mock()
        mock_llm_client.generate_podcast_script.return_value = "Generated script"
        creator.llm_client = mock_llm_client

        long_cleaned_text = "This is cleaned text " * 50  # 250 words
        with patch.object(
                creator, '_clean_text_for_podcast', return_value=long_cleaned_text), \
             patch('audify.podcast_creator.isinstance', return_value=True):
            result = creator.generate_podcast_script("Chapter text", 1, language="en")

            assert result == "Generated script"
            assert "Chapter Title" in creator.chapter_titles
            mock_llm_client.generate_podcast_script.assert_called_once()

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_generate_podcast_script_empty_text(self, mock_init):
        """Test podcast script generation with empty text."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.scripts_path = Path('/fake/scripts')
        creator.confirm = True
        creator.save_text = False
        creator.chapter_titles = []

        result = creator.generate_podcast_script("", 1, language="en")

        assert result == "This chapter contains no readable text content."

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_generate_podcast_script_short_text(self, mock_init):
        """Test podcast script generation with very short text."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.scripts_path = Path('/fake/scripts')
        creator.confirm = True
        creator.save_text = False
        creator.translate = None
        creator.title = "Test Book"
        creator.chapter_titles = []

        # Mock the reader with proper attributes
        mock_reader = Mock()
        mock_reader.get_chapter_title.return_value = "Short Chapter"
        mock_path = Mock()
        mock_path.stem = "Short Chapter"
        mock_reader.path = mock_path
        creator.reader = mock_reader

        short_text = "Very short text."

        with patch.object(creator, '_clean_text_for_podcast', return_value=short_text):
            result = creator.generate_podcast_script(short_text, 1, language="en")

            # Should return original text for very short content
            assert result == short_text
            assert "Short Chapter" in creator.chapter_titles

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_generate_podcast_script_with_existing_script(self, mock_init):
        """Test script generation when script file already exists."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.scripts_path = Path('/fake/scripts')
        creator.confirm = False  # Don't overwrite existing
        creator.save_text = False
        creator.chapter_titles = []

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="Existing script content")):

            result = creator.generate_podcast_script("Chapter text", 1, language="en")

            assert result == "Existing script content"

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_generate_podcast_script_with_translation(self, mock_init):
        """Test script generation with translation enabled."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.scripts_path = Path('/fake/scripts')
        creator.confirm = True
        creator.save_text = False
        creator.translate = 'es'
        creator.resolved_language = 'es'
        creator.title = "Test Book"
        creator.chapter_titles = []

        # Mock the reader and llm_client
        mock_reader = Mock()
        mock_reader.get_chapter_title.return_value = "Capítulo"
        creator.reader = mock_reader

        mock_llm_client = Mock()
        mock_llm_client.generate_podcast_script.return_value = "Script traducido"
        creator.llm_client = mock_llm_client

        long_text = "Long enough text " * 70  # 210 words > 200
        with patch.object(creator, '_clean_text_for_podcast', return_value=long_text), \
             patch('audify.translate.translate_sentence',
                   return_value="Translated prompt"), \
             patch('audify.podcast_creator.isinstance', return_value=True):

            result = creator.generate_podcast_script("Chapter text", 1, language='es')

            assert result == "Script traducido"
            # Verify the LLM was called with correct parameters
            mock_llm_client.generate_podcast_script.assert_called_once()
            call_args = mock_llm_client.generate_podcast_script.call_args
            assert call_args[1]['language'] == 'es'  # Check language parameter
            assert long_text in call_args[0][0]  # Check cleaned text is in prompt

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_generate_podcast_script_save_text(self, mock_init):
        """Test script generation with text saving enabled."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.scripts_path = Path('/fake/scripts')
        creator.confirm = True
        creator.save_text = True
        creator.translate = None
        creator.title = "Test Book"
        creator.chapter_titles = []

        # Mock the reader and llm_client
        mock_reader = Mock()
        mock_reader.get_chapter_title.return_value = "Chapter Title"
        creator.reader = mock_reader

        mock_llm_client = Mock()
        mock_llm_client.generate_podcast_script.return_value = "Generated script"
        creator.llm_client = mock_llm_client

        long_text = "Long enough text " * 70  # 210 words > 200
        with patch.object(creator, '_clean_text_for_podcast', return_value=long_text), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('audify.podcast_creator.isinstance', return_value=True):

            result = creator.generate_podcast_script("Chapter text", 1, language="en")

            assert result == "Generated script"
            # Verify files were written (script and original text)
            assert mock_file.call_count == 2


class TestPodcastCreatorM4BCreation:
    """Test M4B creation functionality."""

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_build_ffmpeg_command_with_cover(self, mock_init):
        """Test FFmpeg command building with cover image."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.temp_m4b_path = Path('/fake/temp.m4b')
        creator.metadata_path = Path('/fake/metadata.txt')
        creator.final_m4b_path = Path('/fake/final.m4b')
        creator.cover_image_path = Path('/fake/cover.jpg')

        with patch('pathlib.Path.exists', return_value=True), \
             patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('shutil.copy') as mock_copy:

            mock_temp_file = Mock()
            mock_temp_file.name = '/tmp/cover_temp.jpg'
            mock_temp.return_value = mock_temp_file

            command, cover_temp_file = creator._build_ffmpeg_command()

            # Verify command structure
            assert 'ffmpeg' in command
            assert str(creator.temp_m4b_path) in command
            assert str(creator.metadata_path) in command
            assert '-map' in command
            assert '0:a' in command
            assert '2:v' in command
            assert '-disposition:v' in command
            assert 'attached_pic' in command

            # Verify cover handling
            assert cover_temp_file == mock_temp_file
            mock_copy.assert_called_once()

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_build_ffmpeg_command_without_cover(self, mock_init):
        """Test FFmpeg command building without cover image."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.temp_m4b_path = Path('/fake/temp.m4b')
        creator.metadata_path = Path('/fake/metadata.txt')
        creator.final_m4b_path = Path('/fake/final.m4b')

        with patch('pathlib.Path.exists', return_value=False):
            command, cover_temp_file = creator._build_ffmpeg_command()

            # Verify command structure without cover
            assert 'ffmpeg' in command
            assert str(creator.temp_m4b_path) in command
            assert str(creator.metadata_path) in command
            assert '-map' in command
            assert '0:a' in command
            assert '2:v' not in command  # No video mapping without cover

            # No cover temp file
            assert cover_temp_file is None

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.glob')
    def test_create_m4b_no_episodes(self, mock_glob, mock_init):
        """Test create_m4b with no episode files."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.episodes_path = Path('/fake/episodes')
        creator.podcast_path = Path('/fake/podcast')
        creator.file_name = 'test'

        # No episode files found
        mock_glob.return_value = []

        result = creator.create_m4b()

        # Should return early with no episodes
        assert result is None

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    @patch('pathlib.Path.glob')
    def test_create_m4b_success(self, mock_glob, mock_init):
        """Test successful M4B creation."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.episodes_path = Path('/fake/episodes')
        creator.podcast_path = Path('/fake/podcast')
        creator.file_name = 'test'

        # Mock episode files
        episode_files = [Path('/fake/episode_001.mp3'), Path('/fake/episode_002.mp3')]
        mock_glob.return_value = episode_files

        with patch.object(creator, '_initialize_metadata_file') as mock_init_meta, \
             patch.object(creator, '_calculate_total_duration', return_value=3600.0), \
             patch.object(creator, '_create_single_m4b') as mock_create:

            creator.create_m4b()

            # Verify paths are set up
            assert creator.temp_m4b_path == Path('/fake/podcast/test.tmp.m4b')
            assert creator.final_m4b_path == Path('/fake/podcast/test.m4b')

            # Verify methods were called
            mock_init_meta.assert_called_once()
            mock_create.assert_called_once_with(episode_files)

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_create_single_m4b_empty_audio(self, mock_init):
        """Test _create_single_m4b with empty audio."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.temp_m4b_path = Path('/fake/temp.m4b')
        creator.chapter_titles = []

        episode_files = [Path('/fake/episode_001.mp3')]

        with patch('pathlib.Path.exists', return_value=False), \
             patch('audify.podcast_creator.AudioSegment') as mock_audio_seg, \
             patch('audify.podcast_creator.tqdm.tqdm',
                   side_effect=lambda x, **kwargs: x):

            # Mock empty audio
            mock_empty = Mock()
            mock_empty.__len__ = Mock(return_value=0)
            mock_audio_seg.empty.return_value = mock_empty
            mock_audio_seg.from_mp3.side_effect = lambda x: Mock(
                __len__=Mock(return_value=1000)
            )

            result = creator._create_single_m4b(episode_files)

            # Should return early with empty audio
            assert result is None

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_create_single_m4b_success(self, mock_init):
        """Test successful _create_single_m4b creation."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.temp_m4b_path = Path('/fake/temp.m4b')
        creator.metadata_path = Path('/fake/metadata.txt')
        creator.final_m4b_path = Path('/fake/final.m4b')
        creator.chapter_titles = ['Chapter 1', 'Chapter 2']

        episode_files = [Path('/fake/episode_001.mp3'), Path('/fake/episode_002.mp3')]

        with patch('pathlib.Path.exists', return_value=False), \
             patch('audify.podcast_creator.AudioSegment') as mock_audio_seg, \
             patch(
                 'audify.podcast_creator.tqdm.tqdm', side_effect=lambda x, **kwargs: x
                 ), \
             patch.object(creator, '_log_episode_metadata', side_effect=[1000, 2000]), \
             patch.object(
                 creator, '_build_ffmpeg_command',
                 return_value=(['ffmpeg', 'cmd'], None)
                ), \
             patch('subprocess.run') as mock_subprocess, \
             patch('pathlib.Path.unlink') as mock_unlink:

            # Mock audio segments
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=5000)  # Non-empty
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_seg.empty.return_value = mock_combined

            mock_episode_audio = Mock()
            mock_episode_audio.__len__ = Mock(return_value=2500)
            mock_audio_seg.from_mp3.return_value = mock_episode_audio

            # Mock successful subprocess
            mock_result = Mock()
            mock_result.stdout = 'Success'
            mock_result.stderr = ''
            mock_subprocess.return_value = mock_result

            creator._create_single_m4b(episode_files)

            # Verify audio processing
            assert mock_audio_seg.from_mp3.call_count == 2
            mock_combined.export.assert_called_once()

            # Verify FFmpeg execution
            mock_subprocess.assert_called_once()

            # Verify cleanup
            mock_unlink.assert_called()

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_create_single_m4b_decode_error(self, mock_init):
        """Test _create_single_m4b with decode error."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.temp_m4b_path = Path('/fake/temp.m4b')
        creator.chapter_titles = []

        episode_files = [Path('/fake/episode_001.mp3')]

        with patch('pathlib.Path.exists', return_value=False), \
             patch('audify.podcast_creator.AudioSegment') as mock_audio_seg, \
             patch(
                 'audify.podcast_creator.tqdm.tqdm', side_effect=lambda x, **kwargs: x):

            # Mock CouldntDecodeError
            from pydub.exceptions import CouldntDecodeError
            mock_audio_seg.from_mp3.side_effect = CouldntDecodeError("Decode error")

            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=0)  # Empty after error
            mock_audio_seg.empty.return_value = mock_combined

            result = creator._create_single_m4b(episode_files)

            # Should handle error gracefully
            assert result is None

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_create_single_m4b_ffmpeg_error(self, mock_init):
        """Test _create_single_m4b with FFmpeg error."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.temp_m4b_path = Path('/fake/temp.m4b')
        creator.chapter_titles = []

        episode_files = [Path('/fake/episode_001.mp3')]

        with patch('pathlib.Path.exists', return_value=False), \
             patch('audify.podcast_creator.AudioSegment') as mock_audio_seg, \
             patch(
                 'audify.podcast_creator.tqdm.tqdm', side_effect=lambda x, **kwargs: x
                ), \
             patch.object(creator, '_log_episode_metadata', return_value=1000), \
             patch.object(
                 creator, '_build_ffmpeg_command', return_value=(['ffmpeg'], None)), \
             patch('subprocess.run') as mock_subprocess:

            # Mock successful audio creation
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=5000)
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_seg.empty.return_value = mock_combined
            mock_audio_seg.from_mp3.return_value = Mock(__len__=Mock(return_value=2500))

            # Mock FFmpeg failure
            error = subprocess.CalledProcessError(1, 'ffmpeg')
            error.stdout = ''
            error.stderr = 'FFmpeg error'
            mock_subprocess.side_effect = error

            # Should raise the subprocess error
            with pytest.raises(subprocess.CalledProcessError):
                creator._create_single_m4b(episode_files)

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_create_single_m4b_ffmpeg_not_found(self, mock_init):
        """Test _create_single_m4b with FFmpeg not found."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.temp_m4b_path = Path('/fake/temp.m4b')
        creator.chapter_titles = []

        episode_files = [Path('/fake/episode_001.mp3')]

        with patch('pathlib.Path.exists', return_value=False), \
             patch('audify.podcast_creator.AudioSegment') as mock_audio_seg, \
             patch(
                 'audify.podcast_creator.tqdm.tqdm', side_effect=lambda x, **kwargs: x
                ), \
             patch.object(creator, '_log_episode_metadata', return_value=1000), \
             patch.object(
                 creator, '_build_ffmpeg_command', return_value=(['ffmpeg'], None)), \
             patch('subprocess.run') as mock_subprocess:

            # Mock successful audio creation
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=5000)
            mock_combined.__add__ = Mock(return_value=mock_combined)
            mock_combined.export = Mock()
            mock_audio_seg.empty.return_value = mock_combined
            mock_audio_seg.from_mp3.return_value = Mock(__len__=Mock(return_value=2500))

            # Mock FFmpeg not found
            mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")

            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError):
                creator._create_single_m4b(episode_files)


class TestPodcastEpubCreator:
    """Test the PodcastEpubCreator specialized class."""

    @patch('audify.readers.ebook.EpubReader')
    @patch('audify.podcast_creator.BaseSynthesizer.__init__')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.mkdir')
    def test_epub_creator_initialization_success(self, mock_mkdir, mock_exists,
                                                 mock_base_init, mock_epub_reader):
        """Test successful PodcastEpubCreator initialization."""
        # Setup mocks
        mock_epub_instance = Mock()
        mock_epub_instance.get_language.return_value = "en"
        mock_epub_instance.title = "Test Book"
        mock_epub_instance.get_cover_image.return_value = None
        mock_epub_instance.get_chapters = Mock(return_value=['Ch1', 'Ch2'])
        mock_epub_reader.return_value = mock_epub_instance

        from audify.podcast_creator import PodcastEpubCreator

        creator = PodcastEpubCreator(path='test.epub', llm_model='test-model')

        # Verify it's properly initialized
        assert hasattr(creator.reader, 'get_chapters')

    @patch('audify.readers.pdf.PdfReader')
    @patch('audify.podcast_creator.BaseSynthesizer.__init__')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.mkdir')
    def test_epub_creator_initialization_failure(self, mock_mkdir, mock_exists,
                                                 mock_base_init, mock_pdf_reader):
        """Test PodcastEpubCreator initialization failure with PDF reader."""
        # Setup PDF reader instead of EPUB
        mock_pdf_instance = Mock()
        mock_pdf_reader.return_value = mock_pdf_instance

        from audify.podcast_creator import PodcastEpubCreator

        with pytest.raises(ValueError, match="requires an EPUB reader"):
            PodcastEpubCreator(path='test.pdf', llm_model='test-model')


class TestPodcastPdfCreator:
    """Test the PodcastPdfCreator specialized class."""

    @patch('audify.readers.pdf.PdfReader')
    @patch('audify.podcast_creator.BaseSynthesizer.__init__')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.mkdir')
    def test_pdf_creator_initialization(self, mock_mkdir, mock_exists,
                                       mock_base_init, mock_pdf_reader):
        """Test PodcastPdfCreator initialization."""
        # Setup mocks
        mock_pdf_instance = Mock()
        mock_pdf_reader.return_value = mock_pdf_instance

        from audify.podcast_creator import PodcastPdfCreator

        creator = PodcastPdfCreator(path='test.pdf', llm_model='test-model')

        # Should initialize without error
        assert creator is not None

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_success(self, mock_init):
        """Test successful PDF podcast series creation."""
        from audify.podcast_creator import PodcastPdfCreator
        from audify.readers.pdf import PdfReader

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = False
        creator.language = 'en'
        creator.resolved_language = 'en'
        creator.translate = None

        # Mock PDF reader
        mock_reader = Mock(spec=PdfReader)
        mock_reader.cleaned_text = "This is a long PDF text content " * 50
        creator.reader = mock_reader

        with patch.object(creator, 'generate_podcast_script',
                         return_value="Generated script"), \
             patch.object(creator, 'synthesize_episode',
                         return_value=Path('/fake/episode_001.mp3')), \
             patch('pathlib.Path.exists', return_value=True):

            result = creator.create_podcast_series()

            assert len(result) == 1
            assert result[0] == Path('/fake/episode_001.mp3')

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_with_confirmation(self, mock_init):
        """Test PDF podcast series creation with user confirmation."""
        from audify.podcast_creator import PodcastPdfCreator
        from audify.readers.pdf import PdfReader

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = True
        creator.language = 'en'
        creator.resolved_language = 'en'
        creator.translate = None

        # Mock PDF reader
        mock_reader = Mock(spec=PdfReader)
        mock_reader.cleaned_text = "PDF content"
        creator.reader = mock_reader

        with patch('builtins.input', return_value='y'), \
             patch.object(creator, 'generate_podcast_script',
                         return_value="Generated script"), \
             patch.object(creator, 'synthesize_episode',
                         return_value=Path('/fake/episode_001.mp3')), \
             patch('pathlib.Path.exists', return_value=True):

            result = creator.create_podcast_series()

            assert len(result) == 1

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_user_cancels(self, mock_init):
        """Test PDF podcast series creation when user cancels."""
        from audify.podcast_creator import PodcastPdfCreator
        from audify.readers.pdf import PdfReader

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = True

        # Mock PDF reader
        mock_reader = Mock(spec=PdfReader)
        mock_reader.cleaned_text = "PDF content"
        creator.reader = mock_reader

        with patch('builtins.input', return_value='n'):
            result = creator.create_podcast_series()

            assert result == []

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_empty_text(self, mock_init):
        """Test PDF podcast series creation with empty text."""
        from audify.podcast_creator import PodcastPdfCreator
        from audify.readers.pdf import PdfReader

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = False

        # Mock PDF reader with empty text
        mock_reader = Mock(spec=PdfReader)
        mock_reader.cleaned_text = ""
        creator.reader = mock_reader

        result = creator.create_podcast_series()

        assert result == []

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_short_text_warning(self, mock_init):
        """Test PDF podcast series creation with short text warning."""
        from audify.podcast_creator import PodcastPdfCreator
        from audify.readers.pdf import PdfReader

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = False
        creator.language = 'en'
        creator.resolved_language = 'en'
        creator.translate = None

        # Mock PDF reader with short text (< 200 words)
        mock_reader = Mock(spec=PdfReader)
        mock_reader.cleaned_text = "Short PDF text content."
        creator.reader = mock_reader

        with patch.object(creator, 'generate_podcast_script',
                         return_value="Generated script"), \
             patch.object(creator, 'synthesize_episode',
                         return_value=Path('/fake/episode_001.mp3')), \
             patch('pathlib.Path.exists', return_value=True):

            result = creator.create_podcast_series()

            assert len(result) == 1

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_with_translation(self, mock_init):
        """Test PDF podcast series creation with translation."""
        from audify.podcast_creator import PodcastPdfCreator
        from audify.readers.pdf import PdfReader

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = False
        creator.language = 'es'  # Spanish
        creator.resolved_language = 'es'
        creator.translate = None

        # Mock PDF reader
        mock_reader = Mock(spec=PdfReader)
        mock_reader.cleaned_text = "PDF content in Spanish"
        creator.reader = mock_reader

        with patch('audify.podcast_creator.translate_sentence',
                   return_value="Translated prompt"), \
             patch.object(creator, 'generate_podcast_script',
                         return_value="Generated script"), \
             patch.object(creator, 'synthesize_episode',
                         return_value=Path('/fake/episode_001.mp3')), \
             patch('pathlib.Path.exists', return_value=True):

            result = creator.create_podcast_series()

            assert len(result) == 1

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_wrong_reader_type(self, mock_init):
        """Test PDF creator with wrong reader type."""
        from audify.podcast_creator import PodcastPdfCreator

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = False

        # Mock wrong reader type (not PdfReader)
        mock_reader = Mock()  # Not spec=PdfReader
        creator.reader = mock_reader

        with pytest.raises(ValueError, match="requires a PDF reader"):
            creator.create_podcast_series()

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_episode_not_created(self, mock_init):
        """Test PDF podcast series when episode file is not created."""
        from audify.podcast_creator import PodcastPdfCreator
        from audify.readers.pdf import PdfReader

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = False
        creator.language = 'en'
        creator.resolved_language = 'en'

        # Mock PDF reader
        mock_reader = Mock(spec=PdfReader)
        mock_reader.cleaned_text = "PDF content"
        creator.reader = mock_reader

        with patch.object(creator, 'generate_podcast_script',
                         return_value="Generated script"), \
             patch.object(creator, 'synthesize_episode',
                         return_value=Path('/fake/episode_001.mp3')), \
             patch('pathlib.Path.exists', return_value=False):  # Episode not created

            result = creator.create_podcast_series()

            assert result == []

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_pdf_create_podcast_series_with_exception(self, mock_init):
        """Test PDF podcast series creation with exception handling."""
        from audify.podcast_creator import PodcastPdfCreator
        from audify.readers.pdf import PdfReader

        creator = PodcastPdfCreator.__new__(PodcastPdfCreator)
        creator.confirm = False
        creator.language = 'en'
        creator.resolved_language = 'en'

        # Mock PDF reader
        mock_reader = Mock(spec=PdfReader)
        mock_reader.cleaned_text = "PDF content"
        creator.reader = mock_reader

        with patch.object(creator, 'generate_podcast_script',
                         side_effect=Exception("Script generation error")):

            result = creator.create_podcast_series()

            assert result == []


class TestPodcastCreatorSynthesize:
    """Test the main synthesize method."""

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_synthesize_success(self, mock_init):
        """Test successful synthesize method."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.path = Path('/fake/test.epub')
        creator.podcast_path = Path('/fake/podcast')

        with patch.object(creator, 'create_podcast_series',
                         return_value=[Path('/fake/ep1.mp3'), Path('/fake/ep2.mp3')]):

            result = creator.synthesize()

            assert result == creator.podcast_path

    @patch("audify.podcast_creator.PodcastCreator.__init__", return_value=None)
    def test_synthesize_no_episodes_created(self, mock_init):
        """Test synthesize method when no episodes are created."""
        creator = PodcastCreator.__new__(PodcastCreator)
        creator.path = Path('/fake/test.epub')
        creator.podcast_path = Path('/fake/podcast')

        with patch.object(creator, 'create_podcast_series', return_value=[]):

            result = creator.synthesize()

            assert result == creator.podcast_path
