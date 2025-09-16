#!/usr/bin/env python3
"""
Tests for audify.create_podcast module.
"""
import tempfile
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from audify.create_podcast import get_creator, main


class TestGetCreator:
    """Tests for get_creator function."""

    def test_get_creator_epub(self):
        """Test get_creator returns PodcastEpubCreator for .epub files."""
        with patch('audify.create_podcast.PodcastEpubCreator') as mock_epub_creator:
            mock_instance = Mock()
            mock_epub_creator.return_value = mock_instance

            creator = get_creator(
                file_extension=".epub",
                path="test.epub",
                language="en",
                voice="af_bella",
                model_name="kokoro",
                translate=None,
                save_text=False,
                llm_base_url="http://localhost:11434",
                llm_model="llama3.1",
                max_chapters=None,
                confirm=True
            )

            assert creator == mock_instance
            mock_epub_creator.assert_called_once_with(
                path="test.epub",
                language="en",
                voice="af_bella",
                model_name="kokoro",
                translate=None,
                save_text=False,
                llm_base_url="http://localhost:11434",
                llm_model="llama3.1",
                max_chapters=None,
                confirm=True
            )

    def test_get_creator_pdf(self):
        """Test get_creator returns PodcastPdfCreator for .pdf files."""
        with patch('audify.create_podcast.PodcastPdfCreator') as mock_pdf_creator:
            mock_instance = Mock()
            mock_pdf_creator.return_value = mock_instance

            creator = get_creator(
                file_extension=".pdf",
                path="test.pdf",
                language="en",
                voice="af_bella",
                model_name="kokoro",
                translate=None,
                save_text=False,
                llm_base_url="http://localhost:11434",
                llm_model="llama3.1",
                max_chapters=5,  # Should be ignored for PDF
                confirm=True
            )

            assert creator == mock_instance
            mock_pdf_creator.assert_called_once_with(
                path="test.pdf",
                language="en",
                voice="af_bella",
                model_name="kokoro",
                translate=None,
                save_text=False,
                llm_base_url="http://localhost:11434",
                llm_model="llama3.1",
                confirm=True
                # Note: max_chapters is not passed to PDF creator
            )

    def test_get_creator_unsupported_format(self):
        """Test get_creator raises TypeError for unsupported file formats."""
        with pytest.raises(TypeError, match="Unsupported file format '.txt'"):
            get_creator(
                file_extension=".txt",
                path="test.txt",
                language="en",
                voice="af_bella",
                model_name="kokoro",
                translate=None,
                save_text=False,
                llm_base_url="http://localhost:11434",
                llm_model="llama3.1",
                max_chapters=None,
                confirm=True
            )


class TestMain:
    """Tests for main CLI function."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch('audify.create_podcast.get_creator')
    @patch('audify.create_podcast.get_file_extension')
    @patch('os.get_terminal_size')
    def test_main_epub_success(self, mock_terminal_size, mock_get_extension, mock_get_creator, runner):
        """Test main function with successful EPUB processing."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = '.epub'

        mock_creator = Mock()
        mock_creator.synthesize.return_value = '/path/to/output'
        mock_get_creator.return_value = mock_creator

        with tempfile.NamedTemporaryFile(suffix='.epub') as temp_file:
            result = runner.invoke(main, [
                temp_file.name,
                '--language', 'en',
                '--voice', 'af_bella',
                '--model-name', 'kokoro',
                '--translate', 'es',
                '--save-scripts',
                '--max-chapters', '5',
                '--confirm'
            ])

        assert result.exit_code == 0
        assert 'Podcast creation complete!' in result.output
        assert '/path/to/output' in result.output
        mock_creator.synthesize.assert_called_once()

    @patch('audify.create_podcast.get_creator')
    @patch('audify.create_podcast.get_file_extension')
    @patch('os.get_terminal_size')
    def test_main_pdf_success(self, mock_terminal_size, mock_get_extension, mock_get_creator, runner):
        """Test main function with successful PDF processing."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = '.pdf'

        mock_creator = Mock()
        mock_creator.synthesize.return_value = '/path/to/output'
        mock_get_creator.return_value = mock_creator

        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            result = runner.invoke(main, [
                temp_file.name,
                '--language', 'fr',
                '--voice', 'custom_voice',
                '--model-name', 'custom_model'
            ])

        assert result.exit_code == 0
        assert 'Podcast creation complete!' in result.output
        mock_creator.synthesize.assert_called_once()

    @patch('audify.create_podcast.get_creator')
    @patch('audify.create_podcast.get_file_extension')
    @patch('os.get_terminal_size')
    def test_main_keyboard_interrupt(self, mock_terminal_size, mock_get_extension, mock_get_creator, runner):
        """Test main function handles KeyboardInterrupt gracefully."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = '.epub'

        mock_creator = Mock()
        mock_creator.synthesize.side_effect = KeyboardInterrupt()
        mock_get_creator.return_value = mock_creator

        with tempfile.NamedTemporaryFile(suffix='.epub') as temp_file:
            result = runner.invoke(main, [temp_file.name])

        assert result.exit_code == 0
        assert 'Podcast creation cancelled by user.' in result.output

    @patch('audify.create_podcast.get_creator')
    @patch('audify.create_podcast.get_file_extension')
    @patch('os.get_terminal_size')
    def test_main_generic_exception(self, mock_terminal_size, mock_get_extension, mock_get_creator, runner):
        """Test main function handles generic exceptions."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = '.epub'

        mock_creator = Mock()
        mock_creator.synthesize.side_effect = Exception("Generic error")
        mock_get_creator.return_value = mock_creator

        with tempfile.NamedTemporaryFile(suffix='.epub') as temp_file:
            result = runner.invoke(main, [temp_file.name])

        assert result.exit_code == 0
        assert 'Error: Generic error' in result.output
        assert 'Please check your configuration and try again.' in result.output

    @patch('audify.create_podcast.get_creator')
    @patch('audify.create_podcast.get_file_extension')
    @patch('os.get_terminal_size')
    def test_main_llm_connection_error(self, mock_terminal_size, mock_get_extension, mock_get_creator, runner):
        """Test main function handles LLM connection errors with helpful tips."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = '.epub'

        mock_creator = Mock()
        mock_creator.synthesize.side_effect = Exception("Could not connect to LLM server")
        mock_get_creator.return_value = mock_creator

        with tempfile.NamedTemporaryFile(suffix='.epub') as temp_file:
            result = runner.invoke(main, [temp_file.name, '--llm-model', 'custom-model'])

        assert result.exit_code == 0
        assert 'Could not connect to LLM' in result.output
        assert 'Make sure Ollama is running:' in result.output
        assert 'ollama serve' in result.output
        assert 'ollama pull custom-model' in result.output

    @patch('audify.create_podcast.get_file_extension')
    @patch('os.get_terminal_size')
    def test_main_configuration_display(self, mock_terminal_size, mock_get_extension, runner):
        """Test main function displays configuration correctly."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = '.epub'

        with patch('audify.create_podcast.get_creator') as mock_get_creator, \
             tempfile.NamedTemporaryFile(suffix='.epub') as temp_file:

            mock_creator = Mock()
            mock_creator.synthesize.return_value = '/path/to/output'
            mock_get_creator.return_value = mock_creator

            result = runner.invoke(main, [
                temp_file.name,
                '--language', 'fr',
                '--llm-model', 'custom-model',
                '--translate', 'en',
                '--max-chapters', '10'
            ])

        assert result.exit_code == 0
        assert f'Source file: {temp_file.name}' in result.output
        assert 'Language: fr' in result.output
        assert 'LLM Model: custom-model' in result.output
        assert 'Translation: fr -> en' in result.output
        assert 'Max episodes: 10' in result.output
