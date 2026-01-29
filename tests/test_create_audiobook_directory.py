#!/usr/bin/env python3
"""Tests for directory mode in audify.create_audiobook CLI."""

import tempfile
from unittest.mock import Mock, patch

from click.testing import CliRunner

from audify.create_audiobook import main


class TestMainDirectoryMode:
    """Tests for directory mode in the main CLI function."""

    def setup_method(self):
        self.runner = CliRunner()

    @patch("audify.create_audiobook.DirectoryAudiobookCreator")
    @patch("os.get_terminal_size")
    def test_directory_mode_success(self, mock_terminal_size, mock_dir_creator):
        """Test main function with a directory path."""
        mock_terminal_size.return_value = (80, 24)
        mock_instance = Mock()
        mock_instance.synthesize.return_value = "/path/to/output"
        mock_dir_creator.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(main, [tmpdir, "--language", "en"])

        assert result.exit_code == 0
        assert "Directory Mode" in result.output
        assert "Directory audiobook creation complete!" in result.output
        mock_dir_creator.assert_called_once()

    @patch("audify.create_audiobook.DirectoryAudiobookCreator")
    @patch("os.get_terminal_size")
    def test_directory_mode_with_translate(self, mock_terminal_size, mock_dir_creator):
        """Test directory mode displays translation info."""
        mock_terminal_size.return_value = (80, 24)
        mock_instance = Mock()
        mock_instance.synthesize.return_value = "/path/to/output"
        mock_dir_creator.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, [tmpdir, "--language", "en", "--translate", "es"]
            )

        assert result.exit_code == 0
        assert "Translation: en -> es" in result.output

    @patch("audify.create_audiobook.DirectoryAudiobookCreator")
    @patch("os.get_terminal_size")
    def test_directory_mode_keyboard_interrupt(
        self, mock_terminal_size, mock_dir_creator
    ):
        """Test directory mode handles KeyboardInterrupt."""
        mock_terminal_size.return_value = (80, 24)
        mock_instance = Mock()
        mock_instance.synthesize.side_effect = KeyboardInterrupt()
        mock_dir_creator.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(main, [tmpdir])

        assert result.exit_code == 0
        assert "Directory audiobook creation cancelled by user." in result.output

    @patch("audify.create_audiobook.DirectoryAudiobookCreator")
    @patch("os.get_terminal_size")
    def test_directory_mode_generic_exception(
        self, mock_terminal_size, mock_dir_creator
    ):
        """Test directory mode handles generic exceptions."""
        mock_terminal_size.return_value = (80, 24)
        mock_instance = Mock()
        mock_instance.synthesize.side_effect = Exception("Something failed")
        mock_dir_creator.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(main, [tmpdir])

        assert result.exit_code == 0
        assert "Error: Something failed" in result.output
        assert "Please check your configuration" in result.output

    @patch("audify.create_audiobook.DirectoryAudiobookCreator")
    @patch("os.get_terminal_size")
    def test_directory_mode_llm_connection_error(
        self, mock_terminal_size, mock_dir_creator
    ):
        """Test directory mode shows LLM connection tip."""
        mock_terminal_size.return_value = (80, 24)
        mock_instance = Mock()
        mock_instance.synthesize.side_effect = Exception(
            "Could not connect to LLM server"
        )
        mock_dir_creator.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, [tmpdir, "--llm-model", "my-model"]
            )

        assert result.exit_code == 0
        assert "Could not connect to LLM" in result.output
        assert "ollama serve" in result.output
        assert "ollama pull my-model" in result.output

    @patch("os.get_terminal_size")
    def test_no_terminal_size_fallback(self, mock_terminal_size):
        """Test fallback when terminal size is not available."""
        mock_terminal_size.side_effect = OSError("No terminal")

        with (
            tempfile.NamedTemporaryFile(suffix=".epub") as temp_file,
            patch("audify.create_audiobook.get_creator") as mock_get_creator,
            patch("audify.create_audiobook.get_file_extension", return_value=".epub"),
        ):
            mock_creator = Mock()
            mock_creator.synthesize.return_value = "/out"
            mock_get_creator.return_value = mock_creator

            result = self.runner.invoke(main, [temp_file.name])

        assert result.exit_code == 0
