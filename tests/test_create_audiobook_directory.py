#!/usr/bin/env python3
"""Tests for directory mode in audify.create_audiobook CLI."""

import logging
import tempfile
from unittest.mock import Mock, patch

from click.testing import CliRunner

from audify.cli import cli


class TestMainDirectoryMode:
    """Tests for directory mode in the main CLI function."""

    def setup_method(self):
        self.runner = CliRunner()


    @patch("audify.cli.DirectoryAudiobookCreator")
    @patch("os.get_terminal_size")
    def test_directory_mode_with_translate(
        self, mock_terminal_size, mock_dir_creator, caplog
    ):
        """Test directory mode displays translation info."""
        mock_terminal_size.return_value = (80, 24)
        mock_instance = Mock()
        mock_instance.synthesize.return_value = "/path/to/output"
        mock_dir_creator.return_value = mock_instance

        with caplog.at_level(logging.INFO):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = self.runner.invoke(
                    cli,
                    ["--language", "en", "--translate", "es", "--verbose", tmpdir],
                )

        assert result.exit_code == 0
        assert any(
            "Translation: en -> es" in record.message for record in caplog.records
        )

    @patch("audify.cli.DirectoryAudiobookCreator")
    @patch("os.get_terminal_size")
    def test_directory_mode_keyboard_interrupt(
        self, mock_terminal_size, mock_dir_creator, caplog
    ):
        """Test directory mode handles KeyboardInterrupt."""
        mock_terminal_size.return_value = (80, 24)
        mock_instance = Mock()
        mock_instance.synthesize.side_effect = KeyboardInterrupt()
        mock_dir_creator.return_value = mock_instance

        with caplog.at_level(logging.INFO):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = self.runner.invoke(cli, ["--verbose", tmpdir])

        assert result.exit_code == 1
        assert any(
            "Directory audiobook creation cancelled by user." in record.message
            for record in caplog.records
        )


    @patch("audify.cli.DirectoryAudiobookCreator")
    @patch("os.get_terminal_size")
    def test_directory_mode_llm_connection_error(
        self, mock_terminal_size, mock_dir_creator, caplog
    ):
        """Test directory mode shows LLM connection tip."""
        mock_terminal_size.return_value = (80, 24)
        mock_instance = Mock()
        mock_instance.synthesize.side_effect = Exception(
            "Could not connect to LLM server"
        )
        mock_dir_creator.return_value = mock_instance

        with caplog.at_level(logging.ERROR):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = self.runner.invoke(
                    cli, ["--llm-model", "my-model", "--verbose", tmpdir]
                )

        assert result.exit_code == 1
        assert any(
            "Could not connect to LLM" in record.message for record in caplog.records
        )
        assert any("ollama serve" in record.message for record in caplog.records)
        assert any(
            "ollama pull my-model" in record.message for record in caplog.records
        )
