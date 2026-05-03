#!/usr/bin/env python3
"""
Tests for audify.create_audiobook module.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from audify.cli import cli
from audify.convert import get_creator


class TestGetCreator:
    """Tests for get_creator function."""

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
                confirm=True,
            )


class TestMain:
    """Tests for main CLI function."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch("audify.convert.get_creator")
    @patch("audify.cli.get_file_extension")
    @patch("os.get_terminal_size")
    def test_main_keyboard_interrupt(
        self, mock_terminal_size, mock_get_extension, mock_get_creator, runner, caplog
    ):
        """Test main function handles KeyboardInterrupt gracefully."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".epub"

        mock_creator = Mock()
        mock_creator.synthesize.side_effect = KeyboardInterrupt()
        mock_get_creator.return_value = mock_creator

        with caplog.at_level(logging.INFO):
            with tempfile.NamedTemporaryFile(suffix=".epub") as temp_file:
                result = runner.invoke(cli, ["--verbose", temp_file.name])

        assert result.exit_code == 1
        # Log messages are captured by caplog
        assert any(
            "Audiobook creation cancelled by user." in record.message
            for record in caplog.records
        )


    @patch("audify.convert.get_creator")
    @patch("audify.cli.get_file_extension")
    @patch("os.get_terminal_size")
    def test_main_llm_connection_error(
        self, mock_terminal_size, mock_get_extension, mock_get_creator, runner, caplog
    ):
        """Test main function handles LLM connection errors with helpful tips."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".epub"

        mock_creator = Mock()
        mock_creator.synthesize.side_effect = Exception(
            "Could not connect to LLM server"
        )
        mock_get_creator.return_value = mock_creator

        with caplog.at_level(logging.ERROR):
            with tempfile.NamedTemporaryFile(suffix=".epub") as temp_file:
                result = runner.invoke(
                    cli, ["--llm-model", "custom-model", "--verbose", temp_file.name]
                )

        assert result.exit_code == 1
        # Error messages are captured by caplog
        assert any(
            "Could not connect to LLM" in record.message for record in caplog.records
        )
        assert any(
            "Make sure Ollama is running:" in record.message
            for record in caplog.records
        )
        assert any("ollama serve" in record.message for record in caplog.records)
        assert any(
            "ollama pull custom-model" in record.message for record in caplog.records
        )

    @patch("audify.convert.get_creator")
    @patch("audify.cli.get_file_extension")
    @patch("os.get_terminal_size")
    def test_main_fails_when_output_has_no_audio_artifacts(
        self, mock_terminal_size, mock_get_extension, mock_get_creator, runner
    ):
        """Test CLI returns non-zero when output dir exists but has no audio files."""
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".pdf"

        with tempfile.TemporaryDirectory() as out_dir:
            output_path = Path(out_dir)
            (output_path / "episodes").mkdir()
            (output_path / "scripts").mkdir()

            mock_creator = Mock()
            mock_creator.synthesize.return_value = str(output_path)
            mock_get_creator.return_value = mock_creator

            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                result = runner.invoke(cli, [temp_file.name])

        assert result.exit_code == 1
        assert "No audio artifacts were generated" in result.output
