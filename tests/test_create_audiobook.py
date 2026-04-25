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

    def test_get_creator_epub(self):
        """Test get_creator returns AudiobookEpubCreator for .epub files."""
        with patch("audify.convert.AudiobookEpubCreator") as mock_epub_creator:
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
                confirm=True,
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
                confirm=True,
                output_dir=None,
                tts_provider=None,
                task=None,
                prompt_file=None,
                mode="full",
            )

    def test_get_creator_pdf(self):
        """Test get_creator returns AudiobookPdfCreator for .pdf files."""
        with patch("audify.convert.AudiobookPdfCreator") as mock_pdf_creator:
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
                confirm=True,
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
                confirm=True,
                output_dir=None,
                tts_provider=None,
                task=None,
                prompt_file=None,
                mode="full",
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
    def test_main_epub_success(
        self, mock_terminal_size, mock_get_extension, mock_get_creator, runner, caplog
    ):
        """Test main function with successful EPUB processing."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".epub"

        mock_creator = Mock()
        mock_creator.synthesize.return_value = "/path/to/output"
        mock_get_creator.return_value = mock_creator

        with caplog.at_level(logging.INFO):
            with tempfile.NamedTemporaryFile(suffix=".epub") as temp_file:
                result = runner.invoke(
                    cli,
                    [
                        "--language",
                        "en",
                        "--voice",
                        "af_bella",
                        "--voice-model",
                        "kokoro",
                        "--translate",
                        "es",
                        "--save-text",
                        "--max-chapters",
                        "5",
                        "--confirm",
                        "--verbose",
                        temp_file.name,
                    ],
                )

        assert result.exit_code == 0
        # Log messages are captured by caplog
        assert any(
            "Audiobook creation complete!" in record.message
            for record in caplog.records
        )
        assert any("/path/to/output" in record.message for record in caplog.records)
        mock_creator.synthesize.assert_called_once()

    @patch("audify.convert.get_creator")
    @patch("audify.cli.get_file_extension")
    @patch("os.get_terminal_size")
    def test_main_pdf_success(
        self, mock_terminal_size, mock_get_extension, mock_get_creator, runner, caplog
    ):
        """Test main function with successful PDF processing."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".pdf"

        mock_creator = Mock()
        mock_creator.synthesize.return_value = "/path/to/output"
        mock_get_creator.return_value = mock_creator

        with caplog.at_level(logging.INFO):
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                result = runner.invoke(
                    cli,
                    [
                        "--language",
                        "fr",
                        "--voice",
                        "custom_voice",
                        "--voice-model",
                        "custom_model",
                        "--verbose",
                        temp_file.name,
                    ],
                )

        assert result.exit_code == 0
        # Log messages are captured by caplog
        assert any(
            "Audiobook creation complete!" in record.message
            for record in caplog.records
        )
        mock_creator.synthesize.assert_called_once()

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
    def test_main_generic_exception(
        self, mock_terminal_size, mock_get_extension, mock_get_creator, runner, caplog
    ):
        """Test main function handles generic exceptions."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".epub"

        mock_creator = Mock()
        mock_creator.synthesize.side_effect = Exception("Generic error")
        mock_get_creator.return_value = mock_creator

        with caplog.at_level(logging.ERROR):
            with tempfile.NamedTemporaryFile(suffix=".epub") as temp_file:
                result = runner.invoke(cli, ["--verbose", temp_file.name])

        assert result.exit_code == 1
        # Error messages are captured by caplog
        assert any(
            "Error: Generic error" in record.message for record in caplog.records
        )
        assert any(
            "Please check your configuration and try again." in record.message
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

    @patch("audify.cli.get_file_extension")
    @patch("os.get_terminal_size")
    def test_main_configuration_display(
        self, mock_terminal_size, mock_get_extension, runner, caplog
    ):
        """Test main function displays configuration correctly."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".epub"

        with caplog.at_level(logging.INFO):
            with (
                patch("audify.convert.get_creator") as mock_get_creator,
                tempfile.NamedTemporaryFile(suffix=".epub") as temp_file,
            ):
                mock_creator = Mock()
                mock_creator.synthesize.return_value = "/path/to/output"
                mock_get_creator.return_value = mock_creator

                result = runner.invoke(
                    cli,
                    [
                        "--language",
                        "fr",
                        "--llm-model",
                        "custom-model",
                        "--translate",
                        "en",
                        "--max-chapters",
                        "10",
                        "--verbose",
                        temp_file.name,
                    ],
                )

        assert result.exit_code == 0
        # Configuration messages are captured by caplog
        assert any(
            f"Source file: {temp_file.name}" in record.message
            for record in caplog.records
        )
        assert any("Language: fr" in record.message for record in caplog.records)
        assert any(
            "LLM Model: custom-model" in record.message for record in caplog.records
        )
        assert any(
            "Translation: fr -> en" in record.message for record in caplog.records
        )
        assert any("Max episodes: 10" in record.message for record in caplog.records)

    @patch("audify.convert.get_creator")
    @patch("audify.cli.get_file_extension")
    @patch("os.get_terminal_size")
    def test_main_short_flags_model_mapping(
        self, mock_terminal_size, mock_get_extension, mock_get_creator, runner
    ):
        """Test that -m maps to llm_model and -vm maps to model_name."""
        # Setup mocks
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".epub"
        mock_creator = Mock()
        mock_creator.synthesize.return_value = "/path/to/output"
        mock_get_creator.return_value = mock_creator

        with tempfile.NamedTemporaryFile(suffix=".epub") as temp_file:
            result = runner.invoke(
                cli,
                [
                    "-m",
                    "my-llm-model",
                    "-vm",
                    "my-tts-model",
                    temp_file.name,
                ],
            )

        assert result.exit_code == 0

        # Verify get_creator was called with correct arguments
        call_args = mock_get_creator.call_args
        assert call_args.kwargs["llm_model"] == "my-llm-model"
        assert call_args.kwargs["model_name"] == "my-tts-model"

    @patch("audify.convert.get_creator")
    @patch("audify.cli.get_file_extension")
    @patch("os.get_terminal_size")
    def test_main_options_after_path_are_parsed(
        self, mock_terminal_size, mock_get_extension, mock_get_creator, runner
    ):
        """Test that options appearing after the path are parsed correctly."""
        mock_terminal_size.return_value = (80, 24)
        mock_get_extension.return_value = ".pdf"

        mock_creator = Mock()
        mock_creator.synthesize.return_value = "/path/to/output"
        mock_get_creator.return_value = mock_creator

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            result = runner.invoke(
                cli,
                [
                    temp_file.name,
                    "--tts-provider",
                    "openai",
                    "--llm-model",
                    "gpt-4",
                    "--task",
                    "audiobook",
                ],
            )

        assert result.exit_code == 0
        call_args = mock_get_creator.call_args
        assert call_args.kwargs["tts_provider"] == "openai"
        assert call_args.kwargs["llm_model"] == "gpt-4"
        assert call_args.kwargs["task"] == "audiobook"

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
