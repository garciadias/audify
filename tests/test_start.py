# tests/test_start.py
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from click.testing import CliRunner

from audify import convert
from audify.cli import cli

# These module-level mocks were removed as they were only used by obsolete tests
# that tested the old EpubSynthesizer and PdfSynthesizer direct code paths.


@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance."""
    return CliRunner()


# DELETED: test_main_epub_synthesis
# Reason: Tests the old EpubSynthesizer direct TTS code path which was removed
# during consolidation. The new CLI uses AudiobookCreator for all synthesis.
# Also tests non-existent --model and --engine flags.

# DELETED: test_main_epub_synthesis_abort_confirmation
# Reason: Tests the old confirmation flow for direct EpubSynthesizer which was
# removed during consolidation into unified CLI.


# reset_mocks_fixture was removed as it was only used by obsolete tests.

# DELETED: test_main_pdf_synthesis
# Reason: Tests the old PdfSynthesizer direct TTS code path which was removed
# during consolidation. The new CLI uses AudiobookCreator for all synthesis.

# DELETED: test_main_unsupported_file_format
# Reason: Tests unsupported file format handling which was part of the old
# direct synthesizer path. This functionality is now integrated into
# get_creator() and tested through AudiobookCreator creation tests.


"""Tests for new CLI functionality in start.py."""


class TestStartCLINewFeatures:
    """Tests for new CLI features in start.py."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()


    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_voices_api_error(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-voices flag when API fails."""
        mock_config = Mock()
        mock_config.get_available_voices.side_effect = Exception("API Error")
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["--list-voices"])

        assert result.exit_code == 0
        assert "Available voices for KOKORO:" in result.output
        assert "Error fetching voices from kokoro" in result.output

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_voices_no_voices_found(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-voices flag when no voices are found."""
        mock_config = Mock()
        mock_config.get_available_voices.return_value = []
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["--list-voices"])

        assert result.exit_code == 0
        assert "Available voices for KOKORO:" in result.output
        assert "No voices found for kokoro." in result.output

class TestListTTSProviders:
    """Tests for --list-tts-providers flag."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_tts_providers_error(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-tts-providers handles provider errors gracefully."""
        mock_get_tts_config.side_effect = Exception("Provider error")

        result = runner.invoke(cli, ["--list-tts-providers"])

        assert result.exit_code == 0
        assert "Not available" in result.output

class TestListVoicesNonKokoro:
    """Tests for --list-voices with non-Kokoro providers."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()


class TestCLISubcommands:
    """Tests for CLI subcommands."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        from click.testing import CliRunner

        return CliRunner()

def test_cli_no_path_shows_help():
    """Test that CLI shows help when no path is provided."""
    import os
    from unittest.mock import patch

    from click.testing import CliRunner

    terminal_size = os.terminal_size((80, 24))
    with patch("os.get_terminal_size", return_value=terminal_size):
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Audify:" in result.output

def test_cli_directory_mode_with_prompt_file():
    """Test CLI directory mode with prompt-file option (covers line 379)."""
    import os
    import tempfile
    from unittest.mock import Mock, patch

    from click.testing import CliRunner

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock Path.exists and Path.is_dir
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                # Prevent container runtime check from activating
                # (Path.exists mock would make _is_container_runtime return True)
                with patch("audify.cli._is_container_runtime", return_value=False):
                    # Create a real output directory with an audio file so
                    # _contains_audio_artifacts doesn't fail on os.scandir.
                    output_dir = os.path.join(tmpdir, "output")
                    os.makedirs(output_dir)
                    with open(os.path.join(output_dir, "test.mp3"), "w") as f:
                        f.write("dummy audio")
                    mock_creator = Mock()
                    mock_creator.synthesize.return_value = output_dir
                    with patch(
                        "audify.cli.DirectoryAudiobookCreator",
                        return_value=mock_creator,
                    ):
                        with patch(
                            "audify.cli._contains_audio_artifacts",
                            return_value=True,
                        ):
                            runner = CliRunner()
                            # Create a dummy prompt file
                            prompt_file = os.path.join(tmpdir, "test.prompt")
                            with open(prompt_file, "w") as f:
                                f.write("Test prompt")
                            # Invoke CLI with directory path and prompt file
                            result = runner.invoke(
                                cli, [tmpdir, "--prompt-file", prompt_file]
                            )
                            assert result.exit_code == 0
                            # Line 379 should be executed (prompt file logging)
                            # No exception means success


def test_cli_single_file_mode_with_prompt_file():
    """Test CLI single file mode with prompt-file option (covers line 433)."""
    import os
    import tempfile
    from unittest.mock import Mock, patch

    from click.testing import CliRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        epub_path = os.path.join(tmpdir, "test.epub")
        with open(epub_path, "w") as f:
            f.write("dummy epub content")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                # Prevent container runtime check from activating
                with patch("audify.cli._is_container_runtime", return_value=False):
                    # Mock convert.get_creator to return a mock creator
                    mock_creator = Mock()
                    output_dir = os.path.join(tmpdir, "output")
                    os.makedirs(output_dir)
                    with open(os.path.join(output_dir, "test.mp3"), "w") as f:
                        f.write("dummy audio")
                    mock_creator.synthesize.return_value = output_dir
                    with patch("audify.convert.get_creator", return_value=mock_creator):
                        with patch(
                            "audify.cli._contains_audio_artifacts",
                            return_value=True,
                        ):
                            with patch(
                                "os.get_terminal_size",
                                return_value=os.terminal_size((80, 24)),
                            ):
                                runner = CliRunner()
                                # Create a dummy prompt file
                                prompt_file = os.path.join(
                                    tmpdir, "test.prompt"
                                )
                                with open(prompt_file, "w") as f:
                                    f.write("Test prompt")
                                # Invoke CLI with file path and prompt file
                                result = runner.invoke(
                                    cli,
                                    [epub_path, "--prompt-file", prompt_file],
                                )
                                assert result.exit_code == 0
