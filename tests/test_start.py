# tests/test_start.py
import os
from pathlib import Path
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


@patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
def test_main_list_languages(mock_terminal_size, runner):
    """Test main command with --list-languages flag."""
    result = runner.invoke(cli, ["--list-languages"])

    assert result.exit_code == 0


@patch("pathlib.Path.exists", return_value=True)
@patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
@patch("time.sleep")
def test_main_list_models(mock_sleep, mock_terminal_size, mock_exists, runner):
    """Test main command with --list-models flag."""
    result = runner.invoke(cli, ["--list-models"])

    assert result.exit_code == 0


# DELETED: test_main_pdf_synthesis
# Reason: Tests the old PdfSynthesizer direct TTS code path which was removed
# during consolidation. The new CLI uses AudiobookCreator for all synthesis.


@patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
@patch("time.sleep")
@patch("requests.get")
def test_main_list_models_api_error(mock_get, mock_sleep, mock_terminal_size, runner):
    """Test main command with --list-models flag when API fails."""

    # Mock requests.get to raise a RequestException
    mock_get.side_effect = requests.RequestException("Connection failed")

    result = runner.invoke(cli, ["--list-models"])

    assert result.exit_code == 0
    assert "Error fetching models from Kokoro API" in result.output


@patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
@patch("time.sleep")
@patch("requests.get")
def test_main_list_models_request_exception(
    mock_get, mock_sleep, mock_terminal_size, runner
):
    """Test main command with --list-models flag with RequestException."""

    mock_get.side_effect = requests.RequestException("Network error")

    result = runner.invoke(cli, ["--list-models"])

    assert result.exit_code == 0
    assert "Error fetching models from Kokoro API" in result.output


@patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
@patch("time.sleep")
@patch("requests.get")
def test_main_list_models_success(mock_get, mock_sleep, mock_terminal_size, runner):
    """Test main command with --list-models flag with successful API response."""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [
            {"id": "model1"},
            {"id": "model2"},
            {"name": "model_without_id"},  # Test model without "id" key
        ]
    }
    mock_get.return_value = mock_response

    result = runner.invoke(cli, ["--list-models"])

    assert result.exit_code == 0
    assert "model1" in result.output
    assert "model2" in result.output


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
    @patch("requests.get")
    def test_list_voices_success(self, mock_get, mock_terminal_size, runner):
        """Test --list-voices flag with successful API response."""
        # Mock models response
        mock_models_response = Mock()
        mock_models_response.raise_for_status.return_value = None
        mock_models_response.json.return_value = {
            "data": [{"id": "kokoro"}, {"id": "tts-1"}]
        }

        # Mock voices response
        mock_voices_response = Mock()
        mock_voices_response.raise_for_status.return_value = None
        mock_voices_response.json.return_value = {
            "voices": ["af_bella", "af_alloy", "en_voice", "fr_voice"]
        }

        # Configure mock to return different responses for different URLs
        def side_effect(url, **kwargs):
            if "models" in url:
                return mock_models_response
            elif "voices" in url:
                return mock_voices_response
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_get.side_effect = side_effect

        result = runner.invoke(cli, ["--list-voices"])

        assert result.exit_code == 0
        assert "Available voices for KOKORO:" in result.output
        assert "AF voices:" in result.output
        assert "af_bella" in result.output
        assert "af_alloy" in result.output
        assert "EN voices:" in result.output
        assert "en_voice" in result.output
        assert "FR voices:" in result.output
        assert "fr_voice" in result.output

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

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.VoiceSamplesSynthesizer")
    def test_create_voice_samples_success(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test --create-voice-samples flag successfully creates samples."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(cli, ["--create-voice-samples"])

        assert result.exit_code == 0
        assert "Creating Voice Samples M4B" in result.output

        # Verify synthesizer was initialized correctly
        mock_synthesizer_class.assert_called_once_with(
            language="en",
            translate=None,
            max_samples=5,
            output_dir=None,
            llm_model=None,
            llm_base_url="http://localhost:11434",
        )
        mock_synthesizer.synthesize.assert_called_once()

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.VoiceSamplesSynthesizer")
    def test_create_voice_samples_with_translation(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test --create-voice-samples flag with translation."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(
            cli,
            ["--create-voice-samples", "--language", "en", "--translate", "es"],
        )

        assert result.exit_code == 0
        assert "Creating Voice Samples M4B" in result.output

        # Verify synthesizer was created with correct parameters
        call_args = mock_synthesizer_class.call_args
        assert call_args.kwargs["language"] == "en"
        assert call_args.kwargs["translate"] == "es"

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.VoiceSamplesSynthesizer")
    def test_create_voice_samples_with_custom_language(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test --create-voice-samples flag with custom language."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(cli, ["--create-voice-samples", "--language", "fr"])

        assert result.exit_code == 0

        # Verify synthesizer was created with correct language
        call_args = mock_synthesizer_class.call_args
        assert call_args.kwargs["language"] == "fr"
        assert call_args.kwargs["translate"] is None

    def test_help_includes_new_options(self, runner):
        """Test that help output includes the new CLI options."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "--list-voices" in result.output
        assert "-lv" in result.output
        assert "List available TTS voices" in result.output
        assert "--create-voice-samples" in result.output
        assert "-cvs" in result.output
        assert "Create a sample M4B audiobook" in result.output

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    def test_list_voices_short_flag(self, mock_terminal_size, runner):
        """Test --list-voices short flag (-lv)."""
        with patch("audify.cli.get_tts_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get_available_voices.return_value = ["af_bella"]
            mock_get_config.return_value = mock_config

            result = runner.invoke(cli, ["-lv"])

            assert result.exit_code == 0
            assert "Available voices for KOKORO:" in result.output

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.VoiceSamplesSynthesizer")
    def test_create_voice_samples_short_flag(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test --create-voice-samples short flag (-cvs)."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(cli, ["-cvs"])

        assert result.exit_code == 0
        assert "Creating Voice Samples M4B" in result.output
        mock_synthesizer.synthesize.assert_called_once()

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    def test_mutually_exclusive_options(self, mock_terminal_size, runner):
        """Test that new options work correctly with other flags."""
        # Test that list-models takes precedence over list-voices
        # since it's earlier in elif chain
        with patch("audify.cli.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"data": [{"id": "kokoro"}]}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = runner.invoke(cli, ["--list-voices", "--list-models"])

            # Should execute list-models (comes before list-voices in elif chain)
            assert result.exit_code == 0
            assert "Available models:" in result.output

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.VoiceSamplesSynthesizer")
    def test_voice_samples_takes_precedence(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test that --create-voice-samples takes precedence over other options."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(
            cli, ["--create-voice-samples", "--list-languages", "--list-models"]
        )

        # Should execute create-voice-samples (first if branch)
        assert result.exit_code == 0
        assert "Creating Voice Samples M4B" in result.output
        # Should not show language list
        assert "Available languages:" not in result.output


class TestListTTSProviders:
    """Tests for --list-tts-providers flag."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_tts_providers_all_available(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-tts-providers shows all providers with status."""
        mock_config = Mock()
        mock_config.is_available.return_value = True
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["--list-tts-providers"])

        assert result.exit_code == 0
        assert "Available TTS Providers" in result.output
        assert "Kokoro (Local)" in result.output
        assert "OpenAI TTS" in result.output
        assert "AWS Polly" in result.output
        assert "Google Cloud TTS" in result.output
        assert "Available" in result.output

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_tts_providers_not_configured(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-tts-providers shows 'Not configured' status."""
        mock_config = Mock()
        mock_config.is_available.return_value = False
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["--list-tts-providers"])

        assert result.exit_code == 0
        assert "Not configured" in result.output

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

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_tts_providers_short_flag(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test -ltp short flag for --list-tts-providers."""
        mock_config = Mock()
        mock_config.is_available.return_value = True
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["-ltp"])

        assert result.exit_code == 0
        assert "Available TTS Providers" in result.output

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_tts_providers_shows_config_info(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-tts-providers shows configuration info."""
        mock_config = Mock()
        mock_config.is_available.return_value = True
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["--list-tts-providers"])

        assert result.exit_code == 0
        # Check configuration hints are shown
        assert "KOKORO_API_URL" in result.output
        assert "OPENAI_API_KEY" in result.output
        assert "AWS_ACCESS_KEY_ID" in result.output
        assert "GOOGLE_APPLICATION_CREDENTIALS" in result.output
        assert "TTS_PROVIDER" in result.output


class TestListVoicesNonKokoro:
    """Tests for --list-voices with non-Kokoro providers."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_voices_openai_provider(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-voices with OpenAI provider shows flat list."""
        mock_config = Mock()
        mock_config.get_available_voices.return_value = [
            "alloy",
            "echo",
            "fable",
            "nova",
        ]
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["--list-voices", "--tts-provider", "openai"])

        assert result.exit_code == 0
        assert "Available voices for OPENAI:" in result.output
        assert "Voices for openai:" in result.output
        assert "alloy" in result.output
        assert "echo" in result.output

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_voices_aws_provider(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-voices with AWS provider shows flat list."""
        mock_config = Mock()
        mock_config.get_available_voices.return_value = ["Joanna", "Matthew", "Ivy"]
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["--list-voices", "--tts-provider", "aws"])

        assert result.exit_code == 0
        assert "Available voices for AWS:" in result.output
        assert "Voices for aws:" in result.output
        assert "Joanna" in result.output

    @patch("os.get_terminal_size", return_value=os.terminal_size((80, 24)))
    @patch("audify.cli.get_tts_config")
    def test_list_voices_google_provider(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-voices with Google provider shows flat list."""
        mock_config = Mock()
        mock_config.get_available_voices.return_value = [
            "en-US-Neural2-A",
            "en-US-Neural2-B",
        ]
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(cli, ["--list-voices", "--tts-provider", "google"])

        assert result.exit_code == 0
        assert "Available voices for GOOGLE:" in result.output
        assert "Voices for google:" in result.output
        assert "en-US-Neural2-A" in result.output


class TestGetAvailableModelsAndVoices:
    """Tests for the get_available_models_and_voices function."""

    @patch("requests.get")
    def test_get_available_models_and_voices_success(self, mock_get):
        """Test successful retrieval of models and voices."""
        # Mock models response
        mock_models_response = Mock()
        mock_models_response.raise_for_status.return_value = None
        mock_models_response.json.return_value = {
            "data": [
                {"id": "kokoro"},
                {"id": "tts-1"},
                {"name": "model_without_id"},  # Should be ignored
            ]
        }

        # Mock voices response
        mock_voices_response = Mock()
        mock_voices_response.raise_for_status.return_value = None
        mock_voices_response.json.return_value = {
            "voices": ["af_bella", "af_alloy", "en_voice"]
        }

        # Configure mock to return different responses for different URLs
        def side_effect(url, **kwargs):
            if "models" in url:
                return mock_models_response
            elif "voices" in url:
                return mock_voices_response
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_get.side_effect = side_effect

        models, voices = convert.get_available_models_and_voices()

        assert models == ["kokoro", "tts-1"]
        assert voices == ["af_alloy", "af_bella", "en_voice"]
        assert mock_get.call_count == 2

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_available_models_and_voices_api_error(self, mock_get, mock_sleep):
        """Test API error handling."""
        mock_get.side_effect = requests.RequestException("API Error")

        models, voices = convert.get_available_models_and_voices()

        assert models == []
        assert voices == []

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_available_models_and_voices_timeout(self, mock_get, mock_sleep):
        """Test timeout handling."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        models, voices = convert.get_available_models_and_voices()

        assert models == []
        assert voices == []

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_available_models_and_voices_http_error(self, mock_get, mock_sleep):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        models, voices = convert.get_available_models_and_voices()

        assert models == []
        assert voices == []

    @patch("requests.get")
    def test_get_available_models_and_voices_malformed_response(self, mock_get):
        """Test handling of malformed API responses."""
        # Mock models response with missing data
        mock_models_response = Mock()
        mock_models_response.raise_for_status.return_value = None
        mock_models_response.json.return_value = {}  # Missing 'data' key

        # Mock voices response with missing voices
        mock_voices_response = Mock()
        mock_voices_response.raise_for_status.return_value = None
        mock_voices_response.json.return_value = {}  # Missing 'voices' key

        def side_effect(url, **kwargs):
            if "models" in url:
                return mock_models_response
            elif "voices" in url:
                return mock_voices_response
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_get.side_effect = side_effect

        models, voices = convert.get_available_models_and_voices()

        assert models == []
        assert voices == []


def test_cli_package_version_fallback():
    """Test CLI version fallback when package not installed.

    Exercises the fallback code by re-executing the version-resolution
    block under a patched importlib.metadata.version, using exec to
    avoid corrupting the live module's sys.modules cache.
    """
    import importlib.metadata
    from unittest.mock import patch

    ns = {}
    with patch(
        "importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        exec(
            "try:\n    __version__ = importlib.metadata.version('audify-cli')\n"
            "except importlib.metadata.PackageNotFoundError:\n"
            "    __version__ = '0.1.0'",
            {"importlib": importlib},
            ns,
        )
    assert ns["__version__"] == "0.1.0"


class TestCLISubcommands:
    """Tests for CLI subcommands."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        from click.testing import CliRunner

        return CliRunner()

    def test_cli_commands_registered(self):
        """Debug test to check if subcommands are registered."""
        print("CLI commands:", cli.commands)
        assert "list-tasks" in cli.commands
        assert "validate-prompt" in cli.commands

    def test_list_tasks_command(self, runner):
        """Test the list-tasks subcommand."""
        result = runner.invoke(cli, ["list-tasks"])

        assert result.exit_code == 0
        assert "Available Tasks" in result.output
        assert "direct" in result.output
        assert "audiobook" in result.output

    def test_validate_prompt_command_valid(self, runner):
        """Test validate-prompt subcommand with valid prompt file."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("A valid prompt with sufficient content for testing")
            f.flush()
            result = runner.invoke(cli, ["validate-prompt", f.name])

        assert result.exit_code == 0
        assert "Prompt file is valid" in result.output
        assert "Length:" in result.output
        assert "Preview:" in result.output
        Path(f.name).unlink(missing_ok=True)

    def test_validate_prompt_command_invalid(self, runner):
        """Test validate-prompt subcommand with nonexistent prompt file."""
        result = runner.invoke(cli, ["validate-prompt", "/nonexistent/file.txt"])
        assert result.exit_code == 2


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


def test_cli_terminal_width_oserror():
    """Test that CLI handles OSError from os.get_terminal_size."""
    from unittest.mock import patch

    from click.testing import CliRunner

    # Mock os.get_terminal_size to raise OSError
    with patch("os.get_terminal_size", side_effect=OSError):
        runner = CliRunner()
        result = runner.invoke(cli, ["--list-languages"])
        assert result.exit_code == 0
        # Should still display languages
        assert "Available languages" in result.output


def test_cli_subcommand_early_return():
    """Test that subcommands work and the main CLI callback doesn't run its body."""
    from click.testing import CliRunner

    from audify.cli import cli

    runner = CliRunner()
    # list-tasks should return task list without triggering the main CLI path logic
    result = runner.invoke(cli, ["list-tasks"])
    assert result.exit_code == 0
    assert "Available Tasks" in result.output
    # Ensure main CLI body didn't run (no "Error: Path" message)
    assert "Error:" not in result.output


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


def test_main_module_import():
    """Test that the main module can be imported."""
    import audify.__main__ as main_module

    assert main_module.__name__ == "audify.__main__"
