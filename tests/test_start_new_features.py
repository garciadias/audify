"""Tests for new CLI functionality in start.py."""

from unittest.mock import Mock, patch

import pytest
import requests
from click.testing import CliRunner

from audify import start


class TestStartCLINewFeatures:
    """Tests for new CLI features in start.py."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch("os.get_terminal_size", return_value=(80, 24))
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

        result = runner.invoke(start.main, ["--list-voices"])

        assert result.exit_code == 0
        assert "Available voices:" in result.output
        assert "AF voices:" in result.output
        assert "af_bella" in result.output
        assert "af_alloy" in result.output
        assert "EN voices:" in result.output
        assert "en_voice" in result.output
        assert "FR voices:" in result.output
        assert "fr_voice" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_available_models_and_voices")
    def test_list_voices_api_error(
            self, mock_get_models_voices, mock_terminal_size, runner
        ):
        """Test --list-voices flag when API fails."""
        mock_get_models_voices.side_effect = Exception("API Error")

        result = runner.invoke(start.main, ["--list-voices"])

        assert result.exit_code == 0
        assert "Available voices:" in result.output
        assert "Error fetching voices from Kokoro API" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_available_models_and_voices")
    def test_list_voices_no_voices_found(
            self, mock_get_models_voices, mock_terminal_size, runner
        ):
        """Test --list-voices flag when no voices are found."""
        mock_get_models_voices.return_value = (["kokoro"], [])

        result = runner.invoke(start.main, ["--list-voices"])

        assert result.exit_code == 0
        assert "Available voices:" in result.output
        assert "No voices found." in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.VoiceSamplesSynthesizer")
    def test_create_voice_samples_success(
            self, mock_synthesizer_class, mock_terminal_size, runner
        ):
        """Test --create-voice-samples flag successfully creates samples."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(start.main, ["--create-voice-samples"])

        assert result.exit_code == 0
        assert "Creating Voice Samples M4B" in result.output

        # Verify synthesizer was initialized correctly
        mock_synthesizer_class.assert_called_once_with(
            language="en",
            translate=None,
            max_samples=5
        )
        mock_synthesizer.synthesize.assert_called_once()

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.VoiceSamplesSynthesizer")
    def test_create_voice_samples_with_translation(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test --create-voice-samples flag with translation."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(start.main, [
            "--create-voice-samples",
            "--language", "en",
            "--translate", "es"
        ])

        assert result.exit_code == 0
        assert "Creating Voice Samples M4B" in result.output

        # Verify synthesizer was created with correct parameters
        call_args = mock_synthesizer_class.call_args
        assert call_args.kwargs["language"] == "en"
        assert call_args.kwargs["translate"] == "es"

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.VoiceSamplesSynthesizer")
    def test_create_voice_samples_with_custom_language(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test --create-voice-samples flag with custom language."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(start.main, [
            "--create-voice-samples",
            "--language", "fr"
        ])

        assert result.exit_code == 0

        # Verify synthesizer was created with correct language
        call_args = mock_synthesizer_class.call_args
        assert call_args.kwargs["language"] == "fr"
        assert call_args.kwargs["translate"] is None

    def test_help_includes_new_options(self, runner):
        """Test that help output includes the new CLI options."""
        result = runner.invoke(start.main, ["--help"])

        assert result.exit_code == 0
        assert "--list-voices" in result.output
        assert "-lv" in result.output
        assert "List available TTS voices" in result.output
        assert "--create-voice-samples" in result.output
        assert "-cvs" in result.output
        assert "Create a sample M4B audiobook" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    def test_list_voices_short_flag(self, mock_terminal_size, runner):
        """Test --list-voices short flag (-lv)."""
        with patch("audify.start.get_available_models_and_voices") as mock_get:
            mock_get.return_value = (["kokoro"], ["af_bella"])

            result = runner.invoke(start.main, ["-lv"])

            assert result.exit_code == 0
            assert "Available voices:" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.VoiceSamplesSynthesizer")
    def test_create_voice_samples_short_flag(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test --create-voice-samples short flag (-cvs)."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(start.main, ["-cvs"])

        assert result.exit_code == 0
        assert "Creating Voice Samples M4B" in result.output
        mock_synthesizer.synthesize.assert_called_once()

    @patch("os.get_terminal_size", return_value=(80, 24))
    def test_mutually_exclusive_options(self, mock_terminal_size, runner):
        """Test that new options work correctly with other flags."""
        # Test that list-models takes precedence over list-voices
        # since it's earlier in elif chain
        with patch("audify.start.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"data": [{"id": "kokoro"}]}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = runner.invoke(start.main, ["--list-voices", "--list-models"])

            # Should execute list-models (comes before list-voices in elif chain)
            assert result.exit_code == 0
            assert "Available models:" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.VoiceSamplesSynthesizer")
    def test_voice_samples_takes_precedence(
        self, mock_synthesizer_class, mock_terminal_size, runner
    ):
        """Test that --create-voice-samples takes precedence over other options."""
        mock_synthesizer = Mock()
        mock_synthesizer_class.return_value = mock_synthesizer

        result = runner.invoke(start.main, [
            "--create-voice-samples",
            "--list-languages",
            "--list-models"
        ])

        # Should execute create-voice-samples (first if branch)
        assert result.exit_code == 0
        assert "Creating Voice Samples M4B" in result.output
        # Should not show language list
        assert "Available languages:" not in result.output


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

        models, voices = start.get_available_models_and_voices()

        assert models == ["kokoro", "tts-1"]
        assert voices == ["af_alloy", "af_bella", "en_voice"]
        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_get_available_models_and_voices_api_error(self, mock_get):
        """Test API error handling."""
        mock_get.side_effect = requests.RequestException("API Error")

        models, voices = start.get_available_models_and_voices()

        assert models == []
        assert voices == []

    @patch("requests.get")
    def test_get_available_models_and_voices_timeout(self, mock_get):
        """Test timeout handling."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        models, voices = start.get_available_models_and_voices()

        assert models == []
        assert voices == []

    @patch("requests.get")
    def test_get_available_models_and_voices_http_error(self, mock_get):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        models, voices = start.get_available_models_and_voices()

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

        models, voices = start.get_available_models_and_voices()

        assert models == []
        assert voices == []
