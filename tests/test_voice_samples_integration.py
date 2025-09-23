"""Integration tests for voice samples functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from audify import start
from audify.text_to_speech import VoiceSamplesSynthesizer


class TestVoiceSamplesIntegration:
    """Integration tests for the complete voice samples workflow."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def mock_api_responses(self):
        """Fixture providing mock API responses."""
        def mock_get_side_effect(url, **kwargs):
            if "models" in url:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {
                    "data": [{"id": "kokoro"}, {"id": "tts-1"}]
                }
                return mock_response
            elif "voices" in url:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {
                    "voices": ["af_bella", "af_alloy"]
                }
                return mock_response
            else:
                raise ValueError(f"Unexpected URL: {url}")

        return mock_get_side_effect

    def test_cli_integration_voice_samples_creation(
        self, runner, mock_api_responses
    ):
        """Test full CLI integration for voice samples creation."""
        with (
            patch("requests.get", side_effect=mock_api_responses),
            patch("os.get_terminal_size", return_value=(80, 24)),
            patch("tempfile.mkdtemp", return_value="/tmp/test_voice_samples"),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
            patch("audify.text_to_speech.BaseSynthesizer") as mock_base_synth,
            patch("audify.text_to_speech.break_text_into_sentences") as mock_break,
            patch("audify.text_to_speech.AudioSegment") as mock_audio,
            patch("subprocess.run") as mock_subprocess,
            patch("shutil.rmtree"),
            patch("pathlib.Path.unlink"),
        ):
            # Setup mocks
            mock_break.return_value = ["Test sentence."]

            # Mock synthesizer
            mock_synth_instance = Mock()
            mock_base_synth.return_value = mock_synth_instance
            mock_synth_instance._convert_to_mp3.return_value = Path("/tmp/test.mp3")

            # Mock audio processing
            mock_audio_instance = MagicMock()
            mock_audio_instance.__len__.return_value = 5000  # 5 seconds
            mock_audio.empty.return_value = mock_audio_instance
            mock_audio.from_mp3.return_value = mock_audio_instance
            mock_audio_instance.__iadd__.return_value = mock_audio_instance

            # Mock FFmpeg
            mock_ffmpeg_result = Mock()
            mock_ffmpeg_result.stdout = "Success"
            mock_ffmpeg_result.stderr = ""
            mock_subprocess.return_value = mock_ffmpeg_result

            result = runner.invoke(start.main, ["--create-voice-samples"])

            assert result.exit_code == 0
            assert "Creating Voice Samples M4B" in result.output

            # Verify that samples were created (2 models × 2 voices = 4 combinations)
            # But limited by max_samples in current config
            assert mock_base_synth.call_count > 0

    def test_voice_samples_with_translation_integration(
        self, runner, mock_api_responses
    ):
        """Test voice samples creation with translation."""
        with (
            patch("requests.get", side_effect=mock_api_responses),
            patch("os.get_terminal_size", return_value=(80, 24)),
            patch("tempfile.mkdtemp", return_value="/tmp/test_voice_samples"),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
            patch("audify.text_to_speech.translate_sentence") as mock_translate,
            patch("audify.text_to_speech.BaseSynthesizer") as mock_base_synth,
            patch("audify.text_to_speech.break_text_into_sentences") as mock_break,
            patch("audify.text_to_speech.AudioSegment") as mock_audio,
            patch("subprocess.run") as mock_subprocess,
            patch("shutil.rmtree"),
            patch("pathlib.Path.unlink"),
        ):
            # Setup translation mock
            mock_translate.return_value = "Texto traducido al español"

            # Setup other mocks
            mock_break.return_value = ["Translated sentence."]
            mock_synth_instance = Mock()
            mock_base_synth.return_value = mock_synth_instance
            mock_synth_instance._convert_to_mp3.return_value = Path("/tmp/test.mp3")

            mock_audio_instance = MagicMock()
            mock_audio_instance.__len__.return_value = 5000
            mock_audio.empty.return_value = mock_audio_instance
            mock_audio.from_mp3.return_value = mock_audio_instance
            mock_audio_instance.__iadd__.return_value = mock_audio_instance

            mock_subprocess.return_value = Mock(stdout="", stderr="")

            result = runner.invoke(start.main, [
                "--create-voice-samples",
                "--translate", "es"
            ])

            assert result.exit_code == 0
            assert "Creating Voice Samples M4B" in result.output

            # Verify translation was called
            mock_translate.assert_called()

    def test_list_voices_integration(self, runner, mock_api_responses):
        """Test voice listing integration."""
        with (
            patch("requests.get", side_effect=mock_api_responses),
            patch("os.get_terminal_size", return_value=(80, 24)),
        ):
            result = runner.invoke(start.main, ["--list-voices"])

            assert result.exit_code == 0
            assert "Available voices:" in result.output
            assert "AF voices:" in result.output
            assert "af_bella" in result.output
            assert "af_alloy" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    def test_error_handling_integration(self, mock_terminal_size, runner):
        """Test error handling in voice samples creation."""
        with (
            patch("tempfile.mkdtemp", return_value="/tmp/test_voice_samples"),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                VoiceSamplesSynthesizer, "_get_available_models_and_voices"
            ) as mock_get_models_voices,
        ):
            # Simulate API failure
            mock_get_models_voices.return_value = ([], [])

            result = runner.invoke(start.main, ["--create-voice-samples"])

            # Should complete without crashing even with no models/voices
            assert result.exit_code == 0
            assert "Creating Voice Samples M4B" in result.output

    def test_cli_precedence_integration(self, runner):
        """Test that CLI option precedence works correctly."""
        with (
            patch("os.get_terminal_size", return_value=(80, 24)),
            patch("tempfile.mkdtemp", return_value="/tmp/test_voice_samples"),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                VoiceSamplesSynthesizer, "_get_available_models_and_voices"
            ) as mock_get,
            patch.object(VoiceSamplesSynthesizer, "synthesize") as mock_synthesize,
        ):
            mock_get.return_value = (["kokoro"], ["af_bella"])
            mock_synthesize.return_value = Path("/tmp/voice_samples.m4b")

            # Even with multiple flags, voice samples should take precedence
            result = runner.invoke(start.main, [
                "--create-voice-samples",
                "--list-languages",
                "--list-models",
                "--list-voices"
            ])

            assert result.exit_code == 0
            assert "Creating Voice Samples M4B" in result.output
            # Should not show other lists
            assert "Available languages:" not in result.output
            assert "Available models:" not in result.output


class TestVoiceSamplesRegressionTests:
    """Regression tests to ensure existing functionality isn't broken."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch("os.get_terminal_size", return_value=(80, 24))
    def test_existing_list_languages_still_works(self, mock_terminal_size, runner):
        """Test that existing --list-languages functionality is preserved."""
        result = runner.invoke(start.main, ["--list-languages"])

        assert result.exit_code == 0
        assert "Available languages:" in result.output
        assert "Language\tCode" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("requests.get")
    def test_existing_list_models_still_works(
            self, mock_get, mock_terminal_size, runner
        ):
        """Test that existing --list-models functionality is preserved."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [{"id": "kokoro"}, {"id": "tts-1"}]
        }
        mock_get.return_value = mock_response

        result = runner.invoke(start.main, ["--list-models"])

        assert result.exit_code == 0
        assert "Available models:" in result.output
        assert "kokoro" in result.output
        assert "tts-1" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_file_extension", return_value=".epub")
    @patch("audify.start.EpubSynthesizer")
    def test_existing_epub_synthesis_still_works(
        self, mock_epub_synthesizer, mock_get_extension, mock_terminal_size, runner
    ):
        """Test that existing EPUB synthesis functionality is preserved."""
        mock_synth_instance = Mock()
        mock_epub_synthesizer.return_value = mock_synth_instance

        with tempfile.NamedTemporaryFile(suffix=".epub") as temp_file:
            result = runner.invoke(start.main, [temp_file.name])

            # Should not interfere with existing functionality
            assert result.exit_code == 0
            assert "Epub to Audiobook" in result.output

            # Verify synthesizer was called
            mock_epub_synthesizer.assert_called_once()
            mock_synth_instance.synthesize.assert_called_once()

    def test_help_output_includes_all_options(self, runner):
        """Test that help output includes both old and new options."""
        result = runner.invoke(start.main, ["--help"])

        assert result.exit_code == 0

        # Existing options
        assert "--list-languages" in result.output
        assert "--list-models" in result.output
        assert "--language" in result.output
        assert "--voice" in result.output
        assert "--translate" in result.output

        # New options
        assert "--list-voices" in result.output
        assert "--create-voice-samples" in result.output

        # Short flags
        assert "-ll" in result.output
        assert "-lm" in result.output
        assert "-lv" in result.output
        assert "-cvs" in result.output


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch("os.get_terminal_size", return_value=(80, 24))
    def test_voice_samples_with_no_max_samples(self, mock_terminal_size, runner):
        """Test voice samples creation without max_samples limit."""
        with (
            patch("tempfile.mkdtemp", return_value="/tmp/test_voice_samples"),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                VoiceSamplesSynthesizer, "_get_available_models_and_voices"
            ) as mock_get,
            patch.object(VoiceSamplesSynthesizer, "synthesize") as mock_synthesize,
        ):
            # Many models and voices to test limiting
            mock_get.return_value = (
                ["model1", "model2", "model3"],
                ["voice1", "voice2", "voice3", "voice4"]
            )
            mock_synthesize.return_value = Path("/tmp/voice_samples.m4b")

            result = runner.invoke(start.main, ["--create-voice-samples"])

            assert result.exit_code == 0
            mock_synthesize.assert_called_once()

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("requests.get")
    def test_malformed_api_responses(self, mock_get, mock_terminal_size, runner):
        """Test handling of malformed API responses in voice listing."""
        # Return malformed response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"invalid": "response"}
        mock_get.return_value = mock_response

        result = runner.invoke(start.main, ["--list-voices"])

        assert result.exit_code == 0
        assert "No voices found." in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_available_models_and_voices")
    def test_voice_grouping_edge_cases(
        self, mock_get_models_voices, mock_terminal_size, runner
    ):
        """Test voice grouping with edge case voice names."""
        mock_get_models_voices.return_value = (
            ["kokoro"],
            ["voice_without_underscore", "multi_part_voice_name", "single"]
        )

        result = runner.invoke(start.main, ["--list-voices"])

        assert result.exit_code == 0
        assert "VOICE voices:" in result.output
        assert "MULTI voices:" in result.output
        assert "OTHER voices:" in result.output  # single -> other
