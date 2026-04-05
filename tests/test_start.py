# tests/test_start.py
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from click.testing import CliRunner
from reportlab.pdfgen import canvas

from audify import start

# Mock the audify modules before importing start
# Keep mocks for underlying synthesizers and utils
mock_epub_synthesizer = MagicMock()
mock_pdf_synthesizer = MagicMock()
mock_inspect_synthesizer = MagicMock()

# Mock the classes themselves
mock_epub_synthesizer_class = MagicMock(return_value=mock_epub_synthesizer)
mock_pdf_synthesizer_class = MagicMock(return_value=mock_pdf_synthesizer)
mock_inspect_synthesizer_class = MagicMock(return_value=mock_inspect_synthesizer)

# Mock the utility function
mock_get_file_extension = MagicMock()


# Apply patches using a dictionary
patches = {
    "audify.text_to_speech.EpubSynthesizer": mock_epub_synthesizer_class,
    "audify.text_to_speech.PdfSynthesizer": mock_pdf_synthesizer_class,
    "audify.text_to_speech.InspectSynthesizer": mock_inspect_synthesizer_class,
    "audify.start.get_file_extension": mock_get_file_extension,
    "os.get_terminal_size": MagicMock(return_value=(80, 24)),  # Mock terminal size
}

# Use pytest's patching capabilities or context managers in tests
# Import start *after* potentially mocking dependencies if needed at module level
# For simplicity here, we'll patch within tests.


@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance."""
    return CliRunner()


@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)  # Mock file existence check
def test_main_epub_synthesis(mock_exists, runner):
    """Test main command with an EPUB file."""
    mock_get_file_extension.return_value = ".epub"
    with TemporaryDirectory() as temp_dir:
        # Create a temporary EPUB file
        epub_file_path = f"{temp_dir}/test.epub"
        with open(epub_file_path, "w") as f:
            f.write("Fake EPUB content")

            result = runner.invoke(
                start.main,
                [
                    epub_file_path,
                    "--language",
                    "en",
                    "--voice",
                    "voice.wav",
                    "--model",
                    "model_name",
                    "--engine",
                    "tts_models",
                    "--save-text",
                    "--translate",
                ],
            )
        result = runner.invoke(
            start.main,
            [
                "fake.epub",
                "--language",
                "en",
                "--voice",
                "voice.wav",
                "--model",
                "model_name",
                "-y",
            ],
        )

        assert result.exit_code == 2
        mock_pdf_synthesizer_class.assert_not_called()
        mock_inspect_synthesizer_class.assert_not_called()


@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)  # Mock file existence check
def test_main_epub_synthesis_abort_confirmation(mock_get_file_extension, runner):
    """Test main command with an EPUB file aborting confirmation."""
    mock_get_file_extension.return_value = ".epub"
    # Simulate user typing 'n' then Enter for confirmation
    result = runner.invoke(start.main, ["fake.epub"], input="n\n")
    assert result.exit_code == 2

    # Apply patches using a dictionary
    patches = {
        "audify.text_to_speech.EpubSynthesizer": mock_epub_synthesizer_class,
        "audify.text_to_speech.PdfSynthesizer": mock_pdf_synthesizer_class,
        "audify.text_to_speech.InspectSynthesizer": mock_inspect_synthesizer_class,
        "audify.start.get_file_extension": mock_get_file_extension,
        "os.get_terminal_size": MagicMock(return_value=(80, 24)),  # Mock terminal size
    }

    # Use pytest's patching capabilities or context managers in tests
    # Import start *after* potentially mocking dependencies if needed at module level
    # For simplicity here, we'll patch within tests.
    # We need to apply patches before importing start if mocks affect import time
    # Using patch.dict on sys.modules handles this if done before import.
    # However, the example structure imports start first, then patches in tests.
    # Let's assume the mocks are intended to be active during test execution only.
    with patch.dict(sys.modules, patches):
        pass  # noqa: E402


def reset_mocks_fixture():
    """Reset mocks before each test function."""
    mock_epub_synthesizer_class.reset_mock()
    mock_pdf_synthesizer_class.reset_mock()
    mock_inspect_synthesizer_class.reset_mock()
    mock_get_file_extension.reset_mock()
    mock_epub_synthesizer.reset_mock()
    mock_pdf_synthesizer.reset_mock()
    mock_inspect_synthesizer.reset_mock()
    # Reset mock model attributes if they were set directly
    if hasattr(mock_inspect_synthesizer, "model"):
        # Ensure the attribute exists before trying to delete it
        try:
            del mock_inspect_synthesizer.model
        except AttributeError:
            pass  # Already deleted or never set


@patch("os.get_terminal_size", return_value=(80, 24))  # Mock terminal size
def test_main_list_languages(mock_exists, runner):
    """Test main command with --list-languages flag."""
    # Setup mock model attribute for languages
    result = runner.invoke(start.main, ["--list-languages"])

    assert result.exit_code == 0


@patch("pathlib.Path.exists", return_value=True)
@patch("os.get_terminal_size", return_value=(80, 24))  # Mock terminal size
def test_main_list_models(mock_exists, mock_terminal_size, runner):
    """Test main command with --list-models flag."""
    # Setup mock model attribute for models
    result = runner.invoke(start.main, ["--list-models"])

    assert result.exit_code == 0


@patch("pathlib.Path.exists", return_value=True)
@patch("os.get_terminal_size", return_value=(80, 24))
def test_main_pdf_synthesis(mock_exists, mock_terminal_size, runner):
    """Test main command with a PDF file."""
    with TemporaryDirectory() as temp_dir:
        # Create a temporary PDF file
        pdf_file_path = f"{temp_dir}/test.pdf"
        # Use reportlab to create a PDF with text content

        c = canvas.Canvas(pdf_file_path, pagesize=(8.27 * 72, 11.7 * 72))
        c.drawString(100, 700, "This is a test PDF content.")
        c.save()
        mock_get_file_extension.return_value = True

        # Invoke the main command but mock the PDF synthesizer
        mock_pdf_synthesizer_class.return_value = mock_pdf_synthesizer
        mock_pdf_synthesizer.synthesize.return_value = None  # Mock synthesize method
        mock_get_file_extension.return_value = ".pdf"
        # Now invoke the main command with the PDF file mocking all API calls
        with (
            patch("audify.utils.api_config.requests.get") as mock_get_voices,
            patch("audify.utils.api_config.requests.post") as mock_post_synthesis,
            patch(
                "audify.translate.OllamaTranslationConfig.translate"
            ) as mock_translate,
            patch("audify.utils.m4b_builder.subprocess.run") as mock_subprocess,
            patch("audify.text_to_speech.AudioSegment") as mock_audio_segment,
            patch("audify.utils.audio.AudioSegment") as mock_audio_segment_utils,
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            # Mock the voices GET request
            mock_voices_response = MagicMock()
            mock_voices_response.status_code = 200
            mock_voices_response.json.return_value = {
                "voices": ["af_bella", "other_voice"]
            }
            mock_voices_response.raise_for_status.return_value = None
            mock_get_voices.return_value = mock_voices_response

            # Mock the synthesis POST request
            mock_synthesis_response = MagicMock()
            mock_synthesis_response.status_code = 200
            mock_synthesis_response.json.return_value = {
                "data": {"audio_url": "http://example.com/audio.mp3"}
            }
            mock_synthesis_response.content = b"fake_audio_data"
            mock_synthesis_response.raise_for_status.return_value = None
            mock_post_synthesis.return_value = mock_synthesis_response

            # Mock the translation method
            mock_translate.return_value = "This is test PDF content."

            # Mock subprocess.run for ffmpeg calls
            mock_ffmpeg_result = MagicMock()
            mock_ffmpeg_result.stdout = "FFmpeg mock output"
            mock_ffmpeg_result.stderr = ""
            mock_ffmpeg_result.returncode = 0
            mock_subprocess.return_value = mock_ffmpeg_result

            # Mock AudioSegment for pydub audio processing
            mock_audio_instance = MagicMock()
            mock_audio_instance.__len__ = MagicMock(return_value=1000)
            mock_audio_instance.__iadd__ = MagicMock(return_value=mock_audio_instance)
            mock_audio_segment.from_wav.return_value = mock_audio_instance
            mock_audio_segment.empty.return_value = mock_audio_instance
            mock_audio_instance.export.return_value = None
            # Also mock AudioSegment in utils.audio (used by combine_wav_segments)
            mock_audio_segment_utils.from_wav.return_value = mock_audio_instance
            mock_audio_segment_utils.empty.return_value = mock_audio_instance

            # Mock file operations
            mock_unlink.return_value = None  # File deletion succeeds

            # Mock AudioProcessor to avoid file operations
            with patch(
                "audify.utils.audio.AudioProcessor.convert_wav_to_mp3"
            ) as mock_convert:

                mock_convert.return_value = Path("/fake/output.mp3")

                result = runner.invoke(
                    start.main,
                    [
                        pdf_file_path,
                        "--language",
                        "fr",
                        "--voice",
                        "af_bella",
                        "--save-text",
                        "--translate",
                        "en",
                    ],
                )

        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
            print(f"Exception: {result.exception}")
        assert result.exit_code == 0
        assert "PDF to mp3" in result.output


@patch("os.get_terminal_size", return_value=(80, 24))
@patch("requests.get")
def test_main_list_models_api_error(mock_get, mock_terminal_size, runner):
    """Test main command with --list-models flag when API fails."""

    # Mock requests.get to raise a RequestException
    mock_get.side_effect = requests.RequestException("Connection failed")

    result = runner.invoke(start.main, ["--list-models"])

    assert result.exit_code == 0
    assert "Error fetching models from Kokoro API" in result.output


@patch("os.get_terminal_size", return_value=(80, 24))
@patch("requests.get")
def test_main_list_models_request_exception(mock_get, mock_terminal_size, runner):
    """Test main command with --list-models flag with RequestException."""

    mock_get.side_effect = requests.RequestException("Network error")

    result = runner.invoke(start.main, ["--list-models"])

    assert result.exit_code == 0
    assert "Error fetching models from Kokoro API" in result.output


@patch("os.get_terminal_size", return_value=(80, 24))
@patch("requests.get")
def test_main_list_models_success(mock_get, mock_terminal_size, runner):
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

    result = runner.invoke(start.main, ["--list-models"])

    assert result.exit_code == 0
    assert "model1" in result.output
    assert "model2" in result.output


@patch("os.get_terminal_size", return_value=(80, 24))
def test_main_unsupported_file_format(mock_terminal_size, runner):
    """Test main command with unsupported file format."""
    with TemporaryDirectory() as temp_dir:
        # Create a temporary file with unsupported extension
        unsupported_file = f"{temp_dir}/test.txt"
        with open(unsupported_file, "w") as f:
            f.write("Test content")

        with patch("audify.start.get_file_extension", return_value=".txt"):
            result = runner.invoke(start.main, [unsupported_file])

        assert result.exit_code == 1
        # The exception message doesn't get printed by Click, just the exit code
        assert isinstance(result.exception, ValueError)
        assert "Unsupported file format" in str(result.exception)


"""Tests for new CLI functionality in start.py."""


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
        assert "Available voices for KOKORO:" in result.output
        assert "AF voices:" in result.output
        assert "af_bella" in result.output
        assert "af_alloy" in result.output
        assert "EN voices:" in result.output
        assert "en_voice" in result.output
        assert "FR voices:" in result.output
        assert "fr_voice" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_voices_api_error(
            self, mock_get_tts_config, mock_terminal_size, runner
        ):
        """Test --list-voices flag when API fails."""
        mock_config = Mock()
        mock_config.get_available_voices.side_effect = Exception("API Error")
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(start.main, ["--list-voices"])

        assert result.exit_code == 0
        assert "Available voices for KOKORO:" in result.output
        assert "Error fetching voices from kokoro" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_voices_no_voices_found(
            self, mock_get_tts_config, mock_terminal_size, runner
        ):
        """Test --list-voices flag when no voices are found."""
        mock_config = Mock()
        mock_config.get_available_voices.return_value = []
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(start.main, ["--list-voices"])

        assert result.exit_code == 0
        assert "Available voices for KOKORO:" in result.output
        assert "No voices found for kokoro." in result.output

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
            max_samples=5,
            output_dir=None,
            llm_model=None,
            llm_base_url="http://localhost:11434",
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
        with patch("audify.start.get_tts_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get_available_voices.return_value = ["af_bella"]
            mock_get_config.return_value = mock_config

            result = runner.invoke(start.main, ["-lv"])

            assert result.exit_code == 0
            assert "Available voices for KOKORO:" in result.output

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


class TestListTTSProviders:
    """Tests for --list-tts-providers flag."""

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner instance."""
        return CliRunner()

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_tts_providers_all_available(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-tts-providers shows all providers with status."""
        mock_config = Mock()
        mock_config.is_available.return_value = True
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(start.main, ["--list-tts-providers"])

        assert result.exit_code == 0
        assert "Available TTS Providers" in result.output
        assert "Kokoro (Local)" in result.output
        assert "OpenAI TTS" in result.output
        assert "AWS Polly" in result.output
        assert "Google Cloud TTS" in result.output
        assert "Available" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_tts_providers_not_configured(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-tts-providers shows 'Not configured' status."""
        mock_config = Mock()
        mock_config.is_available.return_value = False
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(start.main, ["--list-tts-providers"])

        assert result.exit_code == 0
        assert "Not configured" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_tts_providers_error(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-tts-providers handles provider errors gracefully."""
        mock_get_tts_config.side_effect = Exception("Provider error")

        result = runner.invoke(start.main, ["--list-tts-providers"])

        assert result.exit_code == 0
        assert "Not available" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_tts_providers_short_flag(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test -ltp short flag for --list-tts-providers."""
        mock_config = Mock()
        mock_config.is_available.return_value = True
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(start.main, ["-ltp"])

        assert result.exit_code == 0
        assert "Available TTS Providers" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_tts_providers_shows_config_info(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-tts-providers shows configuration info."""
        mock_config = Mock()
        mock_config.is_available.return_value = True
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(start.main, ["--list-tts-providers"])

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

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_voices_openai_provider(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-voices with OpenAI provider shows flat list."""
        mock_config = Mock()
        mock_config.get_available_voices.return_value = [
            "alloy", "echo", "fable", "nova"
        ]
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(
            start.main, ["--list-voices", "--tts-provider", "openai"]
        )

        assert result.exit_code == 0
        assert "Available voices for OPENAI:" in result.output
        assert "Voices for openai:" in result.output
        assert "alloy" in result.output
        assert "echo" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_voices_aws_provider(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-voices with AWS provider shows flat list."""
        mock_config = Mock()
        mock_config.get_available_voices.return_value = ["Joanna", "Matthew", "Ivy"]
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(
            start.main, ["--list-voices", "--tts-provider", "aws"]
        )

        assert result.exit_code == 0
        assert "Available voices for AWS:" in result.output
        assert "Voices for aws:" in result.output
        assert "Joanna" in result.output

    @patch("os.get_terminal_size", return_value=(80, 24))
    @patch("audify.start.get_tts_config")
    def test_list_voices_google_provider(
        self, mock_get_tts_config, mock_terminal_size, runner
    ):
        """Test --list-voices with Google provider shows flat list."""
        mock_config = Mock()
        mock_config.get_available_voices.return_value = [
            "en-US-Neural2-A", "en-US-Neural2-B"
        ]
        mock_get_tts_config.return_value = mock_config

        result = runner.invoke(
            start.main, ["--list-voices", "--tts-provider", "google"]
        )

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
