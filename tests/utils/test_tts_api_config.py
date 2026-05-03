"""Tests for TTS API configuration classes."""

from unittest.mock import Mock, patch

import pytest

from audify.utils.api_config import (
    AWSTTSConfig,
    GoogleTTSConfig,
    KokoroTTSConfig,
    OpenAITTSConfig,
    get_tts_config,
)


class TestKokoroTTSConfig:
    """Test cases for KokoroTTSConfig class."""

    @patch("audify.utils.api_config.requests.get")
    def test_get_available_voices_cached(self, mock_get):
        """Test get_available_voices uses cache on subsequent calls."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"voices": ["voice1"]}
        mock_get.return_value = mock_response

        config = KokoroTTSConfig()
        voices1 = config.get_available_voices()
        voices2 = config.get_available_voices()

        assert voices1 == voices2
        assert mock_get.call_count == 1  # Only called once due to caching

    @patch("audify.utils.api_config.requests.get")
    def test_get_available_voices_error(self, mock_get):
        """Test get_available_voices returns empty list on error."""
        import requests

        mock_get.side_effect = requests.RequestException("Error")

        config = KokoroTTSConfig()
        voices = config.get_available_voices()

        assert voices == []

class TestOpenAITTSConfig:
    """Test cases for OpenAITTSConfig class."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        config = OpenAITTSConfig()
        assert config.provider_name == "openai"
        assert config.language == "en"
        assert config.timeout == 60
        assert config.base_url == "https://api.openai.com/v1/audio/speech"

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_success(self, mock_post, tmp_path):
        """Test synthesize creates audio file on success."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake audio content"
        mock_post.return_value = mock_response

        config = OpenAITTSConfig(api_key="test-key", voice="nova")
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.read_bytes() == b"fake audio content"

        # Verify request was made with correct headers
        call_kwargs = mock_post.call_args.kwargs
        assert "Bearer test-key" in call_kwargs["headers"]["Authorization"]

    def test_synthesize_no_api_key(self, tmp_path):
        """Test synthesize returns False when no API key."""
        config = OpenAITTSConfig(api_key="")
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is False

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_api_error(self, mock_post, tmp_path):
        """Test synthesize returns False on API error."""
        import requests

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "401 Unauthorized", response=mock_response
        )
        mock_post.return_value = mock_response

        config = OpenAITTSConfig(api_key="bad-key")
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is False

class TestAWSTTSConfig:
    """Test cases for AWSTTSConfig class."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        config = AWSTTSConfig()
        assert config.provider_name == "aws"
        assert config.language == "en"
        assert config.timeout == 60


    def test_is_available_without_credentials(self):
        """Test is_available returns False without credentials."""
        config = AWSTTSConfig(access_key_id="", secret_access_key="")
        assert config.is_available() is False

    @patch("audify.utils.api_config.boto3.client")
    def test_is_available_with_credentials(self, mock_boto_client):
        """Test is_available returns True with valid credentials."""
        config = AWSTTSConfig(access_key_id="test-key", secret_access_key="test-secret")
        assert config.is_available() is True

    @patch("audify.utils.api_config.boto3.client")
    def test_is_available_client_error(self, mock_boto_client):
        """Test is_available returns False on client error."""
        mock_boto_client.side_effect = Exception("Client error")

        config = AWSTTSConfig(access_key_id="test-key", secret_access_key="test-secret")
        assert config.is_available() is False

    @patch("audify.utils.api_config.boto3.client")
    def test_get_available_voices_success(self, mock_boto_client):
        """Test get_available_voices returns voices from Polly."""
        mock_client = Mock()
        mock_client.describe_voices.return_value = {
            "Voices": [{"Id": "Joanna"}, {"Id": "Matthew"}, {"Id": "Ivy"}]
        }
        mock_boto_client.return_value = mock_client

        config = AWSTTSConfig(access_key_id="test-key", secret_access_key="test-secret")
        voices = config.get_available_voices()

        assert voices == ["Joanna", "Matthew", "Ivy"]

    @patch("audify.utils.api_config.boto3.client")
    def test_synthesize_success(self, mock_boto_client, tmp_path):
        """Test synthesize creates WAV file on success."""
        mock_audio_stream = Mock()
        mock_audio_stream.read.return_value = b"\x00" * 1000  # Fake PCM data

        mock_client = Mock()
        mock_client.synthesize_speech.return_value = {"AudioStream": mock_audio_stream}
        mock_boto_client.return_value = mock_client

        config = AWSTTSConfig(access_key_id="test-key", secret_access_key="test-secret")
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is True
        assert output_path.exists()

    @patch("audify.utils.api_config.boto3.client")
    def test_synthesize_no_audio_stream(self, mock_boto_client, tmp_path):
        """Test synthesize returns False when no audio stream in response."""
        mock_client = Mock()
        mock_client.synthesize_speech.return_value = {}
        mock_boto_client.return_value = mock_client

        config = AWSTTSConfig(access_key_id="test-key", secret_access_key="test-secret")
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is False

    @patch("audify.utils.api_config.boto3.client")
    def test_synthesize_rejects_oversized_text(self, mock_boto_client, tmp_path):
        """Test synthesize returns False for text exceeding byte limit."""
        config = AWSTTSConfig(access_key_id="test-key", secret_access_key="test-secret")
        output_path = tmp_path / "output.wav"

        long_text = "a" * 5000  # 5000 bytes, exceeds 3000 limit
        result = config.synthesize(long_text, output_path)

        assert result is False
        # Verify the API was never called
        mock_boto_client.return_value.synthesize_speech.assert_not_called()

class TestGoogleTTSConfig:
    """Test cases for GoogleTTSConfig class."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        config = GoogleTTSConfig()
        assert config.provider_name == "google"
        assert config.language == "en"
        assert config.timeout == 60
        assert (config.credentials_path is None) or isinstance(
            config.credentials_path, str
        )

    def test_get_language_code_prefers_voice_locale(self):
        """Voice locale should override requested language when explicitly set."""
        config = GoogleTTSConfig(language="es", voice="en-US-Chirp-HD-F")
        assert config._get_language_code() == "en-US"


    @patch("audify.utils.api_config.GOOGLE_TTS_DEFAULT_VOICE_BY_LANGUAGE", {})
    @patch("audify.utils.api_config.GOOGLE_TTS_VOICE", "en-US-Chirp-HD-F")
    def test_default_voice_fallback_when_dictionary_missing(self):
        """Fallback keeps backward compatibility when mapping has no entry."""
        config = GoogleTTSConfig(language="es")
        assert config.voice.startswith("es-ES-")
        assert config._get_language_code() == "es-ES"


    @patch("audify.utils.api_config.GoogleTTSConfig._get_client")
    def test_is_available_success(self, mock_get_client):
        """Test is_available returns True when client can be created."""
        mock_get_client.return_value = Mock()

        config = GoogleTTSConfig()
        assert config.is_available() is True

    @patch("audify.utils.api_config.GoogleTTSConfig._get_client")
    def test_is_available_error(self, mock_get_client):
        """Test is_available returns False on client creation error."""
        mock_get_client.side_effect = Exception("No credentials")

        config = GoogleTTSConfig()
        assert config.is_available() is False

    @patch("audify.utils.api_config.GoogleTTSConfig._get_client")
    def test_get_available_voices_success(self, mock_get_client):
        """Test get_available_voices returns voices from API."""
        mock_voice1 = Mock()
        mock_voice1.name = "en-US-Neural2-A"
        mock_voice2 = Mock()
        mock_voice2.name = "en-US-Neural2-B"

        mock_response = Mock()
        mock_response.voices = [mock_voice1, mock_voice2]

        mock_client = Mock()
        mock_client.list_voices.return_value = mock_response
        mock_get_client.return_value = mock_client

        config = GoogleTTSConfig()
        voices = config.get_available_voices()

        assert voices == ["en-US-Neural2-A", "en-US-Neural2-B"]

    @patch("audify.utils.api_config.GoogleTTSConfig._get_client")
    def test_synthesize_success(self, mock_get_client, tmp_path):
        """Test synthesize creates audio file on success."""
        mock_response = Mock()
        mock_response.audio_content = b"fake audio content"

        mock_client = Mock()
        mock_client.synthesize_speech.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Mock the texttospeech module
        with patch.dict(
            "sys.modules",
            {"google.cloud": Mock(), "google.cloud.texttospeech": Mock()},
        ):
            config = GoogleTTSConfig()
            output_path = tmp_path / "output.wav"

            result = config.synthesize("Hello world", output_path)

            assert result is True
            assert output_path.exists()
            assert output_path.read_bytes() == b"fake audio content"

    @patch("time.sleep")
    @patch("audify.utils.api_config.GoogleTTSConfig._get_client")
    def test_synthesize_error(self, mock_get_client, mock_sleep, tmp_path):
        """Test synthesize returns False on error."""
        import requests

        mock_client = Mock()
        mock_client.synthesize_speech.side_effect = requests.RequestException(
            "Synthesis error"
        )
        mock_get_client.return_value = mock_client

        with patch.dict(
            "sys.modules",
            {"google.cloud": Mock(), "google.cloud.texttospeech": Mock()},
        ):
            config = GoogleTTSConfig()
            output_path = tmp_path / "output.wav"

            result = config.synthesize("Hello world", output_path)

            assert result is False


class TestGetTTSConfig:
    """Test cases for get_tts_config factory function."""

    def test_get_aws_config(self):
        """Test get_tts_config returns AWSTTSConfig for 'aws'."""
        config = get_tts_config(provider="aws")
        assert isinstance(config, AWSTTSConfig)

    def test_get_google_config(self):
        """Test get_tts_config returns GoogleTTSConfig for 'google'."""
        config = get_tts_config(provider="google")
        assert isinstance(config, GoogleTTSConfig)

    def test_get_config_unsupported_provider(self):
        """Test get_tts_config raises ValueError for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported TTS provider"):
            get_tts_config(provider="unsupported")
