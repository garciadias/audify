"""Tests for TTS API configuration classes."""

from unittest.mock import Mock, patch

import pytest

from audify.utils.api_config import (
    AWSTTSConfig,
    GoogleTTSConfig,
    KokoroTTSConfig,
    OpenAITTSConfig,
    QwenTTSConfig,
    get_tts_config,
)


class TestKokoroTTSConfig:
    """Test cases for KokoroTTSConfig class."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        config = KokoroTTSConfig()
        assert config.provider_name == "kokoro"
        assert config.language == "en"
        assert config.timeout == 60
        assert config.voice is not None  # Uses DEFAULT_SPEAKER

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        config = KokoroTTSConfig(
            voice="test_voice",
            language="es",
            base_url="http://custom:8080/v1/audio",
            timeout=120,
        )
        assert config.voice == "test_voice"
        assert config.language == "es"
        assert config.base_url == "http://custom:8080/v1/audio"
        assert config.timeout == 120

    def test_voices_url(self):
        """Test voices_url property."""
        config = KokoroTTSConfig(base_url="http://test:8080")
        assert config.voices_url == "http://test:8080/voices"

    def test_speech_url(self):
        """Test speech_url property."""
        config = KokoroTTSConfig(base_url="http://test:8080")
        assert config.speech_url == "http://test:8080/speech"

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_true(self, mock_get):
        """Test is_available returns True when API is reachable."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        config = KokoroTTSConfig()
        assert config.is_available() is True

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_false_on_error(self, mock_get):
        """Test is_available returns False on request exception."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection error")

        config = KokoroTTSConfig()
        assert config.is_available() is False

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_false_on_non_200(self, mock_get):
        """Test is_available returns False on non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        config = KokoroTTSConfig()
        assert config.is_available() is False

    @patch("audify.utils.api_config.requests.get")
    def test_get_available_voices_success(self, mock_get):
        """Test get_available_voices returns voices from API."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"voices": ["voice1", "voice2", "voice3"]}
        mock_get.return_value = mock_response

        config = KokoroTTSConfig()
        voices = config.get_available_voices()

        assert voices == ["voice1", "voice2", "voice3"]

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

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_success(self, mock_post, tmp_path):
        """Test synthesize creates audio file on success."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake audio content"
        mock_post.return_value = mock_response

        config = KokoroTTSConfig(voice="test_voice", language="en")
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.read_bytes() == b"fake audio content"

    @patch("time.sleep")
    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_api_error(self, mock_post, mock_sleep, tmp_path):
        """Test synthesize returns False on API error."""
        import requests

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error", response=mock_response
        )
        mock_post.return_value = mock_response

        config = KokoroTTSConfig()
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is False

    @patch("time.sleep")
    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_request_exception(self, mock_post, mock_sleep, tmp_path):
        """Test synthesize returns False on request exception."""
        import requests

        mock_post.side_effect = requests.RequestException("Network error")

        config = KokoroTTSConfig()
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is False

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_uses_language_code(self, mock_post, tmp_path):
        """Test synthesize uses correct language code."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"audio"
        mock_post.return_value = mock_response

        config = KokoroTTSConfig(language="es")
        output_path = tmp_path / "output.wav"
        config.synthesize("Hola mundo", output_path)

        call_json = mock_post.call_args.kwargs["json"]
        # Spanish should map to a specific lang_code
        assert "lang_code" in call_json


class TestOpenAITTSConfig:
    """Test cases for OpenAITTSConfig class."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        config = OpenAITTSConfig()
        assert config.provider_name == "openai"
        assert config.language == "en"
        assert config.timeout == 60
        assert config.base_url == "https://api.openai.com/v1/audio/speech"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        config = OpenAITTSConfig(
            voice="nova",
            language="es",
            api_key="test-key",
            model="tts-1-hd",
            timeout=120,
        )
        assert config.voice == "nova"
        assert config.language == "es"
        assert config.api_key == "test-key"
        assert config.model == "tts-1-hd"
        assert config.timeout == 120

    def test_is_available_with_key(self):
        """Test is_available returns True when API key is set."""
        config = OpenAITTSConfig(api_key="test-key")
        assert config.is_available() is True

    def test_is_available_without_key(self):
        """Test is_available returns False when API key is empty."""
        config = OpenAITTSConfig(api_key="")
        assert config.is_available() is False

    def test_get_available_voices(self):
        """Test get_available_voices returns static voice list."""
        config = OpenAITTSConfig()
        voices = config.get_available_voices()

        assert "alloy" in voices
        assert "echo" in voices
        assert "fable" in voices
        assert "onyx" in voices
        assert "nova" in voices
        assert "shimmer" in voices
        assert len(voices) == 6

    def test_get_available_voices_returns_copy(self):
        """Test get_available_voices returns a copy, not the original."""
        config = OpenAITTSConfig()
        voices1 = config.get_available_voices()
        voices2 = config.get_available_voices()

        voices1.append("modified")
        assert "modified" not in voices2

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

    @patch("time.sleep")
    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_request_exception(self, mock_post, mock_sleep, tmp_path):
        """Test synthesize returns False on request exception."""
        import requests

        mock_post.side_effect = requests.RequestException("Network error")

        config = OpenAITTSConfig(api_key="test-key")
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

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        config = AWSTTSConfig(
            voice="Matthew",
            language="es",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region="eu-west-1",
            engine="standard",
            timeout=120,
        )
        assert config.voice == "Matthew"
        assert config.language == "es"
        assert config.access_key_id == "test-key"
        assert config.secret_access_key == "test-secret"
        assert config.region == "eu-west-1"
        assert config.engine == "standard"

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
    def test_get_available_voices_error_fallback(self, mock_boto_client):
        """Test get_available_voices falls back to default voices on error."""
        mock_client = Mock()
        mock_client.describe_voices.side_effect = Exception("API error")
        mock_boto_client.return_value = mock_client

        config = AWSTTSConfig(
            access_key_id="test-key", secret_access_key="test-secret", language="en"
        )
        voices = config.get_available_voices()

        # Should return default English neural voices
        assert "Joanna" in voices
        assert "Matthew" in voices

    @patch("audify.utils.api_config.boto3.client")
    def test_get_available_voices_unknown_language_fallback(self, mock_boto_client):
        """Test get_available_voices falls back to English for unknown language."""
        mock_client = Mock()
        mock_client.describe_voices.side_effect = Exception("API error")
        mock_boto_client.return_value = mock_client

        config = AWSTTSConfig(
            access_key_id="test-key",
            secret_access_key="test-secret",
            language="unknown",
        )
        voices = config.get_available_voices()

        # Should return default English neural voices
        assert "Joanna" in voices

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
    def test_synthesize_truncates_long_text(self, mock_boto_client, tmp_path):
        """Test synthesize truncates text exceeding 3000 chars."""
        mock_audio_stream = Mock()
        mock_audio_stream.read.return_value = b"\x00" * 100

        mock_client = Mock()
        mock_client.synthesize_speech.return_value = {"AudioStream": mock_audio_stream}
        mock_boto_client.return_value = mock_client

        config = AWSTTSConfig(access_key_id="test-key", secret_access_key="test-secret")
        output_path = tmp_path / "output.wav"

        long_text = "a" * 5000
        result = config.synthesize(long_text, output_path)

        assert result is True
        # Verify text was truncated in the call
        call_kwargs = mock_client.synthesize_speech.call_args.kwargs
        assert len(call_kwargs["Text"]) == 3000

    @patch("time.sleep")
    @patch("audify.utils.api_config.boto3.client")
    def test_synthesize_error(self, mock_boto_client, mock_sleep, tmp_path):
        """Test synthesize returns False on error."""
        from botocore.exceptions import BotoCoreError

        mock_client = Mock()
        mock_client.synthesize_speech.side_effect = BotoCoreError()
        mock_boto_client.return_value = mock_client

        config = AWSTTSConfig(access_key_id="test-key", secret_access_key="test-secret")
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is False


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

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        config = GoogleTTSConfig(
            voice="es-ES-Neural2-A",
            language="es",
            credentials_path="/path/to/creds.json",
            timeout=120,
        )
        assert config.voice == "es-ES-Neural2-A"
        assert config.language == "es"
        assert config.credentials_path == "/path/to/creds.json"
        assert config.timeout == 120

    def test_get_language_code_known(self):
        """Test _get_language_code returns correct code for known languages."""
        config = GoogleTTSConfig(language="es")
        assert config._get_language_code() == "es-ES"

        config = GoogleTTSConfig(language="fr")
        assert config._get_language_code() == "fr-FR"

        config = GoogleTTSConfig(language="zh")
        assert config._get_language_code() == "cmn-CN"

    def test_get_language_code_unknown(self):
        """Test _get_language_code returns default for unknown languages."""
        config = GoogleTTSConfig(language="unknown")
        # Should return default from GOOGLE_TTS_LANGUAGE_CODE
        assert config._get_language_code() is not None

    def test_get_language_code_prefers_voice_locale(self):
        """Voice locale should override requested language when explicitly set."""
        config = GoogleTTSConfig(language="es", voice="en-US-Chirp-HD-F")
        assert config._get_language_code() == "en-US"

    def test_default_voice_uses_language_dictionary(self):
        """Implicit Google voice should come from language voice mapping."""
        config = GoogleTTSConfig(language="es")
        assert config.voice == "es-ES-Neural2-F"
        assert config._get_language_code() == "es-ES"

    @patch("audify.utils.api_config.GOOGLE_TTS_DEFAULT_VOICE_BY_LANGUAGE", {})
    @patch("audify.utils.api_config.GOOGLE_TTS_VOICE", "en-US-Chirp-HD-F")
    def test_default_voice_fallback_when_dictionary_missing(self):
        """Fallback keeps backward compatibility when mapping has no entry."""
        config = GoogleTTSConfig(language="es")
        assert config.voice.startswith("es-ES-")
        assert config._get_language_code() == "es-ES"

    def test_extract_language_code_from_voice_handles_cmn_cn(self):
        """Chinese voices use a three-part locale prefix (cmn-CN-...)."""
        assert (
            GoogleTTSConfig._extract_language_code_from_voice("cmn-CN-Neural2-A")
            == "cmn-CN"
        )

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
    def test_get_available_voices_error_fallback(self, mock_get_client):
        """Test get_available_voices falls back to defaults on error."""
        mock_client = Mock()
        mock_client.list_voices.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        config = GoogleTTSConfig(language="en")
        voices = config.get_available_voices()

        # Should return default English voices
        assert "en-US-Neural2-A" in voices

    @patch("audify.utils.api_config.GoogleTTSConfig._get_client")
    def test_get_available_voices_unknown_language_fallback(self, mock_get_client):
        """Test get_available_voices falls back to English for unknown language."""
        mock_client = Mock()
        mock_client.list_voices.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        config = GoogleTTSConfig(language="unknown")
        voices = config.get_available_voices()

        # Should return default English voices
        assert "en-US-Neural2-A" in voices

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

    def test_get_kokoro_config(self):
        """Test get_tts_config returns KokoroTTSConfig for 'kokoro'."""
        config = get_tts_config(provider="kokoro")
        assert isinstance(config, KokoroTTSConfig)

    def test_get_openai_config(self):
        """Test get_tts_config returns OpenAITTSConfig for 'openai'."""
        config = get_tts_config(provider="openai")
        assert isinstance(config, OpenAITTSConfig)

    def test_get_aws_config(self):
        """Test get_tts_config returns AWSTTSConfig for 'aws'."""
        config = get_tts_config(provider="aws")
        assert isinstance(config, AWSTTSConfig)

    def test_get_google_config(self):
        """Test get_tts_config returns GoogleTTSConfig for 'google'."""
        config = get_tts_config(provider="google")
        assert isinstance(config, GoogleTTSConfig)

    def test_get_qwen_config(self):
        """Test get_tts_config returns QwenTTSConfig for 'qwen'."""
        config = get_tts_config(provider="qwen")
        assert isinstance(config, QwenTTSConfig)

    def test_get_config_with_voice(self):
        """Test get_tts_config passes voice parameter."""
        config = get_tts_config(provider="openai", voice="nova")
        assert config.voice == "nova"

    def test_get_config_with_language(self):
        """Test get_tts_config passes language parameter."""
        config = get_tts_config(provider="kokoro", language="es")
        assert config.language == "es"

    def test_get_config_with_kwargs(self):
        """Test get_tts_config passes additional kwargs."""
        config = get_tts_config(provider="openai", api_key="test-key")
        assert config.api_key == "test-key"

    def test_get_config_unsupported_provider(self):
        """Test get_tts_config raises ValueError for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported TTS provider"):
            get_tts_config(provider="unsupported")

    def test_get_config_default_provider(self):
        """Test get_tts_config uses default provider when none specified."""
        config = get_tts_config()
        # Should use DEFAULT_TTS_PROVIDER which is "kokoro"
        assert config.provider_name in ["kokoro", "openai", "aws", "google", "qwen"]
