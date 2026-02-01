"""
Shared API configuration utilities for various external services.

This module consolidates API configuration classes to reduce code duplication
across different modules that interact with external APIs.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import boto3
import requests
from litellm import completion

from audify.utils.constants import (
    AWS_ACCESS_KEY_ID,
    AWS_POLLY_ENGINE,
    AWS_POLLY_VOICE,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    DEFAULT_SPEAKER,
    DEFAULT_TTS_PROVIDER,
    GOOGLE_TTS_LANGUAGE_CODE,
    GOOGLE_TTS_VOICE,
    KOKORO_API_BASE_URL,
    LANG_CODES,
    OLLAMA_API_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_DEFAULT_TRANSLATION_MODEL,
    OPENAI_API_KEY,
    OPENAI_TTS_MODEL,
    OPENAI_TTS_VOICE,
    QWEN_API_BASE_URL,
    QWEN_TTS_VOICE,
    _keys_config,
)

logger = logging.getLogger(__name__)


class APIConfig:
    """Base class for API configurations."""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base_url='{self.base_url}', "
            f"timeout={self.timeout})"
        )


class KokoroAPIConfig(APIConfig):
    """Configuration for Kokoro TTS API."""

    def __init__(self, base_url: Optional[str] = None, voice: Optional[str] = None):
        base_url = base_url or f"{KOKORO_API_BASE_URL}/audio"
        super().__init__(base_url)
        self.default_voice = voice or DEFAULT_SPEAKER

    @property
    def voices_url(self) -> str:
        """URL for fetching available voices."""
        return f"{self.base_url}/voices"

    @property
    def speech_url(self) -> str:
        """URL for text-to-speech synthesis."""
        return f"{self.base_url}/speech"


# =============================================================================
# TTS API Abstraction Layer
# =============================================================================


class TTSAPIConfig(ABC):
    """Abstract base class for TTS API configurations.

    This provides a common interface for different TTS providers
    (Kokoro, OpenAI, AWS Polly, Google Cloud TTS).
    """

    def __init__(
        self,
        voice: Optional[str] = None,
        language: str = "en",
        timeout: int = 60,
    ):
        self.voice = voice
        self.language = language
        self.timeout = timeout

    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> bool:
        """Synthesize text to audio and save to output_path.

        Args:
            text: The text to synthesize.
            output_path: Path where the audio file will be saved (WAV format).

        Returns:
            True if synthesis was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Get list of available voices for this provider.

        Returns:
            List of voice identifiers.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS service is available and properly configured.

        Returns:
            True if service is available, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the TTS provider."""
        pass


class KokoroTTSConfig(TTSAPIConfig):
    """TTS configuration for local Kokoro API."""

    def __init__(
        self,
        voice: Optional[str] = None,
        language: str = "en",
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(
            voice=voice or DEFAULT_SPEAKER, language=language, timeout=timeout
        )
        self.base_url = base_url or f"{KOKORO_API_BASE_URL}/audio"
        self._available_voices: Optional[List[str]] = None

    @property
    def provider_name(self) -> str:
        return "kokoro"

    @property
    def voices_url(self) -> str:
        """URL for fetching available voices."""
        return f"{self.base_url}/voices"

    @property
    def speech_url(self) -> str:
        """URL for text-to-speech synthesis."""
        return f"{self.base_url}/speech"

    def is_available(self) -> bool:
        """Check if Kokoro API is reachable."""
        try:
            response = requests.get(self.voices_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_available_voices(self) -> List[str]:
        """Get available voices from Kokoro API."""
        if self._available_voices is not None:
            return self._available_voices
        try:
            response = requests.get(self.voices_url, timeout=self.timeout)
            response.raise_for_status()
            self._available_voices = response.json().get("voices", [])
            return self._available_voices
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Kokoro voices: {e}")
            return []

    def synthesize(self, text: str, output_path: Path) -> bool:
        """Synthesize text using Kokoro API."""
        try:
            lang_code = LANG_CODES.get(self.language, "a")
            response = requests.post(
                self.speech_url,
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": self.voice,
                    "response_format": "wav",
                    "lang_code": lang_code,
                    "speed": 1.0,
                },
                timeout=self.timeout,
            )
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                logger.error(f"Kokoro API error: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Kokoro synthesis failed: {e}")
            return False


class OpenAITTSConfig(TTSAPIConfig):
    """TTS configuration for OpenAI TTS API."""

    # OpenAI TTS available voices
    AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(
        self,
        voice: Optional[str] = None,
        language: str = "en",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(
            voice=voice or OPENAI_TTS_VOICE,
            language=language,
            timeout=timeout,
        )
        self.api_key = api_key if api_key is not None else OPENAI_API_KEY
        self.model = model if model is not None else OPENAI_TTS_MODEL
        self.base_url = "https://api.openai.com/v1/audio/speech"

    @property
    def provider_name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.api_key)

    def get_available_voices(self) -> List[str]:
        """Return available OpenAI TTS voices."""
        return self.AVAILABLE_VOICES.copy()

    def synthesize(self, text: str, output_path: Path) -> bool:
        """Synthesize text using OpenAI TTS API."""
        if not self.api_key:
            logger.error("OpenAI API key not configured")
            return False

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": "wav",
            }
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                logger.error(
                    f"OpenAI TTS API error: {response.status_code} - {response.text}"
                )
                return False
        except requests.RequestException as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            return False


class AWSTTSConfig(TTSAPIConfig):
    """TTS configuration for AWS Polly."""

    # Common neural voices by language
    NEURAL_VOICES = {
        "en": [
            "Joanna",
            "Matthew",
            "Ivy",
            "Kendra",
            "Kimberly",
            "Salli",
            "Joey",
            "Justin",
            "Kevin",
            "Ruth",
            "Stephen",
        ],
        "es": ["Lupe", "Pedro", "Mia"],
        "fr": ["Lea", "Remi"],
        "de": ["Vicki", "Daniel"],
        "it": ["Bianca", "Adriano"],
        "pt": ["Camila", "Thiago", "Vitoria", "Ricardo"],
        "ja": ["Takumi", "Kazuha", "Tomoko"],
        "zh": ["Zhiyu"],
        "hi": ["Kajal"],
    }

    def __init__(
        self,
        voice: Optional[str] = None,
        language: str = "en",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region: Optional[str] = None,
        engine: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(
            voice=voice or AWS_POLLY_VOICE,
            language=language,
            timeout=timeout,
        )
        self.access_key_id = access_key_id or AWS_ACCESS_KEY_ID
        self.secret_access_key = secret_access_key or AWS_SECRET_ACCESS_KEY
        self.region = region or AWS_REGION
        self.engine = engine or AWS_POLLY_ENGINE
        self._polly_client = None

    @property
    def provider_name(self) -> str:
        return "aws"

    def _get_polly_client(self):
        """Get or create AWS Polly client."""
        if self._polly_client is None:
            try:
                self._polly_client = boto3.client(
                    "polly",
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name=self.region,
                )
            except ImportError:
                logger.error(
                    "boto3 is required for AWS Polly. Install with: pip install boto3"
                )
                raise
        return self._polly_client

    def is_available(self) -> bool:
        """Check if AWS credentials are configured."""
        if not self.access_key_id or not self.secret_access_key:
            return False
        try:
            self._get_polly_client()
            return True
        except Exception:
            return False

    def get_available_voices(self) -> List[str]:
        """Get available voices from AWS Polly."""
        try:
            client = self._get_polly_client()
            response = client.describe_voices(Engine=self.engine)
            return [voice["Id"] for voice in response.get("Voices", [])]
        except Exception as e:
            logger.error(f"Failed to fetch AWS Polly voices: {e}")
            # Return default voices for the language
            return self.NEURAL_VOICES.get(self.language, self.NEURAL_VOICES["en"])

    def synthesize(self, text: str, output_path: Path) -> bool:
        """Synthesize text using AWS Polly."""
        try:
            client = self._get_polly_client()

            # Polly has a 3000 character limit for standard synthesis
            # Split text if needed
            max_chars = 3000
            if len(text) > max_chars:
                logger.warning(f"Text exceeds {max_chars} chars, truncating for Polly")
                text = text[:max_chars]

            response = client.synthesize_speech(
                Text=text,
                OutputFormat="pcm",
                VoiceId=self.voice,
                Engine=self.engine,
                SampleRate="24000",
            )

            # Convert PCM to WAV
            if "AudioStream" in response:
                import wave

                pcm_data = response["AudioStream"].read()

                with wave.open(str(output_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(24000)
                    wav_file.writeframes(pcm_data)
                return True
            else:
                logger.error("No audio stream in Polly response")
                return False

        except Exception as e:
            logger.error(f"AWS Polly synthesis failed: {e}")
            return False


class GoogleTTSConfig(TTSAPIConfig):
    """TTS configuration for Google Cloud Text-to-Speech."""

    # Common voices by language
    NEURAL_VOICES = {
        "en": [
            "en-US-Neural2-A",
            "en-US-Neural2-C",
            "en-US-Neural2-D",
            "en-US-Neural2-E",
            "en-US-Neural2-F",
            "en-US-Neural2-G",
            "en-US-Neural2-H",
            "en-US-Neural2-I",
            "en-US-Neural2-J",
        ],
        "es": [
            "es-ES-Neural2-A",
            "es-ES-Neural2-B",
            "es-ES-Neural2-C",
            "es-ES-Neural2-D",
            "es-ES-Neural2-E",
            "es-ES-Neural2-F",
        ],
        "fr": [
            "fr-FR-Neural2-A",
            "fr-FR-Neural2-B",
            "fr-FR-Neural2-C",
            "fr-FR-Neural2-D",
            "fr-FR-Neural2-E",
        ],
        "de": [
            "de-DE-Neural2-A",
            "de-DE-Neural2-B",
            "de-DE-Neural2-C",
            "de-DE-Neural2-D",
            "de-DE-Neural2-F",
        ],
        "it": ["it-IT-Neural2-A", "it-IT-Neural2-B", "it-IT-Neural2-C"],
        "pt": ["pt-BR-Neural2-A", "pt-BR-Neural2-B", "pt-BR-Neural2-C"],
        "ja": ["ja-JP-Neural2-B", "ja-JP-Neural2-C", "ja-JP-Neural2-D"],
        "zh": [
            "cmn-CN-Neural2-A",
            "cmn-CN-Neural2-B",
            "cmn-CN-Neural2-C",
            "cmn-CN-Neural2-D",
        ],
        "hi": [
            "hi-IN-Neural2-A",
            "hi-IN-Neural2-B",
            "hi-IN-Neural2-C",
            "hi-IN-Neural2-D",
        ],
    }

    # Language code mapping for Google TTS
    LANGUAGE_CODES = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-BR",
        "ja": "ja-JP",
        "zh": "cmn-CN",
        "hi": "hi-IN",
        "ko": "ko-KR",
        "ru": "ru-RU",
        "ar": "ar-XA",
        "nl": "nl-NL",
        "pl": "pl-PL",
        "tr": "tr-TR",
    }

    def __init__(
        self,
        voice: Optional[str] = None,
        language: str = "en",
        credentials_path: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(
            voice=voice or GOOGLE_TTS_VOICE,
            language=language,
            timeout=timeout,
        )
        self.credentials_path = credentials_path
        self._client = None

    @property
    def provider_name(self) -> str:
        return "google"

    def _get_language_code(self) -> str:
        """Get Google TTS language code from short language code."""
        return self.LANGUAGE_CODES.get(self.language, GOOGLE_TTS_LANGUAGE_CODE)

    def _get_client(self):
        """Get or create Google TTS client."""
        if self._client is None:
            try:
                from google.cloud import texttospeech

                self._client = texttospeech.TextToSpeechClient()
            except ImportError:
                logger.error(
                    "google-cloud-texttospeech is required. "
                    "Install with: pip install google-cloud-texttospeech"
                )
                raise
        return self._client

    def is_available(self) -> bool:
        """Check if Google Cloud TTS is properly configured."""
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def get_available_voices(self) -> List[str]:
        """Get available voices from Google Cloud TTS."""
        try:
            client = self._get_client()
            response = client.list_voices(language_code=self._get_language_code())
            return [voice.name for voice in response.voices]
        except Exception as e:
            logger.error(f"Failed to fetch Google TTS voices: {e}")
            # Return default voices for the language
            return self.NEURAL_VOICES.get(self.language, self.NEURAL_VOICES["en"])

    def synthesize(self, text: str, output_path: Path) -> bool:
        """Synthesize text using Google Cloud TTS."""
        try:
            from google.cloud import texttospeech

            client = self._get_client()

            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=self._get_language_code(),
                name=self.voice,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
            )

            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )

            # Write WAV file
            with open(output_path, "wb") as f:
                f.write(response.audio_content)
            return True

        except Exception as e:
            logger.error(f"Google TTS synthesis failed: {e}")
            return False


class QwenTTSConfig(TTSAPIConfig):
    """TTS configuration for Qwen3-TTS API."""

    def __init__(
        self,
        voice: Optional[str] = None,
        language: str = "en",
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(
            voice=voice or QWEN_TTS_VOICE,
            language=language,
            timeout=timeout,
        )
        self.base_url = base_url or QWEN_API_BASE_URL
        self._available_voices: Optional[List[str]] = None

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def health_url(self) -> str:
        """URL for health check."""
        return f"{self.base_url}/health"

    @property
    def tts_url(self) -> str:
        """URL for text-to-speech synthesis."""
        return f"{self.base_url}/tts"

    def is_available(self) -> bool:
        """Check if Qwen-TTS API is reachable."""
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return (
                    data.get("status") == "healthy"
                    and data.get("model_loaded", False)
                )
            return False
        except requests.RequestException:
            return False

    def get_available_voices(self) -> List[str]:
        """Get available voices for Qwen-TTS.

        Note: Qwen3-TTS supports custom voice cloning, but has default voices.
        This returns a basic list of common voices.
        """
        # Qwen-TTS default voices based on the documentation
        return ["Vivian"]  # Add more voices as they become available

    def synthesize(self, text: str, output_path: Path) -> bool:
        """Synthesize text using Qwen-TTS API."""
        try:
            # Map language codes to Qwen-TTS language format
            # Qwen-TTS uses "Auto" by default for automatic language detection
            language_map = {
                "en": "Auto",
                "es": "Auto",
                "fr": "Auto",
                "de": "Auto",
                "it": "Auto",
                "pt": "Auto",
                "zh": "Auto",
                "ja": "Auto",
                "hi": "Auto",
            }
            qwen_language = language_map.get(self.language, "Auto")

            response = requests.post(
                self.tts_url,
                json={
                    "text": text,
                    "language": qwen_language,
                    "speaker": self.voice,
                    "instruct": None,  # Optional instruction for voice style
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                logger.error(f"Qwen-TTS API error: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Qwen-TTS synthesis failed: {e}")
            return False


def get_tts_config(
    provider: Optional[str] = None,
    voice: Optional[str] = None,
    language: str = "en",
    **kwargs,
) -> TTSAPIConfig:
    """Factory function to get the appropriate TTS configuration.

    Args:
        provider: TTS provider name ("kokoro", "openai", "aws", "google", "qwen").
                  Defaults to DEFAULT_TTS_PROVIDER from environment.
        voice: Voice identifier for the provider.
        language: Language code (e.g., "en", "es", "fr").
        **kwargs: Additional provider-specific arguments.

    Returns:
        TTSAPIConfig instance for the specified provider.

    Raises:
        ValueError: If provider is not supported.
    """
    provider = provider or DEFAULT_TTS_PROVIDER

    # Map Kokoro voices to other providers' voices when needed
    kokoro_to_openai = {
        "af_bella": "nova",  # Female voice
        "af_nicole": "shimmer",  # Female voice
        "af_sarah": "alloy",  # Female voice
        "am_adam": "onyx",  # Male voice
        "am_michael": "echo",  # Male voice
        "bf_emma": "fable",  # British female
        "bf_isabella": "nova",  # British female
        "bm_george": "echo",  # British male
        "bm_lewis": "onyx",  # British male
    }

    if provider == "kokoro":
        return KokoroTTSConfig(voice=voice, language=language, **kwargs)
    elif provider == "openai":
        # Map Kokoro voice to OpenAI voice if needed
        if voice and voice in kokoro_to_openai:
            voice = kokoro_to_openai[voice]
        return OpenAITTSConfig(voice=voice, language=language, **kwargs)
    elif provider == "aws":
        # For AWS, use default voice if Kokoro voice provided
        if voice and voice.startswith(("af_", "am_", "bf_", "bm_")):
            voice = None  # Use AWS default
        return AWSTTSConfig(voice=voice, language=language, **kwargs)
    elif provider == "google":
        # For Google, use default voice if Kokoro voice provided
        if voice and voice.startswith(("af_", "am_", "bf_", "bm_")):
            voice = None  # Use Google default
        return GoogleTTSConfig(voice=voice, language=language, **kwargs)
    elif provider == "qwen":
        # For Qwen, use default voice if Kokoro voice provided
        if voice and voice.startswith(("af_", "am_", "bf_", "bm_")):
            voice = None  # Use Qwen default
        return QwenTTSConfig(voice=voice, language=language, **kwargs)
    else:
        raise ValueError(
            f"Unsupported TTS provider: {provider}. "
            f"Available providers: kokoro, openai, aws, google, qwen"
        )


class CommercialAPIConfig(APIConfig):
    """Configuration for commercial LLM APIs (DeepSeek, Claude, GPT-4, Gemini).

    Uses LiteLLM to provide unified interface to commercial APIs.
    Model format: 'api:provider/model' (e.g., 'api:deepseek-chat')
    API keys are loaded from .keys file or environment variables.
    """

    # Map of shortened names to full LiteLLM provider/model formats
    MODEL_MAPPINGS = {
        "deepseek-chat": "deepseek/deepseek-chat",
        "deepseek-reasoner": "deepseek/deepseek-reasoner",
        "deepseekr1": "deepseek/deepseek-reasoner",
        "claude": "anthropic/claude-3-5-sonnet-20241022",
        "claude-sonnet": "anthropic/claude-3-5-sonnet-20241022",
        "claude-opus": "anthropic/claude-3-opus-20240229",
        "gpt-4": "openai/gpt-4-turbo-preview",
        "gpt-4-turbo": "openai/gpt-4-turbo-preview",
        "gpt-4o": "openai/gpt-4o",
        "gemini": "gemini/gemini-2.0-flash-exp",
        "gemini-pro": "gemini/gemini-pro",
        "gemini-flash": "gemini/gemini-2.0-flash-exp",
    }

    def __init__(
        self,
        model: str,
        timeout: int = 600,
    ):
        """Initialize commercial API config.

        Args:
            model: Model identifier with 'api:' prefix (e.g., 'api:deepseek-chat')
            timeout: Request timeout in seconds
        """
        super().__init__(base_url="", timeout=timeout)

        # Strip 'api:' prefix if present
        if model.startswith("api:"):
            model = model[4:]

        # Map shortened names to full provider/model format
        self.model = self.MODEL_MAPPINGS.get(model, model)

        # Load API keys from .keys file into environment variables
        # LiteLLM expects API keys in environment variables
        self._load_api_keys_to_env()

    def _load_api_keys_to_env(self) -> None:
        """Load API keys from .keys file into environment variables."""
        # Map of possible key names in .keys to environment variable names
        key_mappings = {
            "DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
            "DEEPSEEK": "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
            "ANTHROPIC": "ANTHROPIC_API_KEY",
            "CLAUDE": "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY": "OPENAI_API_KEY",
            "OPENAI": "OPENAI_API_KEY",
            "GEMINI_API_KEY": "GEMINI_API_KEY",
            "GEMINI": "GEMINI_API_KEY",
            "GOOGLE_API_KEY": "GEMINI_API_KEY",
            "GOOGLE": "GEMINI_API_KEY",
        }

        for key_name, env_var in key_mappings.items():
            if key_name in _keys_config and not os.getenv(env_var):
                os.environ[env_var] = _keys_config[key_name]

    def generate(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_ctx: int = 8 * 4096,
        repeat_penalty: float = 1.05,
        seed: Optional[int] = None,
        top_k: int = 60,
        num_predict: int = 4096,
    ) -> str:
        """Generate text using commercial API via LiteLLM.

        Args:
            prompt: Legacy parameter for single user message (deprecated)
            system_prompt: System role message (instructions/context)
            user_prompt: User role message (actual content to process)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_ctx: Context window size (ignored for most commercial APIs)
            repeat_penalty: Penalty for repeating tokens (ignored for most)
            seed: Random seed for reproducibility
            top_k: Top-k sampling parameter (ignored for most)
            num_predict: Maximum tokens to generate

        Returns:
            Generated text content
        """
        # Build messages array
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        elif prompt:
            # Legacy support: single prompt goes to user role
            messages.append({"role": "user", "content": prompt})

        if not messages:
            raise ValueError("Must provide either prompt or user_prompt")

        response = completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=num_predict,
            seed=seed,
            request_timeout=self.timeout,
        )
        return response.choices[0].message.content


class OllamaAPIConfig(APIConfig):
    """Configuration for Ollama LLM API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 600,
    ):
        base_url = base_url or OLLAMA_API_BASE_URL
        super().__init__(base_url, timeout)
        self.model = model or OLLAMA_DEFAULT_MODEL

    def generate(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_ctx: int = 8 * 4096,
        repeat_penalty: float = 1.05,
        seed: Optional[int] = None,
        top_k: int = 60,
        num_predict: int = 4096,
    ) -> str:
        """Generate text using LiteLLM with Ollama.

        Args:
            prompt: Legacy parameter for single user message (deprecated)
            system_prompt: System role message (instructions/context)
            user_prompt: User role message (actual content to process)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_ctx: Context window size
            repeat_penalty: Penalty for repeating tokens
            seed: Random seed for reproducibility
            top_k: Top-k sampling parameter
            num_predict: Maximum tokens to generate

        Returns:
            Generated text content
        """
        # Build messages array
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        elif prompt:
            # Legacy support: single prompt goes to user role
            messages.append({"role": "user", "content": prompt})

        if not messages:
            raise ValueError("Must provide either prompt or user_prompt")

        response = completion(
            model=f"ollama/{self.model}",
            messages=messages,
            api_base=self.base_url,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            num_ctx=num_ctx,
            top_k=top_k,
            max_tokens=num_predict,
            repeat_penalty=repeat_penalty,
            request_timeout=self.timeout,
            # reasoning_effort="high",
        )
        return response.choices[0].message.content


class OllamaTranslationConfig(OllamaAPIConfig):
    """Configuration for Ollama translation API using LiteLLM."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 600,
    ):
        base_url = base_url or OLLAMA_API_BASE_URL
        model = model or OLLAMA_DEFAULT_TRANSLATION_MODEL
        super().__init__(base_url, model, timeout)

    def translate(self, prompt: str) -> str:
        """Generate translation using LiteLLM with optimized parameters."""
        return self.generate(
            prompt=prompt,
            temperature=0.1,  # Low temperature for consistent translation
            top_p=0.9,
            num_ctx=4096,  # Smaller context for translation
            repeat_penalty=1.0,  # No repeat penalty for translation
            num_predict=2048,  # Shorter responses for translation
        )


class CommercialAPIConfig(APIConfig):
    """Configuration for commercial LLM APIs using LiteLLM.

    Supports DeepSeek, Claude, OpenAI, Gemini, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 600,
    ):
        """
        Initialize commercial API configuration.

        Args:
            model: Model identifier (e.g., 'deepseek-chat',
                'claude-3-sonnet', 'gpt-4')
            api_key: API key for the service. If None, will try to get
                from environment
            timeout: Request timeout in seconds
        """
        super().__init__(base_url="", timeout=timeout)
        self.model = model
        self.api_key = api_key

        # If no API key provided, try to load from api_keys module
        if not self.api_key:
            try:
                from audify.utils.api_keys import get_api_key

                # Map model prefixes to API key names
                if 'deepseek' in model.lower():
                    self.api_key = get_api_key('DEEPSEEK')
                elif 'claude' in model.lower():
                    self.api_key = get_api_key('ANTHROPIC') or get_api_key('CLAUDE')
                elif 'gpt' in model.lower() or 'openai' in model.lower():
                    self.api_key = get_api_key('OPENAI')
                elif 'gemini' in model.lower():
                    self.api_key = get_api_key('GOOGLE') or get_api_key('GEMINI')

                if not self.api_key:
                    logger.warning(f"No API key found for model {model}")
            except ImportError:
                logger.warning("Could not import api_keys module")

        # Set API key as environment variable for LiteLLM
        if self.api_key:
            if 'deepseek' in model.lower():
                os.environ['DEEPSEEK_API_KEY'] = self.api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = self.api_key
            elif 'gpt' in model.lower() or 'openai' in model.lower():
                os.environ['OPENAI_API_KEY'] = self.api_key
            elif 'gemini' in model.lower():
                os.environ['GOOGLE_API_KEY'] = self.api_key

    def generate(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_ctx: int = 8 * 4096,
        repeat_penalty: float = 1.05,
        seed: Optional[int] = None,
        top_k: int = 60,
        num_predict: int = 4096,
    ) -> str:
        """Generate text using LiteLLM with commercial API.

        Args:
            prompt: Legacy parameter for single user message (deprecated)
            system_prompt: System role message (instructions/context)
            user_prompt: User role message (actual content to process)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_ctx: Context window size (ignored for most APIs)
            repeat_penalty: Penalty for repeating tokens (ignored for most APIs)
            seed: Random seed for reproducibility
            top_k: Top-k sampling parameter (ignored for most APIs)
            num_predict: Maximum tokens to generate

        Returns:
            Generated text content
        """
        # Build messages array
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        elif prompt:
            # Legacy support: single prompt goes to user role
            messages.append({"role": "user", "content": prompt})

        if not messages:
            raise ValueError("Must provide either prompt or user_prompt")

        # Prepare kwargs for litellm completion
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": num_predict,
            "timeout": self.timeout,
        }

        # Add seed if provided
        if seed is not None:
            kwargs["seed"] = seed

        response = completion(**kwargs)
        return response.choices[0].message.content
