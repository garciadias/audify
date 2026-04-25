import os
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2].resolve()
OUTPUT_BASE_DIR = MODULE_PATH / "data" / "output"

KEYS_FILE = MODULE_PATH / ".keys"
_keys_config: dict[str, str] = {}

if KEYS_FILE.exists():
    with open(KEYS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key and value:
                    _keys_config[key] = value


def _get_config(key: str, default: str = "") -> str:
    """Get configuration value with priority: env var > .keys file > default."""
    return os.getenv(key, _keys_config.get(key, default))


KOKORO_API_BASE_URL = _get_config("KOKORO_API_URL", "http://localhost:8887/v1")
OLLAMA_API_BASE_URL = _get_config("OLLAMA_API_URL", "http://localhost:11434")

DEFAULT_TTS_PROVIDER = _get_config("TTS_PROVIDER", "kokoro")
AVAILABLE_TTS_PROVIDERS = ["kokoro", "openai", "aws", "google", "qwen"]

OPENAI_API_KEY = _get_config("OPENAI_API_KEY", "")
OPENAI_TTS_MODEL = _get_config("OPENAI_TTS_MODEL", "gpt-4o-mini-tts-2025-03-20")
OPENAI_TTS_VOICE = _get_config("OPENAI_TTS_VOICE", "coral")

AWS_ACCESS_KEY_ID = _get_config("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = _get_config("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = _get_config("AWS_REGION", "us-east-1")
AWS_POLLY_VOICE = _get_config("AWS_POLLY_VOICE", "Joanna")
AWS_POLLY_ENGINE = _get_config("AWS_POLLY_ENGINE", "neural")

GOOGLE_APPLICATION_CREDENTIALS = _get_config("GOOGLE_APPLICATION_CREDENTIALS", "")
GOOGLE_TTS_VOICE = _get_config("GOOGLE_TTS_VOICE", "en-US-Neural2-F")
GOOGLE_TTS_LANGUAGE_CODE = _get_config("GOOGLE_TTS_LANGUAGE_CODE", "en-US")

# Language-specific Google TTS defaults used when no explicit voice is provided.
GOOGLE_TTS_DEFAULT_VOICE_BY_LANGUAGE = {
    "en": _get_config("GOOGLE_TTS_VOICE_EN", "en-US-Neural2-F"),
    "es": _get_config("GOOGLE_TTS_VOICE_ES", "es-ES-Neural2-F"),
    "fr": _get_config("GOOGLE_TTS_VOICE_FR", "fr-FR-Neural2-E"),
    "de": _get_config("GOOGLE_TTS_VOICE_DE", "de-DE-Neural2-F"),
    "it": _get_config("GOOGLE_TTS_VOICE_IT", "it-IT-Neural2-C"),
    "pt": _get_config("GOOGLE_TTS_VOICE_PT", "pt-BR-Neural2-C"),
    "ja": _get_config("GOOGLE_TTS_VOICE_JA", "ja-JP-Neural2-D"),
    "zh": _get_config("GOOGLE_TTS_VOICE_ZH", "cmn-CN-Neural2-D"),
    "hi": _get_config("GOOGLE_TTS_VOICE_HI", "hi-IN-Neural2-D"),
    "ko": _get_config("GOOGLE_TTS_VOICE_KO", "ko-KR-Neural2-A"),
    "ru": _get_config("GOOGLE_TTS_VOICE_RU", "ru-RU-Wavenet-D"),
    "ar": _get_config("GOOGLE_TTS_VOICE_AR", "ar-XA-Wavenet-D"),
    "nl": _get_config("GOOGLE_TTS_VOICE_NL", "nl-NL-Wavenet-D"),
    "pl": _get_config("GOOGLE_TTS_VOICE_PL", "pl-PL-Wavenet-B"),
    "tr": _get_config("GOOGLE_TTS_VOICE_TR", "tr-TR-Wavenet-B"),
}


QWEN_API_BASE_URL = _get_config("QWEN_API_URL", "http://localhost:8890")
QWEN_TTS_VOICE = _get_config("QWEN_TTS_VOICE", "Vivian")

OLLAMA_DEFAULT_TRANSLATION_MODEL = _get_config("OLLAMA_TRANSLATION_MODEL", "qwen3:30b")
OLLAMA_DEFAULT_MODEL = _get_config("OLLAMA_MODEL", "magistral:24b")

AVAILABLE_LANGUAGES = {
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi",
    "Italian": "it",
    "Portuguese": "pt",
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
}

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh": "Chinese",
    "hu": "Hungarian",
    "ko": "Korean",
    "ja": "Japanese",
    "hi": "Hindi",
}

LANG_CODES = {
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "en": "a",
    "zh": "z",
    "ja": "j",
}

DEFAULT_LANGUAGE_LIST = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh",
    "hu",
    "ko",
    "ja",
    "hi",
]
DEFAULT_SPEAKER = "ef_dora"
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_ENGINE = "kokoro"
