import os
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1].resolve()
OUTPUT_BASE_DIR = MODULE_PATH / "data" / "output"

# Kokoro API configuration
KOKORO_API_BASE_URL = "http://localhost:8887/v1"
OLLAMA_API_BASE_URL = "http://localhost:11434"
# Allow override via environment variable
KOKORO_API_BASE_URL = os.getenv("KOKORO_API_URL", KOKORO_API_BASE_URL)
OLLAMA_DEFAULT_TRANSLATION_MODEL = os.getenv(
    "OLLAMA_TRANSLATION_MODEL", "mistral-nemo:12b"
)
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:30b")

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

# Language code mapping for better prompts
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
KOKORO_DEFAULT_VOICE = "af_bella"
DEFAULT_SPEAKER = KOKORO_DEFAULT_VOICE
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_ENGINE = "kokoro"
