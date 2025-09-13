import os
from pathlib import Path

MODULE_PATH = Path(__file__).parent.resolve()
OUTPUT_BASE_DIR = MODULE_PATH / "data" / "output"

# Kokoro API configuration
KOKORO_API_BASE_URL = "http://localhost:8887/v1/audio"

# Allow override via environment variable
KOKORO_API_BASE_URL = os.getenv("KOKORO_API_URL", KOKORO_API_BASE_URL)

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
