from typing import Optional

from audify.utils.api_config import OllamaTranslationConfig
from audify.utils.constants import LANGUAGE_NAMES
from audify.utils.logging_utils import get_logger
from audify.utils.prompts import TRANSLATE_PROMPT

# Configure logging
logger = get_logger(__name__)


def translate_sentence(
    sentence: str,
    model: Optional[str] = None,
    src_lang: str | None = "en",
    tgt_lang: str | None = "en",
) -> str:
    """
    Translate a sentence using Ollama API.

    Parameters:
    -----------
    sentence: str
        Text to translate
    model: Optional[str]
        Ollama model to use (optional, kept for compatibility)
    src_lang: Optional[str], default="en"
        Source language code
    tgt_lang: Optional[str], default="en"
        Target language code

    Returns:
        Translated text
    """
    src_lang = src_lang or "en"
    tgt_lang = tgt_lang or "en"

    # If source and target are the same, return original
    if src_lang == tgt_lang:
        return sentence

    # Get language names for better prompts
    src_lang_name = LANGUAGE_NAMES.get(src_lang, src_lang)
    tgt_lang_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)

    # Initialize API config
    config = OllamaTranslationConfig(model=model)

    # Create translation prompt
    prompt = TRANSLATE_PROMPT.format(
        src_lang_name=src_lang_name, tgt_lang_name=tgt_lang_name, sentence=sentence
    )

    try:
        logger.info(f"Translating from {src_lang_name} to {tgt_lang_name} using Ollama")

        # Get translation using LiteLLM
        translated_text = config.translate(prompt).strip()

        # If model is a thinking model, take the text after </think>
        if "</think>" in translated_text:
            translated_text = translated_text.split("</think>", 1)[1].strip()

        if translated_text:
            logger.debug(f"Translation successful: '{sentence}' -> '{translated_text}'")
            return translated_text
        else:
            logger.warning("Empty translation response from Ollama")
            return sentence

    except Exception as e:
        logger.error(f"Failed to translate using Ollama at {config.base_url}: {e}")
        return sentence
