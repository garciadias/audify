from typing import Optional, Union

from audify.utils.api_config import (
    CommercialAPIConfig,
    OllamaTranslationConfig,
)
from audify.utils.constants import LANGUAGE_NAMES
from audify.utils.logging_utils import get_logger
from audify.utils.prompts import TRANSLATE_PROMPT

# Configure logging
logger = get_logger(__name__)


def _get_translation_config(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Union[OllamaTranslationConfig, CommercialAPIConfig]:
    """Create the appropriate LLM config for translation.

    If model starts with 'api:', uses CommercialAPIConfig.
    Otherwise uses OllamaTranslationConfig.
    """
    if model and model.startswith("api:"):
        actual_model = model[4:]
        return CommercialAPIConfig(model=actual_model)
    return OllamaTranslationConfig(model=model, base_url=base_url)


def translate_sentence(
    sentence: str,
    model: Optional[str] = None,
    src_lang: str | None = "en",
    tgt_lang: str | None = "en",
    base_url: Optional[str] = None,
) -> str:
    """
    Translate a sentence using LLM.

    Parameters:
    -----------
    sentence: str
        Text to translate
    model: Optional[str]
        LLM model to use. Use 'api:model_name' for commercial APIs
        (e.g., 'api:deepseek/deepseek-chat'). If None, uses the
        default translation model from config.
    src_lang: Optional[str], default="en"
        Source language code
    tgt_lang: Optional[str], default="en"
        Target language code
    base_url: Optional[str]
        Base URL for Ollama API (ignored for commercial APIs)

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
    config = _get_translation_config(model=model, base_url=base_url)

    # Create translation prompt
    prompt = TRANSLATE_PROMPT.format(
        src_lang_name=src_lang_name, tgt_lang_name=tgt_lang_name, sentence=sentence
    )

    try:
        provider_name = (
            f"commercial API ({config.model})"
            if isinstance(config, CommercialAPIConfig)
            else f"Ollama ({config.model})"
        )
        logger.info(
            f"Translating from {src_lang_name} to {tgt_lang_name}"
            f" using {provider_name}"
        )

        # Generate translation
        if isinstance(config, CommercialAPIConfig):
            translated_text = config.generate(
                user_prompt=prompt,
                temperature=0.1,
                top_p=0.9,
                num_predict=2048,
            ).strip()
        else:
            translated_text = config.translate(prompt).strip()

        # If model is a thinking model, take the text after </think>
        if "</think>" in translated_text:
            translated_text = translated_text.split("</think>", 1)[1].strip()

        if translated_text:
            logger.debug(f"Translation successful: '{sentence}' -> '{translated_text}'")
            return translated_text
        else:
            logger.warning("Empty translation response from LLM")
            return sentence

    except Exception as e:
        provider_info = (
            "commercial API"
            if isinstance(config, CommercialAPIConfig)
            else f"Ollama at {config.base_url}"
        )
        logger.error(f"Failed to translate using {provider_info}: {e}")
        return sentence
