import logging
from typing import Optional

from langchain_ollama import OllamaLLM

from audify.constants import (
    LANGUAGE_NAMES,
    OLLAMA_API_BASE_URL,
    OLLAMA_DEFAULT_TRANSLATION_MODEL,
)
from audify.prompts import TRANSLATE_PROMPT

# Configure logging
logger = logging.getLogger(__name__)


class OllamaTranslationConfig:
    """Configuration for Ollama translation using LangChain."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or OLLAMA_API_BASE_URL
        self.model = model or OLLAMA_DEFAULT_TRANSLATION_MODEL

    def create_llm(self) -> OllamaLLM:
        """Create and configure OllamaLLM instance."""
        return OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=0.1,  # Low temperature for consistent translation
            top_p=0.9,
        )


def translate_sentence(
    sentence: str,
    model: Optional[str] = None,
    tokenizer: Optional[str] = None,  # Kept for compatibility, unused
    src_lang: str | None = "en",
    tgt_lang: str = "en",
) -> str:
    """
    Translate a sentence using Ollama API.

    Args:
        sentence: Text to translate
        model: Ollama model to use (optional, kept for compatibility)
        tokenizer: Unused, kept for compatibility with old interface
        src_lang: Source language code
        tgt_lang: Target language code

    Returns:
        Translated text
    """
    src_lang = src_lang or "en"

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
        src_lang_name=src_lang_name,
        tgt_lang_name=tgt_lang_name,
        sentence=sentence
    )

    try:
        logger.info(f"Translating from {src_lang_name} to {tgt_lang_name} using Ollama")

        # Create LangChain Ollama LLM instance
        llm = config.create_llm()

        # Get translation using LangChain
        translated_text = llm.invoke(prompt).strip()
        # If model is a thinking model, take the text after </think>
        if "</think>" in translated_text:
            translated_text = translated_text.split("</think>", 1)[1].strip()

        if translated_text:
            logger.debug(
                f"Translation successful: '{sentence}' -> '{translated_text}'"
            )
            return translated_text
        else:
            logger.warning("Empty translation response from Ollama")
            return sentence

    except Exception as e:
        logger.error(f"Failed to translate using Ollama at {config.base_url}: {e}")
        return sentence
