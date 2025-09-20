"""
Shared API configuration utilities for various external services.

This module consolidates API configuration classes to reduce code duplication
across different modules that interact with external APIs.
"""

import logging
from typing import Optional

from langchain_ollama import OllamaLLM

from audify.utils.constants import (
    KOKORO_API_BASE_URL,
    KOKORO_DEFAULT_VOICE,
    OLLAMA_API_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_DEFAULT_TRANSLATION_MODEL,
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
        self.default_voice = voice or KOKORO_DEFAULT_VOICE

    @property
    def voices_url(self) -> str:
        """URL for fetching available voices."""
        return f"{self.base_url}/voices"

    @property
    def speech_url(self) -> str:
        """URL for text-to-speech synthesis."""
        return f"{self.base_url}/speech"


class OllamaAPIConfig(APIConfig):
    """Configuration for Ollama LLM API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        base_url = base_url or OLLAMA_API_BASE_URL
        super().__init__(base_url, timeout)
        self.model = model or OLLAMA_DEFAULT_MODEL

    def create_llm(
        self,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_ctx: int = 8 * 4096,
        repeat_penalty: float = 1.05,
        seed: Optional[int] = None,
        top_k: int = 60,
        num_predict: int = 4096,
    ) -> OllamaLLM:
        """Create and configure OllamaLLM instance."""
        return OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            num_ctx=num_ctx,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            top_k=top_k,
            num_predict=num_predict,
        )


class OllamaTranslationConfig(OllamaAPIConfig):
    """Configuration for Ollama translation API using LangChain."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        base_url = base_url or OLLAMA_API_BASE_URL
        model = model or OLLAMA_DEFAULT_TRANSLATION_MODEL
        super().__init__(base_url, model, timeout)

    def create_translation_llm(self) -> OllamaLLM:
        """Create OllamaLLM instance specifically configured for translation."""
        return self.create_llm(
            temperature=0.1,  # Low temperature for consistent translation
            top_p=0.9,
            num_ctx=4096,  # Smaller context for translation
            repeat_penalty=1.0,  # No repeat penalty for translation
            num_predict=2048,  # Shorter responses for translation
        )
