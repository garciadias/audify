"""
Shared API configuration utilities for various external services.

This module consolidates API configuration classes to reduce code duplication
across different modules that interact with external APIs.
"""

import logging
import os
from typing import Optional

from litellm import completion

from audify.utils.constants import (
    DEFAULT_SPEAKER,
    KOKORO_API_BASE_URL,
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
        self.default_voice = voice or DEFAULT_SPEAKER

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
