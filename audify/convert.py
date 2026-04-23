#!/usr/bin/env python3
"""
Helper functions for Audify CLI - Factory and utility functions for conversion.
"""

import requests

from audify.audiobook_creator import (
    AudiobookCreator,
    AudiobookEpubCreator,
    AudiobookPdfCreator,
)
from audify.utils.constants import KOKORO_API_BASE_URL


def get_available_models_and_voices():
    """Get available models and voices from Kokoro API."""
    from audify.utils.api_config import _retry_request

    def _fetch():
        models_response = requests.get(f"{KOKORO_API_BASE_URL}/models", timeout=10)
        models_response.raise_for_status()
        models_data = models_response.json().get("data", [])
        models = sorted([model.get("id") for model in models_data if "id" in model])

        voices_response = requests.get(
            f"{KOKORO_API_BASE_URL}/audio/voices", timeout=10
        )
        voices_response.raise_for_status()
        voices_data = voices_response.json().get("voices", [])
        voices = sorted(voices_data)
        return models, voices

    try:
        return _retry_request(_fetch, api_name=f"Kokoro API ({KOKORO_API_BASE_URL})")
    except RuntimeError:
        return [], []


def get_creator(
    file_extension: str,
    path: str,
    language: str,
    voice: str,
    model_name: str,
    translate: str | None,
    save_text: bool,
    llm_base_url: str,
    llm_model: str,
    max_chapters: int | None,
    confirm: bool,
    output_dir: str | None = None,
    tts_provider: str | None = None,
    task: str | None = None,
    prompt_file: str | None = None,
    mode: str = "full",
) -> AudiobookCreator:
    """Get the appropriate AudiobookCreator subclass based on file extension.

    Args:
        file_extension: The file extension (e.g., '.epub', '.pdf').
        task: Task name for prompt selection (e.g., 'audiobook', 'podcast').
        prompt_file: Path to a custom prompt file.

    Returns:
        The corresponding AudiobookCreator subclass.

    Raises:
        TypeError: If the file extension is unsupported.
    """
    if file_extension == ".epub":
        return AudiobookEpubCreator(
            path=path,
            language=language,
            voice=voice,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            max_chapters=max_chapters,
            confirm=confirm,
            output_dir=output_dir,
            tts_provider=tts_provider,
            task=task,
            prompt_file=prompt_file,
            mode=mode,
        )
    elif file_extension == ".pdf":
        # remove max_chapters for PDF
        return AudiobookPdfCreator(
            path=path,
            language=language,
            voice=voice,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            confirm=confirm,
            output_dir=output_dir,
            tts_provider=tts_provider,
            task=task,
            prompt_file=prompt_file,
            mode=mode,
        )
    else:
        raise TypeError(f"Unsupported file format '{file_extension}'")
