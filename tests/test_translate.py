#!/usr/bin/env python3
"""
Test script for audify.translate module with comprehensive coverage.
"""

from unittest.mock import ANY, Mock, patch

from audify.translate import (
    OllamaTranslationConfig,
    _get_translation_config,
    translate_sentence,
)
from audify.utils.api_config import CommercialAPIConfig
from audify.utils.constants import LANGUAGE_NAMES
from audify.utils.prompts import TRANSLATE_PROMPT

def test_translate_method():
    """Test translation method with proper configuration."""
    config = OllamaTranslationConfig(base_url="http://test:11434", model="test-model")

    with patch("audify.utils.api_config.completion") as mock_completion:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Translated text"
        mock_completion.return_value = mock_response

        result = config.translate("Test prompt")
        assert result == "Translated text"
        mock_completion.assert_called_once()

@patch("audify.translate.OllamaTranslationConfig")
def test_translate_sentence_with_thinking_model(mock_config_class):
    """Test translation with thinking model response."""
    # Mock the config
    mock_config = Mock()
    mock_config.translate.return_value = "<think>This is thinking</think>Hola mundo"
    mock_config_class.return_value = mock_config

    result = translate_sentence("Hello world", src_lang="en", tgt_lang="es")
    assert result == "Hola mundo"


@patch("audify.translate.OllamaTranslationConfig")
def test_translate_sentence_empty_response(mock_config_class):
    """Test translation with empty response."""
    # Mock the config
    mock_config = Mock()
    mock_config.translate.return_value = ""
    mock_config_class.return_value = mock_config

    sentence = "Hello world"
    result = translate_sentence(sentence, src_lang="en", tgt_lang="es")
    assert result == sentence  # Should return original on empty response


@patch("audify.translate.OllamaTranslationConfig")
def test_translate_sentence_exception(mock_config_class):
    """Test translation with exception handling."""
    # Mock the config to raise exception
    mock_config = Mock()
    mock_config.translate.side_effect = Exception("Connection failed")
    mock_config_class.return_value = mock_config
    mock_config.base_url = "http://localhost:11434"

    sentence = "Hello world"
    result = translate_sentence(sentence, src_lang="en", tgt_lang="es")
    assert result == sentence  # Should return original on exception
