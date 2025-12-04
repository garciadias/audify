#!/usr/bin/env python3
"""
Test script for audify.translate module with comprehensive coverage.
"""

from unittest.mock import Mock, patch

from audify.translate import OllamaTranslationConfig, translate_sentence
from audify.utils.constants import LANGUAGE_NAMES
from audify.utils.prompts import TRANSLATE_PROMPT


def test_ollama_config_defaults():
    """Test OllamaTranslationConfig with default values."""
    config = OllamaTranslationConfig()
    assert config.base_url == "http://localhost:11434"  # Default from constants
    assert config.model == "qwen3:30b"  # Default from constants


def test_ollama_config_custom():
    """Test OllamaTranslationConfig with custom values."""
    config = OllamaTranslationConfig(
        base_url="http://custom-host:8080", model="custom-model"
    )
    assert config.base_url == "http://custom-host:8080"
    assert config.model == "custom-model"


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


def test_translate_sentence_same_language():
    """Test translation when source and target languages are the same."""
    sentence = "Hello world"
    result = translate_sentence(sentence, src_lang="en", tgt_lang="en")
    assert result == sentence


def test_translate_sentence_none_src_lang():
    """Test translation with None source language defaults to 'en'."""
    sentence = "Hello world"
    result = translate_sentence(sentence, src_lang=None, tgt_lang="en")
    assert result == sentence  # Same language after defaulting to 'en'


@patch("audify.translate.OllamaTranslationConfig")
def test_translate_sentence_successful(mock_config_class):
    """Test successful translation."""
    # Mock the config
    mock_config = Mock()
    mock_config.translate.return_value = "Hola mundo"
    mock_config_class.return_value = mock_config

    result = translate_sentence("Hello world", src_lang="en", tgt_lang="es")
    assert result == "Hola mundo"


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


def test_litellm_config():
    """Test the OllamaTranslationConfig class with LiteLLM."""
    # Test default configuration
    config = OllamaTranslationConfig()
    assert config.base_url is not None
    assert config.model is not None

    # Test custom configuration
    custom_config = OllamaTranslationConfig(
        base_url="http://custom-host:11434", model="llama3.1"
    )
    assert custom_config.base_url == "http://custom-host:11434"
    assert custom_config.model == "llama3.1"


def test_translation_interface():
    """Test the translation function interface with LiteLLM."""
    # Test sentences for different languages
    test_cases = [
        ("Hello world!", "en", "es"),
        ("How are you today?", "en", "fr"),
        ("This is a test sentence.", "en", "pt"),
        ("Good morning!", "en", "de"),
    ]

    for sentence, src_lang, tgt_lang in test_cases:

        src_lang_name = LANGUAGE_NAMES.get(src_lang, src_lang)
        tgt_lang_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)

        prompt = TRANSLATE_PROMPT.format(
            src_lang_name=src_lang_name, tgt_lang_name=tgt_lang_name, sentence=sentence
        )
        assert len(prompt) > 0
        assert sentence in prompt
