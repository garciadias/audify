"""Tests for commercial API configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest

from audify.utils.api_config import CommercialAPIConfig


class TestCommercialAPIConfig:
    """Test cases for CommercialAPIConfig class."""

    def test_init_claude_model(self):
        """Test initialization with Claude model."""
        with patch('audify.utils.api_keys.get_api_key') as mock_get_key:
            mock_get_key.return_value = "claude-key"
            config = CommercialAPIConfig(model="claude-3-sonnet")
            assert config.model == "claude-3-sonnet"
            # Should try ANTHROPIC first
            assert mock_get_key.call_count >= 1

    def test_init_openai_model(self):
        """Test initialization with OpenAI model."""
        with patch('audify.utils.api_keys.get_api_key') as mock_get_key:
            mock_get_key.return_value = "openai-key"
            config = CommercialAPIConfig(model="gpt-4")
            # Model gets mapped to LiteLLM format
            assert config.model == "openai/gpt-4-turbo-preview"
            mock_get_key.assert_called_once_with('OPENAI')

    def test_init_gemini_model(self):
        """Test initialization with Gemini model."""
        with patch('audify.utils.api_keys.get_api_key') as mock_get_key:
            mock_get_key.return_value = "gemini-key"
            config = CommercialAPIConfig(model="gemini-pro")
            # Model gets mapped to LiteLLM format
            assert config.model == "gemini/gemini-pro"
            # Should try GOOGLE first
            assert mock_get_key.call_count >= 1

    def test_generate_with_system_and_user_prompts(self):
        """Test generate method with system and user prompts."""
        config = CommercialAPIConfig(model="deepseek-chat", api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated response"

        with patch(
            'audify.utils.api_config.completion', return_value=mock_response
        ) as mock_completion:
            result = config.generate(
                system_prompt="You are a helpful assistant",
                user_prompt="Hello, world!"
            )

            assert result == "Generated response"
            mock_completion.assert_called_once()

            # Check the call arguments
            call_args = mock_completion.call_args
            # Model gets mapped to LiteLLM format
            assert call_args[1]['model'] == "deepseek/deepseek-chat"
            messages = call_args[1]['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert messages[0]['content'] == "You are a helpful assistant"
            assert messages[1]['role'] == 'user'
            assert messages[1]['content'] == "Hello, world!"

    def test_generate_with_prompt_only(self):
        """Test generate method with legacy prompt parameter."""
        config = CommercialAPIConfig(model="deepseek-chat", api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated response"

        with patch(
            'audify.utils.api_config.completion', return_value=mock_response
        ) as mock_completion:
            result = config.generate(prompt="Hello, world!")

            assert result == "Generated response"
            messages = mock_completion.call_args[1]['messages']
            assert len(messages) == 1
            assert messages[0]['role'] == 'user'
            assert messages[0]['content'] == "Hello, world!"


    def test_generate_no_prompt_raises_error(self):
        """Test that generate raises error when no prompt is provided."""
        config = CommercialAPIConfig(model="deepseek-chat", api_key="test-key")

        with pytest.raises(
            ValueError, match="Must provide either prompt or user_prompt"
        ):
            config.generate()
