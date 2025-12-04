
from unittest.mock import Mock, patch

import pytest

from audify.utils.api_config import OllamaAPIConfig


class TestOllamaAPIConfig:
    """Test cases for OllamaAPIConfig class."""

    def setup_method(self):
        self.config = OllamaAPIConfig(base_url="http://test:11434", model="test-model")

    def test_generate_legacy_prompt(self):
        """Test generate method with legacy prompt parameter."""
        with patch("audify.utils.api_config.completion") as mock_completion:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Generated content"
            mock_completion.return_value = mock_response

            result = self.config.generate(prompt="Legacy prompt")

            assert result == "Generated content"
            mock_completion.assert_called_once()

            # Verify arguments passed to completion
            call_args = mock_completion.call_args
            assert call_args.kwargs["messages"] == [
                {"role": "user", "content": "Legacy prompt"}
            ]

    def test_generate_user_prompt(self):
        """Test generate method with user_prompt parameter."""
        with patch("audify.utils.api_config.completion") as mock_completion:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Generated content"
            mock_completion.return_value = mock_response

            result = self.config.generate(user_prompt="User prompt")

            assert result == "Generated content"
            mock_completion.assert_called_once()

            # Verify arguments passed to completion
            call_args = mock_completion.call_args
            assert call_args.kwargs["messages"] == [
                {"role": "user", "content": "User prompt"}
            ]

    def test_generate_system_and_user_prompt(self):
        """Test generate method with system_prompt and user_prompt parameters."""
        with patch("audify.utils.api_config.completion") as mock_completion:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Generated content"
            mock_completion.return_value = mock_response

            result = self.config.generate(
                system_prompt="System instructions",
                user_prompt="User content"
            )

            assert result == "Generated content"
            mock_completion.assert_called_once()

            # Verify arguments passed to completion
            call_args = mock_completion.call_args
            expected_messages = [
                {"role": "system", "content": "System instructions"},
                {"role": "user", "content": "User content"}
            ]
            assert call_args.kwargs["messages"] == expected_messages

    def test_generate_no_prompt_raises_error(self):
        """Test generate method raises ValueError when no prompt is provided."""
        with pytest.raises(
            ValueError, match="Must provide either prompt or user_prompt"
        ):
            self.config.generate()

    def test_repr(self):
        """Test __repr__ method."""
        expected = "OllamaAPIConfig(base_url='http://test:11434', timeout=600)"
        assert repr(self.config) == expected
