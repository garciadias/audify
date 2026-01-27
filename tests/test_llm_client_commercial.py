"""Tests for LLMClient with commercial API support."""

from unittest.mock import Mock, patch

from audify.audiobook_creator import LLMClient


class TestLLMClientCommercialAPI:
    """Test cases for LLMClient with commercial API support."""

    def test_init_with_ollama_model(self):
        """Test LLMClient initialization with Ollama model."""
        with patch("audify.audiobook_creator.OllamaAPIConfig") as mock_ollama_config:
            client = LLMClient(model="llama3.2:3b")
            assert client.is_commercial is False
            mock_ollama_config.assert_called_once()

    def test_init_with_commercial_api_model(self):
        """Test LLMClient initialization with commercial API model."""
        with patch(
                "audify.audiobook_creator.CommercialAPIConfig"
            ) as mock_commercial_config:
            LLMClient(model="api:deepseek-chat")
            assert mock_commercial_config.called
            mock_commercial_config.assert_called_once_with(model="deepseek-chat")

    def test_init_with_api_prefix_deepseek(self):
        """Test initialization with api:deepseek-chat model."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            LLMClient(model="api:deepseek-chat")
            mock_config.assert_called_once_with(model="deepseek-chat")

    def test_init_with_api_prefix_claude(self):
        """Test initialization with api:claude model."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            LLMClient(model="api:claude-3-sonnet-20240229")
            mock_config.assert_called_once_with(model="claude-3-sonnet-20240229")

    def test_init_with_api_prefix_gpt4(self):
        """Test initialization with api:gpt-4 model."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            LLMClient(model="api:gpt-4")
            mock_config.assert_called_once_with(model="gpt-4")

    def test_generate_audiobook_script_commercial_api_success(self):
        """Test successful script generation with commercial API."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.model = "deepseek-chat"
            mock_config_instance.generate.return_value = "Generated script from" \
            " commercial API"
            mock_config.return_value = mock_config_instance

            client = LLMClient(model="api:deepseek-chat")

            with patch("audify.audiobook_creator.clean_text") as mock_clean:
                mock_clean.return_value = "Cleaned commercial script"

                result = client.generate_audiobook_script("test chapter", "en")

                assert result == "Cleaned commercial script"
                mock_clean.assert_called_once()

    def test_generate_audiobook_script_commercial_api_empty_response(self):
        """Test handling of empty response from commercial API."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.model = "deepseek-chat"
            mock_config_instance.generate.return_value = ""
            mock_config.return_value = mock_config_instance

            client = LLMClient(model="api:deepseek-chat")

            result = client.generate_audiobook_script("test chapter", "en")

            expected = "Error: Unable to generate audiobook script for this content."
            assert result == expected

    def test_generate_audiobook_script_commercial_api_connection_error(self):
        """Test handling of connection error with commercial API."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.model = "deepseek-chat"
            mock_config_instance.generate.side_effect = Exception("connection refused")
            mock_config.return_value = mock_config_instance

            client = LLMClient(model="api:deepseek-chat")

            result = client.generate_audiobook_script("test chapter", "en")

            assert "Could not connect to commercial API" in result
            assert "API key" in result

    def test_generate_audiobook_script_commercial_api_timeout(self):
        """Test handling of timeout with commercial API."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.model = "deepseek-chat"
            mock_config_instance.generate.side_effect = Exception("timeout exceeded")
            mock_config.return_value = mock_config_instance

            client = LLMClient(model="api:deepseek-chat")

            result = client.generate_audiobook_script("test chapter", "en")

            assert "timed out" in result

    def test_generate_audiobook_script_commercial_api_key_error(self):
        """Test handling of API key error with commercial API."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.model = "deepseek-chat"
            mock_config_instance.generate.side_effect = Exception(
                "Invalid API key provided"
            )
            mock_config.return_value = mock_config_instance

            client = LLMClient(model="api:deepseek-chat")

            result = client.generate_audiobook_script("test chapter", "en")

            assert "API key issue" in result
            assert ".keys file" in result

    def test_generate_audiobook_script_ollama_connection_error(self):
        """Test handling of Ollama connection error."""
        with patch("audify.audiobook_creator.OllamaAPIConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.base_url = "http://localhost:11434"
            mock_config_instance.generate.side_effect = Exception("connection refused")
            mock_config.return_value = mock_config_instance

            client = LLMClient(model="llama3.2:3b")

            result = client.generate_audiobook_script("test chapter", "en")

            assert "Could not connect to local LLM server" in result
            assert "Ollama" in result

    def test_generate_with_reasoning_model_response(self):
        """Test handling of reasoning model response with <think> tags."""
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.model = "deepseek-reasoner"
            response_with_thinking = (
                "<think>This is the thinking process...</think>"
                "This is the actual response"
            )
            mock_config_instance.generate.return_value = response_with_thinking
            mock_config.return_value = mock_config_instance

            client = LLMClient(model="api:deepseek-reasoner")

            with patch("audify.audiobook_creator.clean_text") as mock_clean:
                mock_clean.return_value = "This is the actual response"

                client.generate_audiobook_script("test chapter", "en")

                # Should call clean_text with the part after </think>
                mock_clean.assert_called_once()
                call_arg = mock_clean.call_args[0][0]
                assert "thinking process" not in call_arg
                assert "actual response" in call_arg

    def test_commercial_vs_ollama_logging(self):
        """Test that appropriate logging occurs for commercial vs Ollama."""
        # Test commercial API logging
        with patch("audify.audiobook_creator.CommercialAPIConfig") as mock_config:
            with patch("audify.audiobook_creator.logger") as mock_logger:
                mock_config_instance = Mock()
                mock_config_instance.model = "deepseek-chat"
                mock_config.return_value = mock_config_instance

                LLMClient(model="api:deepseek-chat")

                # Check that commercial API message was logged
                log_calls = [str(call) for call in mock_logger.info.call_args_list]
                assert any("commercial API" in str(call) for call in log_calls)

        # Test Ollama logging
        with patch("audify.audiobook_creator.OllamaAPIConfig") as mock_config:
            with patch("audify.audiobook_creator.logger") as mock_logger:
                mock_config_instance = Mock()
                mock_config.return_value = mock_config_instance

                LLMClient(model="llama3.2:3b")

                # Check that Ollama message was logged
                log_calls = [str(call) for call in mock_logger.info.call_args_list]
                assert any("Ollama" in str(call) for call in log_calls)
