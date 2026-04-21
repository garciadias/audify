"""Tests for Qwen TTS API configuration."""

import os
import subprocess
import sys
import time
from unittest.mock import Mock, patch

import pytest

from audify.utils.api_config import QwenTTSConfig


class TestQwenTTSConfig:
    """Test cases for QwenTTSConfig class."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        config = QwenTTSConfig()
        assert config.provider_name == "qwen"
        assert config.language == "en"
        assert config.timeout == 60
        assert config.base_url == "http://localhost:8890"
        assert config.voice == "Vivian"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        config = QwenTTSConfig(
            voice="custom_voice",
            language="es",
            base_url="http://custom:8891",
            timeout=120,
        )
        assert config.voice == "custom_voice"
        assert config.language == "es"
        assert config.base_url == "http://custom:8891"
        assert config.timeout == 120

    def test_health_url(self):
        """Test health_url property."""
        config = QwenTTSConfig(base_url="http://test:8890")
        assert config.health_url == "http://test:8890/health"

    def test_tts_url(self):
        """Test tts_url property."""
        config = QwenTTSConfig(base_url="http://test:8890")
        assert config.tts_url == "http://test:8890/tts"

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_true(self, mock_get):
        """Test is_available returns True when API is reachable and healthy."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "model_loaded": True}
        mock_get.return_value = mock_response

        config = QwenTTSConfig()
        assert config.is_available() is True

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_false_on_non_healthy(self, mock_get):
        """Test is_available returns False when status not healthy."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "unhealthy", "model_loaded": False}
        mock_get.return_value = mock_response

        config = QwenTTSConfig()
        assert config.is_available() is False
        assert mock_get.call_count == config.health_retries

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_retries_until_healthy(self, mock_get):
        """Test is_available retries unhealthy responses until healthy."""
        unhealthy = Mock()
        unhealthy.status_code = 200
        unhealthy.json.return_value = {"status": "healthy", "model_loaded": False}

        healthy = Mock()
        healthy.status_code = 200
        healthy.json.return_value = {"status": "healthy", "model_loaded": True}

        mock_get.side_effect = [unhealthy, healthy]

        config = QwenTTSConfig()
        assert config.is_available() is True
        assert mock_get.call_count == 2

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_false_on_invalid_json(self, mock_get):
        """Test is_available returns False when health endpoint returns invalid JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        config = QwenTTSConfig()
        assert config.is_available() is False
        assert mock_get.call_count == config.health_retries

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_false_on_error(self, mock_get):
        """Test is_available returns False on request exception."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection error")

        config = QwenTTSConfig()
        assert config.is_available() is False

    @patch("audify.utils.api_config.requests.get")
    def test_is_available_false_on_non_200(self, mock_get):
        """Test is_available returns False on non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        config = QwenTTSConfig()
        assert config.is_available() is False

    def test_get_available_voices(self):
        """Test get_available_voices returns default voice list."""
        config = QwenTTSConfig()
        voices = config.get_available_voices()
        assert voices == ["Vivian"]

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_success(self, mock_post, tmp_path):
        """Test synthesize creates audio file on success."""
        # Content must be >= 44 bytes (WAV header size) to pass validation
        fake_wav = b"\x00" * 44 + b"fake audio content"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = fake_wav
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        config = QwenTTSConfig(voice="Vivian", language="en")
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.read_bytes() == fake_wav

        # Verify request payload
        call_json = mock_post.call_args.kwargs["json"]
        assert call_json["text"] == "Hello world"
        assert call_json["language"] == "Auto"
        assert call_json["speaker"] == "Vivian"

    @patch("time.sleep")
    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_api_error(self, mock_post, mock_sleep, tmp_path):
        """Test synthesize returns False on persistent API error."""
        import requests as req

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = req.HTTPError("500")
        mock_post.return_value = mock_response

        config = QwenTTSConfig()
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is False

    @patch("time.sleep")
    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_request_exception(self, mock_post, mock_sleep, tmp_path):
        """Test synthesize returns False on persistent request exception."""
        import requests

        mock_post.side_effect = requests.RequestException("Network error")

        config = QwenTTSConfig()
        output_path = tmp_path / "output.wav"

        result = config.synthesize("Hello world", output_path)

        assert result is False


class TestQwenTTSIntegration:
    """Integration tests with actual mock server."""

    def test_qwen_tts_config_with_mock_server(self, tmp_path):
        """Test QwenTTSConfig works with a live mock server."""

        # Start mock server
        env = os.environ.copy()
        env["QWEN_TTS_MOCK"] = "1"
        env["QWEN_TTS_PORT"] = "8892"

        # Get project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        script_path = os.path.join(project_root, "scripts", "qwen_tts_api.py")
        print(f"Project root: {project_root}")
        print(f"Script path: {script_path}")
        print(f"Script exists: {os.path.exists(script_path)}")

        if not os.path.exists(script_path):
            pytest.skip("qwen_tts_api.py script not found")

        proc = subprocess.Popen(
            [sys.executable, script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root,
        )

        try:
            config = QwenTTSConfig(base_url="http://localhost:8892")

            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if config.is_available():
                    break
                if proc.poll() is not None:
                    stderr = (proc.stderr.read() or b"").decode("utf-8", "replace")
                    pytest.fail(f"qwen_tts_api.py exited early: {stderr}")
                time.sleep(0.5)
            else:
                pytest.fail("Mock Qwen server did not become healthy in 30s")

            # Get voices
            voices = config.get_available_voices()
            print(f"Voices: {voices}")
            assert isinstance(voices, list)
            assert len(voices) > 0

            # Test synthesis
            output_path = tmp_path / "test.wav"
            result = config.synthesize("Hello integration test", output_path)
            assert result is True
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            print(f"Output file size: {output_path.stat().st_size}")

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
