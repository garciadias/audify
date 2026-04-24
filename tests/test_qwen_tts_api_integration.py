"""Integration tests for Qwen TTS API server."""

import io
import json
import os
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from audify.utils.api_config import QwenTTSConfig


def _make_wav_bytes(duration_s: float = 0.1, sample_rate: int = 24000) -> bytes:
    """Create valid WAV bytes (sine wave) for mocking audio."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


WAV_BYTES = _make_wav_bytes()


# Mock HTTP handlers for Qwen TTS server simulation


class QwenMockHandler(BaseHTTPRequestHandler):
    """Mock Qwen TTS API handler that returns valid WAV audio."""

    def log_message(self, format, *args):
        pass


    def do_GET(self):
        if self.path.startswith("/models"):
            model_name = self.path.split("/")[-1]
            body = json.dumps({"model": model_name}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/tts":
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(WAV_BYTES)
        else:
            self.send_error(404)


class FlakyQwenHandler(BaseHTTPRequestHandler):
    """Mock handler that fails the first 2 requests then succeeds."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps(
                {"status": "healthy", "model_loaded": True, "model": "mock"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        request_count = getattr(self, "_request_count", 0)
        if self.path == "/tts":
            if request_count < 2:
                request_count += 1
                self._request_count = request_count
                self.send_error(503, "Service unavailable - model loading")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                self.wfile.write(WAV_BYTES)
        else:
            self.send_error(404)


class UnrecoverableErrorHandler(BaseHTTPRequestHandler):
    """Mock handler that always fails - simulates permanent outage."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_error(503, "Service unavailable")

    def do_POST(self):
        self.send_error(500, "Internal server error")


class SlowResponseHandler(BaseHTTPRequestHandler):
    """Mock handler that adds delay to simulate slow API responses."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/tts":
            time.sleep(0.1)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(WAV_BYTES)
        else:
            self.send_error(404)


# Test fixtures for server management


@pytest.fixture
def mock_qwen_server():
    """Start a mock Qwen TTS server (healthy)."""
    server_address = ("127.0.0.1", 18890)
    httpd = HTTPServer(server_address, QwenMockHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


@pytest.fixture
def flaky_qwen_server():
    """Start a flaky Qwen TTS server (intermittent failures)."""
    server_address = ("127.0.0.1", 18891)
    httpd = HTTPServer(server_address, FlakyQwenHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


@pytest.fixture
def slow_qwen_server():
    """Start a slow Qwen TTS server (delayed responses)."""
    server_address = ("127.0.0.1", 18892)
    httpd = HTTPServer(server_address, SlowResponseHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


@pytest.fixture(name="unrecov_qwen_server")
def unrecov_qwen_server_fixt():
    """Start an unrecoverable error server for testing permanent failures."""
    server_address = ("127.0.0.1", 18893)
    httpd = HTTPServer(server_address, UnrecoverableErrorHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


@pytest.fixture
def cleanup_qwen_servers():
    """Ensure all mock servers are shut down after test."""
    yield
    # Servers should have been shut down by fixtures themselves
    # This is a safety net


def test_healthy_server(mock_qwen_server):
    """Test that healthy server responds with audio."""
    import requests
    response = requests.post("http://127.0.0.1:18890/tts", timeout=5)
    assert response.status_code == 200
    assert "audio/wav" in response.headers.get("Content-Type", "")
    assert len(response.content) > 0


def test_flaky_server_recovery(flaky_qwen_server):
    """Test that flaky server recovers after initial failures."""
    import requests
    # First two requests should fail
    for i in range(2):
        response = requests.post("http://127.0.0.1:18891/tts", timeout=5)
        assert response.status_code == 503, f"Expected 503 on attempt {i+1}"
    
    # Third request should succeed
    response = requests.post("http://127.0.0.1:18891/tts", timeout=5)
    assert response.status_code == 200


def test_unrecoverable_server(unrecov_qwen_server):
    """Test that unrecoverable server always fails."""
    import requests
    response = requests.post("http://127.0.0.1:18893/tts", timeout=5)
    assert response.status_code == 500
    assert response.status_code == 500


def test_slow_server_response_time(slow_qwen_server):
    """Test that slow server returns within acceptable timeout."""
    import requests
    start = time.time()
    response = requests.post("http://127.0.0.1:18892/tts", timeout=5)
    elapsed = time.time() - start
    assert response.status_code == 200
    assert elapsed < 5, f"Response took {elapsed}s, expected < 5s"


def test_model_get_endpoint(mock_qwen_server):
    """Test that /models/get endpoint returns model info."""
    import requests
    response = requests.get("http://127.0.0.1:18890/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
