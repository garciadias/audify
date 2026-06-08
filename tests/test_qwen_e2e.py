"""End-to-end test for Qwen TTS audiobook pipeline.

Uses a local mock HTTP server that returns valid WAV audio to exercise the
full pipeline: EPUB -> script generation -> Qwen TTS synthesis -> MP3 output.
Also tests retry and recovery logic for transient server failures.
"""

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ebooklib import epub

from audify.audiobook_creator import AudiobookEpubCreator
from audify.utils.api_config import QwenTTSConfig


def _make_wav_bytes(duration_s: float = 0.3, sample_rate: int = 24000) -> bytes:
    """Create valid WAV bytes (sine wave)."""
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


# ---------------------------------------------------------------------------
# Mock HTTP handlers
# ---------------------------------------------------------------------------


class QwenMockHandler(BaseHTTPRequestHandler):
    """Mock Qwen TTS API handler that returns valid WAV audio."""

    def log_message(self, format, *args):
        pass  # suppress noisy request logs during tests

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
        if self.path == "/tts":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            payload = json.loads(body)

            text = payload.get("text", "")
            if not text.strip():
                self.send_error(400, "Empty text")
                return

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(WAV_BYTES)
        else:
            self.send_error(404)


class FlakyQwenHandler(BaseHTTPRequestHandler):
    """Mock handler that fails the first N requests then succeeds.

    Simulates a Qwen TTS server that crashes and recovers.
    Uses a class-level counter shared across all requests.
    """

    fail_count = 0  # how many requests have been made
    fail_until = 2  # fail the first N POST requests

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
        if self.path == "/tts":
            content_len = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_len)  # consume body

            FlakyQwenHandler.fail_count += 1
            if FlakyQwenHandler.fail_count <= FlakyQwenHandler.fail_until:
                self.send_error(500, "Server overloaded")
                return

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(WAV_BYTES)
        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qwen_mock_server():
    """Start a mock Qwen TTS server on a random port."""
    server = HTTPServer(("127.0.0.1", 0), QwenMockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def flaky_qwen_server():
    """Start a flaky mock Qwen TTS server that fails then recovers."""
    FlakyQwenHandler.fail_count = 0
    FlakyQwenHandler.fail_until = 2
    server = HTTPServer(("127.0.0.1", 0), FlakyQwenHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _create_test_epub(path: Path, num_chapters: int = 2) -> Path:
    """Create a minimal EPUB for testing."""
    book = epub.EpubBook()
    book.set_identifier("test-qwen-e2e")
    book.set_title("E2E Test Book")
    book.set_language("en")
    book.add_author("Test Author")

    chapters = []
    for i in range(1, num_chapters + 1):
        ch = epub.EpubHtml(
            title=f"Chapter {i}",
            file_name=f"chapter{i}.xhtml",
            lang="en",
        )
        ch.content = f"""<html><head><title>Chapter {i}</title></head>
<body>
<h1>Chapter {i}</h1>
<p>This is the content of chapter {i}. It has enough text for the
audiobook creator to process it properly. The quick brown fox jumps
over the lazy dog.</p>
<p>Here is a second paragraph with more text to ensure the script
generation has meaningful content to work with during testing.</p>
</body></html>"""
        book.add_item(ch)
        chapters.append(ch)

    book.toc = chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters

    epub_path = path / "test_e2e.epub"
    epub.write_epub(str(epub_path), book, {})
    return epub_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQwenE2EPipeline:
    """End-to-end tests for the Qwen TTS audiobook pipeline."""

    def test_synthesize_episode_with_mock_qwen(self, tmp_path, qwen_mock_server):
        """Full pipeline: EPUB -> LLM script -> Qwen TTS -> MP3."""
        epub_path = _create_test_epub(tmp_path, num_chapters=1)
        output_dir = tmp_path / "output"

        with patch.object(
            AudiobookEpubCreator,
            "_verify_tts_provider_available",
            return_value=None,
        ):
            creator = AudiobookEpubCreator(
                path=str(epub_path),
                language="en",
                voice="Vivian",
                llm_model="api:deepseek/deepseek-chat",
                confirm=False,
                output_dir=str(output_dir),
                tts_provider="qwen",
                save_text=True,
            )

        # Override the TTS config to point at our mock server
        mock_config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url=qwen_mock_server,
        )
        creator._tts_config = mock_config

        # Mock the LLM client to return a simple script
        creator.llm_client.generate_script = MagicMock(
            return_value="Hello world. This is a test audiobook episode."
        )

        # Mock ffmpeg-dependent MP3 conversion: copy WAV content to .mp3
        def _fake_convert(self, wav_path):
            mp3_path = wav_path.with_suffix(".mp3")
            mp3_path.write_bytes(wav_path.read_bytes())
            return mp3_path

        with patch.object(
            type(creator), "_convert_to_mp3", _fake_convert
        ):
            script = creator.generate_audiobook_script("Test chapter content.", 1)
            assert script

            episode_path = creator.synthesize_episode(script, 1)

            assert episode_path.exists(), (
                f"Episode file was not created at {episode_path}. "
                f"Episodes dir contents: {list(creator.episodes_path.iterdir())}"
            )
            assert episode_path.suffix == ".mp3"
            assert episode_path.stat().st_size > 0

    def test_full_audiobook_series_with_mock_qwen(self, tmp_path, qwen_mock_server):
        """Full pipeline with multiple chapters."""
        epub_path = _create_test_epub(tmp_path, num_chapters=2)
        output_dir = tmp_path / "output"

        with patch.object(
            AudiobookEpubCreator,
            "_verify_tts_provider_available",
            return_value=None,
        ):
            creator = AudiobookEpubCreator(
                path=str(epub_path),
                language="en",
                voice="Vivian",
                llm_model="api:deepseek/deepseek-chat",
                confirm=False,
                output_dir=str(output_dir),
                tts_provider="qwen",
                save_text=True,
            )

        # Override TTS config
        mock_config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url=qwen_mock_server,
        )
        creator._tts_config = mock_config

        # Mock LLM
        creator.llm_client.generate_script = MagicMock(
            return_value="Hello world. This is a test audiobook episode."
        )

        # Mock ffmpeg-dependent MP3 conversion
        def _fake_convert(self, wav_path):
            mp3_path = wav_path.with_suffix(".mp3")
            mp3_path.write_bytes(wav_path.read_bytes())
            return mp3_path

        with patch.object(type(creator), "_convert_to_mp3", _fake_convert):
            episode_paths = creator.create_audiobook_series()

            assert len(episode_paths) > 0, "No episodes created"

            for ep in episode_paths:
                assert ep.exists(), f"Episode {ep} does not exist"
                assert ep.stat().st_size > 0, f"Episode {ep} is empty"

    def test_qwen_synthesize_writes_valid_wav(self, tmp_path, qwen_mock_server):
        """Verify QwenTTSConfig.synthesize writes a file that pydub can decode."""
        from pydub import AudioSegment

        config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url=qwen_mock_server,
        )

        output_path = tmp_path / "test_segment.wav"
        success = config.synthesize("Hello world", output_path)

        assert success is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # This is the critical check: can pydub actually decode it?
        segment = AudioSegment.from_wav(output_path)
        assert len(segment) > 0, "AudioSegment is empty after decoding"


class TestQwenRetryBehavior:
    """Tests for retry and recovery logic in Qwen TTS synthesis."""

    @patch("time.sleep")
    def test_synthesize_retries_on_server_error(
        self, mock_sleep, tmp_path, flaky_qwen_server
    ):
        """Synthesis should retry on 5xx errors and eventually succeed."""
        config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url=flaky_qwen_server,
        )

        output_path = tmp_path / "retried.wav"
        # First 2 requests return 500, 3rd succeeds
        success = config.synthesize("Hello retry test", output_path)

        assert success is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_retries_on_timeout(self, mock_post, tmp_path):
        """Synthesis should retry on timeout then succeed."""
        import requests

        # First call times out, second succeeds
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = WAV_BYTES
        mock_post.side_effect = [
            requests.exceptions.Timeout("read timeout"),
            mock_response,
        ]

        config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url="http://fake:8890",
        )

        output_path = tmp_path / "timeout_retry.wav"
        # Patch time.sleep so the test doesn't actually wait
        with patch("time.sleep"):
            success = config.synthesize("Hello timeout", output_path)

        assert success is True
        assert output_path.exists()
        assert mock_post.call_count == 2

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_retries_on_connection_error(self, mock_post, tmp_path):
        """Synthesis should retry on connection error then succeed."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = WAV_BYTES
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.ConnectionError("Connection refused"),
            mock_response,
        ]

        config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url="http://fake:8890",
        )

        output_path = tmp_path / "conn_retry.wav"
        with patch("time.sleep"):
            success = config.synthesize("Hello connection", output_path)

        assert success is True
        assert mock_post.call_count == 3

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_fails_after_all_retries_exhausted(self, mock_post, tmp_path):
        """Synthesis should return False after all 5 retries fail."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url="http://fake:8890",
        )

        output_path = tmp_path / "all_fail.wav"
        with patch("time.sleep"):
            success = config.synthesize("Hello fail", output_path)

        assert success is False
        assert not output_path.exists()
        assert mock_post.call_count == 5  # 5 retries (default)

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_retries_on_client_error(self, mock_post, tmp_path):
        """4xx errors are also retried (server might have been temporarily confused)."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.HTTPError("400 Bad Request")
        )
        mock_post.return_value = mock_response

        config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url="http://fake:8890",
        )

        output_path = tmp_path / "client_err.wav"
        with patch("time.sleep"):
            success = config.synthesize("Hello 400", output_path)

        assert success is False
        assert mock_post.call_count == 5  # retried all 5 attempts

    @patch("audify.utils.api_config.requests.post")
    def test_synthesize_rejects_tiny_response(self, mock_post, tmp_path):
        """A response smaller than a WAV header should be retried."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"tiny"
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        config = QwenTTSConfig(
            voice="Vivian",
            language="en",
            base_url="http://fake:8890",
        )

        output_path = tmp_path / "tiny.wav"
        with patch("time.sleep"):
            success = config.synthesize("Hello tiny", output_path)

        assert success is False
        assert mock_post.call_count == 5  # retried all 5 attempts
