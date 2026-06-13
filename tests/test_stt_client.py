"""Unit tests for :mod:`audify.qa.stt`.

The client is exercised against a tiny in-process HTTP server (same
pattern as ``tests/test_qwen_tts_api_integration.py``) so we cover the
multipart upload, form fields, retry behaviour, and 4xx vs 5xx branching
without spinning up the real faster-whisper container.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from audify.qa.stt import (
    FakeSTTClient,
    STTClient,
    STTServiceError,
    WhisperSTTClient,
)


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake-payload")
    return audio


def _make_handler(
    response_sequence: list[tuple[int, dict]],
    captured: list[dict],
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler that returns ``response_sequence[i]`` for the
    i-th POST and records the parsed form data into ``captured``.
    """

    class _Handler(BaseHTTPRequestHandler):
        _index = 0

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
            if self.path != "/transcribe":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            captured.append({"body": raw, "content_type": content_type})

            idx = type(self)._index
            type(self)._index = idx + 1
            if idx >= len(response_sequence):
                status, payload = response_sequence[-1]
            else:
                status, payload = response_sequence[idx]

            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _Handler


@pytest.fixture
def stt_server() -> Iterator[
    "tuple[str, list[tuple[int, dict]], list[dict]]"
]:
    """Start an HTTP server on a free localhost port.

    Test mutates ``response_sequence`` *before* the first request to set up
    the response programme. ``captured`` collects parsed POST bodies for
    assertion.
    """
    response_sequence: list[tuple[int, dict]] = []
    captured: list[dict] = []
    handler_cls = _make_handler(response_sequence, captured)
    httpd = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield (f"http://127.0.0.1:{port}", response_sequence, captured)
    finally:
        httpd.shutdown()


def test_protocol_membership() -> None:
    """WhisperSTTClient and FakeSTTClient are both ``STTClient`` instances."""
    assert isinstance(
        WhisperSTTClient(base_url="http://localhost:8888"), STTClient
    )
    assert isinstance(FakeSTTClient("hello"), STTClient)


def test_transcribe_returns_text(stt_server, wav_file: Path) -> None:
    base_url, responses, captured = stt_server
    responses.append(
        (200, {"text": "hello world", "language": "en", "duration_s": 1.0})
    )

    client = WhisperSTTClient(base_url=base_url, timeout_s=5.0, max_retries=2)
    assert client.transcribe(wav_file, language="en") == "hello world"

    assert len(captured) == 1
    assert "multipart/form-data" in captured[0]["content_type"]


def test_transcribe_passes_start_end_language_form_fields(
    stt_server, wav_file: Path
) -> None:
    base_url, responses, captured = stt_server
    responses.append((200, {"text": "ok", "language": "en", "duration_s": 0.5}))

    client = WhisperSTTClient(base_url=base_url, timeout_s=5.0, max_retries=2)
    client.transcribe(wav_file, start_s=1.5, end_s=11.5, language="fr")

    body = captured[0]["body"].decode("utf-8", errors="replace")
    assert "name=\"start_s\"" in body
    assert "1.500000" in body
    assert "name=\"end_s\"" in body
    assert "11.500000" in body
    assert "name=\"language\"" in body
    assert "fr" in body
    assert "name=\"audio\"" in body


def test_retries_on_5xx_then_succeeds(stt_server, wav_file: Path) -> None:
    base_url, responses, _ = stt_server
    responses.extend(
        [
            (503, {"detail": "model loading"}),
            (500, {"detail": "transient"}),
            (200, {"text": "third time lucky", "language": "en", "duration_s": 1.0}),
        ]
    )

    client = WhisperSTTClient(base_url=base_url, timeout_s=5.0, max_retries=3)
    assert client.transcribe(wav_file) == "third time lucky"


def test_does_not_retry_on_4xx(stt_server, wav_file: Path) -> None:
    base_url, responses, captured = stt_server
    responses.append((400, {"detail": "bad offsets"}))

    client = WhisperSTTClient(base_url=base_url, timeout_s=5.0, max_retries=3)
    with pytest.raises(STTServiceError, match="400"):
        client.transcribe(wav_file, start_s=10.0, end_s=5.0)

    assert len(captured) == 1, "client must not retry 4xx"


def test_raises_service_error_after_exhausted_retries(
    stt_server, wav_file: Path
) -> None:
    base_url, responses, captured = stt_server
    responses.extend(
        [
            (502, {"detail": "bad gateway"}),
            (502, {"detail": "bad gateway"}),
            (502, {"detail": "bad gateway"}),
        ]
    )

    client = WhisperSTTClient(base_url=base_url, timeout_s=5.0, max_retries=3)
    with pytest.raises(STTServiceError):
        client.transcribe(wav_file)

    assert len(captured) == 3


def test_raises_filenotfound_for_missing_audio(tmp_path: Path) -> None:
    client = WhisperSTTClient(base_url="http://localhost:1", timeout_s=1.0)
    with pytest.raises(FileNotFoundError):
        client.transcribe(tmp_path / "does-not-exist.wav")


def test_fake_client_returns_canned_transcripts(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00")

    fake = FakeSTTClient(["head text", "tail text"])
    assert fake.transcribe(audio, start_s=0.0, end_s=10.0) == "head text"
    assert fake.transcribe(audio, start_s=50.0, end_s=60.0) == "tail text"

    with pytest.raises(AssertionError):
        fake.transcribe(audio)

    assert fake.calls[0]["start_s"] == 0.0
    assert fake.calls[1]["end_s"] == 60.0


def test_fake_client_callable_form(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00")

    def transcript_fn(**kwargs: object) -> str:
        return f"window {kwargs['start_s']}-{kwargs['end_s']}"

    fake = FakeSTTClient(transcript_fn)
    assert fake.transcribe(audio, start_s=0.0, end_s=10.0) == "window 0.0-10.0"


def test_fake_client_static_string(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00")

    fake = FakeSTTClient("always the same")
    assert fake.transcribe(audio) == "always the same"
    assert fake.transcribe(audio, language="fr") == "always the same"
