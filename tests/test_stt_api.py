"""Server-side tests for ``scripts/stt_api.py`` in mock mode.

Uses FastAPI's ``TestClient`` with ``STT_MOCK=true`` to exercise the
endpoint contracts without needing the faster-whisper model or a GPU.
The mock branch is what the docker-compose healthcheck and the
``WhisperSTTClient`` retry tests rely on, so it carries the contract.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest
from fastapi.testclient import TestClient


def _load_module() -> ModuleType:
    """Import ``scripts/stt_api.py`` once with ``STT_MOCK=true`` in place."""
    os.environ["STT_MOCK"] = "true"
    os.environ["STT_MOCK_TRANSCRIPT"] = "mock transcript"

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "stt_api.py"

    spec = importlib.util.spec_from_file_location("stt_api_under_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stt_api_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stt_module() -> ModuleType:
    return _load_module()


@pytest.fixture
def client(stt_module: ModuleType) -> Iterator[TestClient]:
    with TestClient(stt_module.app) as c:
        yield c


def test_health_reports_loaded_in_mock_mode(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["mock"] is True


def test_transcribe_returns_canned_text(client: TestClient) -> None:
    files = {"audio": ("clip.wav", b"RIFF\x00\x00\x00\x00WAVEpayload", "audio/wav")}
    response = client.post("/transcribe", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "mock transcript"
    assert body["language"] == "en"


def test_transcribe_echoes_offsets_in_mock_mode(client: TestClient) -> None:
    files = {"audio": ("clip.wav", b"RIFF\x00\x00\x00\x00WAVEpayload", "audio/wav")}
    response = client.post(
        "/transcribe",
        files=files,
        data={"start_s": "2.5", "end_s": "12.5", "language": "fr"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "fr"
    assert body["start_s"] == 2.5
    assert body["end_s"] == 12.5
    assert body["duration_s"] == 10.0


def test_transcribe_rejects_inverted_offsets(client: TestClient) -> None:
    files = {"audio": ("clip.wav", b"RIFF\x00\x00\x00\x00WAVEpayload", "audio/wav")}
    response = client.post(
        "/transcribe",
        files=files,
        data={"start_s": "10.0", "end_s": "5.0"},
    )
    assert response.status_code == 400


def test_transcribe_rejects_negative_offsets(client: TestClient) -> None:
    files = {"audio": ("clip.wav", b"RIFF\x00\x00\x00\x00WAVEpayload", "audio/wav")}
    for data in ({"start_s": "-1.0"}, {"end_s": "-1.0"}):
        response = client.post("/transcribe", files=files, data=data)
        assert response.status_code == 400


def test_transcribe_rejects_empty_audio(client: TestClient) -> None:
    files = {"audio": ("clip.wav", b"", "audio/wav")}
    response = client.post("/transcribe", files=files)
    assert response.status_code == 400


def test_transcribe_returns_503_when_model_not_loaded(
    stt_module: ModuleType,
) -> None:
    """If the lifespan never ran (no ``TestClient`` context), transcribe 503s."""
    stt_module._state["model_loaded"] = False
    try:
        with TestClient(stt_module.app, raise_server_exceptions=False) as _:
            # Force the unloaded state inside the lifespan-managed client.
            stt_module._state["model_loaded"] = False
            response = _.post(
                "/transcribe",
                files={"audio": ("c.wav", b"abc", "audio/wav")},
            )
            assert response.status_code == 503
    finally:
        # Reset so other tests in this module observe a loaded model.
        stt_module._state["model_loaded"] = True
