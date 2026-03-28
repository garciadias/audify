"""Tests for the Audify REST API routes."""

import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from audify.api.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health():
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /providers
# ---------------------------------------------------------------------------


def test_list_providers():
    """GET /providers returns a list of provider names."""
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert isinstance(data["providers"], list)
    assert len(data["providers"]) > 0


# ---------------------------------------------------------------------------
# /voices
# ---------------------------------------------------------------------------


def test_list_voices_success():
    """GET /voices returns voices for a provider."""
    mock_config = MagicMock()
    mock_config.get_available_voices.return_value = ["voice_a", "voice_b"]

    with patch("audify.api.app.get_tts_config", return_value=mock_config):
        response = client.get("/voices?provider=kokoro&language=en")

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "kokoro"
    assert data["voices"] == ["voice_a", "voice_b"]


def test_list_voices_invalid_provider():
    """GET /voices returns 400 for unknown provider."""
    with patch("audify.api.app.get_tts_config", side_effect=ValueError("Unknown")):
        response = client.get("/voices?provider=unknown")

    assert response.status_code == 400


def test_list_voices_server_error():
    """GET /voices returns 500 on unexpected error."""
    with patch("audify.api.app.get_tts_config", side_effect=RuntimeError("boom")):
        response = client.get("/voices")

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve voices"


# ---------------------------------------------------------------------------
# /synthesize helpers
# ---------------------------------------------------------------------------


def _epub_upload(name: str = "book.epub") -> dict:
    """Return a files dict suitable for multipart upload."""
    return {"file": (name, io.BytesIO(b"fake epub content"), "application/epub+zip")}


def _pdf_upload(name: str = "doc.pdf") -> dict:
    return {"file": (name, io.BytesIO(b"fake pdf content"), "application/pdf")}


# ---------------------------------------------------------------------------
# /synthesize
# ---------------------------------------------------------------------------


def test_synthesize_epub_happy_path(tmp_path):
    """POST /synthesize with EPUB returns a FileResponse."""
    output_mp3 = tmp_path / "result.mp3"
    output_mp3.write_bytes(b"fake mp3")

    mock_synth = MagicMock()
    mock_synth.synthesize.return_value = output_mp3

    with patch("audify.api.app.os.remove"):
        with patch("audify.text_to_speech.EpubSynthesizer", return_value=mock_synth):
            response = client.post(
                "/synthesize",
                files=_epub_upload(),
                data={"voice": "af_bella", "language": "en"},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")


def test_synthesize_pdf_happy_path(tmp_path):
    """POST /synthesize with PDF returns a FileResponse."""
    output_mp3 = tmp_path / "result.mp3"
    output_mp3.write_bytes(b"fake mp3")

    mock_synth = MagicMock()
    mock_synth.synthesize.return_value = output_mp3

    with patch("audify.api.app.os.remove"):
        with patch("audify.text_to_speech.PdfSynthesizer", return_value=mock_synth):
            response = client.post(
                "/synthesize",
                files=_pdf_upload(),
                data={"voice": "af_bella", "language": "en"},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")


def test_synthesize_invalid_extension():
    """POST /synthesize rejects unsupported file types."""
    response = client.post(
        "/synthesize",
        files={"file": ("book.txt", io.BytesIO(b"text"), "text/plain")},
        data={"voice": "af_bella"},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_synthesize_server_error():
    """POST /synthesize returns 500 when synthesis raises unexpectedly."""
    mock_synth = MagicMock()
    mock_synth.synthesize.side_effect = RuntimeError("TTS crashed")

    with patch("audify.text_to_speech.EpubSynthesizer", return_value=mock_synth):
        response = client.post(
            "/synthesize",
            files=_epub_upload(),
            data={"voice": "af_bella"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Synthesis failed"


def test_synthesize_invalid_params():
    """POST /synthesize returns 400 when ValueError is raised."""
    mock_synth = MagicMock()
    mock_synth.synthesize.side_effect = ValueError("bad voice")

    with patch("audify.text_to_speech.EpubSynthesizer", return_value=mock_synth):
        response = client.post(
            "/synthesize",
            files=_epub_upload(),
            data={"voice": "bad_voice"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request parameters"


# ---------------------------------------------------------------------------
# /audiobook
# ---------------------------------------------------------------------------


def test_audiobook_epub_happy_path(tmp_path):
    """POST /audiobook with EPUB returns an M4B FileResponse."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    m4b_file = output_dir / "book.m4b"
    m4b_file.write_bytes(b"fake m4b")

    mock_creator = MagicMock()
    mock_creator.synthesize.return_value = output_dir

    with patch("audify.api.app.os.remove"):
        with patch(
            "audify.audiobook_creator.AudiobookEpubCreator",
            return_value=mock_creator,
        ):
            response = client.post(
                "/audiobook",
                files=_epub_upload(),
                data={"voice": "af_bella", "language": "en"},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mp4")


def test_audiobook_pdf_happy_path(tmp_path):
    """POST /audiobook with PDF returns an M4B FileResponse."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    m4b_file = output_dir / "book.m4b"
    m4b_file.write_bytes(b"fake m4b")

    mock_creator = MagicMock()
    mock_creator.synthesize.return_value = output_dir

    with patch("audify.api.app.os.remove"):
        with patch(
            "audify.audiobook_creator.AudiobookPdfCreator",
            return_value=mock_creator,
        ):
            response = client.post(
                "/audiobook",
                files=_pdf_upload(),
                data={"voice": "af_bella", "language": "en"},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mp4")


def test_audiobook_no_m4b_produced(tmp_path):
    """POST /audiobook returns 500 when no M4B file is produced."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # No .m4b file created

    mock_creator = MagicMock()
    mock_creator.synthesize.return_value = output_dir

    with patch(
        "audify.audiobook_creator.AudiobookEpubCreator",
        return_value=mock_creator,
    ):
        response = client.post(
            "/audiobook",
            files=_epub_upload(),
            data={"voice": "af_bella"},
        )

    assert response.status_code == 500
    assert "no M4B" in response.json()["detail"]


def test_audiobook_server_error():
    """POST /audiobook returns 500 when creator raises unexpectedly."""
    mock_creator = MagicMock()
    mock_creator.synthesize.side_effect = RuntimeError("LLM crashed")

    with patch(
        "audify.audiobook_creator.AudiobookEpubCreator",
        return_value=mock_creator,
    ):
        response = client.post(
            "/audiobook",
            files=_epub_upload(),
            data={"voice": "af_bella"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Audiobook creation failed"


def test_audiobook_invalid_params():
    """POST /audiobook returns 400 when ValueError is raised."""
    mock_creator = MagicMock()
    mock_creator.synthesize.side_effect = ValueError("bad language")

    with patch(
        "audify.audiobook_creator.AudiobookEpubCreator",
        return_value=mock_creator,
    ):
        response = client.post(
            "/audiobook",
            files=_epub_upload(),
            data={"voice": "af_bella"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request parameters"


def test_audiobook_invalid_extension():
    """POST /audiobook rejects unsupported file types."""
    response = client.post(
        "/audiobook",
        files={"file": ("book.docx", io.BytesIO(b"docx"), "application/octet-stream")},
        data={"voice": "af_bella"},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
