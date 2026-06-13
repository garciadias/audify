#!/usr/bin/env python3
"""FastAPI server wrapping faster-whisper for boundary-sampling STT.

Exposes ``/health`` (probed by the docker-compose healthcheck) and
``/transcribe`` (consumed by ``audify.qa.stt.WhisperSTTClient``). The
``STT_MOCK=true`` env var short-circuits the model load and returns a
deterministic transcript so tests can spin up the server without a GPU or
the ~3 GB model download.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("STT_MODEL", "large-v3")
DEVICE = os.getenv("STT_DEVICE", "cuda:0")
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")
HOST = os.getenv("STT_HOST", "127.0.0.1")
PORT = int(os.getenv("STT_PORT", "8888"))
MOCK_MODE = os.getenv("STT_MOCK", "false").lower() in ("true", "1", "yes")
MOCK_TRANSCRIPT = os.getenv(
    "STT_MOCK_TRANSCRIPT",
    "the quick brown fox jumps over the lazy dog",
)
DOWNLOAD_ROOT = os.getenv("STT_DOWNLOAD_ROOT", "/models")

# State shared between lifespan + handlers.
_state: dict = {"model": None, "model_loaded": False, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load faster-whisper once at startup; expose status via /health."""
    if MOCK_MODE:
        logger.warning("STT_MOCK=true; skipping model load and serving canned transcripts.")
        _state["model_loaded"] = True
        yield
        _state["model_loaded"] = False
        return

    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error("faster-whisper is not installed: %s", e)
        _state["load_error"] = f"faster-whisper not installed: {e}"
        yield
        return

    try:
        logger.info(
            "Loading faster-whisper model '%s' on %s (compute_type=%s, cache=%s)",
            MODEL_NAME,
            DEVICE,
            COMPUTE_TYPE,
            DOWNLOAD_ROOT,
        )
        device, _, index = DEVICE.partition(":")
        kwargs: dict = {"compute_type": COMPUTE_TYPE}
        if Path(DOWNLOAD_ROOT).exists():
            kwargs["download_root"] = DOWNLOAD_ROOT
        if index:
            kwargs["device_index"] = int(index)
        _state["model"] = WhisperModel(MODEL_NAME, device=device, **kwargs)
        _state["model_loaded"] = True
        logger.info("faster-whisper model loaded.")
    except Exception as e:  # noqa: BLE001 — surface load errors via /health
        logger.error("Failed to load faster-whisper model: %s", e, exc_info=True)
        _state["load_error"] = str(e)

    yield

    _state["model"] = None
    _state["model_loaded"] = False


app = FastAPI(
    title="Audify STT API",
    description="faster-whisper wrapper for boundary-sampling fidelity checks.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> JSONResponse:
    """Readiness probe. 200 once the model is loaded, 503 otherwise."""
    payload = {
        "model_loaded": bool(_state["model_loaded"]),
        "model": MODEL_NAME,
        "device": DEVICE,
        "mock": MOCK_MODE,
    }
    if _state["load_error"]:
        payload["error"] = _state["load_error"]
    code = (
        status.HTTP_200_OK
        if _state["model_loaded"]
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=code, content=payload)


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    start_s: Optional[float] = Form(None),
    end_s: Optional[float] = Form(None),
    language: Optional[str] = Form(None),
) -> JSONResponse:
    """Transcribe an uploaded audio segment.

    ``start_s`` and ``end_s`` are optional. When provided, the server slices
    the uploaded audio with ffmpeg before handing it to whisper so cycle-3
    only pays for the boundary window, not the full episode.
    """
    if not _state["model_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )

    if start_s is not None and end_s is not None and end_s <= start_s:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_s must be greater than start_s",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        upload_path = tmp / f"upload{suffix}"
        upload_bytes = await audio.read()
        if not upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio is empty",
            )
        upload_path.write_bytes(upload_bytes)

        if MOCK_MODE:
            return JSONResponse(
                content={
                    "text": MOCK_TRANSCRIPT,
                    "language": language or "en",
                    "duration_s": (
                        float(end_s - start_s)
                        if start_s is not None and end_s is not None
                        else 0.0
                    ),
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )

        target_path = (
            _slice_with_ffmpeg(upload_path, tmp, start_s, end_s)
            if (start_s is not None or end_s is not None)
            else upload_path
        )

        try:
            segments, info = _state["model"].transcribe(
                str(target_path),
                language=language,
                beam_size=5,
                vad_filter=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return JSONResponse(
                content={
                    "text": text,
                    "language": info.language,
                    "duration_s": float(info.duration),
                }
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Transcription failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"transcription failed: {e}",
            ) from e


def _slice_with_ffmpeg(
    source: Path, tmpdir: Path, start_s: Optional[float], end_s: Optional[float]
) -> Path:
    """Cut ``[start_s, end_s]`` out of ``source`` into a 16 kHz mono WAV.

    Whisper expects mono 16 kHz; the conversion happens here so the model
    never has to resample. Returns the source path unchanged when ffmpeg is
    not available — degrades gracefully so tests on minimal environments
    still exercise the upload path.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not on PATH; passing full upload to whisper.")
        return source

    sliced = tmpdir / "slice.wav"
    cmd = [ffmpeg, "-loglevel", "error", "-y"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.3f}"]
    if end_s is not None and start_s is not None:
        cmd += ["-t", f"{(end_s - start_s):.3f}"]
    elif end_s is not None:
        cmd += ["-to", f"{end_s:.3f}"]
    cmd += ["-i", str(source), "-ar", "16000", "-ac", "1", str(sliced)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"ffmpeg slice failed: {stderr or e}",
        ) from e
    return sliced


def main() -> None:
    logger.info("Starting Audify STT API on %s:%s", HOST, PORT)
    logger.info("Model: %s, device: %s, mock: %s", MODEL_NAME, DEVICE, MOCK_MODE)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    sys.exit(main())
