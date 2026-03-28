"""
Audify REST API

Exposes the core Audify functionality (TTS synthesis and LLM-powered audiobook
creation) as a Docker-deployable FastAPI service.

Endpoints
---------
GET  /health            – liveness probe
GET  /voices            – list voices available from the configured TTS provider
GET  /providers         – list all supported TTS providers
POST /synthesize        – convert an uploaded EPUB/PDF to MP3 (simple TTS)
POST /audiobook         – convert an uploaded EPUB/PDF to an M4B audiobook via LLM
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from audify.utils.api_config import get_tts_config
from audify.utils.constants import (
    AVAILABLE_TTS_PROVIDERS,
    DEFAULT_SPEAKER,
    DEFAULT_TTS_PROVIDER,
    OLLAMA_API_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
)
from audify.utils.logging_utils import setup_logging

logger = setup_logging(module_name=__name__)

app = FastAPI(
    title="Audify API",
    description=(
        "Convert ebooks (EPUB, PDF, TXT) into audiobooks using Kokoro TTS "
        "and LLM-powered script generation."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
def health() -> JSONResponse:
    """Liveness probe — always returns 200 when the server is up."""
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Metadata endpoints
# ---------------------------------------------------------------------------


@app.get("/providers", tags=["metadata"])
def list_providers() -> JSONResponse:
    """Return all supported TTS provider names."""
    return JSONResponse({"providers": AVAILABLE_TTS_PROVIDERS})


@app.get("/voices", tags=["metadata"])
def list_voices(
    provider: str = DEFAULT_TTS_PROVIDER,
    language: str = "en",
) -> JSONResponse:
    """Return available voices for *provider*.

    Query parameters
    ----------------
    provider : str
        One of the values returned by ``GET /providers`` (default: the value of
        the ``TTS_PROVIDER`` environment variable, which defaults to ``kokoro``).
    language : str
        BCP-47 language code such as ``en``, ``es``, ``fr`` (default: ``en``).
    """
    try:
        tts_config = get_tts_config(provider=provider, language=language)
        voices = tts_config.get_available_voices()
        return JSONResponse(
            {"provider": provider, "language": language, "voices": voices}
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Error listing voices for provider '{provider}': {exc}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve voices"
        ) from exc


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    """Write *upload* to *dest_dir* and return its path."""
    suffix = Path(upload.filename or "upload").suffix or ".bin"
    dest = dest_dir / f"input{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _validate_extension(path: Path, allowed: tuple) -> None:
    if path.suffix.lower() not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{path.suffix}'. "
                f"Allowed: {', '.join(allowed)}"
            ),
        )


# ---------------------------------------------------------------------------
# /synthesize  –  simple TTS (no LLM)
# ---------------------------------------------------------------------------


@app.post("/synthesize", tags=["synthesis"])
def synthesize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="EPUB or PDF file to synthesize"),
    voice: str = Form(DEFAULT_SPEAKER, description="Voice ID"),
    language: Optional[str] = Form(None, description="Override language (e.g. 'en')"),
    tts_provider: str = Form(DEFAULT_TTS_PROVIDER, description="TTS provider name"),
    translate: Optional[str] = Form(
        None, description="Translate to this language before synthesis"
    ),
) -> FileResponse:
    """Convert an uploaded EPUB or PDF file to an MP3 audio file.

    The response is the MP3 file as a binary download.
    """
    with tempfile.TemporaryDirectory(prefix="audify_api_synth_") as tmp:
        tmp_path = Path(tmp)
        input_path = _save_upload(file, tmp_path)
        _validate_extension(input_path, (".epub", ".pdf"))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        try:
            from audify.text_to_speech import (
                BaseSynthesizer,
                EpubSynthesizer,
                PdfSynthesizer,
            )

            synth: BaseSynthesizer
            ext = input_path.suffix.lower()
            if ext == ".epub":
                synth = EpubSynthesizer(
                    path=input_path,
                    language=language,
                    speaker=voice,
                    translate=translate,
                    output_dir=output_dir,
                    tts_provider=tts_provider,
                )
            else:
                synth = PdfSynthesizer(
                    pdf_path=input_path,
                    language=language or "en",
                    speaker=voice,
                    translate=translate,
                    output_dir=output_dir,
                    tts_provider=tts_provider,
                )

            result_path = synth.synthesize()

            if not result_path.exists():
                raise HTTPException(
                    status_code=500, detail="Synthesis produced no output file"
                )

            # Copy result out before temp directory is cleaned up
            permanent_tmp = tempfile.NamedTemporaryFile(
                suffix=result_path.suffix, delete=False
            )
            shutil.copy(result_path, permanent_tmp.name)
            permanent_tmp.close()
            background_tasks.add_task(os.remove, permanent_tmp.name)

            return FileResponse(
                path=permanent_tmp.name,
                media_type="audio/mpeg",
                filename=result_path.name,
                background=background_tasks,
            )

        except HTTPException:
            raise
        except ValueError as exc:
            logger.warning(f"Invalid synthesis request: {exc}")
            raise HTTPException(
                status_code=400, detail="Invalid request parameters"
            ) from exc
        except Exception as exc:
            logger.error(f"Synthesis failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Synthesis failed") from exc


# ---------------------------------------------------------------------------
# /audiobook  –  LLM-powered M4B creation
# ---------------------------------------------------------------------------


@app.post("/audiobook", tags=["synthesis"])
def create_audiobook(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="EPUB or PDF file"),
    voice: str = Form(DEFAULT_SPEAKER, description="Voice ID"),
    language: Optional[str] = Form(None, description="Override language"),
    tts_provider: str = Form(DEFAULT_TTS_PROVIDER, description="TTS provider name"),
    llm_model: str = Form(
        OLLAMA_DEFAULT_MODEL, description="LLM model for script generation"
    ),
    llm_base_url: str = Form(OLLAMA_API_BASE_URL, description="Ollama base URL"),
    translate: Optional[str] = Form(None, description="Translate to this language"),
    max_chapters: Optional[int] = Form(
        None, description="Limit number of chapters (useful for testing)"
    ),
) -> FileResponse:
    """Convert an uploaded EPUB or PDF to an M4B audiobook using LLM script generation.

    The response is the M4B audiobook file as a binary download.
    """
    with tempfile.TemporaryDirectory(prefix="audify_api_audiobook_") as tmp:
        tmp_path = Path(tmp)
        input_path = _save_upload(file, tmp_path)
        _validate_extension(input_path, (".epub", ".pdf"))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        try:
            from audify.audiobook_creator import (
                AudiobookCreator,
                AudiobookEpubCreator,
                AudiobookPdfCreator,
            )

            creator: AudiobookCreator
            ext = input_path.suffix.lower()
            if ext == ".epub":
                creator = AudiobookEpubCreator(
                    path=input_path,
                    language=language,
                    voice=voice,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    translate=translate,
                    max_chapters=max_chapters,
                    confirm=False,
                    output_dir=output_dir,
                    tts_provider=tts_provider,
                )
            else:
                creator = AudiobookPdfCreator(
                    path=input_path,
                    language=language,
                    voice=voice,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    translate=translate,
                    confirm=False,
                    output_dir=output_dir,
                    tts_provider=tts_provider,
                )

            result_path = creator.synthesize()

            # Find the M4B inside the result directory
            m4b_files = list(result_path.glob("*.m4b"))
            if not m4b_files:
                raise HTTPException(
                    status_code=500,
                    detail="Audiobook creation produced no M4B file",
                )
            m4b_path = m4b_files[0]

            permanent_tmp = tempfile.NamedTemporaryFile(suffix=".m4b", delete=False)
            shutil.copy(m4b_path, permanent_tmp.name)
            permanent_tmp.close()
            background_tasks.add_task(os.remove, permanent_tmp.name)

            return FileResponse(
                path=permanent_tmp.name,
                media_type="audio/mp4",
                filename=m4b_path.name,
                background=background_tasks,
            )

        except HTTPException:
            raise
        except ValueError as exc:
            logger.warning(f"Invalid audiobook request: {exc}")
            raise HTTPException(
                status_code=400, detail="Invalid request parameters"
            ) from exc
        except Exception as exc:
            logger.error(f"Audiobook creation failed: {exc}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Audiobook creation failed"
            ) from exc
