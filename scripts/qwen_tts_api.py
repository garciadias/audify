#!/usr/bin/env python3
"""
Simple FastAPI server for Qwen3-TTS.
Provides endpoints compatible with Audify's QwenTTSConfig.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any

# Import torch with fallback for mock mode
try:
    import torch
except ImportError:
    # Check if we're in mock mode via environment variable
    if os.getenv("QWEN_TTS_MOCK", "false").lower() in ("true", "1", "yes"):
        # Create a dummy torch module for mock mode
        import types

        torch = types.ModuleType("torch")
        torch.cuda = types.ModuleType("torch.cuda")
        torch.cuda.is_available = lambda: False
        torch.bfloat16 = "bfloat16"
        torch.float16 = "float16"
        torch.float32 = "float32"
        torch.is_tensor = lambda x: False
        # Add to sys.modules so other imports can find it
        sys.modules["torch"] = torch
        sys.modules["torch.cuda"] = torch.cuda
    else:
        raise

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_NAME = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEVICE = os.getenv("QWEN_TTS_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = os.getenv("QWEN_TTS_DTYPE", "bfloat16")
PORT = int(os.getenv("QWEN_TTS_PORT", "8890"))
HOST = os.getenv("QWEN_TTS_HOST", "0.0.0.0")
MOCK_MODE = os.getenv("QWEN_TTS_MOCK", "false").lower() in ("true", "1", "yes")

# Try to import Qwen3TTSModel
QWEN_TTS_AVAILABLE = False
if MOCK_MODE:
    logger.info("Running in mock mode - qwen-tts import will be skipped")
    QWEN_TTS_AVAILABLE = True
else:
    try:
        from qwen_tts import Qwen3TTSModel

        QWEN_TTS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"qwen-tts package not available: {e}")
        QWEN_TTS_AVAILABLE = False


class MockTTSModel:
    """Mock TTS model for testing."""

    def __init__(self):
        self.model = MockModel()

    def generate_custom_voice(
        self, text, language="Auto", speaker="Vivian", instruct=None
    ):
        """Generate mock audio."""
        import numpy as np
        import torch

        # Generate a simple sine wave (1 second at 440Hz)
        sample_rate = 24000
        duration = 1.0  # seconds
        frequency = 440.0  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

        # Ensure we return the same format as real model
        return [audio_data], sample_rate


class MockModel:
    """Mock inner model with speaker methods."""

    def get_supported_speakers(self):
        return ["Vivian", "TestSpeaker", "MockVoice"]

    def get_supported_languages(self):
        return ["Auto", "en", "zh"]


# Global model instance
tts_model = None

app = FastAPI(
    title="Qwen3-TTS API",
    description="Simple FastAPI wrapper for Qwen3-TTS models",
    version="0.1.0",
)


class TTSRequest(BaseModel):
    """Request model for TTS synthesis."""

    text: str
    language: Optional[str] = "Auto"
    speaker: Optional[str] = "Vivian"
    instruct: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Load the TTS model on startup."""
    global tts_model

    if not QWEN_TTS_AVAILABLE:
        logger.error(
            "qwen-tts package not installed. Please install with: pip install qwen-tts"
        )
        return

    if MOCK_MODE:
        logger.info("Mock mode enabled - using mock model")
        tts_model = MockTTSModel()
        return

    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Device: {DEVICE}, Dtype: {DTYPE}")

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(DTYPE.lower(), torch.bfloat16)

        tts_model = Qwen3TTSModel.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE,
            dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        tts_model = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if tts_model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": "Model not loaded"
            if QWEN_TTS_AVAILABLE
            else "qwen-tts package not installed",
        }

    # Check if model seems ready
    try:
        # Simple check: try to get supported languages
        if hasattr(tts_model.model, "get_supported_languages"):
            _ = tts_model.model.get_supported_languages()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model": MODEL_NAME,
            "device": DEVICE,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e),
        }


@app.get("/voices")
async def get_voices():
    """Get available voices."""
    if tts_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        if hasattr(tts_model.model, "get_supported_speakers"):
            voices = tts_model.model.get_supported_speakers()
            return {"voices": voices}
        else:
            # Default voices
            return {"voices": ["Vivian"]}
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        return {"voices": ["Vivian"]}


@app.post("/tts")
async def synthesize_speech(request: TTSRequest):
    """Synthesize text to speech."""
    if tts_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Generate audio
        wavs, sample_rate = tts_model.generate_custom_voice(
            text=request.text,
            language=request.language or "Auto",
            speaker=request.speaker or "Vivian",
            instruct=request.instruct,
        )

        # Convert to WAV bytes
        import numpy as np
        import soundfile as sf
        import io

        # wavs is a list of numpy arrays
        if isinstance(wavs, list) and len(wavs) > 0:
            audio_data = wavs[0]
        else:
            audio_data = wavs

        # Ensure audio data is numpy array
        if torch.is_tensor(audio_data):
            audio_data = audio_data.cpu().numpy()

        # Normalize audio
        audio_data = np.asarray(audio_data, dtype=np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val

        # Write to WAV bytes
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, audio_data, sample_rate, format="WAV")
        wav_bytes.seek(0)

        return Response(
            content=wav_bytes.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}",
        )


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Qwen3-TTS API",
        "version": "0.1.0",
        "model": MODEL_NAME if tts_model else "Not loaded",
        "status": "healthy" if tts_model else "unhealthy",
        "endpoints": ["/health", "/voices", "/tts", "/docs"],
    }


def main():
    """Main entry point."""
    if not QWEN_TTS_AVAILABLE:
        logger.error("qwen-tts package is not installed.")
        logger.error("Install it with: pip install qwen-tts")
        sys.exit(1)

    logger.info(f"Starting Qwen3-TTS API server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    if MOCK_MODE:
        logger.info("Running in MOCK mode - no real TTS model will be used")

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
