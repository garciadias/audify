"""End-to-end round-trip: text → Kokoro TTS → STT → text.

Gated on ``RUN_INTEGRATION=1`` so the standard ``task test`` run on
machines without docker/GPU keeps passing. To run locally:

.. code-block:: shell

    task up   # docker compose --profile stt --profile kokoro --profile ollama up
    RUN_INTEGRATION=1 uv run pytest tests/integration/test_stt_roundtrip.py -v

Asserts that the transcript's WER vs the canonical input sentence is
below ``0.15``. ``large-v3`` on synthesized speech sits well under that;
the cushion absorbs voice quirks and ASR punctuation differences.
"""

from __future__ import annotations

import io
import os
import re
import wave
from pathlib import Path

import pytest
import requests

from audify.qa.stt import WhisperSTTClient

CANONICAL_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the river bank."
)

KOKORO_BASE_URL = os.getenv("KOKORO_BASE_URL", "http://localhost:8887/v1")
STT_BASE_URL = os.getenv("STT_BASE_URL", "http://localhost:8888")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_bella")


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 with docker stack up to run",
)


def _normalize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein-distance-based WER over normalized word lists."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    prev_row = list(range(len(hyp) + 1))
    for i, r_word in enumerate(ref, start=1):
        curr_row = [i]
        for j, h_word in enumerate(hyp, start=1):
            ins = curr_row[j - 1] + 1
            dele = prev_row[j] + 1
            sub = prev_row[j - 1] + (0 if r_word == h_word else 1)
            curr_row.append(min(ins, dele, sub))
        prev_row = curr_row
    return prev_row[-1] / len(ref)


def _synthesize_with_kokoro(sentence: str, out_path: Path) -> Path:
    """Call Kokoro's OpenAI-compatible speech endpoint and save WAV bytes."""
    response = requests.post(
        f"{KOKORO_BASE_URL}/audio/speech",
        json={
            "model": "kokoro",
            "input": sentence,
            "voice": KOKORO_VOICE,
            "response_format": "wav",
        },
        timeout=60,
    )
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


@pytest.fixture
def kokoro_clip(tmp_path: Path) -> Path:
    out = tmp_path / "kokoro.wav"
    _synthesize_with_kokoro(CANONICAL_SENTENCE, out)
    assert out.stat().st_size > 0
    # Quick sanity check: file is a real WAV.
    with wave.open(io.BytesIO(out.read_bytes())) as wf:
        assert wf.getnframes() > 0
    return out


def test_roundtrip_wer_under_15_percent(kokoro_clip: Path) -> None:
    client = WhisperSTTClient(base_url=STT_BASE_URL, timeout_s=120.0)
    transcript = client.transcribe(kokoro_clip, language="en")
    assert transcript.strip(), "transcript came back empty"

    wer = _word_error_rate(CANONICAL_SENTENCE, transcript)
    assert wer < 0.15, (
        f"WER={wer:.3f} between transcript and canonical sentence "
        f"exceeds 0.15 tolerance.\nCanonical: {CANONICAL_SENTENCE!r}\n"
        f"Transcript: {transcript!r}"
    )


def test_roundtrip_with_offsets(kokoro_clip: Path) -> None:
    """Slicing the audio at the server doesn't break the round-trip."""
    client = WhisperSTTClient(base_url=STT_BASE_URL, timeout_s=120.0)
    # Whole-clip slice — should match the no-offset transcript closely.
    transcript = client.transcribe(
        kokoro_clip, start_s=0.0, end_s=10.0, language="en"
    )
    assert transcript.strip()
