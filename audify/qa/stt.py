"""STT client for the boundary-sampling fidelity check.

The cycle-3 detector (issue #38) calls :meth:`STTClient.transcribe` with a
short audio window and an optional language hint. Two implementations:

* :class:`WhisperSTTClient` — HTTPs the docker-compose STT service from
  issue #36 (``scripts/stt_api.py`` running ``faster-whisper-large``).
* :class:`FakeSTTClient` — deterministic transcripts for tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Union, runtime_checkable

import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class STTServiceError(RuntimeError):
    """Raised when the STT service fails after exhausting retries."""


@runtime_checkable
class STTClient(Protocol):
    """Boundary-sampling STT client interface.

    A single ``transcribe`` call returns the text of one audio window.
    Implementations are expected to be thread-safe enough for sequential
    use from a single graph node — concurrent calls are not required.
    """

    def transcribe(
        self,
        audio_path: Path,
        *,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        language: Optional[str] = None,
    ) -> str:
        """Return the transcription of ``audio_path`` (or a slice of it)."""
        ...


_TRANSIENT_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class _ServerError(RuntimeError):
    """Internal — wraps 5xx so tenacity treats it like a transient error."""


class WhisperSTTClient:
    """HTTP client for the docker-compose ``stt`` service.

    Retries on connection errors, timeouts, and 5xx responses with
    exponential backoff. Does **not** retry on 4xx — those indicate a
    client-side mistake (missing audio, bad offsets) and surface as
    :class:`STTServiceError` immediately.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8888",
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def transcribe(
        self,
        audio_path: Path,
        *,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        language: Optional[str] = None,
    ) -> str:
        if not audio_path.exists():
            raise FileNotFoundError(f"audio_path does not exist: {audio_path}")

        data: dict[str, str] = {}
        if start_s is not None:
            data["start_s"] = f"{start_s:.6f}"
        if end_s is not None:
            data["end_s"] = f"{end_s:.6f}"
        if language is not None:
            data["language"] = language

        try:
            payload = self._post_with_retry(audio_path, data)
        except RetryError as e:
            raise STTServiceError(
                f"STT service unreachable after {self.max_retries} attempts: "
                f"{e.last_attempt.exception()}"
            ) from e
        return str(payload.get("text", ""))

    def _post_with_retry(self, audio_path: Path, data: dict[str, str]) -> dict:
        retryer = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS + (_ServerError,)),
        )
        return retryer(self._post_once)(audio_path, data)

    def _post_once(self, audio_path: Path, data: dict[str, str]) -> dict:
        with audio_path.open("rb") as fh:
            files = {"audio": (audio_path.name, fh, "application/octet-stream")}
            response = requests.post(
                f"{self.base_url}/transcribe",
                files=files,
                data=data,
                timeout=self.timeout_s,
            )
        if 500 <= response.status_code < 600:
            raise _ServerError(
                f"STT server returned {response.status_code}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise STTServiceError(
                f"STT service rejected request ({response.status_code}): "
                f"{response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as e:
            raise STTServiceError(
                f"STT service returned a non-JSON body "
                f"({response.status_code}): {response.text[:200]}"
            ) from e
        if not isinstance(payload, dict):
            raise STTServiceError(
                f"STT service returned an unexpected payload type "
                f"({type(payload).__name__}): {response.text[:200]}"
            )
        return payload


# Accepts either a static list (consumed in order) or a callable that
# computes the transcript from the call arguments.
FakeTranscript = Union[
    str,
    Iterable[str],
    Callable[..., str],
]


class FakeSTTClient:
    """Deterministic STT for tests.

    Pass either a fixed string (returned every call), an iterable of
    strings (returned in order — raises if exhausted), or a callable that
    receives the same kwargs as :meth:`transcribe` and returns a string.
    """

    def __init__(self, transcripts: FakeTranscript) -> None:
        self.calls: list[dict] = []
        if isinstance(transcripts, str):
            self._fn: Callable[..., str] = lambda **_: transcripts
        elif callable(transcripts):
            self._fn = transcripts
        else:
            iterator = iter(transcripts)

            def _next(**_kwargs: object) -> str:
                try:
                    return next(iterator)
                except StopIteration as e:
                    raise AssertionError(
                        "FakeSTTClient ran out of canned transcripts"
                    ) from e

            self._fn = _next

    def transcribe(
        self,
        audio_path: Path,
        *,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        language: Optional[str] = None,
    ) -> str:
        call = {
            "audio_path": audio_path,
            "start_s": start_s,
            "end_s": end_s,
            "language": language,
        }
        self.calls.append(call)
        return self._fn(**call)
