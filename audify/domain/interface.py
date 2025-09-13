from abc import ABC, abstractmethod
from pathlib import Path

from TTS.api import TTS


class Synthesizer(ABC):
    @abstractmethod
    def __init__(
        self,
        path: str | Path,
        model_name: str,
        language: str,
        speaker: str,
        engine: str,
    ):
        self.model: TTS

    @abstractmethod
    def synthesize(self) -> str | Path: ...
