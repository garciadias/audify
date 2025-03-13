from abc import ABC, abstractmethod
from pathlib import Path

from TTS.api import TTS


class Reader(ABC):
    @abstractmethod
    def __init__(self, path: str | Path): ...

    @abstractmethod
    def get_chapters(self) -> list[str]: ...

    @abstractmethod
    def get_language(self) -> str: ...

    @abstractmethod
    def get_title(self) -> str: ...

    @abstractmethod
    def get_cover_image(self) -> str | None: ...


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
