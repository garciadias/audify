from abc import ABC, abstractmethod
from pathlib import Path


class Reader(ABC):
    @abstractmethod
    def __init__(self, path: str | Path):
        self.path: Path
        self.cleaned_text: str
        ...

    @abstractmethod
    def read(self): ...
