from abc import ABC, abstractmethod


class Reader(ABC):
    @abstractmethod
    def get_chapters(self, path: str) -> list[str]:
        ...

    @abstractmethod
    def get_language(self) -> str:
        ...

    @abstractmethod
    def get_title(self) -> str:
        ...

    @abstractmethod
    def get_cover_image(self) -> str | None:
        ...
