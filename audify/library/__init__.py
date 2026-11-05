"""Library delivery: Book metadata, its carriers, and Library items.

See the "Library delivery" section of CONTEXT.md for the vocabulary.
"""

from audify.library.metadata import BookMetadata, SeriesEntry, book_metadata_from_epub

__all__ = ["BookMetadata", "SeriesEntry", "book_metadata_from_epub"]
