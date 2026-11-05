"""Metadata override: a person's deliberate corrections for one book.

The override is ``metadata.override.json`` in the book's folder inside the
Work directory. It wins over every other source (override > Source ebook >
catalogue > LLM) and is the place to correct a book: edits made in the
audiobook server's UI are overwritten by the next Re-publish, because the
server ranks ``metadata.json`` highest.

Example::

    {
      "title": "A Ordem do Capital",
      "authors": ["Clara E. Mattei"],
      "series": ["Ensaios #2"],
      "published_date": "2023",
      "description": "..."
    }
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from audify.library.metadata import METADATA_FIELDS, BookMetadata, SeriesEntry

logger = logging.getLogger(__name__)

OVERRIDE_FILE = "metadata.override.json"
_LIST_FIELDS = {"authors", "narrators", "genres", "tags"}
_SERIES_RE = re.compile(r"^(?P<name>.+?)\s*#\s*(?P<seq>[^#]+)$")


def load_override(path: Path) -> Optional[dict[str, Any]]:
    """Read an override file, or None when it does not exist or is invalid."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Ignoring unreadable metadata override {path}: {e}")
        return None
    if not isinstance(data, dict):
        logger.error(f"Ignoring metadata override {path}: expected a JSON object")
        return None
    unknown = sorted(set(data) - set(METADATA_FIELDS))
    if unknown:
        logger.warning(f"Unknown fields in {path.name} ignored: {', '.join(unknown)}")
    return {k: v for k, v in data.items() if k in METADATA_FIELDS}


def _series(value: Any) -> list[SeriesEntry]:
    entries = value if isinstance(value, list) else [value]
    result: list[SeriesEntry] = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name"):
            seq = entry.get("sequence")
            result.append(SeriesEntry(str(entry["name"]), str(seq) if seq else None))
        elif isinstance(entry, str) and entry.strip():
            match = _SERIES_RE.match(entry.strip())
            if match:
                result.append(SeriesEntry(match["name"], match["seq"].strip()))
            else:
                result.append(SeriesEntry(entry.strip()))
    return result


def apply_override(metadata: BookMetadata, data: dict[str, Any]) -> BookMetadata:
    """Set every field present in *data* on *metadata*, marked ``override``.

    A field set to ``null``, ``""`` or ``[]`` in the override is cleared on
    purpose, which is how a wrong value from the ebook is removed.
    """
    for name, value in data.items():
        if name == "series":
            new: Any = _series(value) if value else []
        elif name in _LIST_FIELDS:
            if value is None:
                new = []
            elif isinstance(value, list):
                new = [str(v) for v in value if str(v).strip()]
            else:
                new = [str(value)]
        else:
            new = "" if value is None else str(value)
        setattr(metadata, name, new)
        if new:
            metadata.provenance[name] = "override"
        else:
            metadata.provenance.pop(name, None)
    return metadata
