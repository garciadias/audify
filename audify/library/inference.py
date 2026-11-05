"""Metadata enrichment, rung 2: LLM inference of Text-derivable fields.

After the catalogues, fields that can be read off the book's own text are
inferred from an opening sample: description, genres, tags and subtitle.
Factual fields (ISBN, dates, publisher, series, authors, title, language)
are never inferred: a model would guess them. The policy is the same as
rung 1: only empty fields are filled, a field an override cleared stays
empty, and every filled field records ``llm`` as its provenance.

The step never fails a run. No LLM, an error reply or unparseable output
leaves the fields empty.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from audify.library.metadata import BookMetadata

logger = logging.getLogger(__name__)

# The only fields an LLM may fill. Tests assert nothing else gets "llm".
TEXT_DERIVABLE_FIELDS = ("description", "genres", "tags", "subtitle")
SAMPLE_CHARS = 8000
_MAX_LIST = 5

# (system prompt, user text) -> reply. Errors are raised or returned as text
# starting with "Error:", which is how the run's LLM client reports them.
Generate = Callable[[str, str], str]

PROMPT = """You catalogue books for a library. From the opening of a book
below, write JSON with exactly these keys:
- "description": a 2 to 4 sentence blurb of what the book is about, in the
  book's language, with no spoilers and no praise;
- "genres": up to 3 broad genres in English (e.g. "Fiction", "History");
- "tags": up to 5 short lowercase topic tags in English;
- "subtitle": the subtitle only if the title page in the text shows one,
  else "".
Use only what the text shows. Reply with the JSON object and nothing else."""


def _parse(reply: str) -> Optional[dict[str, Any]]:
    reply = reply.split("</think>")[-1]
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in items and len(text) <= 40:
            items.append(text)
    return items[:_MAX_LIST]


def infer_text_fields(
    metadata: BookMetadata, book_text: str, generate: Generate
) -> list[str]:
    """Fill empty Text-derivable fields of *metadata* from *book_text*.

    Returns the names of the fields that were filled.
    """
    wanted = [
        name
        for name in TEXT_DERIVABLE_FIELDS
        if not getattr(metadata, name) and metadata.provenance.get(name) != "override"
    ]
    sample = book_text.strip()[:SAMPLE_CHARS]
    if not wanted or not sample:
        return []
    header = f"Title: {metadata.title}\n" if metadata.title else ""
    try:
        reply = generate(PROMPT, f"{header}\n{sample}")
    except Exception as e:
        logger.info(f"Skipping LLM metadata inference: {e}")
        return []
    if not reply or reply.lstrip().startswith("Error:"):
        logger.info("Skipping LLM metadata inference: no usable LLM reply.")
        return []
    data = _parse(reply)
    if data is None:
        logger.warning("LLM metadata reply was not JSON; ignoring it.")
        return []

    filled: list[str] = []
    for name in wanted:
        if name in ("genres", "tags"):
            value: Any = _clean_list(data.get(name))
            if name == "tags":
                value = [t.lower() for t in value]
        else:
            raw = data.get(name)
            value = raw.strip() if isinstance(raw, str) else ""
            if name == "subtitle" and len(value) > 120:
                value = ""
        if value:
            setattr(metadata, name, value)
            metadata.provenance[name] = "llm"
            filled.append(name)
    if filled:
        logger.info(f"Inferred from the text: {', '.join(filled)}")
    return filled
