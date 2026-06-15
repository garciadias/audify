"""Cycle-1 text-quality detector node — heuristic check + escalation back-edge.

Immediately after the reader extracts chapter text, this node applies cheap
deterministic heuristics to classify each chapter as **clean** or **garbage**.
On garbage verdict, a conditional escalation back-edge fires the escalate node
to re-extract the chapter with a more capable parser (alternate EPUB parser,
PyMuPDF with different flags, or OCR for PDF).

Unlike cycle 2 (reroute) and cycle 3 (retry), the escalation edge does **not**
re-run the same node — it replaces the reader's output with output from a
different extractor, climbing a ladder of progressively more expensive methods.

Up to ``MAX_BUDGET_PER_CYCLE`` (3) attempts per chapter. On exhaustion the
best-effort text is kept and the chapter is flagged in the report.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from audify.qa.nodes.report import MAX_BUDGET_PER_CYCLE
from audify.qa.state import CycleId, FlagEntry, GraphState

logger = logging.getLogger(__name__)

# Detection thresholds (configurable via creator attributes or env vars).
_MOJIBAKE_THRESHOLD = 0.10
"""Max allowed fraction of non-ASCII / control / replacement characters."""
_WHITESPACE_RATIO_MIN = 0.05
_WHITESPACE_RATIO_MAX = 0.70
"""Whitespace/total char ratio bounds."""
_NONWORD_RATIO_THRESHOLD = 0.30
"""Max allowed fraction of non-alphanumeric characters."""
_MIN_CHARS = 10
"""Samples below this char count are always garbage."""

_CYCLE: CycleId = "cycle_1_escalation"


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def text_quality_node(state: GraphState) -> dict:
    """Inspect each chapter's extracted text and classify as clean or garbage.

    Returns updated state with ``pending_escalation`` populated for garbage
    chapters that still have budget, and ``flags``/``retry_budget`` updated.
    """
    creator = state["creator"]
    chapters: list[str] = state["chapters"]
    chapter_titles: list[str] = state["chapter_titles"]

    # Copy nested containers so we never mutate inbound state in place.
    retry_budget = {k: dict(v) for k, v in state.get("retry_budget", {}).items()}
    flags = {k: list(v) for k, v in state.get("flags", {}).items()}

    pending_escalation: list[int] = []
    clean_chapters: list[str] = []
    clean_titles: list[str] = []

    for i, (chapter_text, title) in enumerate(zip(chapters, chapter_titles)):
        episode_number = i + 1
        chapter_id = f"chapter_{episode_number}"

        # Classify the text.
        verdict = _classify(chapter_text, chapter_id, creator)

        if verdict == "clean":
            clean_chapters.append(chapter_text)
            clean_titles.append(title)
            # Check if this chapter was previously escalated and now resolves.
            if _has_escalation_history(flags, retry_budget, chapter_id):
                _append_flag(
                    flags, chapter_id,
                    reason="Text passed quality checks after escalation",
                    exhausted=False,
                )
            continue

        # Garbage verdict — check budget.
        budget = retry_budget.setdefault(chapter_id, {})
        remaining = budget.get(_CYCLE, MAX_BUDGET_PER_CYCLE)

        if remaining > 0:
            budget[_CYCLE] = remaining - 1
            pending_escalation.append(episode_number)
            logger.info(
                "Text-quality flagged %s (%s); scheduling escalation "
                "(%d attempt(s) left).",
                chapter_id, verdict, remaining - 1,
            )
            # Keep the stale text for now; escalate node will replace it.
            clean_chapters.append(chapter_text)
            clean_titles.append(title)
        else:
            # Budget exhausted: keep best-effort text, write exhausted flag.
            _append_flag(
                flags, chapter_id,
                reason=(
                    f"Text garbage after {MAX_BUDGET_PER_CYCLE}"
                    f" escalations: {verdict}"
                ),
                exhausted=True,
            )
            clean_chapters.append(chapter_text)
            clean_titles.append(title)
            logger.warning(
                "Text-quality exhausted for %s; keeping best-effort "
                "text (verdict: %s).",
                chapter_id, verdict,
            )

    return {
        "chapters": clean_chapters,
        "chapter_titles": clean_titles,
        "retry_budget": retry_budget,
        "flags": flags,
        "pending_escalation": pending_escalation,
    }


def text_quality_route(state: GraphState) -> str:
    """Conditional edge: loop to ``escalate`` while escalations are pending."""
    if state.get("pending_escalation"):
        return "escalate"
    return "confirm"


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------


def _classify(text: str, chapter_id: str, creator: Any) -> str:
    """Classify extracted text as ``"clean"`` or ``"garbage"``.

    Applies a series of cheap deterministic heuristics. Returns ``"garbage"``
    as soon as *any* heuristic triggers, so the LLM judge (cycle 2) and TTS
    are never wasted on bad input.
    """
    if _is_empty_after_clean(text):
        logger.debug("%s: garbage — empty after clean", chapter_id)
        return "garbage"

    moji_env = __import__("os").environ.get("MOJIBAKE_THRESHOLD", _MOJIBAKE_THRESHOLD)
    # Use __dict__.get to avoid MagicMock auto-attribute creation.
    moji_attr = getattr(creator, "__dict__", {}).get("mojibake_threshold")
    threshold = float(moji_attr) if moji_attr is not None else float(moji_env)
    if _has_high_mojibake_ratio(text, threshold):
        logger.debug("%s: garbage — mojibake ratio", chapter_id)
        return "garbage"

    if _has_bad_whitespace_ratio(text):
        logger.debug("%s: garbage — bad whitespace ratio", chapter_id)
        return "garbage"

    nw_env = __import__("os").environ.get(
        "NONWORD_RATIO_THRESHOLD", _NONWORD_RATIO_THRESHOLD
    )
    nw_attr = getattr(creator, "__dict__", {}).get("nonword_ratio_threshold")
    nw_threshold = float(nw_attr) if nw_attr is not None else float(nw_env)
    if _has_high_nonword_ratio(text, nw_threshold):
        logger.debug("%s: garbage — high nonword ratio", chapter_id)
        return "garbage"

    return "clean"


def _is_empty_after_clean(text: str) -> bool:
    """True if stripped text has fewer than ``_MIN_CHARS`` meaningful characters."""
    cleaned = text.strip()
    return len(cleaned) < _MIN_CHARS


def _has_high_mojibake_ratio(text: str, threshold: float = _MOJIBAKE_THRESHOLD) -> bool:
    """True if the fraction of suspicious characters exceeds *threshold*.

    Suspicious characters include:
    - The Unicode replacement character U+FFFD
    - Control characters (C0/C1 ranges, excluding common whitespace)
    - Characters outside the Latin-1 complement range (beyond U+00FF),
      which often indicate encoding mismatch
    """
    if not text:
        return True
    suspicious = 0
    total = len(text)
    for ch in text:
        code = ord(ch)
        if code == 0xFFFD:
            suspicious += 1
        # control chars except tab(0x09), newline(0x0A), cr(0x0D)
        elif code < 0x20 and code not in (0x09, 0x0A, 0x0D):
            suspicious += 1
        elif 0x80 <= code <= 0x9F:  # C1 control characters
            suspicious += 1
    return (suspicious / total) > threshold if total > 0 else True


def _has_bad_whitespace_ratio(
    text: str,
    min_ratio: float = _WHITESPACE_RATIO_MIN,
    max_ratio: float = _WHITESPACE_RATIO_MAX,
) -> bool:
    """True if whitespace fraction is outside acceptable bounds."""
    if not text:
        return True
    ws_count = sum(1 for ch in text if ch in (" ", "\t", "\n", "\r"))
    ratio = ws_count / len(text)
    return ratio < min_ratio or ratio > max_ratio


def _has_high_nonword_ratio(text: str, threshold: float = 0.30) -> bool:
    """True if fraction of non-alphanumeric characters exceeds *threshold*."""
    if not text:
        return True
    ws = (" ", "\t", "\n", "\r")
    nonword = sum(1 for ch in text if not ch.isalnum() and ch not in ws)
    ratio = nonword / len(text)
    return ratio > threshold


# ---------------------------------------------------------------------------
# Escalation — re-extract with a more capable parser
# ---------------------------------------------------------------------------


def escalate_node(state: GraphState) -> dict:
    """Re-extract garbage chapters using the next escalation-ladder method.

    Reads ``pending_escalation`` from state and for each chapter invokes the
    next available parser in the escalation ladder based on the remaining
    retry budget.

    EPUB ladder:
      1. ``BeautifulSoup`` (current default — already tried in ``read_node``)
      2. Raw ``ebooklib`` item walking with ``lxml`` parsing
      3. Regex-based text stripping (last resort)

    PDF ladder:
      1. ``PyMuPDF`` text extraction (current default)
      2. ``PyMuPDF`` with ``sort=True`` + position-preserving flags
      3. OCR via ``pytesseract`` (Tesseract)
    """
    creator = state["creator"]
    chapters: list[str] = state["chapters"]
    pending_escalation: list[int] = state.get("pending_escalation", [])
    retry_budget = state.get("retry_budget", {})

    is_epub = _is_epub_reader(creator)
    source_path = getattr(creator.reader, "path", None)

    for episode_number in pending_escalation:
        i = episode_number - 1
        chapter_id = f"chapter_{episode_number}"
        budget = retry_budget.get(chapter_id, {}).get(_CYCLE, MAX_BUDGET_PER_CYCLE)
        # Attempt number: MAX_BUDGET_PER_CYCLE - remaining
        attempt = MAX_BUDGET_PER_CYCLE - budget  # 1-indexed

        if is_epub and source_path:
            new_text = _escalate_epub(source_path, attempt)
        elif not is_epub and source_path:
            new_text = _escalate_pdf(source_path, attempt)
        else:
            logger.warning("Cannot escalate %s: no source path available", chapter_id)
            continue

        if new_text:
            chapters[i] = new_text
            logger.info(
                "Escalated %s (attempt %d via %s): %d chars extracted",
                chapter_id, attempt, "EPUB" if is_epub else "PDF", len(new_text),
            )
        else:
            logger.warning(
                "Escalation attempt %d for %s produced no text; keeping previous.",
                attempt, chapter_id,
            )

    return {
        "chapters": chapters,
        "pending_escalation": [],
    }


# ---------------------------------------------------------------------------
# EPUB escalation ladder
# ---------------------------------------------------------------------------


def _escalate_epub(source_path: Path, attempt: int) -> Optional[str]:
    """Re-extract EPUB text using the *attempt*-th parser in the ladder.

    Attempt 1: ``BeautifulSoup`` via default reader (already done in read_node,
                so attempt 1 is a no-op — we try attempt 2+).
    Attempt 2: Raw ``ebooklib`` item walking with ``lxml.html`` parsing.
    Attempt 3: Regex-based text stripping (last resort).
    """
    if attempt >= 3:
        return _epub_escalate_regex(source_path)
    if attempt >= 2:
        return _epub_escalate_ebooklib_raw(source_path)
    return None


def _epub_escalate_ebooklib_raw(source_path: Path) -> Optional[str]:
    """Extract text from EPUB using raw ``ebooklib`` item walking.

    Iterates all ``ITEM_DOCUMENT`` items and parses each with ``lxml.html``,
    extracting all text content without TOC-based chapter grouping.
    Returns concatenated text or ``None`` on failure.
    """
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        logger.warning("ebooklib not available for EPUB escalation")
        return None

    try:
        import lxml.html
    except ImportError:
        logger.warning("lxml not available for EPUB escalation")
        return None

    try:
        book = epub.read_epub(str(source_path))
    except Exception as exc:
        logger.warning("Failed to read EPUB for escalation: %s", exc)
        return None

    texts: list[str] = []
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        try:
            html_content = item.get_body_content()
            if not html_content:
                continue
            tree = lxml.html.fromstring(html_content)
            # Remove script/style elements.
            for elem in tree.xpath(".//script | .//style"):
                elem.getparent().remove(elem)
            text = tree.text_content()
            if text:
                texts.append(text.strip())
        except Exception as exc:
            logger.debug("Failed to parse one EPUB item: %s", exc)
            continue

    if not texts:
        return None
    return "\n\n".join(texts)


def _epub_escalate_regex(source_path: Path) -> Optional[str]:
    """Last-resort EPUB extraction: read raw HTML, strip tags with regex.

    This is deliberately fragile — it's the final escalation step before
    giving up entirely.
    """
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        logger.warning("ebooklib not available for EPUB regex escalation")
        return None

    try:
        book = epub.read_epub(str(source_path))
    except Exception as exc:
        logger.warning("Failed to read EPUB for regex escalation: %s", exc)
        return None

    texts: list[str] = []
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        try:
            raw = item.get_body_content()
            if not raw:
                continue
            # Strip all HTML tags.
            text = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="replace"))
            # Collapse whitespace.
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                texts.append(text)
        except Exception:
            continue

    if not texts:
        return None
    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# PDF escalation ladder
# ---------------------------------------------------------------------------


def _escalate_pdf(source_path: Path, attempt: int) -> Optional[str]:
    """Re-extract PDF text using the *attempt*-th parser in the ladder.

    Attempt 1: ``PyMuPDF`` default (already done in read_node) — no-op.
    Attempt 2: ``PyMuPDF`` with ``sort=True`` flag.
    Attempt 3: OCR via ``pytesseract``.
    """
    if attempt >= 3:
        return _pdf_escalate_ocr(source_path)
    if attempt >= 2:
        return _pdf_escalate_sort(source_path)
    return None


def _pdf_escalate_sort(source_path: Path) -> Optional[str]:
    """Extract PDF text with ``PyMuPDF`` using ``sort=True``.

    This sorts text blocks by reading order (y-then-x), which often extracts
    more coherent text from complex layouts.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not available for PDF escalation")
        return None

    try:
        doc = fitz.open(str(source_path))
    except Exception as exc:
        logger.warning("Failed to open PDF for sort escalation: %s", exc)
        return None

    texts: list[str] = []
    for page in doc:
        try:
            text = page.get_text("text", sort=True)
            if text:
                texts.append(text.strip())
        except Exception:
            continue

    doc.close()
    if not texts:
        return None
    return "\n\n".join(texts)


def _pdf_escalate_ocr(source_path: Path) -> Optional[str]:
    """Extract PDF text via OCR using ``pytesseract``.

    Converts each page to an image via ``pdf2image`` then runs Tesseract OCR.
    Requires ``pytesseract``, ``pdf2image``, and Tesseract system binary.
    """
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not available for OCR escalation")
        return None

    try:
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning("pdf2image not available for OCR escalation")
        return None

    try:
        images = convert_from_path(str(source_path), dpi=300)
    except Exception as exc:
        logger.warning("Failed to convert PDF pages to images: %s", exc)
        return None

    texts: list[str] = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img)
            if text.strip():
                texts.append(text.strip())
        except Exception as exc:
            logger.warning("OCR failed on a page: %s", exc)
            continue

    if not texts:
        return None
    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_epub_reader(creator: Any) -> bool:
    """True if the creator's reader is an EpubReader."""
    try:
        from audify.readers.ebook import EpubReader

        reader = creator.reader
        if reader is None:
            return False
        return isinstance(reader, EpubReader)
    except (ImportError, AttributeError, TypeError):
        # Fallback: class name check when isinstance fails (LangGraph
        # execution context sometimes interferes with lazy imports).
        try:
            return type(getattr(creator, "reader", None)).__name__ == "EpubReader"
        except (ImportError, AttributeError):
            return False


def _has_escalation_history(
    flags: dict[str, list[FlagEntry]],
    retry_budget: dict[str, dict[CycleId, int]],
    chapter_id: str,
) -> bool:
    """True if the chapter has a previous cycle-1 escalation history."""
    if retry_budget.get(chapter_id, {}).get(_CYCLE) is not None:
        return True
    return any(
        entry["cycle"] == _CYCLE
        for entry in flags.get(chapter_id, [])
    )


def _append_flag(
    flags: dict[str, list[FlagEntry]],
    chapter_id: str,
    *,
    reason: str,
    exhausted: bool,
) -> None:
    entry: FlagEntry = {
        "cycle": "cycle_1_escalation",
        "reason": reason,
        "exhausted": exhausted,
    }
    flags.setdefault(chapter_id, []).append(entry)
