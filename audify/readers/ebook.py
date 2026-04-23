import logging
import re
from pathlib import Path
from typing import Optional, Union

import bs4
from bs4 import Tag
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE, epub
from audify.domain.reader import Reader

from audify.utils.api_config import CommercialAPIConfig, OllamaAPIConfig

# Minimum number of TOC-to-spine matches before we trust TOC-based grouping.
# If fewer items match, the TOC structure probably doesn't align with the
# spine and we fall back to legacy per-item extraction.
_MIN_TOC_MATCHES = 3

logger = logging.getLogger(__name__)

# Regex patterns for chapter-like titles
CHAPTER_PATTERNS = [
    # "Chapter 1", "Chapter I", "Chapter One", with optional colon/dash and subtitle
    re.compile(
        r"^(chapter|ch\.?)\s+[\divxlcdm]+[\s.:—\-]*.*",
        re.IGNORECASE,
    ),
    # "Part 1", "Part I", etc.
    re.compile(
        r"^(part|section|book)\s+[\divxlcdm]+[\s.:—\-]*.*",
        re.IGNORECASE,
    ),
    # Standalone Roman numerals (I, II, III, IV, ..., possibly with title)
    re.compile(
        r"^[IVXLCDM]+[\s.:—\-]+.+",
    ),
    # Standalone numbers like "1", "2.", "1 -", possibly followed by a title
    re.compile(
        r"^\d+[\s.:—\-]+\S.*",
    ),
    # Prologue, Epilogue, Introduction, Foreword, Preface, Afterword, Appendix
    re.compile(
        r"^(prologue|epilogue|introduction|foreword|preface|afterword|appendix"
        r"|acknowledgments|dedication|interlude|intermission)"
        r"[\s.:—\-]*.*",
        re.IGNORECASE,
    ),
]

# CSS class/id patterns that commonly indicate titles.
# Uses token boundaries (start/end of string or common separators) to avoid
# false positives on substrings like "section-content" matching "title".
_TITLE_TOKENS = (
    "title|heading|chapter|chaptertitle|chaptitle|chapter-title|"
    "chapter-head|chapter-heading|book-title|section-title|"
    "calibre_heading|sgc-toc-level"
)
TITLE_ATTR_PATTERNS = re.compile(
    rf"(?:^|[\s_\-\.])({_TITLE_TOKENS})(?:$|[\s_\-\.])",
    re.IGNORECASE,
)

CHAPTER_TITLE_LLM_PROMPT = (
    "Extract the chapter title from the following text excerpted from the "
    "beginning of an ebook chapter. Return ONLY the chapter title, nothing else. "
    "If there is no identifiable chapter title, return 'Unknown'. "
    "Do not add quotes or any extra formatting.\n\n"
    "Text:\n{text}"
)

# Common front/back-matter file name tokens seen across multilingual EPUBs.
NON_CHAPTER_FILENAME_TOKENS = [
    "toc",
    "nav",
    "titlepage",
    "cover",
    "cubierta",
    "sinopsis",
    "synopsis",
    "titulo",
    "título",
    "info",
    "copyright",
    "colophon",
    "autor",
    "author",
    "biography",
    "bio",
    "notes",
    "notas",
    "appendix",
    "apendice",
    "apéndice",
]


class EpubReader(Reader):
    def __init__(
        self,
        path: str | Path,
        llm_config: Optional[Union[OllamaAPIConfig, CommercialAPIConfig]] = None,
    ):
        self.path = Path(path).resolve()
        self.book = self.read()
        self.title = self.get_title()
        self.llm_config = llm_config

    def read(self) -> epub.EpubBook:
        return epub.read_epub(self.path)

    def get_chapters(self) -> list[str]:
        """Get chapter content in spine order, grouped by TOC boundaries.

        Uses the EPUB table of contents to merge spine items that belong to
        the same logical chapter. Falls back to per-spine-item extraction when
        the TOC structure doesn't align with the spine.
        """
        chapters = self._get_chapters_grouped_by_toc()
        if chapters:
            return chapters
        logger.info(
            "TOC grouping produced no chapters, using legacy per-item extraction"
        )
        return self._get_chapters_legacy()

    # ------------------------------------------------------------------
    # TOC-based chapter grouping
    # ------------------------------------------------------------------

    def _get_chapters_grouped_by_toc(self) -> list[str]:
        """Extract chapters by grouping spine items via EPUB TOC entries.

        Spine items that fall between two TOC chapter boundaries are merged
        into a single logical chapter.  Returns an empty list when the TOC
        cannot be used (empty, unparseable, or mismatched with the spine).
        """
        toc_item_names = self._build_toc_item_name_set()
        if not toc_item_names:
            return []

        chapters: list[str] = []
        current_group: list = []
        matches_found = 0

        for spine_id, linear in self.book.spine:
            item = self.book.get_item_with_id(spine_id)
            if not item or item.get_type() != ITEM_DOCUMENT:
                continue

            item_name = item.get_name().lower()

            if self._should_skip_document_by_name(item_name):
                continue

            is_toc_boundary = item_name in toc_item_names

            if is_toc_boundary and current_group:
                merged = self._merge_items(current_group)
                if merged and self._is_valid_chapter(merged):
                    chapters.append(merged)
                current_group = [item]
                matches_found += 1
            else:
                current_group.append(item)

        if current_group:
            merged = self._merge_items(current_group)
            if merged and self._is_valid_chapter(merged):
                chapters.append(merged)

        # If we didn't find enough TOC-to-spine matches or ended up with
        # a single blob, the TOC likely doesn't align with the spine —
        # fall back to legacy extraction.
        if matches_found < min(_MIN_TOC_MATCHES, len(toc_item_names)):
            logger.debug(
                f"TOC-based grouping: only {matches_found} spine matches "
                f"(need ≥{_MIN_TOC_MATCHES}), falling back"
            )
            return []
        if len(chapters) <= 1 and len(toc_item_names) > 2:
            logger.debug(
                f"TOC-based grouping: produced {len(chapters)} chapter(s) "
                f"from {len(toc_item_names)} TOC entries, falling back"
            )
            return []

        return chapters

    def _flatten_toc_hrefs(self) -> list[str]:
        """Flatten the hierarchical TOC into a flat list of href strings."""
        raw = getattr(self.book, "toc", None)
        if not isinstance(raw, list):
            return []

        hrefs: list[str] = []

        def _walk(entries: list) -> None:
            for entry in entries:
                if isinstance(entry, tuple) and len(entry) == 2:
                    nav_point, sub = entry
                    href = getattr(nav_point, "href", None)
                    if href:
                        hrefs.append(href)
                    _walk(sub)
                else:
                    href = getattr(entry, "href", None)
                    if href:
                        hrefs.append(href)

        try:
            _walk(raw)
        except (TypeError, Exception):
            return []

        return hrefs

    def _build_toc_item_name_set(self) -> set[str]:
        """Normalise TOC hrefs into a set of spine item names for fast lookup."""
        hrefs = self._flatten_toc_hrefs()
        names: set[str] = set()
        for href in hrefs:
            # Strip fragment, leading './' and '/', case-fold.
            name = href.split("#")[0].lstrip("./").lower()
            if name:
                names.add(name)
        return names

    def _merge_items(self, items: list) -> str | None:
        """Merge the body contents of *items* into a single chapter HTML.

        Returns *None* when no content could be extracted from any item.
        """
        if not items:
            return None
        if len(items) == 1:
            try:
                return items[0].get_body_content().decode("utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Could not decode item {items[0].get_name()}: {e}")
                return None

        bodies: list[str] = []
        for item in items:
            try:
                content = item.get_body_content().decode("utf-8", errors="ignore")
                soup = bs4.BeautifulSoup(content, "html.parser")
                body = soup.find("body")
                if body is not None:
                    bodies.append(str(body))
            except Exception as e:
                logger.warning(f"Could not decode/parse item {item.get_name()}: {e}")

        if not bodies:
            return None

        return "<html><body>" + "\n".join(bodies) + "</body></html>"

    @staticmethod
    def _looks_like_toc(soup: bs4.BeautifulSoup, text: str) -> bool:
        """Heuristic: return True when *soup* looks like a table of contents."""
        toc_indicators = [
            "table of contents",
            "contents",
            "目录",
            "chapter",
            "part",
            "section",
            "章",
        ]
        indicator_count = sum(1 for ind in toc_indicators if ind in text)
        links = soup.find_all("a")
        list_items = soup.find_all(["li", "dt", "dd"])
        return (len(links) > 5 or len(list_items) > 5) and indicator_count > 1

    @staticmethod
    def _looks_like_copyright(text: str) -> bool:
        """Heuristic: return True when *text* resembles a copyright page."""
        indicators = [
            "copyright",
            "©",
            "isbn",
            "published by",
            "all rights reserved",
            "责任编辑",
            "封面设计",
            "图书在版编目",
        ]
        return sum(1 for ind in indicators if ind in text) > 2

    def _is_valid_chapter(self, merged_html: str) -> bool:
        """Return True when *merged_html* has enough content to be a chapter."""
        if len(merged_html.strip()) < 100:
            return False
        soup = bs4.BeautifulSoup(merged_html, "html.parser")
        visible_text = soup.get_text(separator=" ", strip=True)
        if len(visible_text) < 80:
            return False
        text = visible_text.lower()
        if self._looks_like_toc(soup, text):
            return False
        chinese_patterns = re.findall(r"第[一二三四五六七八九十\d]+章", text)
        if len(chinese_patterns) > 2:
            return False
        if self._looks_like_copyright(text):
            return False
        return True

    # ------------------------------------------------------------------
    # Legacy per-spine-item extraction (fallback)
    # ------------------------------------------------------------------

    def _get_chapters_legacy(self) -> list[str]:
        """Original per-spine-item chapter extraction (unchanged logic)."""
        chapters: list[str] = []

        for spine_id, linear in self.book.spine:
            item = self.book.get_item_with_id(spine_id)
            if not item or item.get_type() != ITEM_DOCUMENT:
                continue

            item_name = item.get_name().lower()

            if self._should_skip_document_by_name(item_name):
                continue

            try:
                content = item.get_body_content().decode("utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Could not decode item {item.get_name()}: {e}")
                continue

            if len(content.strip()) < 100:
                continue

            soup = bs4.BeautifulSoup(content, "html.parser")
            visible_text = soup.get_text(separator=" ", strip=True)
            text = visible_text.lower()

            if len(visible_text) < 80:
                continue

            if self._looks_like_toc(soup, text):
                logger.info(f"Skipping TOC document: {item.get_name()}")
                continue

            chinese_chapter_patterns = re.findall(
                r"第[一二三四五六七八九十\d]+章",
                text,
            )
            if len(chinese_chapter_patterns) > 2:
                logger.info(
                    "Skipping document with multiple chapter titles "
                    f"(likely TOC): {item.get_name()}"
                )
                continue

            if self._looks_like_copyright(text):
                logger.info(f"Skipping copyright/credits page: {item.get_name()}")
                continue

            chapters.append(content)

        if not chapters:
            logger.warning(
                "No chapters found with filtering, falling back to all documents"
            )
            for item in self.book.get_items():
                if item.get_type() != ITEM_DOCUMENT:
                    continue
                item_name = item.get_name().lower()
                if self._should_skip_document_by_name(item_name):
                    continue
                try:
                    content = item.get_body_content().decode("utf-8", errors="ignore")
                except Exception as e:
                    logger.warning(f"Could not decode item {item.get_name()}: {e}")
                    continue
                visible_text = bs4.BeautifulSoup(content, "html.parser").get_text(
                    separator=" ", strip=True
                )
                if len(visible_text) < 20:
                    continue
                chapters.append(content)

        return chapters

    @staticmethod
    def _should_skip_document_by_name(item_name: str) -> bool:
        return any(token in item_name for token in NON_CHAPTER_FILENAME_TOKENS)

    def extract_text(self, chapter: str) -> str:
        return bs4.BeautifulSoup(chapter, "html.parser").get_text()

    def get_chapter_title(self, chapter: str) -> str:
        soup = bs4.BeautifulSoup(chapter, "html.parser")

        # Strategy 1: Look for heading tags (h1-h6, title, hgroup, header)
        title = self._extract_from_heading_tags(soup)
        if title:
            return title

        # Strategy 2: Look for elements with title-related class/id attributes
        title = self._extract_from_title_attributes(soup)
        if title:
            return title

        # Strategy 3: Look for bold/strong/em text in early elements
        title = self._extract_from_emphasis_tags(soup)
        if title:
            return title

        # Strategy 4: Check early paragraphs for chapter-like patterns via regex
        title = self._extract_from_paragraph_patterns(soup)
        if title:
            return title

        # Strategy 5: Check if a short early paragraph looks like a title
        title = self._extract_short_paragraph_title(soup)
        if title:
            return title

        # Strategy 6: LLM fallback — ask a language model to extract the title
        title = self._extract_via_llm(soup)
        if title:
            return title

        return "Unknown"

    def _extract_from_heading_tags(self, soup: bs4.BeautifulSoup) -> str:
        body = self._find_body(soup)
        # First: search within the body for real content headings
        body_heading_tags = [f"h{i}" for i in range(1, 7)]
        body_heading_tags += ["hgroup", "header"]
        tag = body.find(body_heading_tags)
        if tag:
            text = tag.get_text(separator=" ", strip=True)
            if text:
                return text
        # Fallback: check <title> and other metadata tags in <head>
        fallback_tag = soup.find("title")
        if fallback_tag:
            text = fallback_tag.get_text(separator=" ", strip=True)
            if text:
                return text
        return ""

    def _find_body(self, soup: bs4.BeautifulSoup) -> Tag:
        body = soup.find("body")
        if isinstance(body, Tag):
            return body
        return soup  # type: ignore[return-value]

    def _extract_from_title_attributes(self, soup: bs4.BeautifulSoup) -> str:
        for tag in soup.find_all(True):
            if not isinstance(tag, Tag):
                continue
            classes = " ".join(tag.get("class", []))  # type: ignore[arg-type]
            tag_id = tag.get("id", "")
            combined = f"{classes} {tag_id}"
            if TITLE_ATTR_PATTERNS.search(combined):
                text = tag.get_text(separator=" ", strip=True)
                if text:
                    return text
        return ""

    def _extract_from_emphasis_tags(self, soup: bs4.BeautifulSoup) -> str:
        body = self._find_body(soup)
        early_elements = list(body.children)[:10]
        for elem in early_elements:
            if not isinstance(elem, Tag):
                continue
            for emphasis_tag in elem.find_all(["strong", "b", "em"]):
                text = emphasis_tag.get_text(separator=" ", strip=True)
                if text and len(text) < 150 and not text.endswith("."):
                    for pattern in CHAPTER_PATTERNS:
                        if pattern.match(text):
                            return text
        return ""

    @staticmethod
    def _is_leaf_paragraph(tag: bs4.element.PageElement) -> bool:
        """Return True if the tag has no nested paragraph-like children."""
        if not isinstance(tag, Tag):
            return True
        return tag.find(["p", "div", "span"]) is None

    def _extract_from_paragraph_patterns(self, soup: bs4.BeautifulSoup) -> str:
        body = self._find_body(soup)
        paragraphs = body.find_all(["p", "div", "span"], limit=10)
        for p in paragraphs:
            if not self._is_leaf_paragraph(p):
                continue
            text = p.get_text(separator=" ", strip=True)
            if not text:
                continue
            for pattern in CHAPTER_PATTERNS:
                if pattern.match(text):
                    return text
        return ""

    def _extract_short_paragraph_title(self, soup: bs4.BeautifulSoup) -> str:
        body = self._find_body(soup)
        paragraphs = body.find_all(["p", "div"], limit=5)
        for p in paragraphs:
            if not self._is_leaf_paragraph(p):
                continue
            text = p.get_text(separator=" ", strip=True)
            if not text:
                continue
            # A short line (< 80 chars) that doesn't end with sentence
            # punctuation is likely a title
            if (
                len(text) < 80
                and not text.endswith((".", "!", "?", ",", ";", ":"))
                and len(text.split()) >= 1
                and len(text.split()) <= 12
            ):
                return text
        return ""

    def _extract_via_llm(self, soup: bs4.BeautifulSoup) -> str:
        if self.llm_config is None:
            return ""
        body = self._find_body(soup)
        paragraphs = body.find_all(["p", "div", "span", "h1", "h2", "h3"], limit=10)
        text_parts = []
        for p in paragraphs[:5]:
            text = p.get_text(separator=" ", strip=True)
            if text:
                text_parts.append(text)
        if not text_parts:
            return ""
        combined_text = "\n".join(text_parts)
        prompt = CHAPTER_TITLE_LLM_PROMPT.format(text=combined_text)
        try:
            response = self.llm_config.generate(
                user_prompt=prompt,
                temperature=0.1,
                num_predict=100,
            )
            if response:
                title = response.strip().strip("\"'")
                # Remove <think>...</think> blocks from reasoning models
                if "<think>" in title.lower():
                    title = re.sub(
                        r"<think>.*?</think>",
                        "",
                        title,
                        flags=re.DOTALL | re.IGNORECASE,
                    ).strip()
                if title and title.lower() != "unknown" and len(title) < 200:
                    return title
        except Exception as e:
            logger.warning(f"LLM title extraction failed: {e}")
        return ""

    def get_title(self) -> str:
        title = self.book.title
        if not title and self.book.get_metadata("DC", "title"):
            if self.book.get_metadata("DC", "title")[0]:
                title = self.book.get_metadata("DC", "title")[0][0]
        if not title:
            title = "missing title"
        return title

    def get_cover_image(self, output_path: str | Path) -> Path | None:
        # If ITEM_COVER is available, use it
        cover_image = next(
            (item for item in self.book.get_items() if item.get_type() == ITEM_COVER),
            None,
        )
        if not cover_image:
            # If not, use the first image
            cover_image = next(
                (
                    item
                    for item in self.book.get_items()
                    if item.get_type() == ITEM_IMAGE
                ),
                None,
            )
        if not cover_image:
            return None
        cover_path = f"{output_path}/cover.jpg"
        with open(cover_path, "wb") as f:
            f.write(cover_image.content)
        return Path(cover_path)

    def get_language(self) -> str:
        language = self.book.get_metadata("DC", "language")[0][0]
        return language
