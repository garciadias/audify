import logging
import re
from pathlib import Path
from typing import Optional, Union

import bs4
from bs4 import Tag
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE, epub
from typing_extensions import Reader

from audify.utils.api_config import CommercialAPIConfig, OllamaAPIConfig

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

    def read(self):
        return epub.read_epub(self.path)

    def get_chapters(self) -> list[str]:
        """Get chapter content in spine order, filtering out TOC and other non-chapter documents."""
        chapters = []

        # Process items in spine order (reading order)
        for spine_id, linear in self.book.spine:
            item = self.book.get_item_with_id(spine_id)
            if not item or item.get_type() != ITEM_DOCUMENT:
                continue

            item_name = item.get_name().lower()

            # Skip obvious non-chapter files
            if any(
                pattern in item_name for pattern in ["toc", "nav", "titlepage", "cover"]
            ):
                continue

            try:
                content = item.get_body_content().decode("utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Could not decode item {item.get_name()}: {e}")
                continue

            # Skip empty or very short content
            if len(content.strip()) < 100:
                continue

            # Parse HTML to check document type
            soup = bs4.BeautifulSoup(content, "html.parser")
            text = soup.get_text().lower()

            # Check for TOC indicators in content
            toc_indicators = [
                "table of contents",
                "contents",
                "目录",  # Chinese for TOC
                "chapter",  # Might appear in TOC listing chapters
                "part",
                "section",
                "章",  # Chinese chapter character
            ]

            # Count how many TOC indicators appear
            toc_indicator_count = sum(
                1 for indicator in toc_indicators if indicator in text
            )

            # Check if this looks like a TOC by analyzing structure
            # TOCs often have many links or list items
            links = soup.find_all("a")
            list_items = soup.find_all(["li", "dt", "dd"])

            # If it has many links/list items AND contains TOC indicators, likely a TOC
            if (len(links) > 5 or len(list_items) > 5) and toc_indicator_count > 1:
                logger.info(f"Skipping TOC document: {item.get_name()}")
                continue

            # Check for Chinese chapter patterns like "第一章", "第二章"
            chinese_chapter_patterns = re.findall(
                r"第[一二三四五六七八九十\d]+章", text
            )
            if len(chinese_chapter_patterns) > 2:
                # If we see multiple chapter titles in one document, it's likely a TOC
                logger.info(
                    f"Skipping document with multiple chapter titles (likely TOC): {item.get_name()}"
                )
                continue

            # Check for copyright/credits pages
            copyright_indicators = [
                "copyright",
                "©",
                "isbn",
                "published by",
                "all rights reserved",
                "责任编辑",
                "封面设计",
                "版式设计",
                "图书在版编目",
            ]

            copyright_indicator_count = sum(
                1 for indicator in copyright_indicators if indicator in text
            )
            if copyright_indicator_count > 2:
                logger.info(f"Skipping copyright/credits page: {item.get_name()}")
                continue

            chapters.append(content)

        # If no chapters found with filtering, fall back to all documents
        if not chapters:
            logger.warning(
                "No chapters found with filtering, falling back to all documents"
            )
            for item in self.book.get_items():
                if item.get_type() != ITEM_DOCUMENT:
                    continue
                item_name = item.get_name().lower()
                # Skip obvious non-chapter files
                if any(
                    pattern in item_name
                    for pattern in ["toc", "nav", "titlepage", "cover"]
                ):
                    continue
                try:
                    content = item.get_body_content().decode("utf-8", errors="ignore")
                except Exception as e:
                    logger.warning(f"Could not decode item {item.get_name()}: {e}")
                    continue
                # Skip empty or very short content (likely metadata)
                if len(content.strip()) < 20:
                    continue
                chapters.append(content)

        return chapters

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
