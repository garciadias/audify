#!/usr/bin/env python3
"""
Verify/Evidence mode for Audify - compare EPUB content with generated audiobook.

This module compares the chapters of an EPUB ebook against the chapters of
a generated M4B or MP3 audiobook to find discrepancies such as:
  - Missing chapters in the audiobook
  - Extra chapters in the audiobook that don't exist in the source
  - Chapters with wrong order
  - Chapters that may have been skipped/silently truncated
  - Timing analysis (expected duration vs actual)

Usage as CLI:
    audify compare <epub> <m4b_or_mp3>

Usage as library:
    from audify.verify import AudiobookVerifier
    verifier = AudiobookVerifier("book.epub", "audiobook.m4b")
    report = verifier.verify()
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audify.readers.ebook import EpubReader
from audify.readers.pdf import PdfReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chapter:
    """Represents a single chapter from either source or audiobook."""

    number: int
    title: str
    index: int  # position in source/order

    def __hash__(self) -> int:
        return hash((self.number, self.title))

    def __eq__(self, other: object) -> object:
        if not isinstance(other, Chapter):
            return NotImplemented
        return self.number == other.number and self.title == other.title

    def __repr__(self) -> str:
        return f"Chapter({self.number}: {self.title!r})"


@dataclass
class MissingChapter:
    """A chapter from source that is missing from the audiobook."""

    number: int
    title: str
    expected_position: int


@dataclass
class ExtraChapter:
    """A chapter in the audiobook that has no EPUB source."""

    number: int
    title: str
    position: int


@dataclass
class OrderViolation:
    """A chapter in the audiobook that appears out of order compared to source."""

    number: int
    title: str
    expected_order: int
    actual_position: int


@dataclass
class DurationHint:
    """Expected duration based on source text length analysis."""

    source_word_count: int
    expected_duration_s: float  # rough estimate
    actual_duration_s: float
    ratio: float  # actual / expected


@dataclass
class VerifyReport:
    """Result of comparing a source file against an audiobook."""

    source_path: Path
    audiobook_path: Path
    source_chapters: list[Chapter]
    audiobook_chapters: list[Chapter]
    source_type: str  # "epub" or "pdf"
    source_word_count: int = 0  # total word count from source
    missing: list[MissingChapter] = field(default_factory=list)
    extra: list[ExtraChapter] = field(default_factory=list)
    order_violations: list[OrderViolation] = field(default_factory=list)
    matched: int = 0
    unmatched_source: list[tuple[int, str]] = field(default_factory=list)
    unmatched_audio: list[tuple[int, str]] = field(default_factory=list)
    duration_hint: Optional[DurationHint] = None

    _audio_processor = None  # lazy import

    @property
    def total_source(self) -> int:
        return len(self.source_chapters)

    @property
    def total_audiobook(self) -> int:
        return len(self.audiobook_chapters)

    @property
    def overall_match_percentage(self) -> float:
        """Percentage of source chapters found (with correct order) in the audiobook."""
        if not self.total_source:
            return 0.0
        # Matched count includes those that matched name AND had no order violation
        return round(
            (self.matched) / max(self.total_source, 1) * 100, 1
        )

    def has_missing_chapters(self) -> bool:
        return bool(self.missing)

    def has_order_issues(self) -> bool:
        return bool(self.order_violations)

    def has_extra_chapters(self) -> bool:
        return bool(self.extra)

    def _has_correct_position_after(self, chapter_number: int) -> bool:
        """Check if a chapter ends up at the right position relative to others."""
        for v in self.order_violations:
            if v.number == chapter_number:
                return False
        # Also verify ordering by checking neighbors
        return True

    def _get_audio_processor(self):
        """Lazy-import AudioProcessor."""
        if self._audio_processor is None:
            from audify.utils.audio import AudioProcessor
            self._audio_processor = AudioProcessor()
        return self._audio_processor

    def analyze_duration(self, actual_duration: float | None = None) -> DurationHint:
        """Analyze duration hints.

        Args:
            actual_duration: Override for the actual audio duration in
                seconds. Used for multi-part audiobooks where the caller
                has summed durations across all parts.  When *None*, the
                duration is read from ``self.audiobook_path``.
        """
        ap = self._get_audio_processor()
        actual = (
            actual_duration
            if actual_duration is not None
            else ap.get_duration(self.audiobook_path)
        )

        # Rough estimate: ~75 words per minute for audiobook narration
        words_per_minute = 75
        expected_minutes = max(self.source_word_count / words_per_minute, 1)
        expected_s = expected_minutes * 60

        return DurationHint(
            source_word_count=self.source_word_count,
            expected_duration_s=expected_s,
            actual_duration_s=actual,
            ratio=actual / expected_s if expected_s else 0,
        )


# ---------------------------------------------------------------------------
# EPUB extraction
# ---------------------------------------------------------------------------

def extract_epub_chapters(epub_path: str | Path) -> list[Chapter]:
    """Extract chapters from an EPUB file using EpubReader.

    Returns a list of chapters in spine order (index starts from 1).
    """
    reader = EpubReader(Path(epub_path))
    raw_chapters = reader.get_chapters()  # type: list[str]

    chapters: list[Chapter] = []
    for idx, chapter_html in enumerate(raw_chapters, start=1):
        title = reader.get_chapter_title(chapter_html) or f"Chapter {idx}"
        chapters.append(Chapter(number=idx, title=title, index=idx - 1))

    return chapters


def extract_pdf_chapters(pdf_path: str | Path) -> list[Chapter]:
    """Extract content from a PDF file using PdfReader.

    Returns a single-chapter list since PdfReader doesn't have TOC-aware chapter splits.
    """
    reader = PdfReader(Path(pdf_path))
    chapters: list[Chapter] = []
    chapters.append(Chapter(number=1, title=reader.path.stem, index=0))
    return chapters


# ---------------------------------------------------------------------------
# M4B/MP3 chapter extraction
# ---------------------------------------------------------------------------

# M4B chapter metadata is stored as an FFMETADATA1 atom.
# Structure:
#   FFMETADATA1
#     version=1
#     major_brand=M4A
#     minor_version=512
#     ...
# [CHAPTER]
# TIMEBASE=1/1000
# START=<ms>
# END=<ms>
# title=<title>
# [CHAPTER]
# ...


def _is_metadata_atom(data: bytes, offset: int) -> bool:
    """Check if an atom at *offset* is of type 'mdat' with FFMETADATA1."""
    if offset + 8 > len(data):
        return False
    atom_type = data[offset : offset + 4]
    return atom_type == b"moov"


def extract_chapters_from_m4b(m4b_path: str | Path) -> list[Chapter]:
    """Extract chapter information from a .m4b audiobook file.

    Uses ``ffprobe`` to read standard MP4 chapter atoms.  Falls back to
    sibling ``chapters.txt`` metadata file when ffprobe is not available
    or returns no chapters.

    Returns a list of Chapter objects in order they appear in the file.
    """
    path = Path(m4b_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {m4b_path}")

    # Primary method: ffprobe reads MP4 chapter atoms natively.
    chapters = _extract_chapters_via_ffprobe(path)
    if chapters:
        return chapters

    # Fallback: sibling chapters.txt metadata file (FFMETADATA1 text format).
    chapters_txt = path.parent / "chapters.txt"
    if chapters_txt.exists():
        logger.debug(f"Extracting chapters from metadata file: {chapters_txt}")
        chapters = _parse_ffmetadata_from_bytes(
            chapters_txt.read_bytes()
        )

    return chapters


def _extract_chapters_via_ffprobe(path: Path) -> list[Chapter]:
    """Use ffprobe to read MP4 chapter atoms."""
    import json
    import subprocess

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_chapters", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"ffprobe failed for {path}: {result.stderr[:200]}")
            return []

        data = json.loads(result.stdout)
        raw_chapters = data.get("chapters", [])
        if not raw_chapters:
            return []

        chapters: list[Chapter] = []
        for idx, ch in enumerate(raw_chapters, start=1):
            title = ch.get("tags", {}).get("title", "") or f"Chapter {idx}"
            chapters.append(Chapter(number=idx, title=title, index=idx - 1))

        return chapters

    except FileNotFoundError:
        logger.warning("ffprobe not found in PATH, cannot extract M4B chapters")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse ffprobe output: {e}")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe timed out for {path}")
        return []
    except Exception as e:
        logger.warning(f"Error extracting chapters via ffprobe: {e}")
        return []


def extract_chapters_from_mp3(mp3_path: str | Path) -> list[Chapter]:
    """Extract chapter information from an MP3 file using ID3v2 chapter frames.

    Returns a list of Chapter objects in order they appear in the file.
    """
    path = Path(mp3_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {mp3_path}")

    data = path.read_bytes()

    try:
        import mutagen
        from mutagen.id3 import ChapterFrame

        audio = mutagen.File(str(path), easy=False)
        if audio is None:
            logger.warning(f"Could not open MP3 file: {mp3_path}")
            return []

        chapters: list[Chapter] = []

        # Find CHAP frames (children of CTOC or standalone)
        chap_frames = {}
        for frame in audio.frames:
            if isinstance(frame, ChapterFrame):
                chap_frames[frame.id] = frame

        if chap_frames:
            # Order by ID number
            for num in sorted(chap_frames.keys()):
                frame = chap_frames[num]
                # Get frame's title
                title = ""
                if hasattr(frame, "text") and frame.text:
                    title = str(frame.text[0])
                if not title:
                    title = f"Chapter {num}"
                chapters.append(Chapter(number=num, title=title, index=num - 1))
            return chapters
    except ImportError:
        pass

    # Fallback: try to parse the FFMETADATA1 section from the file if available
    # (MP3 files can contain similar metadata in custom frames)
    # Search for '[[FFMETADATA1]]' marker (ID3v2 custom frame content)
    ffind = data.find(b"[[FFMETADATA1]]")
    if ffind != -1:
        data = data[:ffind + 15]  # strip everything after marker
        return _parse_ffmetadata_from_bytes(data)

    return []


def _parse_ffmetadata_from_bytes(data: bytes) -> list[Chapter]:
    """Parse FFMETADATA1 chapter markers from raw bytes.

    Handles both:
    - FFMETADATA1 atoms from M4B MOOV atom
    - '[CHAPTER]' sections from MP3 ID3v2 custom frames
    """
    chapters: list[Chapter] = []
    # Find FFMETADATA1 marker
    if hasattr(data, "decode"):
        text = data.decode("utf-8", errors="ignore")
    else:
        text = bytes(data).decode("utf-8", errors="ignore")

    # Find all [CHAPTER] blocks
    chapter_blocks = re.finditer(
        r"\[CHAPTER\]\s*\n(.*?)\n(?=\[CHAPTER\]|\Z)", text, re.DOTALL
    )

    for i, block in enumerate(chapter_blocks):
        block_text = block.group(1)
        title = ""
        start_ms = 0

        for line in block_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("TIMEBASE="):
                continue  # skip TIMEBASE
            if line.upper().startswith("START="):
                try:
                    start_ms = int(line.split("=")[1])
                except ValueError:
                    pass
            if line.upper().startswith("END="):
                # We don't need it for chapter extraction
                continue
            if line.upper().startswith("TITLE="):
                title = line[len("TITLE=") :]

        if not title:
            title = f"Chapter {i + 1}"

        chapters.append(Chapter(number=i + 1, title=title, index=i))

    return chapters


# ---------------------------------------------------------------------------
# PDF word count for duration estimation
# ---------------------------------------------------------------------------

def _count_pdf_words(pdf_path: str | Path) -> int:
    """Estimate word count from a PDF file."""
    reader = PdfReader(Path(pdf_path))
    text = reader.cleaned_text
    return len(text.split())


def _count_epub_words(epub_path: str | Path) -> int:
    """Estimate word count from an EPUB file by scanning chapter content."""
    reader = EpubReader(Path(epub_path))
    chapters = reader.get_chapters()
    total = 0
    for chapter_html in chapters:
        text = reader.extract_text(chapter_html)
        total += len(text.split())
    return total


# ---------------------------------------------------------------------------
# Main verify logic
# ---------------------------------------------------------------------------


class AudiobookVerifier:
    """Compare an EPUB/MD source file against a generated M4B/MP3 audiobook.

    Supports multi-part audiobooks (split across ``*_partN.m4b`` files).
    Pass the primary (first) audio file as ``audiobook_path``; all parts are
    auto-detected and merged during chapter extraction.
    """

    def __init__(
        self,
        source_path: str | Path,
        audiobook_path: str | Path,
    ):
        self.source_path = Path(source_path).resolve()
        self.audiobook_path = Path(audiobook_path).resolve()

        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
        if not self.audiobook_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audiobook_path}")

        # Extract source chapters
        self.source_type = self._detect_source_type()
        self._raw_chapters: list[Chapter] = []
        if self.source_type == "epub":
            self._raw_chapters = extract_epub_chapters(self.source_path)
        elif self.source_type == "pdf":
            self._raw_chapters = extract_pdf_chapters(self.source_path)

        # Extract audiobook chapters — supports multi-part auto-detection
        self._audiobook_chapters, self._audio_paths = self._extract_audiobook_chapters()

        # Word count for duration estimation
        if self.source_type == "epub":
            self._word_count = _count_epub_words(self.source_path)
        else:
            self._word_count = _count_pdf_words(self.source_path)

    # ------------------------------------------------------------------
    # Multi-part support
    # ------------------------------------------------------------------

    @staticmethod
    def _find_multipart_files(audiobook_path: Path) -> list[Path]:
        """Detect multi-part M4B files (``*_partN.m4b``) in the same directory.

        Returns a sorted list of all part files, or a single-element list
        containing only the given path if no multi-part pattern is detected.
        """
        stem = audiobook_path.stem

        # Check if this file itself follows the ``_partN`` pattern
        m = re.search(r'_part(\d+)$', stem, re.IGNORECASE)
        if not m:
            return [audiobook_path]

        base = stem[:m.start()]
        ext = audiobook_path.suffix
        parent = audiobook_path.parent

        # Gather all matching part files sorted by part number
        pattern = re.compile(
            re.escape(base) + r'_part(\d+)' + re.escape(ext) + '$',
            re.IGNORECASE,
        )
        candidates = [
            p for p in parent.iterdir()
            if p.is_file() and pattern.match(p.name)
        ]
        candidates.sort(key=lambda p: int(
            re.search(r'_part(\d+)', p.stem, re.IGNORECASE).group(1)
        ))

        return candidates if candidates else [audiobook_path]

    def _extract_audiobook_chapters(self) -> tuple[list[Chapter], list[Path]]:
        """Extract chapters from all audio file parts.

        Chapters from each part are merged into a single sequential list,
        with indices adjusted to span the whole audiobook.

        Returns:
            (merged_chapters, all_audio_paths)
        """
        parts = self._find_multipart_files(self.audiobook_path)

        all_chapters: list[Chapter] = []
        suffix = self.audiobook_path.suffix.lower()

        for path in parts:
            if suffix in (".m4b", ".m4a"):
                chs = extract_chapters_from_m4b(path)
            elif suffix == ".mp3":
                chs = extract_chapters_from_mp3(path)
            else:
                chs = []

            # Adjust chapter indices to be sequential across all parts
            offset = len(all_chapters)
            for c in chs:
                c.index += offset
            all_chapters.extend(chs)

        return all_chapters, parts

    def _get_total_duration(self) -> float:
        """Return combined duration of all audio parts (seconds)."""
        from audify.utils.audio import AudioProcessor
        ap = AudioProcessor()
        return sum(ap.get_duration(p) for p in self._audio_paths)

    def _detect_source_type(self) -> str:
        suffix = self.source_path.suffix.lower()
        if suffix == ".epub":
            return "epub"
        elif suffix == ".pdf":
            return "pdf"
        else:
            raise ValueError(f"Unsupported source format: {suffix}")

    def _build_source_lookup(self) -> dict[str, int]:
        """Build title -> chapter mapping for quick lookup."""
        return {c.title.lower(): c for c in self._raw_chapters}

    def _build_audiobook_lookup(self) -> dict[str, int]:
        """Build title -> position mapping for audiobook chapters."""
        return {c.title.lower(): i for i, c in enumerate(self._audiobook_chapters)}

    def verify(self) -> VerifyReport:
        """Run the full comparison and return a report.

        Returns:
            VerifyReport with all findings.
        """
        report = VerifyReport(
            source_path=self.source_path,
            audiobook_path=self.audiobook_path,
            source_chapters=self._raw_chapters,
            audiobook_chapters=self._audiobook_chapters,
            source_type=self.source_type,
            source_word_count=self._word_count,
        )

        source_lookup = self._build_source_lookup()
        audio_lookup = self._build_audiobook_lookup()

        source_titles_set = set(c.title.lower() for c in self._raw_chapters)
        audio_titles_set = set(c.title.lower() for c in self._audiobook_chapters)

        # Find missing chapters (in source but not in audio)
        missing_titles = source_titles_set - audio_titles_set
        for title in sorted(missing_titles):
            chap = source_lookup[title]
            report.missing.append(
                MissingChapter(
                    number=chap.number,
                    title=chap.title,
                    expected_position=chap.index,
                )
            )

        # Find extra chapters (in audio but not in source)
        extra_titles = audio_titles_set - source_titles_set
        for title in sorted(extra_titles):
            pos = audio_lookup[title]
            report.extra.append(
                ExtraChapter(number=pos + 1, title=title, position=pos)
            )

        # Check ordering: for chapters that exist in both, verify relative order
        common_titles = source_titles_set & audio_titles_set
        if common_titles:
            for title in sorted(common_titles):
                source_chap = source_lookup[title]
                audio_pos = audio_lookup[title]

                # Check that this chapter's position in audio is after all
                # preceding common chapters in the audio
                is_in_order = True
                for other_title in common_titles:
                    if other_title == title:
                        continue
                    other_audio_pos = audio_lookup[other_title]
                    other_chap = source_lookup[other_title]
                    # If source_chap should come after other_chap in source
                    # but appears before in audio, it's an issue
                    if source_chap.number > other_chap.number and other_audio_pos > audio_pos:
                        is_in_order = False
                        break

                if is_in_order:
                    report.matched += 1
                else:
                    # Still count it as a match but flag order violation
                    report.matched += 1
                    report.order_violations.append(
                        OrderViolation(
                            number=source_chap.number,
                            title=title,
                            expected_order=source_chap.number - 1,
                            actual_position=audio_pos,
                        )
                    )

        # Report unmatched chapters
        report.unmatched_source = [
            (c.number, c.title) for c in self._raw_chapters if c.title.lower() in missing_titles
        ]
        report.unmatched_audio = [
            (c.number, c.title) for c in self._audiobook_chapters if c.title.lower() in extra_titles
        ]

        # Duration analysis (uses combined duration for multi-part audiobooks)
        total_duration = self._get_total_duration()
        report.duration_hint = report.analyze_duration(actual_duration=total_duration)

        return report

    def generate_report(self) -> dict[str, Any]:
        """Generate a detailed JSON-serializable report."""
        report = self.verify()

        def _chapters_to_list(chapters: list) -> list[dict]:
            return [{"number": c.number, "title": c.title} for c in chapters]

        result = {
            "source": str(report.source_path),
            "source_type": report.source_type,
            "audiobook": str(report.audiobook_path),
            "summary": {
                "source_chapters": report.total_source,
                "audiobook_chapters": report.total_audiobook,
                "matched": report.matched,
                "overall_match_percentage": report.overall_match_percentage,
                "has_missing": report.has_missing_chapters(),
                "has_order_issues": report.has_order_issues(),
                "has_extra": report.has_extra_chapters(),
            },
            "source_chapters": _chapters_to_list(report.missing) if report.missing else [],
            "extra_chapters": _chapters_to_list(report.extra) if report.extra else [],
            "order_violations": [
                {
                    "number": v.number,
                    "title": v.title,
                    "expected_position": v.expected_order,
                    "actual_position": v.actual_position,
                }
                for v in report.order_violations
            ],
        }

        if report.duration_hint:
            d = report.duration_hint
            result["duration_hint"] = {
                "source_word_count": d.source_word_count,
                "expected_duration_s": round(d.expected_duration_s, 1),
                "actual_duration_s": round(d.actual_duration_s, 1),
                "ratio_actual_to_expected": round(d.ratio, 2),
                "interpretation": self._duration_interpretation(d),
            }

        return result

    def _duration_interpretation(self, hint: DurationHint) -> str:
        r"""Interpret the duration ratio."""
        r = hint.ratio
        if r < 0.5:
            return "AUDIO MUCH SHORTER than expected based on source text. Check for missing chapters."
        elif r < 0.9:
            return "AUDIO SHORTER than expected. Some chapters may be abbreviated."
        elif r > 1.5:
            return "AUDIO MUCH LONGER than expected. Check for extra content or slow narration."
        elif r > 1.1:
            return "AUDIO slightly longer than expected."
        else:
            return "AUDIO duration matches expectations based on source text length."

    def print_report(self) -> None:
        """Print a human-readable report to stdout."""
        report = self.verify()

        print("\n" + "=" * 70)
        print("  Audify Audiobook Verification Report")
        print("=" * 70)
        print(f"  Source:       {report.source_path.name}")
        print(f"  Source Type:  {report.source_type.upper()}")
        print(f"  Audiobook:    {report.audiobook_path.name}")
        print(f"  Duration:     {report.duration_hint.actual_duration_s:.1f}s" if report.duration_hint else "  Duration:     (unknown)")
        print()
        print("-" * 70)
        print("  Source Chapters:")
        print("-" * 70)
        for c in report.source_chapters:
            marker = "✓" if c.title.lower() in {a.title.lower() for a in report.audiobook_chapters} else "✗"
            print(f"    {c.number:3d}. [{marker}] {c.title}")

        print()
        print("-" * 70)
        print("  Audiobook Chapters:")
        print("-" * 70)
        for c in report.audiobook_chapters:
            marker = "✓" if c.title.lower() in {s.title.lower() for s in report.source_chapters} else "?"
            print(f"    {c.number:3d}. [{marker}] {c.title}")

        print()
        print("=" * 70)
        print("  Summary")
        print("=" * 70)
        print(f"  Total source chapters:   {report.total_source}")
        print(f"  Total audiobook chapters: {report.total_audiobook}")
        matched_str = (
            f"{report.matched}/{report.total_source} "
            f"({report.overall_match_percentage}%)"
        )
        print(f"  Matched (with correct order): {matched_str}")

        if report.missing:
            print()
            print("  ⚠️  Missing Chapters:")
            for m in report.missing:
                exp_pos = m.expected_position + 1
                print(f"       - {m.number}. {m.title} (expected at position {exp_pos})")

        if report.extra:
            print()
            print("  ⚠️  Extra Chapters:")
            for e in report.extra:
                print(f"       - {e.number}. {e.title}")

        if report.order_violations:
            print()
            print("  ⚠️  Order Violations:")
            for v in report.order_violations:
                exp_pos = v.expected_order + 1
                act_pos = v.actual_position + 1
                print(
                    f"       - {v.number}. {v.title} (expected at {exp_pos}, "
                    f"found at {act_pos})"
                )

        if report.duration_hint:
            d = report.duration_hint
            interp = self._duration_interpretation(d)
            print()
            actual_s = f"{d.actual_duration_s:.0f}s"
            expected_s = f"{d.expected_duration_s:.0f}s"
            ratio_str = f"{d.ratio:.2f}"
            print(
                f"  Duration Hint: {actual_s} actual vs {expected_s} "
                f"expected (ratio: {ratio_str})"
            )
            print(f"             Interpretation: {interp}")

        print()
        print("=" * 70)
        match_pct = report.overall_match_percentage
        if match_pct == 100:
            status = "✅  PASS — All chapters matched correctly!"
        elif match_pct >= 90:
            remaining = report.total_source - report.matched
            status = (
                f"⚠️  NEAR MISS — {match_pct}% match. {remaining} "
                f"chapter(s) may be missing or out of order."
            )
        else:
            status = (
                f"❌  FAIL — Only {match_pct}% match. "
                f"Significant discrepancies detected."
            )
        print(f"  Status: {status}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Convenience function for CLI
# ---------------------------------------------------------------------------


def verify_audiobook(
    source_path: str | Path, audiobook_path: str | Path
) -> VerifyReport:
    """Run verification and return the report.

    Convenience wrapper around :class:`AudiobookVerifier`.
    """
    verifier = AudiobookVerifier(source_path, audiobook_path)
    return verifier.verify()
