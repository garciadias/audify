"""Tests for Book metadata built from an EPUB and its M4B tag carrier."""

import subprocess
from pathlib import Path

import pytest
from ebooklib import epub

from audify.library.metadata import (
    BookMetadata,
    SeriesEntry,
    book_metadata_from_epub,
    narrator_label,
)
from audify.utils.m4b_builder import escape_ffmetadata, write_metadata_header

OPF = "http://www.idpf.org/2007/opf"


def _book(**fields) -> epub.EpubBook:
    book = epub.EpubBook()
    book.set_identifier(fields.get("identifier", "urn:uuid:1234"))
    if "title" in fields:
        book.set_title(fields["title"])
    if "language" in fields:
        book.set_language(fields["language"])
    for author in fields.get("authors", []):
        book.add_author(author)
    for name in ("date", "publisher", "description"):
        if name in fields:
            book.add_metadata("DC", name, fields[name])
    for subject in fields.get("subjects", []):
        book.add_metadata("DC", "subject", subject)
    if "isbn" in fields:
        book.add_metadata("DC", "identifier", fields["isbn"], {"id": "isbn"})
    if "series" in fields:
        book.add_metadata(
            None, "meta", "", {"name": "calibre:series", "content": fields["series"]}
        )
        book.add_metadata(
            None,
            "meta",
            "",
            {"name": "calibre:series_index", "content": fields["series_index"]},
        )
    return book


def _round_trip(book: epub.EpubBook, tmp_path: Path) -> epub.EpubBook:
    """Write and re-read, so the test exercises a parsed OPF as Audify sees it."""
    chapter = epub.EpubHtml(title="One", file_name="one.xhtml", lang="en")
    chapter.content = "<h1>One</h1><p>Text.</p>"
    book.add_item(chapter)
    book.toc = [chapter]
    book.spine = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)
    return epub.read_epub(str(path))


def test_full_opf_maps_to_book_metadata(tmp_path):
    book = _round_trip(
        _book(
            title="A Ordem do Capital",
            language="pt",
            authors=["Clara E. Mattei"],
            date="2023-05-01",
            publisher="Boitempo",
            description="<p>Como economistas <b>inventaram</b> a austeridade.</p>",
            subjects=["Economics", "History"],
            isbn="urn:isbn:978-65-5717-123-4",
            series="Ensaios",
            series_index="2.0",
        ),
        tmp_path,
    )
    meta = book_metadata_from_epub(book, narrators=["Kokoro (af_bella)"])

    assert meta.title == "A Ordem do Capital"
    assert meta.authors == ["Clara E. Mattei"]
    assert meta.language == "pt"
    assert meta.published_year == "2023"
    assert meta.publisher == "Boitempo"
    assert meta.description == "Como economistas inventaram a austeridade."
    assert meta.genres == ["Economics", "History"]
    assert meta.isbn == "9786557171234"
    assert meta.series == [SeriesEntry("Ensaios", "2")]
    assert meta.narrators == ["Kokoro (af_bella)"]


def test_missing_fields_stay_empty(tmp_path):
    book = _round_trip(_book(title="Only a title", language="en"), tmp_path)
    meta = book_metadata_from_epub(book)

    assert meta.authors == []
    assert meta.isbn == ""
    assert meta.description == ""
    assert meta.series == []
    tags = meta.m4b_tags()
    assert set(tags) == {"title", "album"}
    assert "Unknown" not in " ".join(tags.values())


def test_uuid_identifier_is_not_an_isbn(tmp_path):
    book = _round_trip(_book(title="T", language="en"), tmp_path)
    assert book_metadata_from_epub(book).isbn == ""


def test_m4b_tags_use_standard_mp4_keys_only():
    meta = BookMetadata(
        title="T",
        authors=["A", "B"],
        narrators=["Kokoro (af_bella)"],
        published_date="1999-01-01",
        genres=["Fiction"],
        description="D",
        series=[SeriesEntry("S", "3")],
        language="pt",
        publisher="P",
        isbn="9780000000002",
    )
    tags = meta.m4b_tags(track=1, total_tracks=2)

    assert tags == {
        "title": "T",
        "album": "T",
        "artist": "A, B",
        "album_artist": "A, B",
        "composer": "Kokoro (af_bella)",
        "date": "1999",
        "genre": "Fiction",
        "comment": "D",
        "description": "D",
        "show": "S",
        "grouping": "S",
        "episode_id": "3",
        "track": "1/2",
    }


@pytest.mark.parametrize(
    ("engine", "voice", "expected"),
    [
        ("kokoro", "af_bella", "Kokoro (af_bella)"),
        ("openai", None, "Openai"),
        ("", "x", "TTS (x)"),
    ],
)
def test_narrator_label(engine, voice, expected):
    assert narrator_label(engine, voice) == expected


def test_escape_ffmetadata():
    assert escape_ffmetadata("a=b;c#d\\e") == "a\\=b\\;c\\#d\\\\e"
    assert escape_ffmetadata("line1\nline2") == "line1\\\nline2"


def test_header_writes_tags_before_chapters(tmp_path):
    path = tmp_path / "chapters.txt"
    write_metadata_header(path, {"title": "X = Y", "artist": "", "genre": "G"})
    text = path.read_text()

    assert text.startswith(";FFMETADATA1\n")
    assert "title=X \\= Y\n" in text
    assert "artist=" not in text
    assert "genre=G\n" in text


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg not installed",
)
def test_tags_survive_into_m4b(tmp_path):
    """End to end through FFmpeg: the tags land in the M4B and ffprobe reads them."""
    from audify.utils.m4b_builder import assemble_m4b

    raw = tmp_path / "in.tmp.m4b"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=22050:cl=mono",
            "-t",
            "1",
            "-c:a",
            "aac",
            "-f",
            "mp4",
            str(raw),
        ],
        check=True,
    )
    meta = BookMetadata(
        title="Título; com = sinais",
        authors=["Clara E. Mattei"],
        narrators=["Kokoro (pf_dora)"],
        published_date="2023",
        description="Linha 1\nLinha 2",
    )
    metadata_path = tmp_path / "chapters.txt"
    write_metadata_header(metadata_path, meta.m4b_tags())
    out = tmp_path / "out.m4b"
    assemble_m4b(raw, metadata_path, out)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags",
            "-of",
            "default=nw=1",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "TAG:title=Título; com = sinais" in probe
    assert "TAG:artist=Clara E. Mattei" in probe
    assert "TAG:composer=Kokoro (pf_dora)" in probe
    assert "TAG:date=2023" in probe


def test_creator_builds_tags_from_a_real_epub(tmp_path):
    """The audiobook creator turns the Source ebook's OPF into M4B tags."""
    from audify.audiobook_creator import AudiobookCreator
    from audify.readers.ebook import EpubReader

    book = _book(
        title="Everything Is Illuminated",
        language="en",
        authors=["Jonathan Safran Foer"],
        date="2002",
    )
    chapter = epub.EpubHtml(title="One", file_name="one.xhtml", lang="en")
    chapter.content = "<h1>One</h1><p>Text.</p>"
    book.add_item(chapter)
    book.toc = [chapter]
    book.spine = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)

    creator = object.__new__(AudiobookCreator)
    creator.reader = EpubReader(path)
    creator.title = creator.reader.title
    creator.language = "en"
    creator.tts_provider = "kokoro"
    creator.speaker = "af_bella"
    creator.book_metadata = creator._build_book_metadata()

    tags = creator._book_tags(track=2, total_tracks=3)
    assert tags["title"] == "Everything Is Illuminated"
    assert tags["artist"] == "Jonathan Safran Foer"
    assert tags["composer"] == "Kokoro (af_bella)"
    assert tags["date"] == "2002"
    assert tags["track"] == "2/3"


def test_authors_drop_separators_and_non_author_roles():
    from audify.library.metadata import OPF_NS, _authors

    book = epub.EpubBook()
    book.add_author("Clara E. Mattei;", role="aut", uid="c1")
    book.add_author("A. Translator", role="trl", uid="c2")  # EPUB 3 refines
    book.add_author("Ann One; Bob Two", uid="c3")
    book.add_metadata("DC", "creator", "E. Ditor", {"{%s}role" % OPF_NS: "edt"})
    assert _authors(book) == ["Clara E. Mattei", "Ann One", "Bob Two"]
