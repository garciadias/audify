"""Tests for long books: ordered Parts and one continuous chapter list."""

import shutil
import subprocess
from pathlib import Path

import pytest

from audify.audiobook_creator import AudiobookCreator
from audify.utils.constants import MAX_PART_HOURS

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _mp3(path: Path, seconds: float) -> Path:
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
            str(seconds),
            "-c:a",
            "libmp3lame",
            str(path),
        ],
        check=True,
    )
    return path


def test_split_threshold_is_named():
    assert MAX_PART_HOURS == 14.0


@needs_ffmpeg
def test_chapter_list_runs_continuously_across_episodes(tmp_path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    _mp3(episodes / "episode_001.mp3", 2)
    _mp3(episodes / "episode_002.mp3", 3)
    _mp3(episodes / "episode_003.mp3", 1)

    creator = object.__new__(AudiobookCreator)
    creator.episodes_path = episodes
    creator.chapter_titles = ["Prologue", "", "Epilogue"]

    marks = creator._whole_book_chapters()

    assert [m.title for m in marks] == ["Prologue", "Episode 2", "Epilogue"]
    assert marks[0].start == 0.0
    for previous, current in zip(marks, marks[1:]):
        assert current.start == pytest.approx(previous.end)
    assert marks[-1].end == pytest.approx(6.0, abs=0.2)


def test_no_episodes_means_no_chapter_list(tmp_path):
    creator = object.__new__(AudiobookCreator)
    creator.episodes_path = tmp_path / "missing"
    assert creator._whole_book_chapters() == []
