"""Tests for the LangGraph QA pipeline skeleton (issue #35)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from audify.qa.graph import build_graph, render_mermaid, run_graph


def _make_creator(tmp_path: Path, chapters: list[str] | None = None) -> MagicMock:
    """Return a minimal mock AudiobookCreator sufficient for the graph nodes."""
    if chapters is None:
        chapters = ["Chapter one text.", "Chapter two text."]

    creator = MagicMock()
    creator.max_chapters = None
    creator.audiobook_path = tmp_path
    creator.chapter_titles = []

    # reader is a non-EpubReader so we get the simple path
    creator.reader.cleaned_text = "\n\n".join(chapters)
    # make isinstance(..., EpubReader) return False

    creator.reader.__class__ = MagicMock  # not EpubReader

    # script_gen returns a fake script per episode
    def _gen_script(chapter_content: str, episode_number: int) -> str:
        return f"Narration for episode {episode_number}."

    creator.generate_audiobook_script.side_effect = _gen_script

    # synthesize_episode creates a dummy mp3 file
    def _synth(script: str, episode_number: int) -> Path:
        p = tmp_path / f"episode_{episode_number:03d}.mp3"
        p.write_bytes(b"FAKE")
        return p

    creator.synthesize_episode.side_effect = _synth

    creator._validate_chapters.return_value = None
    creator._save_chapter_titles.return_value = None
    creator.create_m4b.return_value = None
    creator.progress = MagicMock()

    return creator


class TestBuildGraph:
    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        nodes = set(graph.get_graph().nodes)
        expected = {"read", "script_gen", "synthesize", "assemble", "report"}
        assert expected.issubset(nodes)

    def test_graph_is_acyclic(self):
        """The skeleton must have no back-edges — all edges must go forward."""
        graph = build_graph()
        node_order = ["read", "script_gen", "synthesize", "assemble", "report"]
        rank = {name: i for i, name in enumerate(node_order)}

        for edge in graph.get_graph().edges:
            src, dst = edge[0], edge[1]
            if src in rank and dst in rank:
                assert rank[src] < rank[dst], (
                    f"Back-edge detected: {src} → {dst}; "
                    "skeleton must be acyclic."
                )


class TestRunGraph:
    def test_run_graph_calls_all_phases(self, tmp_path):
        creator = _make_creator(tmp_path)

        with patch("audify.readers.ebook.EpubReader") as _mock_epub:
            run_graph(creator)

        creator.generate_audiobook_script.assert_called()
        creator.synthesize_episode.assert_called()
        creator.create_m4b.assert_called_once()

    def test_run_graph_writes_quality_report(self, tmp_path):
        creator = _make_creator(tmp_path)

        with patch("audify.readers.ebook.EpubReader"):
            run_graph(creator)

        report_path = tmp_path / "quality_report.json"
        assert report_path.exists(), "Quality report JSON must be written"
        report = json.loads(report_path.read_text())
        assert "chapter_1" in report
        assert report["chapter_1"]["verdict"] == "ok"
        assert report["chapter_1"]["flags"] == []

    def test_run_graph_returns_audiobook_path(self, tmp_path):
        creator = _make_creator(tmp_path)

        with patch("audify.readers.ebook.EpubReader"):
            result = run_graph(creator)

        assert result == tmp_path

    def test_run_graph_with_max_chapters(self, tmp_path):
        from audify.readers.ebook import EpubReader

        creator = _make_creator(tmp_path)
        creator.max_chapters = 1
        # Simulate an EpubReader with 3 chapters
        creator.reader.__class__ = EpubReader
        creator.reader.get_chapters.return_value = ["Ch1", "Ch2", "Ch3"]
        creator.reader.get_chapter_title.side_effect = lambda c: f"Title of {c}"

        run_graph(creator)

        # max_chapters=1 → only 1 episode synthesized
        assert creator.synthesize_episode.call_count == 1


class TestRenderMermaid:
    def test_render_mermaid_writes_file(self, tmp_path):
        out = tmp_path / "docs" / "graph.md"
        render_mermaid(out)
        assert out.exists()
        content = out.read_text()
        assert "```mermaid" in content
        assert "read" in content

    def test_render_mermaid_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "c" / "graph.md"
        render_mermaid(out)
        assert out.exists()
