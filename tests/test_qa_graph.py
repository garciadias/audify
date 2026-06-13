"""Tests for the LangGraph QA pipeline skeleton (issue #35)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audify.qa.graph import (
    _strip_mermaid_frontmatter,
    build_graph,
    render_mermaid,
    run_graph,
)
from audify.qa.nodes.assemble import assemble_node
from audify.qa.nodes.script_gen import script_gen_node
from audify.qa.nodes.synthesize import synthesize_node
from audify.readers.pdf import PdfReader


def _make_creator(tmp_path: Path, chapters: list[str] | None = None) -> MagicMock:
    """Return a minimal mock AudiobookCreator sufficient for the graph nodes."""
    if chapters is None:
        chapters = ["Chapter one text.", "Chapter two text."]

    creator = MagicMock()
    creator.max_chapters = None
    creator.audiobook_path = tmp_path
    creator.chapter_titles = []
    creator.mode = "full"
    # Default to non-interactive (matches `-y/--confirm` flag): skip the
    # confirm_node prompt so the test doesn't hang on stdin.
    creator.confirm = False

    # reader is a non-EpubReader so we get the simple path. Using
    # ``MagicMock(spec=PdfReader)`` gives a durable ``isinstance`` answer
    # (False for EpubReader, True for PdfReader) instead of rebinding
    # ``__class__``, which would break if EpubReader joined a wider MRO.
    creator.reader = MagicMock(spec=PdfReader)
    creator.reader.cleaned_text = "\n\n".join(chapters)

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

    def test_process_graph_skips_synthesis(self):
        """`process` mode: scripts only, no synthesize/assemble nodes."""
        graph = build_graph("process")
        nodes = set(graph.get_graph().nodes)
        assert {"read", "script_gen", "report"}.issubset(nodes)
        assert "synthesize" not in nodes
        assert "assemble" not in nodes

    def test_synthesize_graph_skips_read_and_script_gen(self):
        """`synthesize` mode: load saved scripts, no read/script_gen nodes."""
        graph = build_graph("synthesize")
        nodes = set(graph.get_graph().nodes)
        assert {"load_scripts", "synthesize", "assemble", "report"}.issubset(nodes)
        assert "read" not in nodes
        assert "script_gen" not in nodes

    def test_build_graph_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown graph mode"):
            build_graph("nonsense")


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
        # Top-level pipeline status mirrors completion against expected
        # chapter count.
        assert report["pipeline_status"] == "complete"
        assert report["episodes_synthesised"] == report["chapters_expected"]
        chapters = report["chapters"]
        assert "chapter_1" in chapters
        # Until the cyclic detectors run, "skeleton" is the honest verdict —
        # no cycle node has yet populated ``best_wer``/``retry_budget``.
        assert chapters["chapter_1"]["verdict"] == "skeleton"
        assert chapters["chapter_1"]["flags"] == []

    def test_run_graph_returns_audiobook_path(self, tmp_path):
        creator = _make_creator(tmp_path)

        with patch("audify.readers.ebook.EpubReader"):
            result = run_graph(creator)

        assert result == tmp_path

    def test_run_graph_with_max_chapters(self, tmp_path):
        from audify.readers.ebook import EpubReader

        creator = _make_creator(tmp_path)
        creator.max_chapters = 1
        # Simulate an EpubReader with 3 chapters. ``spec=EpubReader`` gives a
        # durable ``isinstance`` answer without rebinding ``__class__``.
        creator.reader = MagicMock(spec=EpubReader)
        creator.reader.get_chapters.return_value = ["Ch1", "Ch2", "Ch3"]
        creator.reader.get_chapter_title.side_effect = lambda c: f"Title of {c}"

        run_graph(creator)

        # max_chapters=1 → only 1 episode synthesized
        assert creator.synthesize_episode.call_count == 1


class TestPreflightGuard:
    """`run_graph` invokes `_verify_tts_provider_available` before TTS modes."""

    def test_full_mode_runs_preflight(self, tmp_path):
        creator = _make_creator(tmp_path)
        creator.mode = "full"
        run_graph(creator)
        creator._verify_tts_provider_available.assert_called_once()

    def test_synthesize_mode_runs_preflight(self, tmp_path):
        scripts_path = tmp_path / "scripts"
        scripts_path.mkdir()
        creator = _make_creator(tmp_path)
        creator.mode = "synthesize"
        creator.scripts_path = scripts_path
        run_graph(creator)
        creator._verify_tts_provider_available.assert_called_once()

    def test_process_mode_skips_preflight(self, tmp_path):
        """Process mode performs no TTS, so the preflight is unnecessary."""
        creator = _make_creator(tmp_path)
        creator.mode = "process"
        run_graph(creator)
        creator._verify_tts_provider_available.assert_not_called()

    def test_preflight_failure_aborts_before_graph(self, tmp_path):
        """A failing preflight surfaces as an exception, never reaches read."""
        creator = _make_creator(tmp_path)
        creator.mode = "full"
        creator._verify_tts_provider_available.side_effect = RuntimeError(
            "TTS provider 'kokoro' is not available."
        )
        with pytest.raises(RuntimeError, match="not available"):
            run_graph(creator)
        creator.generate_audiobook_script.assert_not_called()


class TestConfirmNode:
    """`confirm_node` prints the TOC and may abort the pipeline."""

    def test_confirm_disabled_passes_through(self, tmp_path):
        """When `creator.confirm` is False, run_graph proceeds without prompting."""
        creator = _make_creator(tmp_path)
        creator.confirm = False  # default in fixture; made explicit here
        run_graph(creator)
        # TOC is always shown
        creator.progress.print_table_of_contents.assert_called_once()
        # Generation still ran (no abort)
        creator.generate_audiobook_script.assert_called()

    def test_confirm_yes_continues(self, tmp_path, monkeypatch):
        """User typing `y` continues the pipeline."""
        creator = _make_creator(tmp_path)
        creator.confirm = True
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        run_graph(creator)
        creator.generate_audiobook_script.assert_called()

    def test_confirm_no_aborts(self, tmp_path, monkeypatch):
        """User typing `n` aborts: empty chapters → no script_gen, no synth."""
        creator = _make_creator(tmp_path)
        creator.confirm = True
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        run_graph(creator)
        creator.generate_audiobook_script.assert_not_called()
        creator.synthesize_episode.assert_not_called()
        creator.create_m4b.assert_not_called()

    def test_confirm_non_tty_aborts_safely(self, tmp_path, monkeypatch):
        """OSError/EOFError on stdin (non-tty / captured) aborts, no hang."""
        creator = _make_creator(tmp_path)
        creator.confirm = True

        def _no_stdin(_prompt):
            raise OSError("no stdin")

        monkeypatch.setattr("builtins.input", _no_stdin)
        run_graph(creator)
        creator.synthesize_episode.assert_not_called()


class TestRunGraphModeDispatch:
    """`run_graph` routes to the right sub-graph based on ``creator.mode``."""

    def test_process_mode_skips_synthesis(self, tmp_path):
        """`process`: scripts saved, no TTS, no M4B."""
        creator = _make_creator(tmp_path)
        creator.mode = "process"

        run_graph(creator)

        creator.generate_audiobook_script.assert_called()
        # No synthesis / assembly should happen in process mode
        creator.synthesize_episode.assert_not_called()
        creator.create_m4b.assert_not_called()

    def test_synthesize_mode_loads_scripts_and_skips_read(self, tmp_path):
        """`synthesize`: loads saved scripts, no LLM, no `read`."""
        scripts_path = tmp_path / "scripts"
        scripts_path.mkdir()
        # Two pre-existing scripts on disk, plus a chapter_titles.json
        (scripts_path / "episode_001_script.txt").write_text("Script for chapter 1.")
        (scripts_path / "episode_002_script.txt").write_text("Script for chapter 2.")
        (scripts_path / "chapter_titles.json").write_text(
            json.dumps(["Chapter 1", "Chapter 2"])
        )

        creator = _make_creator(tmp_path)
        creator.mode = "synthesize"
        creator.scripts_path = scripts_path
        creator._load_chapter_titles.return_value = ["Chapter 1", "Chapter 2"]

        run_graph(creator)

        # No LLM should be called in synthesize mode
        creator.generate_audiobook_script.assert_not_called()
        # But both pre-existing scripts should be synthesised
        assert creator.synthesize_episode.call_count == 2
        creator.create_m4b.assert_called_once()

    def test_synthesize_mode_with_no_saved_scripts_is_quiet(self, tmp_path):
        """`synthesize` with no scripts on disk: log error, no exception."""
        scripts_path = tmp_path / "scripts"
        scripts_path.mkdir()

        creator = _make_creator(tmp_path)
        creator.mode = "synthesize"
        creator.scripts_path = scripts_path

        run_graph(creator)

        creator.synthesize_episode.assert_not_called()
        creator.create_m4b.assert_not_called()


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

    def test_strip_mermaid_frontmatter_no_frontmatter(self):
        """Pass through plain Mermaid without --- blocks."""
        diagram = "graph TD;\nA-->B;\n"
        assert _strip_mermaid_frontmatter(diagram) == diagram

    def test_strip_mermaid_frontmatter_strips_config(self):
        """Remove leading ---config:...--- block."""
        diagram = (
            "---\nconfig:\n  flowchart:\n    curve: linear\n"
            "---\ngraph TD;\nA-->B;\n"
        )
        result = _strip_mermaid_frontmatter(diagram)
        assert "config" not in result
        assert "graph TD" in result

    def test_strip_mermaid_frontmatter_missing_closing(self):
        """_strip_mermaid_frontmatter handles incomplete --- blocks gracefully."""
        diagram = "---\nconfig: true\ngraph TD;\n"
        assert _strip_mermaid_frontmatter(diagram) == diagram


class TestAssembleNode:
    def test_assemble_node_with_paths(self):
        """assemble_node calls create_m4b when episode_paths is non-empty."""
        creator = MagicMock()
        state = {"creator": creator, "episode_paths": ["ep1.mp3", "ep2.mp3"]}
        result = assemble_node(state)
        creator.create_m4b.assert_called_once()
        assert result == {}

    def test_assemble_node_empty_paths(self):
        """assemble_node skips create_m4b when episode_paths is empty."""
        creator = MagicMock()
        state = {"creator": creator, "episode_paths": []}
        result = assemble_node(state)
        creator.create_m4b.assert_not_called()
        assert result == {}


class TestScriptGenNode:
    def test_script_gen_node_error_handling(self):
        """script_gen_node handles exceptions during script generation."""
        creator = MagicMock()
        creator.generate_audiobook_script.side_effect = Exception("LLM error")
        creator.progress = MagicMock()
        creator.chapter_titles = []

        state = {
            "creator": creator,
            "chapters": ["Chapter content"],
            "chapter_titles": ["Chapter 1"],
        }
        result = script_gen_node(state)
        assert "chapter_scripts" in result
        # Script generation failed, so chapter_scripts should be empty
        assert result["chapter_scripts"] == []


class TestSynthesizeNode:
    def test_synthesize_node_success(self, tmp_path):
        """synthesize_node returns episode_paths for successful episodes."""
        creator = MagicMock()
        creator.progress = MagicMock()
        ep_path = tmp_path / "episode_001.mp3"
        ep_path.write_bytes(b"fake")
        creator.synthesize_episode.return_value = ep_path

        state = {
            "creator": creator,
            "chapter_scripts": [(1, "Script text.")],
            "chapter_titles": ["Chapter 1"],
        }
        result = synthesize_node(state)
        assert len(result["episode_paths"]) == 1
        assert result["episode_paths"][0] == ep_path

    def test_synthesize_node_tts_error_re_raised(self):
        """synthesize_node re-raises TTSSynthesisError."""
        from audify.text_to_speech import TTSSynthesisError

        creator = MagicMock()
        creator.progress = MagicMock()
        creator.synthesize_episode.side_effect = TTSSynthesisError(
            "TTS failed", failed_batches=1, total_batches=1
        )

        state = {
            "creator": creator,
            "chapter_scripts": [(1, "Script text.")],
            "chapter_titles": ["Chapter 1"],
        }
        with pytest.raises(TTSSynthesisError):
            synthesize_node(state)

    def test_synthesize_node_generic_error_re_raised(self):
        """synthesize_node re-raises generic exceptions."""
        creator = MagicMock()
        creator.progress = MagicMock()
        creator.synthesize_episode.side_effect = RuntimeError("Unexpected error")

        state = {
            "creator": creator,
            "chapter_scripts": [(1, "Script text.")],
            "chapter_titles": ["Chapter 1"],
        }
        with pytest.raises(RuntimeError):
            synthesize_node(state)

    def test_synthesize_node_episode_not_found(self, tmp_path):
        """synthesize_node logs warning when episode file does not exist."""
        creator = MagicMock()
        creator.progress = MagicMock()
        creator.synthesize_episode.return_value = tmp_path / "episode_001.mp3"

        state = {
            "creator": creator,
            "chapter_scripts": [(1, "Script.")],
            "chapter_titles": ["Chapter 1"],
        }
        result = synthesize_node(state)
        assert result["episode_paths"] == []
