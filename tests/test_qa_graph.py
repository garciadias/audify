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
from audify.utils.text import format_title_announcement


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
    # Default `warn_stop` to False so the report node's warn-stop prompt
    # never fires in tests that don't set it explicitly. A MagicMock would
    # otherwise read as truthy here.
    creator.warn_stop = False
    # Cycle-3 fidelity check is opt-in; keep it off for the skeleton tests so
    # they don't route through the STT round-trip. A MagicMock attribute would
    # otherwise read as truthy here. The dedicated fidelity tests below set it.
    creator.fidelity_check = False

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
    def _synth(script: str, episode_number: int, **kwargs) -> Path:
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
        expected = {
            "read",
            "text_quality",
            "escalate",
            "script_gen",
            "script_validity",
            "synthesize",
            "fidelity",
            "assemble",
            "report",
        }
        assert expected.issubset(nodes)

    def test_full_graph_has_cycle_1_back_edge(self):
        """Cycle 1: full mode must contain the text_quality → escalate back-edge."""
        graph = build_graph()
        edges = {(e[0], e[1]) for e in graph.get_graph().edges}
        assert ("text_quality", "escalate") in edges, (
            "cycle-1 escalation edge missing: text_quality must route to escalate"
        )
        assert ("escalate", "text_quality") in edges, (
            "cycle-1 escalation back-edge missing: escalate must loop back "
            "to text_quality"
        )

    def test_full_graph_has_fidelity_back_edge(self):
        """Cycle 3: full mode must contain the fidelity → synthesize back-edge."""
        graph = build_graph()
        edges = {(e[0], e[1]) for e in graph.get_graph().edges}
        assert ("synthesize", "fidelity") in edges
        assert ("fidelity", "synthesize") in edges, (
            "cycle-3 retry edge missing: fidelity must be able to loop back "
            "to synthesize"
        )

    def test_process_graph_has_cycle_1_back_edge(self):
        """`process` mode has the cycle-1 ``text_quality → escalate`` back-edge."""
        graph = build_graph("process")
        edges = {(e[0], e[1]) for e in graph.get_graph().edges}
        assert ("text_quality", "escalate") in edges, (
            "process mode must have cycle-1 escalation edge"
        )
        assert ("escalate", "text_quality") in edges, (
            "process mode must have cycle-1 escalation back-edge"
        )

    def test_process_graph_has_cycle_2_back_edge(self):
        """`process` mode has the cycle-2 ``script_validity → script_gen`` reroute."""
        graph = build_graph("process")
        edges = {(e[0], e[1]) for e in graph.get_graph().edges}
        assert ("script_validity", "script_gen") in edges, (
            "process mode must have cycle-2 reroute back-edge"
        )

    def test_process_graph_has_no_cycle_3(self):
        """`process` mode performs no TTS, so it must not have fidelity or cycle-3."""
        graph = build_graph("process")
        edges = {(e[0], e[1]) for e in graph.get_graph().edges}
        cycle_3_edges = {e for e in edges if "fidelity" in e[0] or "synthesize" in e[0]}
        assert not cycle_3_edges, (
            f"process mode must not have cycle-3 edges, found: {cycle_3_edges}"
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
        # No cycle has fired against the stub-data path, so every chapter
        # is ``clean``. The detectors (#3/#4/#5) will populate ``flags`` as
        # they ship; the aggregator already speaks their schema.
        assert chapters["chapter_1"]["verdict"] == "clean"
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


class TestResynthesize:
    """Retry pass of ``synthesize_node`` (``pending_retry`` populated)."""

    def _retry_creator(self, tmp_path: Path) -> MagicMock:
        creator = MagicMock()
        creator.progress = MagicMock()
        creator.episodes_path = tmp_path
        creator._get_tts_config.return_value.max_text_length = 4000
        return creator

    def test_skips_episode_without_script(self, tmp_path):
        """A pending episode number absent from chapter_scripts is skipped."""
        creator = self._retry_creator(tmp_path)

        state = {
            "creator": creator,
            "chapter_scripts": [(1, "Script text.")],
            "pending_retry": [2],  # not in chapter_scripts
            "retry_budget": {},
        }
        result = synthesize_node(state)

        assert result == {"episodes_to_check": [2], "pending_retry": []}
        creator.synthesize_episode.assert_not_called()

    def test_warns_when_retry_produces_no_audio(self, tmp_path):
        """A retry whose synthesis emits no file logs a warning, not an error."""
        creator = self._retry_creator(tmp_path)
        creator.synthesize_episode.return_value = tmp_path / "episode_001.mp3"

        state = {
            "creator": creator,
            "chapter_scripts": [(1, "Script text here.")],
            "pending_retry": [1],
            "retry_budget": {},
        }
        result = synthesize_node(state)

        assert result["episodes_to_check"] == [1]
        creator.synthesize_episode.assert_called_once()

    def test_retry_re_raises_tts_error(self, tmp_path):
        from audify.text_to_speech import TTSSynthesisError

        creator = self._retry_creator(tmp_path)
        creator.synthesize_episode.side_effect = TTSSynthesisError(
            "TTS failed", failed_batches=1, total_batches=1
        )

        state = {
            "creator": creator,
            "chapter_scripts": [(1, "Script text.")],
            "pending_retry": [1],
            "retry_budget": {},
        }
        with pytest.raises(TTSSynthesisError):
            synthesize_node(state)

    def test_retry_re_raises_generic_error(self, tmp_path):
        creator = self._retry_creator(tmp_path)
        creator.synthesize_episode.side_effect = RuntimeError("boom")

        state = {
            "creator": creator,
            "chapter_scripts": [(1, "Script text.")],
            "pending_retry": [1],
            "retry_budget": {},
        }
        with pytest.raises(RuntimeError):
            synthesize_node(state)

    def test_tts_base_length_returns_none_on_error(self):
        from audify.qa.nodes.synthesize import _tts_base_length

        creator = MagicMock()
        creator._get_tts_config.side_effect = RuntimeError("no config")
        assert _tts_base_length(creator) is None


class TestReportNode:
    """Verdict derivation and artifact generation for the report aggregator.

    Covers issue #37's required flag types: clean, cycle-1 escalation
    exhausted, cycle-2 reroute exhausted, cycle-3 retry exhausted. Uses
    ``report_node`` directly with a synthesised state so each verdict path
    is exercised in isolation — independent of the upstream detectors,
    which have not shipped yet.
    """

    def _stub_creator(self, tmp_path: Path, warn_stop: bool = False) -> MagicMock:
        creator = MagicMock()
        creator.audiobook_path = tmp_path
        creator.warn_stop = warn_stop
        return creator

    def _state(
        self,
        creator: MagicMock,
        *,
        chapter_titles: list[str],
        flags: dict | None = None,
        retry_budget: dict | None = None,
        best_wer: dict | None = None,
        episode_paths: list | None = None,
    ) -> dict:
        return {
            "creator": creator,
            "chapters": [],
            "chapter_titles": chapter_titles,
            "chapter_scripts": [],
            # Default: one dummy episode path per chapter, so the default
            # ``pipeline_status`` is ``"complete"`` and tests focus on
            # verdict logic. Tests can override by passing ``episode_paths``.
            "episode_paths": (
                episode_paths
                if episode_paths is not None
                else [Path(f"ep_{i}.mp3") for i in range(len(chapter_titles))]
            ),
            "retry_budget": retry_budget or {},
            "best_wer": best_wer or {},
            "flags": flags or {},
        }

    def _load_report(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "quality_report.json").read_text())

    def test_clean_when_no_flags(self, tmp_path):
        """No flags → every chapter is `clean`."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path)
        state = self._state(creator, chapter_titles=["Ch 1", "Ch 2"])
        report_node(state)

        report = self._load_report(tmp_path)
        assert report["chapters"]["chapter_1"]["verdict"] == "clean"
        assert report["chapters"]["chapter_2"]["verdict"] == "clean"
        assert report["chapters"]["chapter_1"]["flags"] == []
        assert report["verdict_counts"] == {
            "clean": 2, "flagged": 0, "unrecoverable": 0,
        }

    def test_flagged_when_cycle_resolved(self, tmp_path):
        """A FlagEntry with `exhausted=False` yields `flagged`, not unrecoverable."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_3_retry",
                        "reason": "WER 0.32 exceeded threshold once",
                        "exhausted": False,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_3_retry": 2}},
            best_wer={"chapter_1": 0.18},
        )
        report_node(state)

        report = self._load_report(tmp_path)
        assert report["chapters"]["chapter_1"]["verdict"] == "flagged"
        assert report["chapters"]["chapter_1"]["attempts_used"] == {
            "cycle_3_retry": 1,
        }
        assert report["chapters"]["chapter_1"]["best_wer"] == 0.18
        assert report["verdict_counts"]["flagged"] == 1

    def test_unrecoverable_cycle_3_retry_exhausted(self, tmp_path):
        """Cycle-3 retry exhaustion: unrecoverable, 3/3 attempts, best WER surfaced."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_3_retry",
                        "reason": "WER 0.42 exceeds 0.3 threshold",
                        "exhausted": True,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_3_retry": 0}},
            best_wer={"chapter_1": 0.42},
        )
        report_node(state)

        report = self._load_report(tmp_path)
        entry = report["chapters"]["chapter_1"]
        assert entry["verdict"] == "unrecoverable"
        assert entry["attempts_used"] == {"cycle_3_retry": 3}
        assert entry["best_wer"] == 0.42
        assert entry["flags"][0]["exhausted"] is True
        assert report["verdict_counts"]["unrecoverable"] == 1

    def test_unrecoverable_cycle_2_reroute_exhausted(self, tmp_path):
        """Cycle-2 reroute exhaustion: verdict `unrecoverable`, no best_wer."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_2_reroute",
                        "reason": "LLM produced a summary, not a narration",
                        "exhausted": True,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_2_reroute": 0}},
        )
        report_node(state)

        report = self._load_report(tmp_path)
        entry = report["chapters"]["chapter_1"]
        assert entry["verdict"] == "unrecoverable"
        assert entry["attempts_used"] == {"cycle_2_reroute": 3}
        assert entry["best_wer"] is None

    def test_unrecoverable_cycle_1_escalation_exhausted(self, tmp_path):
        """Cycle-1 escalation exhaustion: verdict `unrecoverable`."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_1_escalation",
                        "reason": "text-extraction: mojibake after OCR fallback",
                        "exhausted": True,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_1_escalation": 0}},
        )
        report_node(state)

        report = self._load_report(tmp_path)
        entry = report["chapters"]["chapter_1"]
        assert entry["verdict"] == "unrecoverable"
        assert entry["attempts_used"] == {"cycle_1_escalation": 3}

    def test_human_readable_report_written(self, tmp_path):
        """`quality_report.txt` is written alongside the JSON report."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path)
        state = self._state(creator, chapter_titles=["Ch 1"])
        report_node(state)

        text_path = tmp_path / "quality_report.txt"
        assert text_path.exists(), "Human-readable report must be written"
        content = text_path.read_text()
        # Rich table header + the verdict glyph for ``clean`` must be present.
        assert "Quality Report" in content
        assert "clean" in content
        assert "Summary:" in content

    def test_warn_stop_prompts_on_unrecoverable(self, tmp_path, monkeypatch):
        """`warn_stop=True` with an unrecoverable verdict invokes the input prompt."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path, warn_stop=True)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_3_retry",
                        "reason": "boundary STT mismatch",
                        "exhausted": True,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_3_retry": 0}},
        )

        prompts: list[str] = []

        def _fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return "y"

        monkeypatch.setattr("builtins.input", _fake_input)
        report_node(state)
        assert len(prompts) == 1
        assert "unrecoverable" in prompts[0]

    def test_warn_stop_user_rejects(self, tmp_path, monkeypatch, caplog):
        """Rejecting at the prompt logs a warning; the run still completes."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path, warn_stop=True)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_3_retry",
                        "reason": "boundary STT mismatch",
                        "exhausted": True,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_3_retry": 0}},
        )

        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        with caplog.at_level("WARNING", logger="audify.qa.nodes.report"):
            report_node(state)
        # JSON is still written even when the user rejects.
        assert (tmp_path / "quality_report.json").exists()
        assert any(
            "User did not accept audiobook" in r.message for r in caplog.records
        )

    def test_warn_stop_off_does_not_prompt(self, tmp_path, monkeypatch):
        """`warn_stop=False` (default) never prompts, even with unrecoverables."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path, warn_stop=False)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_3_retry",
                        "reason": "boundary STT mismatch",
                        "exhausted": True,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_3_retry": 0}},
        )

        def _should_not_be_called(prompt: str) -> str:
            raise AssertionError(f"input() called unexpectedly with: {prompt!r}")

        monkeypatch.setattr("builtins.input", _should_not_be_called)
        report_node(state)  # Must not raise

    def test_warn_stop_skipped_when_only_flagged(self, tmp_path, monkeypatch):
        """`warn_stop` only triggers on unrecoverable, not on soft flags."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path, warn_stop=True)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_3_retry",
                        "reason": "single retry resolved",
                        "exhausted": False,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_3_retry": 2}},
        )

        def _should_not_be_called(prompt: str) -> str:
            raise AssertionError(f"input() called unexpectedly with: {prompt!r}")

        monkeypatch.setattr("builtins.input", _should_not_be_called)
        report_node(state)

    @pytest.mark.parametrize(
        "stdin_exc",
        [EOFError("no stdin"), OSError("no stdin")],
        ids=["eof", "oserror"],
    )
    def test_warn_stop_non_tty_does_not_hang(
        self, tmp_path, monkeypatch, stdin_exc
    ):
        """EOFError/OSError on stdin (CI/non-tty) is swallowed; run completes."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path, warn_stop=True)
        state = self._state(
            creator,
            chapter_titles=["Ch 1"],
            flags={
                "chapter_1": [
                    {
                        "cycle": "cycle_1_escalation",
                        "reason": "mojibake",
                        "exhausted": True,
                    },
                ],
            },
            retry_budget={"chapter_1": {"cycle_1_escalation": 0}},
        )

        def _no_stdin(_prompt: str) -> str:
            raise stdin_exc

        monkeypatch.setattr("builtins.input", _no_stdin)
        report_node(state)  # Must not raise

    def test_pipeline_status_partial_when_some_episodes_missing(self, tmp_path):
        """`pipeline_status=partial` when episodes < expected and > 0."""
        from audify.qa.nodes.report import report_node

        creator = self._stub_creator(tmp_path)
        state = self._state(
            creator,
            chapter_titles=["Ch 1", "Ch 2"],
            episode_paths=[tmp_path / "ep_1.mp3"],
        )
        report_node(state)

        report = self._load_report(tmp_path)
        assert report["pipeline_status"] == "partial"
        assert report["episodes_synthesised"] == 1
        assert report["chapters_expected"] == 2


# ---------------------------------------------------------------------------
# Cycle 3 — boundary-sampling fidelity check + re-chunk retry edge (issue #38)
# ---------------------------------------------------------------------------


class _FileBackedSTT:
    """STT stub that 'transcribes' the text written into the episode file.

    The head window returns the first few words of the file content, the tail
    window the last few. Because a truncated episode's file is missing its
    tail, the tail-window transcript will not match the script's closing words.
    """

    def __init__(self, window_words: int = 4):
        self.window_words = window_words
        self.calls: list[tuple] = []

    def transcribe(self, audio_path, *, start_s=None, end_s=None, language=None):
        self.calls.append((Path(audio_path).name, start_s, end_s))
        words = Path(audio_path).read_text().split()
        if start_s == 0.0:  # head window
            return " ".join(words[: self.window_words])
        return " ".join(words[-self.window_words :])  # tail window


class _RaisingSTT:
    """STT stub that is always unreachable (exercises graceful degradation)."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio_path, *, start_s=None, end_s=None, language=None):
        from audify.qa.stt import STTServiceError

        self.calls += 1
        raise STTServiceError("STT service unreachable")


class _FidelityCreator:
    """Minimal real (non-Mock) creator for synthesize-mode graph runs.

    ``truncate_after`` controls how many synthesis attempts produce truncated
    audio before full audio is emitted: ``1`` means the first pass truncates
    and the first re-chunk recovers; a large value means it never recovers.
    """

    def __init__(self, tmp_path: Path, scripts: dict[int, str], *, truncate_after=1):
        self.audiobook_path = tmp_path / "out"
        self.audiobook_path.mkdir(parents=True, exist_ok=True)
        self.scripts_path = tmp_path / "scripts"
        self.scripts_path.mkdir(parents=True, exist_ok=True)
        self.episodes_path = tmp_path / "episodes"
        self.episodes_path.mkdir(parents=True, exist_ok=True)

        self._scripts = scripts
        self._titles = [f"Chapter {n}" for n in sorted(scripts)]
        for n, text in scripts.items():
            (self.scripts_path / f"episode_{n:03d}_script.txt").write_text(text)

        self.mode = "synthesize"
        self.fidelity_check = True
        self.stt_client = _FileBackedSTT()
        self.language = "en"
        self.resolved_language = "en"
        self.warn_stop = False
        self.progress = MagicMock()

        self._truncate_after = truncate_after
        self._attempts: dict[int, int] = {}
        self.synth_calls: list[tuple[int, int | None]] = []
        self.synth_titles: list[str | None] = []
        self.m4b_calls = 0

    def _verify_tts_provider_available(self):
        return None

    def _load_chapter_titles(self):
        return list(self._titles)

    def _get_tts_config(self):
        cfg = MagicMock()
        cfg.max_text_length = 4000
        return cfg

    def synthesize_episode(
        self, script, episode_number, max_text_length=None, title=None
    ):
        self.synth_calls.append((episode_number, max_text_length))
        self.synth_titles.append(title)
        self._attempts[episode_number] = self._attempts.get(episode_number, 0) + 1
        path = self.episodes_path / f"episode_{episode_number:03d}.mp3"
        words = script.split()
        if self._attempts[episode_number] <= self._truncate_after:
            content = " ".join(words[: len(words) // 2])  # drop the tail
        else:
            content = script  # re-chunk restored full fidelity
        # Mirror production audio: the episode opens with the spoken
        # chapter-title announcement followed by the narration.
        announcement = format_title_announcement(title or "")
        if announcement:
            content = f"{announcement} {content}"
        path.write_text(content)
        return path

    def create_m4b(self):
        self.m4b_calls += 1


# A 20-word script with distinct tokens so head/tail windows are unambiguous.
_SCRIPT_20 = " ".join(f"word{i:02d}" for i in range(20))


@pytest.fixture
def fixed_duration(monkeypatch):
    """Pin episode duration so WER (not duration ratio) drives detection."""
    from audify.utils.audio import AudioProcessor

    monkeypatch.setattr(AudioProcessor, "get_duration", lambda path: 60.0)


class TestFidelityCycle:
    def _load_report(self, creator) -> dict:
        return json.loads(
            (Path(creator.audiobook_path) / "quality_report.json").read_text()
        )

    def test_truncation_triggers_rechunk_retry_and_resolves(
        self, tmp_path, fixed_duration
    ):
        """A truncating first pass fires the cycle; re-chunk resolves it."""
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=1)

        run_graph(creator)

        # Initial synthesis used the provider default (None); the single retry
        # used a smaller batch size — proving the re-chunk back-edge fired.
        assert creator.synth_calls == [(1, None), (1, 2000)]
        assert creator.m4b_calls == 1

        report = self._load_report(creator)
        chapter = report["chapters"]["chapter_1"]
        assert chapter["verdict"] == "flagged"
        assert chapter["attempts_used"]["cycle_3_retry"] == 1
        assert chapter["best_wer"] == 0.0
        assert len(chapter["flags"]) == 1
        flag = chapter["flags"][0]
        assert flag["cycle"] == "cycle_3_retry"
        assert flag["exhausted"] is False

        # The re-chunked audio on disk now carries the full script (the
        # chapter-title announcement is still prepended, as in production).
        final = (creator.episodes_path / "episode_001.mp3").read_text()
        assert final == f"Chapter 1. {_SCRIPT_20}"

    def test_clean_episode_never_triggers_cycle(self, tmp_path, fixed_duration):
        """An episode that transcribes faithfully raises no flag, no retry."""
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=0)

        run_graph(creator)

        assert creator.synth_calls == [(1, None)]  # no retry
        report = self._load_report(creator)
        chapter = report["chapters"]["chapter_1"]
        assert chapter["verdict"] == "clean"
        assert chapter["flags"] == []

    def test_title_announcement_does_not_trip_head_window(
        self, tmp_path, fixed_duration
    ):
        """Audio opening with the spoken chapter title passes the head check.

        The synthesize node passes the chapter title through so the episode
        audio begins with the announcement; the fidelity node must include
        that announcement in its head-window reference or every announced
        episode would be flagged as truncated.
        """
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=0)

        run_graph(creator)

        # The synthesize node passed the chapter title through.
        assert creator.synth_titles == ["Chapter 1"]
        # The audio on disk really does open with the announcement.
        audio_text = (creator.episodes_path / "episode_001.mp3").read_text()
        assert audio_text.startswith("Chapter 1.")
        # And the fidelity check still reports the episode as clean.
        assert creator.synth_calls == [(1, None)]  # no retry scheduled
        report = self._load_report(creator)
        assert report["chapters"]["chapter_1"]["verdict"] == "clean"

    def test_title_passed_through_on_rechunk_retry(self, tmp_path, fixed_duration):
        """The re-chunk retry keeps announcing the chapter title."""
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=1)

        run_graph(creator)

        assert creator.synth_titles == ["Chapter 1", "Chapter 1"]

    def test_exhaustion_keeps_best_effort_and_never_aborts(
        self, tmp_path, fixed_duration
    ):
        """When re-chunk never recovers, budget exhausts but the book completes."""
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=99)

        run_graph(creator)

        # 1 initial attempt + 3 bounded retries.
        assert len(creator.synth_calls) == 4
        assert creator.synth_calls[0] == (1, None)
        assert [mtl for _, mtl in creator.synth_calls[1:]] == [2000, 1333, 1000]
        # Book is still assembled — the cycle is warn-only.
        assert creator.m4b_calls == 1

        report = self._load_report(creator)
        chapter = report["chapters"]["chapter_1"]
        assert chapter["verdict"] == "unrecoverable"
        assert chapter["attempts_used"]["cycle_3_retry"] == 3
        assert chapter["flags"][-1]["exhausted"] is True
        assert report["verdict_counts"]["unrecoverable"] == 1

    def test_unreachable_stt_degrades_to_passthrough(
        self, tmp_path, fixed_duration
    ):
        """An unreachable STT service skips detection rather than aborting."""
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=1)
        creator.stt_client = _RaisingSTT()

        run_graph(creator)

        assert creator.synth_calls == [(1, None)]  # no retry attempted
        assert creator.m4b_calls == 1
        report = self._load_report(creator)
        assert report["chapters"]["chapter_1"]["verdict"] == "clean"

    def test_disabled_flag_skips_fidelity(self, tmp_path, fixed_duration):
        """fidelity_check=False routes straight through to assemble."""
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=1)
        creator.fidelity_check = False

        run_graph(creator)

        assert creator.synth_calls == [(1, None)]
        report = self._load_report(creator)
        assert report["chapters"]["chapter_1"]["verdict"] == "clean"

    def test_duration_ratio_alone_flags_truncation(self, tmp_path, monkeypatch):
        """Even with WER=0, a short duration ratio corroborates truncation."""
        from audify.qa.nodes.fidelity import fidelity_node
        from audify.utils.audio import AudioProcessor

        # Perfect transcripts (file holds the full script) but the audio is far
        # shorter than the ~16s the 20-word script implies at 75 wpm.
        monkeypatch.setattr(AudioProcessor, "get_duration", lambda path: 2.0)

        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=0)
        episode_path = creator.episodes_path / "episode_001.mp3"
        episode_path.write_text(_SCRIPT_20)  # full, faithful content

        state = {
            "creator": creator,
            "chapter_scripts": [(1, _SCRIPT_20)],
            "chapter_titles": ["Chapter 1"],
            "episode_paths": [episode_path],
            "episodes_to_check": [1],
            "retry_budget": {},
            "best_wer": {},
            "flags": {},
            "pending_retry": [],
        }

        out = fidelity_node(state)
        assert out["pending_retry"] == [1]
        assert out["retry_budget"]["chapter_1"]["cycle_3_retry"] == 2


class TestFidelityHelpers:
    """Direct coverage of fidelity branches not reached by the cycle tests."""

    def _state(self, creator, *, to_check, scripts):
        return {
            "creator": creator,
            "chapter_scripts": list(scripts.items()),
            "chapter_titles": [f"Chapter {n}" for n in scripts],
            "episode_paths": [],
            "episodes_to_check": to_check,
            "retry_budget": {},
            "best_wer": {},
            "flags": {},
            "pending_retry": [],
        }

    def test_no_stt_client_passes_through(self, tmp_path, monkeypatch):
        """When no STT client can be obtained, the node is a no-op."""
        from audify.qa.nodes import fidelity as fid

        monkeypatch.setattr(fid, "_get_stt_client", lambda creator: None)
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=0)

        state = self._state(creator, to_check=[1], scripts={1: _SCRIPT_20})
        assert fid.fidelity_node(state) == {}

    def test_skips_episode_without_script(self, tmp_path, fixed_duration):
        """An episode_to_check absent from chapter_scripts is skipped."""
        from audify.qa.nodes.fidelity import fidelity_node

        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=0)
        out = fidelity_node(self._state(creator, to_check=[2], scripts={1: _SCRIPT_20}))
        assert out["pending_retry"] == []

    def test_skips_episode_without_audio(self, tmp_path, fixed_duration):
        """No audio file on disk means a different cycle owns the chapter."""
        from audify.qa.nodes.fidelity import fidelity_node

        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=0)
        # No episode_001.mp3 is ever written.
        out = fidelity_node(self._state(creator, to_check=[1], scripts={1: _SCRIPT_20}))
        assert out["pending_retry"] == []

    def test_get_stt_client_builds_when_not_injected(self, monkeypatch):
        from audify.qa.nodes import fidelity as fid

        sentinel = object()
        monkeypatch.setattr(fid, "WhisperSTTClient", lambda base_url: sentinel)

        class _NoClient:
            pass

        assert fid._get_stt_client(_NoClient()) is sentinel

    def test_save_best_candidate_handles_oserror(self, tmp_path, monkeypatch):
        from audify.qa.nodes import fidelity as fid

        creator = MagicMock()
        creator.episodes_path = tmp_path
        source = tmp_path / "episode_001.mp3"
        source.write_text("audio")

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(fid.shutil, "copy2", _boom)
        # Must not raise — failure to cache is best-effort.
        fid._save_best_candidate(creator, 1, source)

    def test_restore_best_candidate_noop_without_cache(self, tmp_path):
        from audify.qa.nodes import fidelity as fid

        creator = MagicMock()
        creator.episodes_path = tmp_path
        dest = tmp_path / "episode_001.mp3"
        fid._restore_best_candidate(creator, 1, dest)
        assert not dest.exists()

    def test_restore_best_candidate_handles_oserror(self, tmp_path, monkeypatch):
        from audify.qa.nodes import fidelity as fid

        creator = MagicMock()
        creator.episodes_path = tmp_path
        cache = tmp_path / ".fidelity_best"
        cache.mkdir()
        (cache / "episode_001.mp3").write_text("best")
        dest = tmp_path / "episode_001.mp3"

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(fid.shutil, "copy2", _boom)
        fid._restore_best_candidate(creator, 1, dest)  # must not raise

    def test_window_wers_zero_duration_is_total_failure(self, tmp_path, monkeypatch):
        from audify.qa.nodes import fidelity as fid
        from audify.utils.audio import AudioProcessor

        monkeypatch.setattr(AudioProcessor, "get_duration", lambda path: 0.0)
        episode = tmp_path / "episode_001.mp3"
        episode.write_text("content")

        assert fid._window_wers(
            _FileBackedSTT(), episode, _SCRIPT_20, 8.0, "en"
        ) == (1.0, 1.0)

    def test_suspect_reason_lists_every_signal(self):
        from audify.qa.nodes.fidelity import _suspect_reason

        reason = _suspect_reason(0.9, 0.8, 0.1, 0.3, 0.5)
        assert "head WER" in reason
        assert "tail WER" in reason
        assert "duration ratio" in reason

    def test_suspect_reason_below_thresholds(self):
        from audify.qa.nodes.fidelity import _suspect_reason

        assert _suspect_reason(0.0, 0.0, 1.0, 0.3, 0.5) == "below thresholds"


# ---------------------------------------------------------------------------
# Cycle 2 — script-validity LLM judge + reroute back-edge (issue #39)
# ---------------------------------------------------------------------------


class _ScriptValidityStubLLM:
    """LLM stub for script-validity testing. Returns configurable verdicts."""

    def __init__(self, verdicts: dict[int, tuple[str, str]] | None = None):
        """
        Args:
            verdicts: Mapping episode_number -> (verdict, reason).
                      Unmapped episodes default to ("pass", "").
        """
        self.verdicts = verdicts or {}
        self.calls: list[tuple[int, str, str]] = []

    def generate_script(self, *, text: str, prompt: str, language, temperature):
        # Store the call info (no access to episode_number here, we rely on context)
        self._last_text = text
        self._last_prompt = prompt
        # Return the appropriate verdict for this call based on the script content
        # Determine which episode from context (source/script in text)
        # We track by call order
        idx = len(self.calls)
        self.calls.append((idx, text[:50], prompt[:50]))
        # Default: pass
        return '{"verdict": "pass", "reason": "Faithful narration."}'


class _ScriptValidityCreator:
    """Minimal creator for script-validity graph runs."""

    def __init__(
        self,
        tmp_path: Path,
        chapters: list[str],
        llm_verdicts: dict[int, tuple[str, str]] | None = None,
        *,
        mode: str = "full",
    ):
        self.audiobook_path = tmp_path / "out"
        self.audiobook_path.mkdir(parents=True, exist_ok=True)
        self.llm_client = _ScriptValidityStubLLM(llm_verdicts)
        self.mode = mode
        self.language = "en"
        self.resolved_language = "en"
        self.warn_stop = False
        self.progress = MagicMock()
        self._chapters = chapters
        self._titles = [f"Chapter {i + 1}" for i in range(len(chapters))]

    def _verify_tts_provider_available(self):
        return None


class TestScriptValidityNode:
    """Unit tests for the script_validity_node in isolation."""

    def _state(
        self,
        creator,
        *,
        chapters: list[str] | None = None,
        chapter_scripts: list[tuple[int, str]] | None = None,
    ):
        if chapters is None:
            chapters = ["Source text for a real narration. " * 50]
        if chapter_scripts is None:
            script = "Long healthy narration script text. " * 50
            chapter_scripts = [(1, script)]
        return {
            "creator": creator,
            "chapters": chapters,
            "chapter_titles": [f"Chapter {i + 1}" for i in range(len(chapters))],
            "chapter_scripts": chapter_scripts,
            "episode_paths": [],
            "retry_budget": {},
            "best_wer": {},
            "flags": {},
            "pending_reroute": [],
            "pending_retry": [],
            "episodes_to_check": [],
        }

    def test_clean_script_passes(self):
        """A script with healthy word count skips the LLM judge entirely."""
        from audify.qa.nodes.script_validity import script_validity_node

        creator = MagicMock()
        chapters = ["Source chapter text. " * 100]
        script = "Long healthy narration script text. " * 100

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        assert result["pending_reroute"] == []
        assert result["flags"] == {}
        assert (1, script) in result["chapter_scripts"]

    def test_short_script_triggers_llm_judge(self):
        """A suspiciously short script reaches the LLM judge."""
        from audify.qa.nodes.script_validity import script_validity_node

        creator = MagicMock()
        llm = _ScriptValidityStubLLM()
        creator.llm_client = llm
        chapters = ["Source chapter with real content. " * 100]
        # Short script that triggers the judge
        script = "Short."

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        # LLM was called (since script is short)
        assert len(llm.calls) >= 1
        assert result["pending_reroute"] == []

    def test_summary_script_triggers_reroute(self):
        """A summary-like script gets rerouted."""
        from audify.qa.nodes.script_validity import script_validity_node

        llm = _ScriptValidityStubLLM()
        # Override generate_script to return a reroute verdict

        def _reroute(text, prompt, language, temperature):
            llm.calls.append((len(llm.calls), text[:50], prompt[:50]))
            return ('{"verdict": "reroute", '
                    '"reason": "Script is a summary, not a narration."}')
        llm.generate_script = _reroute

        creator = MagicMock()
        creator.llm_client = llm
        chapters = ["Source chapter with real content. " * 100]
        script = "Short summary. The chapter is about X."

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        assert result["pending_reroute"] == [1]
        assert result["retry_budget"]["chapter_1"]["cycle_2_reroute"] == 2

    def test_refusal_script_triggers_reroute(self):
        """A refusal script gets rerouted."""
        from audify.qa.nodes.script_validity import script_validity_node

        llm = _ScriptValidityStubLLM()

        def _refusal(text, prompt, language, temperature):
            llm.calls.append((len(llm.calls), text[:50], prompt[:50]))
            return ('{"verdict": "reroute", '
                    '"reason": "Script contains refusal language."}')
        llm.generate_script = _refusal

        creator = MagicMock()
        creator.llm_client = llm
        chapters = ["Source chapter. " * 100]
        script = "I'm sorry, I cannot narrate this content."

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        assert result["pending_reroute"] == [1]

    def test_exhaustion_keeps_best_effort(self):
        """After 3 reroute attempts, budget exhausts and script is kept."""
        from audify.qa.nodes.script_validity import script_validity_node

        llm = _ScriptValidityStubLLM()

        def _always_reroute(text, prompt, language, temperature):
            llm.calls.append((len(llm.calls), text[:50], prompt[:50]))
            return '{"verdict": "reroute", "reason": "Bad script."}'
        llm.generate_script = _always_reroute

        creator = MagicMock()
        creator.llm_client = llm
        chapters = ["Source chapter. " * 100]
        script = "Bad script."

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        # Set retry_budget to 0 so the first call exhausts budget
        state["retry_budget"] = {"chapter_1": {"cycle_2_reroute": 0}}

        result = script_validity_node(state)

        assert result["pending_reroute"] == []  # No more retries
        flags = result["flags"]["chapter_1"]
        assert flags[-1]["exhausted"] is True
        assert flags[-1]["cycle"] == "cycle_2_reroute"

    def test_no_llm_client_skips_judge(self):
        """When no LLM client is available, the node passes through."""
        from audify.qa.nodes.script_validity import script_validity_node

        creator = MagicMock()
        creator.llm_client = None  # Explicitly None (MagicMock auto-creates attrs)
        chapters = ["Source chapter. " * 100]
        script = "Short script."  # Would trigger judge if client existed

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        assert result["pending_reroute"] == []
        assert result["flags"] == {}

    def test_borderline_passes_through(self):
        """A borderline verdict passes through without reroute."""
        from audify.qa.nodes.script_validity import script_validity_node

        llm = _ScriptValidityStubLLM()

        def _borderline(text, prompt, language, temperature):
            llm.calls.append((len(llm.calls), text[:50], prompt[:50]))
            return '{"verdict": "borderline", "reason": "Slightly abbreviated."}'
        llm.generate_script = _borderline

        creator = MagicMock()
        creator.llm_client = llm
        chapters = ["Source chapter. " * 100]
        script = "Short script."

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        assert result["pending_reroute"] == []  # borderline = pass through
        assert result["flags"] == {}  # No flag for borderline

    def test_routing_reroute_to_script_gen(self):
        """script_validity_route returns 'script_gen' when reroutes pending."""
        from audify.qa.nodes.script_validity import script_validity_route

        state = {"pending_reroute": [1], "creator": MagicMock()}
        assert script_validity_route(state) == "script_gen"

    def test_routing_pass_to_synthesize_full_mode(self):
        """script_validity_route returns 'synthesize' in full mode."""
        from audify.qa.nodes.script_validity import script_validity_route

        creator = MagicMock()
        creator.mode = "full"
        state = {"pending_reroute": [], "creator": creator}
        assert script_validity_route(state) == "synthesize"

    def test_routing_pass_to_report_process_mode(self):
        """script_validity_route returns 'report' in process mode."""
        from audify.qa.nodes.script_validity import script_validity_route

        creator = MagicMock()
        creator.mode = "process"
        state = {"pending_reroute": [], "creator": creator}
        assert script_validity_route(state) == "report"

    def test_parse_judge_response_handles_markdown_fences(self):
        """_parse_judge_response strips markdown code fences."""
        from audify.qa.nodes.script_validity import _parse_judge_response

        response = """```json
{"verdict": "reroute", "reason": "Summary detected."}
```"""
        parsed = _parse_judge_response(response)
        assert parsed["verdict"] == "reroute"
        assert parsed["reason"] == "Summary detected."

    def test_multi_chapter_reroute_only_failing_chapters(self):
        """Only chapters with bad scripts get rerouted; clean ones pass."""
        from audify.qa.nodes.script_validity import script_validity_node

        def _mixed_verdicts(text, prompt, language, temperature):
            # Only called for chapter 2 (short script); chapter 1 is long enough
            # to skip the LLM judge entirely.
            return '{"verdict": "reroute", "reason": "Summary."}'

        llm = _ScriptValidityStubLLM()
        llm.generate_script = _mixed_verdicts

        creator = MagicMock()
        creator.llm_client = llm
        chapters = [
            "Source chapter one. " * 100,
            "Source chapter two. " * 100,
        ]
        scripts = [
            (1, "Long healthy script that passes the pre-filter. " * 100),  # Clean
            (2, "Short script."),  # Will trigger LLM and fail
        ]

        state = self._state(creator, chapters=chapters, chapter_scripts=scripts)
        result = script_validity_node(state)

        # Only chapter 2 should be in pending_reroute
        assert result["pending_reroute"] == [2]
        assert "chapter_2" not in result["flags"]

    def test_pass_after_reroute_adds_resolved_flag(self):
        """A short script that passes LLM judge after a previous reroute gets a flag."""
        from audify.qa.nodes.script_validity import script_validity_node

        call_count = 0

        def _first_reroute_then_pass(text, prompt, language, temperature):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '{"verdict": "reroute", "reason": "Summary detected."}'
            return '{"verdict": "pass", "reason": "Now faithful."}'

        llm = _ScriptValidityStubLLM()
        llm.generate_script = _first_reroute_then_pass

        creator = MagicMock()
        creator.llm_client = llm
        chapters = ["Source chapter. " * 100]
        scripts = [(1, "Short script.")]

        state = self._state(creator, chapters=chapters, chapter_scripts=scripts)
        # First call: reroute
        result1 = script_validity_node(state)
        assert result1["pending_reroute"] == [1]
        # Merge result1 into a fresh state for the second pass
        state2 = self._state(creator, chapters=chapters, chapter_scripts=scripts)
        state2.update(result1)

        # Second call: pass (simulate reroute back-edge completed)
        result2 = script_validity_node(state2)
        assert result2["pending_reroute"] == []
        flags = result2["flags"]["chapter_1"]
        assert len(flags) == 1
        assert flags[0]["cycle"] == "cycle_2_reroute"
        assert flags[0]["exhausted"] is False

    def test_llm_exception_degrades_gracefully(self):
        """LLM judge exception is caught, defaults to pass."""
        from audify.qa.nodes.script_validity import script_validity_node

        llm = _ScriptValidityStubLLM()

        def _explode(text, prompt, language, temperature):
            raise RuntimeError("LLM service unavailable")

        llm.generate_script = _explode

        creator = MagicMock()
        creator.llm_client = llm
        chapters = ["Source chapter. " * 100]
        script = "Short busted script."

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        assert result["pending_reroute"] == []

    def test_unknown_judge_verdict_defaults_to_pass(self):
        """An unrecognized LLM verdict defaults to pass."""
        from audify.qa.nodes.script_validity import script_validity_node

        llm = _ScriptValidityStubLLM()

        def _unknown(text, prompt, language, temperature):
            return '{"verdict": "maybe", "reason": "Not sure."}'

        llm.generate_script = _unknown

        creator = MagicMock()
        creator.llm_client = llm
        chapters = ["Source chapter. " * 100]
        script = "Short ambiguous script."

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        assert result["pending_reroute"] == []

    def test_unparseable_judge_response_defaults_to_pass(self):
        """LLM returns garbage JSON, defaults to pass."""
        from audify.qa.nodes.script_validity import script_validity_node

        llm = _ScriptValidityStubLLM()

        def _garbage(text, prompt, language, temperature):
            return "This is not JSON at all."

        llm.generate_script = _garbage

        creator = MagicMock()
        creator.llm_client = llm
        chapters = ["Source chapter. " * 100]
        script = "Short garbage script."

        state = self._state(creator, chapters=chapters, chapter_scripts=[(1, script)])
        result = script_validity_node(state)

        assert result["pending_reroute"] == []


class TestScriptValidityCycle:
    """Integration tests: script_validity back-edge firing in the compiled graph."""

    def test_reroute_back_edge_fires_and_resolves(self, tmp_path):
        """A reroutable script triggers the back-edge; script_gen re-runs."""
        from audify.qa.graph import run_graph

        call_log: list[int] = []

        def _gen_script(chapter_content: str, episode_number: int) -> str:
            call_log.append(episode_number)
            if len(call_log) == 1:
                # First call produces bad script
                return "Short bad script."
            # Retry produces good script
            return "Long enough narration script that passes the pre-filter. " * 100

        # Build a fresh MagicMock with configure_mock to override attributes
        creator = MagicMock()
        creator.max_chapters = None
        creator.audiobook_path = tmp_path / "out"
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        creator.chapter_titles = []
        creator.mode = "full"
        creator.confirm = False
        creator.warn_stop = False
        creator.fidelity_check = False
        creator.language = "en"
        creator.resolved_language = "en"
        from audify.readers.pdf import PdfReader
        creator.reader = MagicMock(spec=PdfReader)
        creator.reader.cleaned_text = "\n\n".join(["Chapter one text. " * 200])
        creator.generate_audiobook_script.side_effect = _gen_script
        creator.progress = MagicMock()
        creator._verify_tts_provider_available.return_value = None

        # Add an LLM client that returns reroute for short scripts
        def _judge(text, prompt, language, temperature):
            if "Short bad script" in text:
                return ('{"verdict": "reroute", '
                        '"reason": "Script too short to be faithful."}')
            return '{"verdict": "pass", "reason": "Looks good."}'

        llm = _ScriptValidityStubLLM()
        llm.generate_script = _judge
        creator.llm_client = llm

        # synthesize needs to create a file
        def _synth(script, episode_number, **kwargs):
            p = tmp_path / "out" / f"episode_{episode_number:03d}.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"FAKE")
            return p

        creator.synthesize_episode.side_effect = _synth
        creator._validate_chapters.return_value = None
        creator._save_chapter_titles.return_value = None
        creator.create_m4b.return_value = None

        with patch("audify.readers.ebook.EpubReader"):
            run_graph(creator)

        # script_gen was called 2 times: 1 initial + 1 reroute
        assert len(call_log) == 2, f"Expected 2 calls, got {len(call_log)}"

        report_path = tmp_path / "out" / "quality_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        # Chapter should have been flagged as resolved
        chapter = report["chapters"]["chapter_1"]
        v = chapter["verdict"]
        assert v == "flagged", f"Expected flagged, got {v}"

    def test_process_mode_reroute_also_fires(self, tmp_path):
        """Cycle-2 reroute edge applies in process mode too."""
        from audify.qa.graph import run_graph

        call_log: list[int] = []

        def _gen_script(chapter_content: str, episode_number: int) -> str:
            call_log.append(episode_number)
            return "Short bad script."  # Always bad

        def _judge(text, prompt, language, temperature):
            return '{"verdict": "reroute", "reason": "Summary detected."}'

        llm = _ScriptValidityStubLLM()
        llm.generate_script = _judge

        creator = MagicMock()
        creator.max_chapters = None
        creator.audiobook_path = tmp_path / "out"
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        creator.chapter_titles = []
        creator.mode = "process"
        creator.confirm = False
        creator.warn_stop = False
        creator.fidelity_check = False
        creator.language = "en"
        creator.resolved_language = "en"
        from audify.readers.pdf import PdfReader
        creator.reader = MagicMock(spec=PdfReader)
        creator.reader.cleaned_text = "\n\n".join(["Chapter one text. " * 200])
        creator.generate_audiobook_script.side_effect = _gen_script
        creator.llm_client = llm
        creator.progress = MagicMock()
        creator._validate_chapters.return_value = None
        creator._save_chapter_titles.return_value = None
        creator.create_m4b.return_value = None
        creator.synthesize_episode = MagicMock()  # Not called in process mode

        with patch("audify.readers.ebook.EpubReader"):
            run_graph(creator)

        # script_gen was called at least 2 times (initial + reroute back-edge)
        assert len(call_log) >= 2, f"Expected >=2 calls, got {call_log}"

        report_path = tmp_path / "out" / "quality_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        chapter = report["chapters"]["chapter_1"]
        assert chapter["attempts_used"]["cycle_2_reroute"] >= 1


# ---------------------------------------------------------------------------
# Cycle 1 — text-quality heuristic detector + escalation back-edge (issue #40)
# ---------------------------------------------------------------------------


class TestTextQualityHeuristics:
    """Unit tests for individual text-quality heuristics."""

    def test_is_empty_after_clean_empty(self):
        from audify.qa.nodes.text_quality import _is_empty_after_clean

        assert _is_empty_after_clean("")
        assert _is_empty_after_clean("   \n  ")
        assert _is_empty_after_clean("  \t  ")

    def test_is_empty_after_clean_not_empty(self):
        from audify.qa.nodes.text_quality import _is_empty_after_clean

        assert not _is_empty_after_clean("Hello world")
        assert not _is_empty_after_clean("Normal text here. " * 5)

    def test_high_mojibake_ratio_detects_garbage(self):
        from audify.qa.nodes.text_quality import _has_high_mojibake_ratio

        # Text with many control characters (C1 range)
        garbage = "\x80\x81\x82\x83\x84" + "normal"
        assert _has_high_mojibake_ratio(garbage, threshold=0.3)

        # Clean ASCII text
        assert not _has_high_mojibake_ratio("Clean readable text.", threshold=0.3)

    def test_mojibake_detects_replacement_char(self):
        from audify.qa.nodes.text_quality import _has_high_mojibake_ratio

        # Text with Unicode replacement characters
        text = "Hello \ufffd World \ufffd"
        assert _has_high_mojibake_ratio(text, threshold=0.1)

    def test_bad_whitespace_ratio_too_much(self):
        from audify.qa.nodes.text_quality import _has_bad_whitespace_ratio

        # Mostly spaces
        text = "   " * 100
        assert _has_bad_whitespace_ratio(text)

    def test_bad_whitespace_ratio_too_little(self):
        from audify.qa.nodes.text_quality import _has_bad_whitespace_ratio

        # No whitespace at all in a long string
        text = "a" * 100 + "b" * 100 + "c" * 100
        assert _has_bad_whitespace_ratio(text)

    def test_good_whitespace_ratio_passes(self):
        from audify.qa.nodes.text_quality import _has_bad_whitespace_ratio

        text = "This is a normal sentence with good whitespace. " * 5
        assert not _has_bad_whitespace_ratio(text)

    def test_high_nonword_ratio(self):
        from audify.qa.nodes.text_quality import _has_high_nonword_ratio

        # Lots of punctuation/symbols
        text = ">>>>>>..,,,,,;;;;;!!!!!%%%%%#####"
        assert _has_high_nonword_ratio(text, threshold=0.3)

    def test_normal_nonword_ratio_passes(self):
        from audify.qa.nodes.text_quality import _has_high_nonword_ratio

        text = "This is a normal English sentence with standard punctuation."
        assert not _has_high_nonword_ratio(text, threshold=0.3)


class TestTextQualityNode:
    """Unit tests for text_quality_node with different chapter content."""

    def _state(self, creator, *, chapters=None, titles=None):
        if chapters is None:
            chapters = ["Clean text here. " * 10]
        if titles is None:
            titles = [f"Chapter {i + 1}" for i in range(len(chapters))]
        return {
            "creator": creator,
            "chapters": chapters,
            "chapter_titles": titles,
            "chapter_scripts": [],
            "episode_paths": [],
            "retry_budget": {},
            "best_wer": {},
            "flags": {},
            "pending_escalation": [],
            "pending_reroute": [],
            "pending_retry": [],
            "episodes_to_check": [],
        }

    def test_clean_text_passes(self):
        from audify.qa.nodes.text_quality import text_quality_node

        creator = MagicMock()
        state = self._state(creator, chapters=["Clean readable text. " * 20])
        result = text_quality_node(state)

        assert result["pending_escalation"] == []
        assert result["flags"] == {}

    def test_empty_text_triggers_escalation(self):
        from audify.qa.nodes.text_quality import text_quality_node

        creator = MagicMock()
        state = self._state(creator, chapters=[""])
        result = text_quality_node(state)

        assert result["pending_escalation"] == [1]

    def test_mojibake_text_triggers_escalation(self):
        from audify.qa.nodes.text_quality import text_quality_node

        creator = MagicMock()
        text = "\x80\x81\x82\x83\x84" * 20 + "some readable parts "
        state = self._state(creator, chapters=[text])
        result = text_quality_node(state)

        assert result["pending_escalation"] == [1]

    def test_clean_and_garbage_mixed(self):
        """Only garbage chapters trigger escalation; clean ones pass."""
        from audify.qa.nodes.text_quality import text_quality_node

        creator = MagicMock()
        chapters = [
            "Clean readable text. " * 20,
            "",  # Garbage
        ]
        state = self._state(creator, chapters=chapters)
        result = text_quality_node(state)

        assert result["pending_escalation"] == [2]

    def test_exhaustion_keeps_best_effort(self):
        from audify.qa.nodes.text_quality import text_quality_node

        creator = MagicMock()
        state = self._state(creator, chapters=[""])
        state["retry_budget"] = {"chapter_1": {"cycle_1_escalation": 0}}

        result = text_quality_node(state)

        assert result["pending_escalation"] == []
        flags = result["flags"]["chapter_1"]
        assert flags[-1]["exhausted"] is True

    def test_routing_escalate_when_pending(self):
        from audify.qa.nodes.text_quality import text_quality_route

        assert text_quality_route({"pending_escalation": [1]}) == "escalate"

    def test_routing_confirm_when_clean(self):
        from audify.qa.nodes.text_quality import text_quality_route

        assert text_quality_route({"pending_escalation": []}) == "confirm"

    def test_resolved_flag_after_escalation(self):
        """A previously-escalated chapter that passes on retry gets a resolved flag."""
        from audify.qa.nodes.text_quality import text_quality_node

        creator = MagicMock()
        state = self._state(creator, chapters=["Clean text now. " * 10])
        state["retry_budget"] = {"chapter_1": {"cycle_1_escalation": 2}}

        result = text_quality_node(state)

        assert result["pending_escalation"] == []
        flags = result["flags"]["chapter_1"]
        assert len(flags) == 1
        assert flags[0]["cycle"] == "cycle_1_escalation"
        assert flags[0]["exhausted"] is False


class TestTextQualityCycle:
    """Integration tests: escalation back-edge firing in the compiled graph."""

    def test_escalation_back_edge_fires(self, tmp_path):
        """Empty chapter text triggers escalation; cycles back through text_quality."""
        from audify.qa.graph import run_graph

        creator = MagicMock()
        creator.max_chapters = None
        creator.audiobook_path = tmp_path / "out"
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        creator.chapter_titles = []
        creator.mode = "process"
        creator.confirm = False
        creator.warn_stop = False
        creator.fidelity_check = False
        creator.language = "en"
        creator.resolved_language = "en"
        from audify.readers.pdf import PdfReader
        creator.reader = MagicMock(spec=PdfReader)
        # Empty text triggers garbage detection in text_quality
        creator.reader.cleaned_text = ""
        creator.reader.path = tmp_path / "dummy.pdf"
        # Create a dummy file so the escalate node can attempt to read it
        (tmp_path / "dummy.pdf").write_bytes(b"%PDF-1.4 dummy")
        creator.generate_audiobook_script.side_effect = lambda c, n: f"Script for {n}"
        creator.progress = MagicMock()
        creator._verify_tts_provider_available.return_value = None
        creator._validate_chapters.return_value = None
        creator._save_chapter_titles.return_value = None
        creator.create_m4b.return_value = None
        creator.synthesize_episode = MagicMock()

        with patch("audify.readers.ebook.EpubReader"):
            run_graph(creator)

        # The escalation should fire (empty text), and the graph should
        # complete with best-effort text (escalate will try but fail to
        # parse the dummy PDF, keeping the original empty text).
        report_path = tmp_path / "out" / "quality_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        # The chapter should be flagged (exhausted after 3 attempts).
        chapter = report["chapters"]["chapter_1"]
        assert chapter["verdict"] in ("unrecoverable", "flagged"), chapter["verdict"]

    def test_clean_text_no_escalation(self, tmp_path):
        """Clean text passes through without escalation."""
        from audify.qa.graph import run_graph

        creator = MagicMock()
        creator.max_chapters = None
        creator.audiobook_path = tmp_path / "out"
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        creator.chapter_titles = []
        creator.mode = "process"
        creator.confirm = False
        creator.warn_stop = False
        creator.fidelity_check = False
        creator.language = "en"
        creator.resolved_language = "en"
        from audify.readers.pdf import PdfReader
        creator.reader = MagicMock(spec=PdfReader)
        creator.reader.cleaned_text = "\n\n".join(["Clean chapter text. " * 50])
        creator.reader.path = tmp_path / "dummy.pdf"
        (tmp_path / "dummy.pdf").write_bytes(b"%PDF-1.4")
        creator.generate_audiobook_script.side_effect = lambda c, n: f"Script for {n}"
        creator.progress = MagicMock()
        creator._verify_tts_provider_available.return_value = None
        creator._validate_chapters.return_value = None
        creator._save_chapter_titles.return_value = None
        creator.create_m4b.return_value = None
        creator.synthesize_episode = MagicMock()

        with patch("audify.readers.ebook.EpubReader"):
            run_graph(creator)

        report_path = tmp_path / "out" / "quality_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["chapters"]["chapter_1"]["verdict"] == "clean"

    def test_synthesize_mode_skips_text_quality(self, tmp_path):
        """synthesize mode has no text_quality node."""
        from audify.qa.graph import build_graph

        graph = build_graph("synthesize")
        nodes = set(graph.get_graph().nodes)
        assert "text_quality" not in nodes

    def test_full_mode_includes_text_quality(self):
        """full mode includes text_quality node."""
        from audify.qa.graph import build_graph

        graph = build_graph()
        nodes = set(graph.get_graph().nodes)
        assert "text_quality" in nodes


class TestTextQualityHelpers:
    """Direct coverage of text-quality helpers not reached by node tests."""

    def test_is_epub_reader_true(self):
        from audify.qa.nodes.text_quality import _is_epub_reader

        creator = MagicMock()
        # Create a mock reader that isinstance recognises as EpubReader.
        import audify.readers.ebook as eb
        reader = MagicMock(spec=eb.EpubReader)
        creator.reader = reader
        assert _is_epub_reader(creator)

    def test_is_epub_reader_false(self):
        from audify.qa.nodes.text_quality import _is_epub_reader
        from audify.readers.pdf import PdfReader

        creator = MagicMock()
        reader = MagicMock(spec=PdfReader)
        creator.reader = reader
        assert not _is_epub_reader(creator)

    def test_is_epub_reader_none(self):
        from audify.qa.nodes.text_quality import _is_epub_reader

        creator = MagicMock()
        creator.reader = None
        assert not _is_epub_reader(creator)

    def test_has_escalation_history_from_budget(self):
        from audify.qa.nodes.text_quality import _has_escalation_history

        flags = {}
        retry_budget = {"chapter_1": {"cycle_1_escalation": 2}}
        assert _has_escalation_history(flags, retry_budget, "chapter_1")

    def test_has_escalation_history_from_flags(self):
        from audify.qa.nodes.text_quality import _has_escalation_history

        flags = {
            "chapter_1": [
                {"cycle": "cycle_1_escalation", "reason": "test", "exhausted": False}
            ]
        }
        retry_budget = {}
        assert _has_escalation_history(flags, retry_budget, "chapter_1")

    def test_has_escalation_history_false(self):
        from audify.qa.nodes.text_quality import _has_escalation_history

        flag = {"cycle": "cycle_2_reroute", "reason": "x", "exhausted": False}
        flags = {"chapter_2": [flag]}
        retry_budget = {}
        assert not _has_escalation_history(flags, retry_budget, "chapter_1")

    def test_classify_empty_text(self):
        from audify.qa.nodes.text_quality import _classify

        creator = MagicMock()
        assert _classify("", "ch1", creator) == "garbage"

    def test_classify_clean_text(self):
        from audify.qa.nodes.text_quality import _classify

        creator = MagicMock()
        text = "This is a normal paragraph of text that should pass all checks. " * 10
        assert _classify(text, "ch1", creator) == "clean"

    def test_escalate_no_source_path(self):
        """escalate_node handles missing source path gracefully."""
        from audify.qa.nodes.text_quality import escalate_node

        creator = MagicMock()
        creator.reader = MagicMock()
        creator.reader.path = None  # No path available

        state = {
            "creator": creator,
            "chapters": [""],
            "chapter_titles": ["Ch 1"],
            "pending_escalation": [1],
            "retry_budget": {"chapter_1": {"cycle_1_escalation": 2}},
        }
        result = escalate_node(state)
        # Should not crash; returns with empty pending_escalation
        assert result["pending_escalation"] == []

    def test_classify_mojibake_text(self):
        """_classify returns garbage for text with high mojibake ratio."""
        from audify.qa.nodes.text_quality import _classify

        creator = MagicMock()
        text = "\x80\x81\x82\x83\x84" * 10 + "normal "
        assert _classify(text, "ch1", creator) == "garbage"

    def test_classify_high_nonword_ratio(self):
        """_classify returns garbage for text with high nonword ratio."""
        from audify.qa.nodes.text_quality import _classify

        creator = MagicMock()
        # Text that passes whitespace check but fails nonword check.
        # Ratio of non-alphanumeric chars must be > 0.30.
        norm = "Normal words here "
        symbols = ">>>>..,,,,,;;;;;!!!!!%%%%%"
        text = norm * 2 + symbols  # ~52 alnum + ~35 symbols = ~40% nonword
        assert _classify(text, "ch1", creator) == "garbage"

    def test_classify_bad_whitespace_ratio(self):
        """_classify returns garbage for text with extreme whitespace."""
        from audify.qa.nodes.text_quality import _classify

        creator = MagicMock()
        # Text with content but very high whitespace ratio (>0.70).
        # Must pass empty check first (len > 10 after strip).
        text = "word" + " " * 80 + "another"
        assert _classify(text, "ch1", creator) == "garbage"

    def test_mojibake_empty_text(self):
        """_has_high_mojibake_ratio returns True for empty text."""
        from audify.qa.nodes.text_quality import _has_high_mojibake_ratio

        assert _has_high_mojibake_ratio("")

    def test_mojibake_c1_control_chars(self):
        """_has_high_mojibake_ratio detects C1 control characters."""
        from audify.qa.nodes.text_quality import _has_high_mojibake_ratio

        # C1 range is 0x80-0x9F. 3 C1 chars out of 10 total = 0.3
        text = chr(0x85) + chr(0x90) + chr(0x95) + "normal"
        assert _has_high_mojibake_ratio(text, threshold=0.25)

    def test_mojibake_c0_control_chars(self):
        """_has_high_mojibake_ratio detects C0 control characters."""
        from audify.qa.nodes.text_quality import _has_high_mojibake_ratio

        # C0 range below 0x20, excluding tab(0x09), lf(0x0A), cr(0x0D)
        text = chr(0x01) + chr(0x02) + chr(0x03) + "normal"
        assert _has_high_mojibake_ratio(text, threshold=0.25)

    def test_whitespace_empty_text(self):
        """_has_bad_whitespace_ratio returns True for empty text."""
        from audify.qa.nodes.text_quality import _has_bad_whitespace_ratio

        assert _has_bad_whitespace_ratio("")

    def test_nonword_empty_text(self):
        """_has_high_nonword_ratio returns True for empty text."""
        from audify.qa.nodes.text_quality import _has_high_nonword_ratio

        assert _has_high_nonword_ratio("")

    def test_is_epub_reader_fallback(self, monkeypatch):
        """_is_epub_reader fallback: TypeError triggers class-name check."""
        from audify.qa.nodes.text_quality import _is_epub_reader

        creator = MagicMock()
        creator.reader = MagicMock()

        # Patch isinstance globally so the first isinstance call raises TypeError.
        import builtins
        real_isinstance = builtins.isinstance

        call_count = 0

        def _broken_isinstance(obj, cls):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("simulated isinstance failure")
            return real_isinstance(obj, cls)

        monkeypatch.setattr(builtins, "isinstance", _broken_isinstance)
        # Fallback checks type().__name__ which is "MagicMock", not "EpubReader"
        assert not _is_epub_reader(creator)

    def test_escalate_epub_reader_success(self, monkeypatch):
        """escalate_node with EPUB reader succeeds when escalation returns text."""
        from audify.qa.nodes.text_quality import escalate_node as enode
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._is_epub_reader",
            lambda c: True,
        )
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._escalate_epub",
            lambda path, attempt: ["Item 1", "Item 2"],
        )
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._group_epub_spine_texts",
            lambda path, item_texts: ["Escalated clean text."],
        )
        creator = MagicMock()
        creator.reader = MagicMock()
        creator.reader.path = "/fake/book.epub"
        state = {
            "creator": creator,
            "chapters": [""],
            "chapter_titles": ["Ch 1"],
            "pending_escalation": [1],
            "retry_budget": {"chapter_1": {"cycle_1_escalation": 0}},
        }
        result = enode(state)
        assert result["pending_escalation"] == []
        assert result["chapters"] == ["Escalated clean text."]

    def test_escalate_epub_dispatch(self, monkeypatch):
        """_escalate_epub dispatches to correct ladder rung by attempt number."""
        from audify.qa.nodes.text_quality import _escalate_epub

        # attempt 1 returns None (already done by read_node)
        assert _escalate_epub("/fake/book.epub", 1) is None

        # attempt 2 tries ebooklib raw — missing dep returns None
        assert _escalate_epub("/fake/book.epub", 2) is None

        # attempt 3 tries regex — missing dep returns None
        assert _escalate_epub("/fake/book.epub", 3) is None

        # attempt 4+ also uses regex
        assert _escalate_epub("/fake/book.epub", 4) is None

    def test_escalate_pdf_dispatch(self, monkeypatch):
        """_escalate_pdf dispatches to correct ladder rung by attempt number."""
        from audify.qa.nodes.text_quality import _escalate_pdf

        # attempt 1 returns None (already done by read_node)
        assert _escalate_pdf("/fake/doc.pdf", 1) is None

        # attempt 2 tries sort-mode — missing fitz returns None
        assert _escalate_pdf("/fake/doc.pdf", 2) is None

        # attempt 3 tries OCR — missing deps returns None
        assert _escalate_pdf("/fake/doc.pdf", 3) is None

    def test_escalate_pdf_reader_success(self, monkeypatch):
        """escalate_node with PDF reader succeeds when escalation returns text."""
        from audify.qa.nodes.text_quality import escalate_node as enode
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._is_epub_reader",
            lambda c: False,
        )
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._escalate_pdf",
            lambda path, attempt: "OCR extracted text.",
        )
        creator = MagicMock()
        creator.reader = MagicMock()
        creator.reader.path = "/fake/doc.pdf"
        state = {
            "creator": creator,
            "chapters": [""],
            "chapter_titles": ["Ch 1"],
            "pending_escalation": [1],
            "retry_budget": {"chapter_1": {"cycle_1_escalation": 0}},
        }
        result = enode(state)
        assert result["pending_escalation"] == []
        assert result["chapters"][0] == "OCR extracted text."


def _write_test_epub(tmp_path: Path) -> Path:
    """Build a minimal two-chapter EPUB fixture with a nested TOC."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test-epub-escalation")
    book.set_title("Escalation Test Book")
    book.set_language("en")

    c1 = epub.EpubHtml(title="Chapter 1", file_name="chap_01.xhtml", lang="en")
    c1.content = (
        "<html><body><h1>Chapter 1</h1>"
        "<p>First chapter body text for escalation.</p>"
        "<script>var skipped = true;</script>"
        "<style>p { color: red; }</style>"
        "</body></html>"
    )
    c2 = epub.EpubHtml(title="Chapter 2", file_name="chap_02.xhtml", lang="en")
    c2.content = (
        "<html><body><h1>Chapter 2</h1>"
        "<p>Second chapter body text for escalation.</p></body></html>"
    )
    book.add_item(c1)
    book.add_item(c2)
    # Nested TOC: flat link + (section, [link]) tuple to exercise both
    # branches of the TOC walker.
    book.toc = (
        epub.Link("chap_01.xhtml", "Chapter 1", "ch1"),
        (epub.Section("Part 1"), [epub.Link("chap_02.xhtml", "Chapter 2", "ch2")]),
    )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2]

    path = tmp_path / "escalation_test.epub"
    epub.write_epub(str(path), book)
    return path


def _write_test_pdf(tmp_path: Path) -> Path:
    """Build a minimal one-page PDF fixture with real text."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sorted extraction sample text for escalation.")
    path = tmp_path / "escalation_test.pdf"
    doc.save(str(path))
    doc.close()
    return path


class TestTextQualityEscalationExtraction:
    """Coverage of the concrete EPUB/PDF escalation-ladder extractors."""

    # --- threshold parsing -------------------------------------------------

    def test_safe_float_threshold_none(self):
        from audify.qa.nodes.text_quality import _safe_float_threshold

        assert _safe_float_threshold(None, 0.1, "X") == 0.1

    def test_safe_float_threshold_invalid(self):
        from audify.qa.nodes.text_quality import _safe_float_threshold

        assert _safe_float_threshold("not-a-number", 0.25, "X") == 0.25

    def test_safe_float_threshold_valid_string(self):
        from audify.qa.nodes.text_quality import _safe_float_threshold

        assert _safe_float_threshold("0.42", 0.1, "X") == pytest.approx(0.42)

    # --- escalate_node edge branches ---------------------------------------

    def test_escalate_node_no_pending(self):
        """escalate_node with nothing pending returns chapters unchanged."""
        from audify.qa.nodes.text_quality import escalate_node

        creator = MagicMock()
        state = {
            "creator": creator,
            "chapters": ["keep me"],
            "pending_escalation": [],
        }
        result = escalate_node(state)
        assert result["chapters"] == ["keep me"]
        assert result["pending_escalation"] == []

    def test_escalate_epub_grouping_failure_keeps_previous(self, monkeypatch):
        """EPUB escalation extracting text but failing to group keeps old text."""
        from audify.qa.nodes.text_quality import escalate_node as enode

        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._is_epub_reader", lambda c: True
        )
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._escalate_epub",
            lambda path, attempt: ["raw item text"],
        )
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._group_epub_spine_texts",
            lambda path, texts: None,
        )
        creator = MagicMock()
        creator.reader = MagicMock()
        creator.reader.path = "/fake/book.epub"
        state = {
            "creator": creator,
            "chapters": ["previous text"],
            "chapter_titles": ["Ch 1"],
            "pending_escalation": [1],
            "retry_budget": {"chapter_1": {"cycle_1_escalation": 2}},
        }
        result = enode(state)
        assert result["chapters"] == ["previous text"]
        assert result["pending_escalation"] == []

    def test_escalate_epub_no_text_keeps_previous(self, monkeypatch):
        """EPUB escalation producing no text keeps the previous chapters."""
        from audify.qa.nodes.text_quality import escalate_node as enode

        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._is_epub_reader", lambda c: True
        )
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._escalate_epub",
            lambda path, attempt: None,
        )
        creator = MagicMock()
        creator.reader = MagicMock()
        creator.reader.path = "/fake/book.epub"
        state = {
            "creator": creator,
            "chapters": ["previous text"],
            "chapter_titles": ["Ch 1"],
            "pending_escalation": [1],
            "retry_budget": {"chapter_1": {"cycle_1_escalation": 2}},
        }
        result = enode(state)
        assert result["chapters"] == ["previous text"]

    def test_escalate_pdf_no_text_keeps_previous(self, monkeypatch):
        """PDF escalation producing no text keeps the previous chapters."""
        from audify.qa.nodes.text_quality import escalate_node as enode

        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._is_epub_reader", lambda c: False
        )
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._escalate_pdf",
            lambda path, attempt: None,
        )
        creator = MagicMock()
        creator.reader = MagicMock()
        creator.reader.path = "/fake/doc.pdf"
        state = {
            "creator": creator,
            "chapters": ["previous text"],
            "chapter_titles": ["Ch 1"],
            "pending_escalation": [1],
            "retry_budget": {"chapter_1": {"cycle_1_escalation": 2}},
        }
        result = enode(state)
        assert result["chapters"] == ["previous text"]

    # --- EPUB raw ebooklib walking ------------------------------------------

    def test_ebooklib_raw_extracts_text(self, tmp_path):
        from audify.qa.nodes.text_quality import _epub_escalate_ebooklib_raw

        path = _write_test_epub(tmp_path)
        texts = _epub_escalate_ebooklib_raw(path)
        assert texts is not None
        joined = "\n".join(texts)
        assert "First chapter body text for escalation." in joined
        assert "Second chapter body text for escalation." in joined
        # script/style content must be stripped.
        assert "var skipped" not in joined
        assert "color: red" not in joined

    def test_ebooklib_raw_missing_ebooklib(self, monkeypatch, tmp_path):
        import sys

        from audify.qa.nodes.text_quality import _epub_escalate_ebooklib_raw

        monkeypatch.setitem(sys.modules, "ebooklib", None)
        assert _epub_escalate_ebooklib_raw(tmp_path / "book.epub") is None

    def test_ebooklib_raw_missing_lxml(self, monkeypatch, tmp_path):
        import sys

        from audify.qa.nodes.text_quality import _epub_escalate_ebooklib_raw

        monkeypatch.setitem(sys.modules, "lxml.html", None)
        assert _epub_escalate_ebooklib_raw(tmp_path / "book.epub") is None

    # --- EPUB regex last resort ----------------------------------------------

    def test_regex_extracts_text(self, tmp_path):
        from audify.qa.nodes.text_quality import _epub_escalate_regex

        path = _write_test_epub(tmp_path)
        texts = _epub_escalate_regex(path)
        assert texts is not None
        joined = "\n".join(texts)
        assert "First chapter body text for escalation." in joined
        assert "Second chapter body text for escalation." in joined
        # No HTML tags may survive.
        assert "<p>" not in joined

    def test_regex_missing_ebooklib(self, monkeypatch, tmp_path):
        import sys

        from audify.qa.nodes.text_quality import _epub_escalate_regex

        monkeypatch.setitem(sys.modules, "ebooklib", None)
        assert _epub_escalate_regex(tmp_path / "book.epub") is None

    # --- TOC-boundary chapter grouping ----------------------------------------

    def test_group_spine_texts(self, tmp_path):
        from audify.qa.nodes.text_quality import _group_epub_spine_texts

        path = _write_test_epub(tmp_path)
        chapters = _group_epub_spine_texts(path, ["Text one.", "Text two."])
        assert chapters == ["Text one.", "Text two."]

    def test_group_spine_texts_bad_path(self, tmp_path):
        from audify.qa.nodes.text_quality import _group_epub_spine_texts

        assert _group_epub_spine_texts(tmp_path / "missing.epub", ["x"]) is None

    def test_group_spine_texts_missing_ebooklib(self, monkeypatch, tmp_path):
        import sys

        from audify.qa.nodes.text_quality import _group_epub_spine_texts

        monkeypatch.setitem(sys.modules, "ebooklib", None)
        assert _group_epub_spine_texts(tmp_path / "book.epub", ["x"]) is None

    def test_group_spine_texts_empty_items_falls_back(self, tmp_path):
        """No item texts to consume: falls back to a single joined slab."""
        from audify.qa.nodes.text_quality import _group_epub_spine_texts

        path = _write_test_epub(tmp_path)
        assert _group_epub_spine_texts(path, []) == [""]

    # --- PDF sort-mode extraction -----------------------------------------------

    def test_pdf_sort_extracts_text(self, tmp_path):
        from audify.qa.nodes.text_quality import _pdf_escalate_sort

        path = _write_test_pdf(tmp_path)
        text = _pdf_escalate_sort(path)
        assert text is not None
        assert "Sorted extraction sample text for escalation." in text

    def test_pdf_sort_missing_fitz(self, monkeypatch, tmp_path):
        import sys

        from audify.qa.nodes.text_quality import _pdf_escalate_sort

        monkeypatch.setitem(sys.modules, "fitz", None)
        assert _pdf_escalate_sort(tmp_path / "doc.pdf") is None

    # --- PDF OCR extraction ------------------------------------------------------

    def test_ocr_missing_pytesseract(self, monkeypatch, tmp_path):
        import sys

        from audify.qa.nodes.text_quality import _pdf_escalate_ocr

        monkeypatch.setitem(sys.modules, "pytesseract", None)
        assert _pdf_escalate_ocr(tmp_path / "doc.pdf") is None

    def test_ocr_missing_pdf2image(self, monkeypatch, tmp_path):
        import sys

        from audify.qa.nodes.text_quality import _pdf_escalate_ocr

        monkeypatch.setitem(sys.modules, "pdf2image", None)
        assert _pdf_escalate_ocr(tmp_path / "doc.pdf") is None

    def test_ocr_success(self, monkeypatch, tmp_path):
        """OCR joins non-empty page texts and skips whitespace-only pages."""
        import sys

        from audify.qa.nodes.text_quality import _pdf_escalate_ocr

        fake_tesseract = MagicMock()
        fake_tesseract.image_to_string.side_effect = [
            "Page one text.",
            "   ",
            "Page three text.",
        ]
        fake_pdf2image = MagicMock()
        fake_pdf2image.convert_from_path.return_value = ["img1", "img2", "img3"]
        monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)
        monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)

        result = _pdf_escalate_ocr(tmp_path / "doc.pdf")
        assert result == "Page one text.\n\nPage three text."

    def test_ocr_all_pages_fail(self, monkeypatch, tmp_path):
        """OCR failing on every page returns None instead of crashing."""
        import sys

        from audify.qa.nodes.text_quality import _pdf_escalate_ocr

        fake_tesseract = MagicMock()
        fake_tesseract.image_to_string.side_effect = RuntimeError("tesseract died")
        fake_pdf2image = MagicMock()
        fake_pdf2image.convert_from_path.return_value = ["img1", "img2"]
        monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)
        monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)

        assert _pdf_escalate_ocr(tmp_path / "doc.pdf") is None

    # --- escalation attempt mapping --------------------------------------------

    @pytest.mark.parametrize(
        ("remaining_budget", "expected_attempt"),
        [
            (2, 2),  # first escalation: budget already decremented 3 -> 2
            (1, 3),  # second escalation: last-resort rung
            (0, 4),  # third escalation: stays on the last rung
        ],
    )
    def test_escalation_attempt_starts_at_first_fallback(
        self, monkeypatch, remaining_budget, expected_attempt
    ):
        """First escalation must use ladder rung 2, not the no-op rung 1.

        Rung 1 (the default parser) already ran in read_node, so mapping the
        first escalation to attempt 1 would burn a cycle without re-extracting.
        """
        from audify.qa.nodes.text_quality import escalate_node as enode

        seen: dict[str, int] = {}

        def _capture(path, attempt):
            seen["attempt"] = attempt
            return None

        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._is_epub_reader", lambda c: True
        )
        monkeypatch.setattr(
            "audify.qa.nodes.text_quality._escalate_epub", _capture
        )
        creator = MagicMock()
        creator.reader = MagicMock()
        creator.reader.path = "/fake/book.epub"
        state = {
            "creator": creator,
            "chapters": [""],
            "chapter_titles": ["Ch 1"],
            "pending_escalation": [1],
            "retry_budget": {"chapter_1": {"cycle_1_escalation": remaining_budget}},
        }
        enode(state)
        assert seen["attempt"] == expected_attempt

    # --- _is_epub_reader inner fallback ---------------------------------------

    def test_is_epub_reader_fallback_broken_type_name(self, monkeypatch):
        """Fallback class-name check returning False when __name__ itself raises."""
        import builtins

        from audify.qa.nodes.text_quality import _is_epub_reader

        class _NoName(type):
            @property
            def __name__(cls):
                raise AttributeError("no name")

        class _Reader(metaclass=_NoName):
            pass

        reader = _Reader()
        creator = MagicMock()
        creator.reader = reader

        real_isinstance = builtins.isinstance
        raised = {"done": False}

        def _broken_isinstance(obj, cls):
            if not raised["done"] and obj is reader:
                raised["done"] = True
                raise TypeError("simulated isinstance failure")
            return real_isinstance(obj, cls)

        monkeypatch.setattr(builtins, "isinstance", _broken_isinstance)
        assert not _is_epub_reader(creator)
