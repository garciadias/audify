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
        expected = {
            "read",
            "script_gen",
            "synthesize",
            "fidelity",
            "assemble",
            "report",
        }
        assert expected.issubset(nodes)

    def test_full_graph_has_fidelity_back_edge(self):
        """Cycle 3: full mode must contain the fidelity → synthesize back-edge."""
        graph = build_graph()
        edges = {(e[0], e[1]) for e in graph.get_graph().edges}
        assert ("synthesize", "fidelity") in edges
        assert ("fidelity", "synthesize") in edges, (
            "cycle-3 retry edge missing: fidelity must be able to loop back "
            "to synthesize"
        )

    def test_process_graph_is_acyclic(self):
        """`process` mode performs no TTS, so it stays a forward-only DAG."""
        graph = build_graph("process")
        node_order = ["read", "confirm", "script_gen", "report"]
        rank = {name: i for i, name in enumerate(node_order)}

        for edge in graph.get_graph().edges:
            src, dst = edge[0], edge[1]
            if src in rank and dst in rank:
                assert rank[src] < rank[dst], (
                    f"Back-edge detected: {src} → {dst}; "
                    "process mode must be acyclic."
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
        self.m4b_calls = 0

    def _verify_tts_provider_available(self):
        return None

    def _load_chapter_titles(self):
        return list(self._titles)

    def _get_tts_config(self):
        cfg = MagicMock()
        cfg.max_text_length = 4000
        return cfg

    def synthesize_episode(self, script, episode_number, max_text_length=None):
        self.synth_calls.append((episode_number, max_text_length))
        self._attempts[episode_number] = self._attempts.get(episode_number, 0) + 1
        path = self.episodes_path / f"episode_{episode_number:03d}.mp3"
        words = script.split()
        if self._attempts[episode_number] <= self._truncate_after:
            content = " ".join(words[: len(words) // 2])  # drop the tail
        else:
            content = script  # re-chunk restored full fidelity
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

        # The re-chunked audio on disk now carries the full script.
        final = (creator.episodes_path / "episode_001.mp3").read_text()
        assert final == _SCRIPT_20

    def test_clean_episode_never_triggers_cycle(self, tmp_path, fixed_duration):
        """An episode that transcribes faithfully raises no flag, no retry."""
        creator = _FidelityCreator(tmp_path, {1: _SCRIPT_20}, truncate_after=0)

        run_graph(creator)

        assert creator.synth_calls == [(1, None)]  # no retry
        report = self._load_report(creator)
        chapter = report["chapters"]["chapter_1"]
        assert chapter["verdict"] == "clean"
        assert chapter["flags"] == []

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
