from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from audify.qa.nodes.assemble import assemble_node
from audify.qa.nodes.confirm import confirm_node
from audify.qa.nodes.load_scripts import load_scripts_node
from audify.qa.nodes.read import read_node
from audify.qa.nodes.report import report_node
from audify.qa.nodes.script_gen import script_gen_node
from audify.qa.nodes.synthesize import synthesize_node
from audify.qa.state import GraphState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from audify.audiobook_creator import AudiobookCreator


def build_graph(mode: str = "full") -> "CompiledStateGraph":
    """Assemble the QA pipeline graph for the requested *mode*.

    Three topologies, each acyclic (no back-edges in this skeleton):

    * ``"full"`` — ``read → script_gen → synthesize → assemble → report → END``
    * ``"process"`` — ``read → script_gen → report → END`` (no TTS)
    * ``"synthesize"`` — ``load_scripts → synthesize → assemble → report → END``
      (loads previously-saved scripts, no LLM)

    Each shape mirrors the corresponding branch of the legacy
    ``AudiobookCreator.create_audiobook_series`` orchestrator.
    """
    if mode == "process":
        return _build_process_graph()
    if mode == "synthesize":
        return _build_synthesize_graph()
    if mode != "full":
        raise ValueError(
            f"Unknown graph mode '{mode}': expected 'full', 'process', "
            "or 'synthesize'"
        )
    return _build_full_graph()


def _build_full_graph() -> "CompiledStateGraph":
    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("read", read_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("script_gen", script_gen_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("assemble", assemble_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("read")
    builder.add_edge("read", "confirm")
    builder.add_edge("confirm", "script_gen")
    builder.add_edge("script_gen", "synthesize")
    builder.add_edge("synthesize", "assemble")
    builder.add_edge("assemble", "report")
    builder.add_edge("report", END)
    return builder.compile()


def _build_process_graph() -> "CompiledStateGraph":
    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("read", read_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("script_gen", script_gen_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("read")
    builder.add_edge("read", "confirm")
    builder.add_edge("confirm", "script_gen")
    builder.add_edge("script_gen", "report")
    builder.add_edge("report", END)
    return builder.compile()


def _build_synthesize_graph() -> "CompiledStateGraph":
    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("load_scripts", load_scripts_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("assemble", assemble_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("load_scripts")
    builder.add_edge("load_scripts", "synthesize")
    builder.add_edge("synthesize", "assemble")
    builder.add_edge("assemble", "report")
    builder.add_edge("report", END)
    return builder.compile()


def run_graph(creator: "AudiobookCreator") -> Path:
    """Run the LangGraph pipeline for *creator* and return the output path.

    Dispatches to the mode-specific sub-graph based on ``creator.mode``
    (``"full"``, ``"process"``, or ``"synthesize"``). Runs the TTS
    preflight guard before invoking the graph for modes that perform
    synthesis (``"full"`` and ``"synthesize"``), so misconfigured
    providers fail fast before any LLM/IO spend.
    """
    mode = getattr(creator, "mode", "full")

    # Preflight guard: matches the legacy
    # ``_verify_tts_provider_available`` calls in
    # ``create_audiobook_series`` and ``_synthesize_from_existing_scripts``.
    # Skipped for ``"process"`` since no TTS happens.
    if mode != "process":
        creator._verify_tts_provider_available()

    graph = build_graph(mode)

    initial_state: GraphState = {
        "creator": creator,
        "chapters": [],
        "chapter_titles": [],
        "chapter_scripts": [],
        "episode_paths": [],
        "retry_budget": {},
        "best_wer": {},
        "flags": {},
    }

    graph.invoke(initial_state)
    return Path(creator.audiobook_path)


def render_mermaid(output_path: Path = Path("docs/graph.md")) -> None:
    """Render the full-mode graph topology as a Mermaid diagram.

    The diagram reflects the ``"full"`` topology; the ``"process"`` and
    ``"synthesize"`` sub-graphs are subsets and are documented prose-only
    in the doc page.
    """
    graph = build_graph("full")
    diagram = graph.get_graph().draw_mermaid()
    diagram = _strip_mermaid_frontmatter(diagram)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# QA Pipeline Graph\n\n"
        "Topology of the LangGraph QA pipeline in `full` mode "
        "(`read → script_gen → synthesize → assemble → report`).\n\n"
        f"```mermaid\n{diagram}\n```\n\n"
        "## Sub-graphs\n\n"
        "* **`process` mode** (`--process-only`): "
        "`read → confirm → script_gen → report → END` — no TTS, no M4B.\n"
        "* **`synthesize` mode** (`--synthesize-only`): "
        "`load_scripts → synthesize → assemble → report → END` — "
        "scripts are loaded from a previous `--process-only` run.\n"
    )
    print(f"Graph diagram written to {output_path}")


def _strip_mermaid_frontmatter(diagram: str) -> str:
    """Remove the leading ``---``-delimited YAML front-matter from Mermaid output.

    ``draw_mermaid`` emits a ``---config:...---`` block that Sphinx's MyST
    ``mermaid`` fence-to-directive conversion misparses as JSON directive
    options, breaking the docs build. The block only carries cosmetic curve
    styling, so it is safe to drop.
    """
    stripped = diagram.lstrip()
    if stripped.startswith("---"):
        parts = stripped.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip("\n")
    return diagram
