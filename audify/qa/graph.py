from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from audify.qa.nodes.assemble import assemble_node
from audify.qa.nodes.read import read_node
from audify.qa.nodes.report import report_node
from audify.qa.nodes.script_gen import script_gen_node
from audify.qa.nodes.synthesize import synthesize_node
from audify.qa.state import GraphState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from audify.audiobook_creator import AudiobookCreator, DirectoryAudiobookCreator


def build_graph() -> "CompiledStateGraph":
    """Assemble the linear QA pipeline graph (no back-edges in this skeleton).

    Topology:
        read → script_gen → synthesize → assemble → report → END
    """
    builder: StateGraph = StateGraph(GraphState)

    builder.add_node("read", read_node)
    builder.add_node("script_gen", script_gen_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("assemble", assemble_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("read")
    builder.add_edge("read", "script_gen")
    builder.add_edge("script_gen", "synthesize")
    builder.add_edge("synthesize", "assemble")
    builder.add_edge("assemble", "report")
    builder.add_edge("report", END)

    return builder.compile()


def run_graph(creator: "AudiobookCreator | DirectoryAudiobookCreator") -> Path:
    """Run the LangGraph pipeline for *creator* and return the output path.

    Initialises GraphState with the creator instance and empty collection
    fields, then invokes the compiled graph.
    """
    graph = build_graph()

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
    """Render the graph topology as a Mermaid diagram and write to *output_path*."""
    graph = build_graph()
    diagram = graph.get_graph().draw_mermaid()
    diagram = _strip_mermaid_frontmatter(diagram)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# QA Pipeline Graph\n\n"
        "Topology of the LangGraph QA pipeline "
        "(`read → script_gen → synthesize → assemble → report`).\n\n"
        f"```mermaid\n{diagram}\n```\n"
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
