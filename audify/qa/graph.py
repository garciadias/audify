from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph

from audify.qa.nodes.assemble import assemble_node
from audify.qa.nodes.read import read_node
from audify.qa.nodes.report import report_node
from audify.qa.nodes.script_gen import script_gen_node
from audify.qa.nodes.synthesize import synthesize_node
from audify.qa.state import GraphState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


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


def run_graph(creator: Any) -> Path:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"```mermaid\n{diagram}\n```\n")
    print(f"Graph diagram written to {output_path}")
