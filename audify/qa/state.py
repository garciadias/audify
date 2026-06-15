from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    # Only imported for type-checkers; the runtime field is kept as ``Any``
    # because LangGraph resolves TypedDict annotations via
    # ``typing.get_type_hints``, which would fail on a forward reference to
    # ``AudiobookCreator`` (importing it at runtime here causes a cycle).
    # The graph pipeline only operates on single-file creators; directory
    # processing has its own non-graph path (see audify/cli.py).
    from audify.audiobook_creator import AudiobookCreator

    CreatorT: TypeAlias = AudiobookCreator
else:
    CreatorT = Any


CycleId = Literal[
    "cycle_1_escalation",
    "cycle_2_reroute",
    "cycle_3_retry",
]
"""Canonical identifiers for the three remediation cycles (see ADR-0001)."""


class FlagEntry(TypedDict):
    """One quality-check flag raised against a chapter during the run.

    Cyclic detectors (#3 retry, #4 reroute, #5 escalation) append a
    ``FlagEntry`` to ``GraphState.flags[chapter_id]`` each time a Quality
    Check fires. The aggregator consumes the list to derive a per-chapter
    verdict and surface diagnostics in the final quality report.

    Fields:
        cycle: which remediation cycle raised the flag.
        reason: short, human-readable diagnostic, e.g.
            ``"text-extraction: mojibake"`` or
            ``"WER 0.42 exceeds 0.3 threshold"``.
        exhausted: ``True`` when the cycle's retry budget reached zero and
            the best-effort artifact was kept. ``False`` when the detector
            fired but the cycle resolved within its budget.
    """

    cycle: CycleId
    reason: str
    exhausted: bool


class GraphState(TypedDict):
    """Shared state carried by the LangGraph QA pipeline between nodes.

    Fields prefixed with ``retry_budget``, ``best_wer``, and ``flags`` are
    keyed by ``chapter_id`` (``"chapter_{episode_number}"``).  They are
    populated as the graph processes each chapter and are consumed by the
    report node at the end.

    ``retry_budget`` is two-level: outer key is the chapter id, inner key is
    the cycle id from :data:`CycleId`. Each cycle owns an independent
    counter so the aggregator can report attempts-used per cycle. Empty
    dictionaries are valid at every level — the cyclic detectors populate
    their own slots lazily as they fire.
    """

    # ``CreatorT`` is ``Any`` at runtime (LangGraph evaluates this) and
    # ``AudiobookCreator`` under mypy, so node bodies still get attribute-level
    # type checking when annotated against this state. Directory processing
    # does not flow through the graph, so a union is not needed here.
    creator: CreatorT
    chapters: list[str]
    chapter_titles: list[str]
    # list of (episode_number, script) pairs produced by script_gen
    chapter_scripts: list[tuple[int, str]]
    episode_paths: list[Path]
    # retry_budget[chapter_id][cycle_id] = remaining attempts for that cycle
    retry_budget: dict[str, dict[CycleId, int]]
    # best_wer[chapter_id] = lowest WER seen across cycle-3 retry attempts
    best_wer: dict[str, float]
    # flags[chapter_id] = ordered list of FlagEntry raised against the chapter
    flags: dict[str, list[FlagEntry]]
    # episode numbers the cycle-1 escalation back-edge scheduled for
    # re-extraction on the next ``escalate`` visit. Empty on the first pass;
    # populated by the text-quality detector and consumed by the escalation
    # node; cleared once ``escalate`` re-extracts the chapter.
    pending_escalation: list[int]
    # episode numbers the cycle-3 fidelity check scheduled for re-synthesis on
    # the next ``synthesize`` visit. Empty on the first pass; populated by the
    # back-edge and cleared once ``synthesize`` consumes it.
    pending_retry: list[int]
    # episode numbers the cycle-2 reroute back-edge scheduled for
    # script regeneration on the next ``script_gen`` visit. Empty on the
    # first pass; populated by the script-validity judge and consumed by
    # the back-edge routing once regeneration completes.
    pending_reroute: list[int]
    # episode numbers the next ``fidelity`` visit should evaluate. Set by
    # ``synthesize`` to all episodes on the first pass, and to just the
    # re-synthesized episodes on a retry pass.
    episodes_to_check: list[int]
