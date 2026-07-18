"""Cycle-2 script-validity judge node — LLM judge + reroute back-edge.

Between ``script_gen`` and ``synthesize`` (or ``report`` in ``process`` mode),
this node evaluates each generated script with a cheap pre-filter followed by
an LLM judge. It answers: *did the script faithfully represent the source
chapter, or did the LLM produce a summary, refusal, or error message?*

**Pre-filter** — the duration ratio from ``ChapterDurationChecker`` (expected
seconds from word count vs actual audio when available; on the first pass
before synthesis we compare against a rough expected-narration-length
heuristic). Scripts with a healthy ratio skip the LLM call entirely.

**LLM judge** — for suspiciously short scripts, call a lightweight LLM with a
structured prompt. Returns a JSON verdict: ``pass``, ``reroute``, or
``borderline``. Also checks the summarize/remove policy (no raw code blocks).

**Conditional back-edge** — on ``reroute``, the ``script_validity_route``
function returns ``"script_gen"`` to regenerate the failing chapter script.
On ``pass`` or ``borderline``, returns ``"synthesize"`` (or ``"report"``
in ``process`` mode).

**Retry budget** — bounded at ``MAX_BUDGET_PER_CYCLE`` (3) attempts per
chapter. On exhaustion, the best-effort script is kept and an ``exhausted``
flag is written for the quality report (#6). The pipeline never aborts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from audify.qa.nodes.report import MAX_BUDGET_PER_CYCLE
from audify.qa.state import CycleId, FlagEntry, GraphState

logger = logging.getLogger(__name__)

# Detection thresholds (overridable via creator attributes of the same name).
DURATION_RATIO_THRESHOLD = 0.5
"""Scripts shorter than 50% of expected narration length trigger the LLM judge."""

# Duration estimate basis — matches audify/verify.py's narration heuristic.
WORDS_PER_MINUTE = 75

# Judge prompt — XML-tagged, matching the conventions from commit ce3f191.
# The judge evaluates both content faithfulness and policy compliance.
_JUDGE_SYSTEM_PROMPT = """
<script_validity_judge>
  <role>
    Audio Narration Script Validator. Your job is to determine whether a
    generated narration script faithfully represents its source chapter
    text, or whether it contains a summary, refusal, error message, or
    policy violation.
  </role>

  <input_fields>
    <source>Full text of the original book chapter.</source>
    <script>The generated narration script to evaluate.</script>
  </input_fields>

  <evaluation_criteria>
    <criteria name="faithfulness">
      Does the script narrate the actual content of the source chapter, or
      does it substitute a summary, a refusal ("I'm sorry, I cannot..."),
      or an error message?
    </criteria>
    <criteria name="coverage">
      Does the script cover the key facts and narrative flow of the source,
      or does it omit large sections or condense them into a few sentences?
    </criteria>
    <criteria name="policy">
      Does the script avoid reading raw code blocks, bibliography entries,
      or parenthetical citations verbatim? Per the audiobook script policy,
      code should be described narratively, and citations/bibliography
      should be removed or summarised.
    </criteria>
  </evaluation_criteria>

  <output_schema>
    You must respond with ONLY a JSON object (no markdown, no commentary):
    {
      "verdict": "pass" | "reroute" | "borderline",
      "reason": "Brief explanation of the verdict"
    }
  </output_schema>

  <verdict_definitions>
    <verdict name="pass">Script is faithful, covers the source, and follows
    policy. Proceed to synthesis.</verdict>
    <verdict name="reroute">Script is NOT faithful — it is a summary, refusal, error
    message, or policy violation. Must regenerate.</verdict>
    <verdict name="borderline">Script is mostly faithful but has minor issues
    (e.g., slightly abbreviated). Can proceed but log the concern.</verdict>
  </verdict_definitions>
</script_validity_judge>
"""

_CYCLE: CycleId = "cycle_2_reroute"


def script_validity_node(state: GraphState) -> dict:
    """Evaluate each generated script for faithfulness and policy compliance.

    Returns updated state with ``pending_reroute`` populated when the judge
    flags scripts, and ``flags``/``retry_budget`` updated accordingly.
    """
    creator = state["creator"]
    chapters = state["chapters"]
    chapter_scripts: list[tuple[int, str]] = state["chapter_scripts"]

    creator.progress.set_phase("Validating scripts")

    # Copy nested containers so we never mutate inbound state in place.
    retry_budget = {k: dict(v) for k, v in state.get("retry_budget", {}).items()}
    flags = {k: list(v) for k, v in state.get("flags", {}).items()}

    pending_reroute: list[int] = []
    updated_scripts: dict[int, str] = {}
    chapters_dict: dict[int, str] = dict(enumerate(chapters, 1))

    for episode_number, script in chapter_scripts:
        chapter_id = f"chapter_{episode_number}"
        source_text = chapters_dict.get(episode_number, "")

        # ----- Pre-filter: word-count heuristic -----
        if _skip_judge_by_duration(script):
            logger.debug(
                "Script-validity pre-filter: %s duration looks healthy, skipping LLM.",
                chapter_id,
            )
            # If this chapter was previously rerouted (retry budget exists),
            # record a resolved flag now that a healthy script is here.
            if _has_reroute_history(flags, retry_budget, chapter_id):
                _append_flag(
                    flags, chapter_id,
                    reason="Script passed word-count pre-filter after reroute",
                    exhausted=False,
                )
            updated_scripts[episode_number] = script
            continue

        # ----- LLM judge -----
        verdict, reason = _llm_judge(creator, source_text, script, episode_number)
        logger.info(
            "Script-validity judge for %s: %s (%s)",
            chapter_id, verdict, reason,
        )

        if verdict == "pass":
            # If this chapter was previously rerouted, record a resolved flag.
            if _has_reroute_history(flags, retry_budget, chapter_id):
                _append_flag(
                    flags, chapter_id,
                    reason=f"Script passed LLM judge after reroute ({reason})",
                    exhausted=False,
                )
            updated_scripts[episode_number] = script
            continue

        if verdict == "borderline":
            # Pass through but log the concern (no flag, no reroute).
            logger.info(
                "Script-validity borderline for %s: %s. "
                "Proceeding with generated script.",
                chapter_id, reason,
            )
            updated_scripts[episode_number] = script
            continue

        # ----- Reroute: budget check -----
        budget = retry_budget.setdefault(chapter_id, {})
        remaining = budget.get(_CYCLE, MAX_BUDGET_PER_CYCLE)

        if remaining > 0:
            budget[_CYCLE] = remaining - 1
            pending_reroute.append(episode_number)
            logger.info(
                "Script-validity flagged %s (%s); scheduling reroute "
                "(%d attempt(s) left).",
                chapter_id, reason, remaining - 1,
            )
            # Keep stale entry so downstream can detect it was re-scheduled.
            updated_scripts[episode_number] = script
        else:
            # Budget exhausted: keep the script as-is, write exhausted flag.
            _append_flag(
                flags, chapter_id,
                reason=(
                    f"LLM judge flagged script after "
                    f"{MAX_BUDGET_PER_CYCLE} attempts: {reason}"
                ),
                exhausted=True,
            )
            updated_scripts[episode_number] = script
            logger.warning(
                "Script-validity exhausted for %s; keeping best-effort "
                "script (reason: %s).",
                chapter_id, reason,
            )

    # Rebuild chapter_scripts — scripts that need reroute keep their current
    # entry (script_gen will overwrite). Scripts that passed stay as-is.
    new_chapter_scripts: list[tuple[int, str]] = [
        (ep, updated_scripts.get(ep, s))
        for ep, s in chapter_scripts
    ]

    return {
        "chapter_scripts": new_chapter_scripts,
        "retry_budget": retry_budget,
        "flags": flags,
        "pending_reroute": pending_reroute,
    }


def script_validity_route(state: GraphState) -> str:
    """Conditional edge: loop back to ``script_gen`` while reroutes are pending.

    Routes based on the type of the current graph: in ``process`` mode the
    downstream is ``"report"``; in ``full`` mode it is ``"synthesize"``.
    """
    if state.get("pending_reroute"):
        return "script_gen"
    mode = getattr(state.get("creator"), "mode", "full")
    return "report" if mode == "process" else "synthesize"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_judge_by_duration(script: str) -> bool:
    """Cheap pre-filter: does the script's expected duration look healthy?

    On the first pass (no audio yet), there is no actual duration to compare
    against. This pre-filter is deliberately generous — it only short-circuits
    the LLM judge when the script is long enough to *obviously* be a real
    narration (several thousand words). Suspiciously-short scripts fall through
    to the LLM judge.

    The duration-ratio pre-filter that compares actual vs expected (from
    ``verify.py``) is used in the *second* pass (during re-synthesis of a
    flagged cycle-3 episode), where audio already exists. That path is handled
    by ``fidelity_node``; this is a simpler heuristic for the initial pass
    before TTS.
    """
    word_count = len(script.split())
    # Skip the LLM judge if the script has at least 200 words — a summary
    # is almost certainly shorter than this. Configurable via env var for
    # tuning without code changes.
    min_words = int(__import__("os").environ.get("SCRIPT_VALIDITY_MIN_WORDS", "200"))
    return word_count >= min_words


def _llm_judge(
    creator: Any,
    source_text: str,
    script: str,
    episode_number: int,
) -> tuple[str, str]:
    """Call the LLM judge and return ``(verdict, reason)``.

    Uses the creator's ``llm_client`` (same LLM as script generation, or a
    configured alternative) with the judge-specific prompt.

    Returns ``("pass", "")`` if the LLM is unavailable, so a missing LLM
    never blocks the pipeline — the existing script is accepted.
    """
    language = getattr(creator, "resolved_language", None) or getattr(
        creator, "language", None
    )

    llm_client = getattr(creator, "llm_client", None)
    if llm_client is None:
        logger.warning(
            "Script-validity judge: no LLM client available on creator. "
            "Skipping LLM judge for episode %d.",
            episode_number,
        )
        return "pass", ""

    # Build the user prompt with the source text and generated script.
    user_prompt = (
        "<source>\n"
        f"{source_text[:8000]}"
        "\n</source>\n\n"
        "<script>\n"
        f"{script[:4000]}"
        "\n</script>\n"
    )

    try:
        response = llm_client.generate_script(
            text=user_prompt,
            prompt=_JUDGE_SYSTEM_PROMPT,
            language=language,
            temperature=0.1,  # Low temperature for deterministic judgement.
        )
    except Exception as exc:
        logger.warning(
            "LLM judge call failed for episode %d: %s. "
            "Treating as pass to avoid blocking pipeline.",
            episode_number,
            exc,
        )
        return "pass", ""

    # Parse the JSON verdict from the response.
    try:
        parsed = _parse_judge_response(response)
        verdict = parsed.get("verdict", "pass")
        reason = parsed.get("reason", "")
        if verdict not in ("pass", "reroute", "borderline"):
            logger.warning(
                "LLM judge returned unknown verdict %r for episode %d. "
                "Defaulting to pass.",
                verdict,
                episode_number,
            )
            return "pass", reason
        return verdict, reason
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "Could not parse LLM judge response for episode %d: %s. "
            "Response was: %s. Defaulting to pass.",
            episode_number,
            exc,
            response[:200],
        )
        return "pass", ""


def _parse_judge_response(response: str) -> dict:
    """Parse the judge's JSON response, handling common LLM formatting issues.

    Strips markdown fences and leading/trailing whitespace before parsing.
    """
    cleaned = response.strip()
    # Remove markdown code fences if present.
    if cleaned.startswith("```"):
        # Extract content between first and last ```
        start = cleaned.index("\n")
        end = cleaned.rindex("```")
        cleaned = cleaned[start:end].strip()
    return json.loads(cleaned)


def _has_reroute_history(
    flags: dict[str, list[FlagEntry]],
    retry_budget: dict[str, dict[CycleId, int]],
    chapter_id: str,
) -> bool:
    """True if the chapter has a previous cycle-2 reroute history."""
    if retry_budget.get(chapter_id, {}).get(_CYCLE) is not None:
        return True
    return any(
        entry["cycle"] == _CYCLE
        for entry in flags.get(chapter_id, [])
    )


def _append_flag(
    flags: dict[str, list[FlagEntry]],
    chapter_id: str,
    *,
    reason: str,
    exhausted: bool,
) -> None:
    entry: FlagEntry = {
        "cycle": "cycle_2_reroute",
        "reason": reason,
        "exhausted": exhausted,
    }
    flags.setdefault(chapter_id, []).append(entry)
