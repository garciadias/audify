# Quality assurance is a cyclic graph, not a central LLM agent

We are turning Audify's linear DAG pipeline into a self-correcting cyclic graph
in which quality checks can fire back-edges that re-run earlier nodes. The
"agent" is the **graph topology itself** — autonomous re-execution driven by
quality feedback — not a central LLM that reasons about routing. LLMs are used
only at the nodes that genuinely require judgment (the script-validity judge and
metadata infer-and-fill); every other detector is deterministic (WER, text
extraction quality, duration ratio).

## Status

accepted

## Considered options

- **Central LLM Critic** that ingests all signals and chooses the back-edge via
  reasoning/tool-calls. Rejected: slower, costlier, non-deterministic to test,
  and indefensible under review ("why is an LLM choosing between three `if`
  branches?").
- **Graph-is-the-agent (chosen).** Deterministic router + LLM only at judgment
  nodes. Testable, cheap, and the autonomy lives in the topology — which is
  the architectural shift this change is built around.

## Consequences

- Each back-edge must have a **bounded** halting condition (max attempts) and a
  **terminal action** (keep best-effort result + flag) so the graph provably
  halts — no infinite loops.
- Detectors are unit-testable in isolation; the graph is testable by asserting
  which edge fires for a given state.
