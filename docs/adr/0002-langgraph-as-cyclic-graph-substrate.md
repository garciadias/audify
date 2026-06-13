# LangGraph as the cyclic-graph substrate

We will build the QA pipeline (see ADR-0001) as an explicit LangGraph graph
rather than hand-rolling the loop in the existing `audiobook_creator`
orchestration. LangGraph is the industry-standard substrate for cyclic, stateful
agent graphs in Python, and it makes the back-edges first-class, legible objects
in the code (`add_conditional_edges` pointing backward), which is the point: the
DAG→cyclic shift becomes inspectable rather than an emergent property of
control flow.

## Status

accepted

## Considered options

- **Hand-rolled `while` loops** in the current orchestrator. No new dependency,
  but the cyclic graph lives implicitly in control flow — harder to reason
  about as cycles multiply, and the topology stops being inspectable.
- **LangGraph (chosen).** A dependency and some lock-in, in exchange for the
  graph topology being a first-class, inspectable artifact and for using the
  substrate the wider ecosystem has standardised on for cyclic agent graphs.

## Consequences

- Adds a LangGraph dependency to a project that is currently plain Python + API
  clients.
- Pipeline state (per-chapter retry budgets, best-so-far WER, flags for the
  report) becomes an explicit shared state object — the thing the graph carries
  between nodes and across back-edges.

## Skeleton scope (deferred behaviours)

The skeleton expresses the legacy linear orchestration as an acyclic graph and
deliberately defers anything cyclic. Two legacy behaviours map onto the
Quality Checks defined in `CONTEXT.md` at the repo root and are therefore
deferred to the cyclic-detector PRs, not reimplemented inside the skeleton:

- **Per-chapter duration check** (`check_chapter_during_synthesis`,
  `verification_integration.py`): maps to the **Fidelity check** plus the
  short-LLM-output signal of the **Script-validity check**. The skeleton would
  reimplement it; the cyclic detectors replace it.
- **Post-M4B audiobook verification** (`verify_complete_audiobook`,
  `verify.py`): is the seed of the **Coverage check** (CONTEXT.md says so
  explicitly). Same reasoning.

Three legacy behaviours are *not* Quality Checks and are wired into the
skeleton as ordinary infrastructure rather than deferred:

- **TTS preflight** (`_verify_tts_provider_available`) — runs in `run_graph`
  before invoking the compiled graph, conditioned on `creator.mode != "process"`.
- **TOC preview + confirm prompt** — encapsulated in `confirm_node`, which sits
  between `read` and `script_gen` in the `full` and `process` graphs. An abort
  returns `{"chapters": []}`, and downstream nodes naturally no-op on empty
  state (no conditional edges needed).
- **Process / synthesize modes** — `build_graph(mode)` dispatches to three
  topologies: `full`, `process` (`read → confirm → script_gen → report`), and
  `synthesize` (`load_scripts → synthesize → assemble → report`). The
  `--graph` flag therefore honours `--process-only` and `--synthesize-only`
  identically to the legacy orchestrator.
