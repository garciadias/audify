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
