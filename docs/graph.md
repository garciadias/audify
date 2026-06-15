# QA Pipeline Graph

Topology of the LangGraph QA pipeline in `full` mode (`read → confirm → script_gen → synthesize → fidelity → assemble → report`).

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	read(read)
	confirm(confirm)
	script_gen(script_gen)
	synthesize(synthesize)
	fidelity(fidelity)
	assemble(assemble)
	report(report)
	__end__([<p>__end__</p>]):::last
	__start__ --> read;
	assemble --> report;
	confirm --> script_gen;
	fidelity -.-> assemble;
	fidelity -.-> synthesize;
	read --> confirm;
	script_gen --> synthesize;
	synthesize --> fidelity;
	report --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## Cycle 3 — fidelity retry edge

`fidelity` boundary-samples each freshly-synthesized episode (head/tail STT round-trip + duration ratio). When it suspects TTS truncation it loops back to `synthesize` to re-chunk the offending episode on a smaller batch size, bounded to 3 retries per chapter (the `fidelity → synthesize` back-edge above). On exhaustion the lowest-WER attempt is kept and the chapter is flagged in the report; the run never aborts.

## Sub-graphs

* **`process` mode** (`--process-only`): `read → confirm → script_gen → report → END` — no TTS, no M4B.
* **`synthesize` mode** (`--synthesize-only`): `load_scripts → synthesize → fidelity → assemble → report → END` — scripts are loaded from a previous `--process-only` run; the cycle-3 retry edge applies here too.
