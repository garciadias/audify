# QA Pipeline Graph

Topology of the LangGraph QA pipeline in `full` mode (`read → script_gen → synthesize → assemble → report`).

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	read(read)
	confirm(confirm)
	script_gen(script_gen)
	synthesize(synthesize)
	assemble(assemble)
	report(report)
	__end__([<p>__end__</p>]):::last
	__start__ --> read;
	assemble --> report;
	confirm --> script_gen;
	read --> confirm;
	script_gen --> synthesize;
	synthesize --> assemble;
	report --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## Sub-graphs

* **`process` mode** (`--process-only`): `read → confirm → script_gen → report → END` — no TTS, no M4B.
* **`synthesize` mode** (`--synthesize-only`): `load_scripts → synthesize → assemble → report → END` — scripts are loaded from a previous `--process-only` run.
