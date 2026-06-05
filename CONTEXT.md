# Audify — Context Glossary

Domain language for the agentic Quality-Assurance pipeline. Glossary only — no
implementation detail. Terms are canonical names; use them in code and docs.

## Pipeline shift

- **Linear pipeline (current):** CLI → Creator → Synthesizer → Reader, executed
  once per chapter as a Directed Acyclic Graph (DAG). No node ever re-runs.
- **Agentic pipeline (target):** the same nodes plus a **Critic** that can close
  **back-edges**, turning the DAG into a cyclic graph. A back-edge re-runs an
  earlier node when a quality check fails.

## Quality checks

A **Quality Check** is a verdict on a produced artifact. Each check is either
**cyclic** (failure triggers automated remediation that feeds back) or
**terminal** (failure is logged/reported; a human decides).

- **Fidelity check:** does the produced *audio* faithfully carry the **script**
  it was synthesized from? Detected by an STT round-trip (transcribe audio,
  WER against the script). Scoped to **TTS truncation/skip** — the residual TTS
  failure that yields audio rather than a 404. Mispronunciation is explicitly
  out of scope: STT normalizes pronunciation, so a text-vs-text comparison
  cannot see it. Cyclic (retry edge → re-chunk smaller).
- **Text-quality check:** is the text extracted from EPUB/PDF coherent
  (non-empty, not mojibake, mostly real words)? Runs post-reader, before any
  LLM/TTS spend. Cyclic (escalation edge → alternate parser / OCR).
- **Script-validity check:** is the LLM script a real narration, not a summary,
  refusal, or error message? Part of the tier-(c) LLM-judge. Cyclic (reroute
  edge → regenerate). Also asserts the script honored the summarize/remove
  policy (e.g. did not read out a raw code block).
- **Coverage check:** was every source chapter actually produced? A set
  difference between source chapters and produced episodes — no audio analysis.
  Cyclic. (Today's structural verifier in `verify.py` is the seed of this.)
- **Metadata check:** are chapter names / cover / ordering correct? Mixed: name
  correctness is cyclic (see infer-and-fill); cover presence is terminal.

## Where errors are actually born

Operational reality (per project owner): failures are overwhelmingly in **text
processing**, not in TTS. BeautifulSoup mis-extracts EPUB text; PDFs decode to
mojibake; the LLM returns a summary, a refusal, or an error message. TTS itself
either voices its input faithfully or fails hard (HTTP 404). The one TTS failure
that still produces *audio* is **truncation/skip on over-long input**.

Consequence: the cyclic remediation lives **upstream of TTS**, where the errors
are born and can be caught cheaply at the text layer — *before* spending TTS
compute. The STT round-trip is kept, but scoped to the single residual TTS
failure it can uniquely see (truncation/skip).

## Remediation patterns (back-edge topologies)

Three structurally distinct cycles, each with a different back-edge target:

- **Escalation edge:** *replace* a failed deterministic node with a more capable
  one. Used when text extraction produces garbage — re-running the same
  deterministic parser would fail identically, so the loop escalates to an
  alternate parser / OCR. (Also covers infer-and-fill for metadata.)
- **Reroute edge:** re-run an *upstream* node. Used when the LLM-judge finds the
  script is a summary / refusal / error message — regenerate the script. The
  error is born at script-generation, not at TTS, so retrying TTS is meaningless.
- **Retry edge:** re-run the *same* node with a changed input. Used by the STT
  loop against TTS truncation/skip — the only surviving adjustment is
  **re-chunk smaller** (perturbing speed or swapping voice were rejected: they
  don't address the real failure and harm voice consistency).

A plain **transient retry** on TTS HTTP 404/5xx is ordinary plumbing (a retry
decorator), not one of the three remediation topologies above.

## Units (disambiguating the overloaded word "chunk")

- **Episode:** the audio artifact for one source chapter (`episode_NNN.mp3`).
  The unit of the coverage check.
- **TTS batch:** ≤5000 chars of text synthesized into one WAV
  (`batch_N.wav`). The unit of the fidelity check and the retry edge.
- **LLM chunk:** ≤2500 words, a long chapter split to fit the LLM context
  window. Internal to script generation; not a QA unit.

## Agents

- **Critic:** the set of Quality Checks plus the routing logic that decides
  which back-edge (if any) to fire. The Critic is **the graph itself**, not a
  central LLM — autonomy lives in the topology (see ADR-0001). LLMs appear only
  at judgment nodes (script-validity judge, metadata infer-and-fill).
- **Boundary-sampling:** the fidelity check's detection strategy — transcribe
  only a head window and a tail window of the episode audio and fuzzy-match each
  against the script's opening/closing sentences. Cheap, and aimed at
  context-window **truncation** (which drops the tail). Blind to mid-chapter
  drops; corroborated by the duration-ratio signal. Replaces full-episode STT.
  Backed by **faster-whisper `large`** running locally alongside Kokoro/Ollama
  in `docker-compose.yml`. Bounded cost: two short windows per episode.
- **Graph state:** the shared object LangGraph carries between nodes and across
  back-edges — per-chapter retry budget, best-so-far WER, and the flags that
  feed the final quality report.
