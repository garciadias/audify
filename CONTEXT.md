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
- **Metadata check:** are chapter names, cover, ordering, and Book metadata
  correct and complete? Mixed: chapter-name correctness and empty Book
  metadata fields are cyclic (see infer-and-fill and Metadata enrichment). A
  cover still missing after enrichment is terminal.

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
- **Script:** the LLM-rewritten narration text for one source chapter
  (`scripts/episode_NNN_script.txt`). The output of `script_gen`, the input
  of `synthesize`, and the persisted artifact that lets `synthesize` mode
  pick up where `process` mode left off.
- **TTS batch:** ≤5000 chars of text synthesized into one WAV
  (`batch_N.wav`). The unit of the fidelity check and the retry edge.
- **LLM chunk:** ≤2500 words, a long chapter split to fit the LLM context
  window. Internal to script generation; not a QA unit.

## Graph modes

The acyclic skeleton expresses three canonical sub-graphs, each selected by a
CLI flag and dispatched by `build_graph(mode)`:

- **`full`** (default): `read → confirm → script_gen → synthesize → assemble → report`.
- **`process`** (`--process-only`): `read → confirm → script_gen → report`. Stops
  after Scripts are written to disk; no TTS, no M4B.
- **`synthesize`** (`--synthesize-only`): `load_scripts → synthesize → assemble → report`.
  Bypasses `read`/`script_gen` by loading previously-saved Scripts. The
  companion to `process`.

The TTS preflight and the user-confirm prompt are infrastructure, not Quality
Checks: preflight runs in `run_graph` before the graph for modes that do TTS;
the confirm prompt is a `confirm_node` between `read` and `script_gen`, and an
abort propagates as `{"chapters": []}` so downstream nodes naturally no-op
without conditional edges.

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

## Library delivery

- **Library item:** Audify's final deliverable. It is one self-contained folder
  that an audiobook server (Audiobookshelf) imports as a single book. It holds
  the M4B (or every part of it), the Source ebook, `cover.jpg`, and a metadata
  sidecar. The folder sits at `Author/[Series/]Title/`, so it can be dropped
  into the library root unchanged. One Library item is one book. The M4B and
  the ebook of the same book never live in separate folders, because the server
  would then list them as two books.
- **Source ebook:** the EPUB or PDF a run was made from. A Book rendition
  copies it into the Library item as the server's ebook. Its Book metadata
  comes from the EPUB's OPF, or from a PDF's info dictionary, which is often
  empty and leaves more to Metadata enrichment.
- **Metadata override:** a small per-book file of field values, kept in the
  Work directory, that a person writes deliberately. It wins over every other
  source. It is the place to correct a book: edits made in the audiobook
  server's UI are overwritten by the next Re-publish, because the server ranks
  `metadata.json` highest. The precedence is override > Source ebook >
  catalogue > LLM.
- **Book metadata:** the single record of a book's identity (title, subtitle,
  authors, narrators, series and position, genres, published date, publisher,
  description, ISBN, language). Every **metadata carrier** is generated from
  it, and none is edited by hand.
- **Metadata carrier:** a place where Book metadata is written for a reader to
  find. A Library item has two: the tags embedded in the M4B (read by any
  player) and the `metadata.json` sidecar (read by Audiobookshelf, and the one
  that wins there).
- **Metadata enrichment:** filling empty Book metadata fields. It is the
  escalation edge of **infer-and-fill** applied to book-level fields, in two
  rungs: (1) **catalogue lookup** in a public catalogue (Open Library first,
  then Google Books), matched by ISBN, or by title and author when there is no
  ISBN; (2) **LLM inference** for whatever is still empty (for example a
  description summarised from the text, or guessed genres). Policy is **fill
  gaps only**: a value present in a Metadata override or the Source ebook is
  never replaced, and rung 2 never replaces rung 1. Every filled field records
  its **provenance**: `override`, `ebook`, the catalogue's name, or `llm`. A
  missing cover is filled by rung 1 only (Open Library, by ISBN). A cover is
  never generated.
- **Text-derivable field:** a Book metadata field that the book's own text can
  support: description, genres, tags, and a subtitle shown on the title page.
  Only these may come from LLM inference. **Factual fields** (ISBN, published
  date, publisher, series and position, authors) come only from an override,
  the Source ebook or a catalogue, and otherwise stay empty. An empty factual
  field is correct; an invented one is a defect.
- **Book rendition:** a Library item that is the book itself, read whole. Only
  the `audiobook` and `direct` tasks produce one. It holds the Source ebook.
- **Derived work:** a Library item made from a book but not the book itself.
  The `podcast`, `summary`, `lecture` and `meditation` tasks produce one. It is
  its own Library item titled `Title (Task)`, for example `The Three-Body
  Problem (Podcast)`. It carries a tag naming the task and contains no ebook,
  so it never merges into the book's Library item.
- **Narrator:** the TTS engine and voice that spoke the audio, written as
  `Engine (voice)`, for example `Kokoro (af_bella)`. There is never a human
  narrator.
- **LLM-script tag:** the tag `llm-script`, set on a Book rendition from the
  `audiobook` task, whose audio speaks an LLM-rewritten Script rather than the
  verbatim text. A `direct` rendition carries no such tag because it is
  verbatim. The Narrator field alone says the voice is synthetic. The
  `abridged` flag stays false. _Avoid_: `ai-narrated` (it reads as "synthetic
  voice", which every rendition is).
- **Translated rendition:** a Book rendition whose audio is in a different
  language from its Source ebook (`--translate`). Its language field is the
  translation target. Its title stays the ebook's title (no machine-translated
  title is invented). It still holds the Source ebook, and it carries the tag
  `translated-from-<lang>`. It lives in its own folder, `Title [<lang>]`, so it
  never merges with an untranslated rendition of the same book.
- **Work directory:** where a run keeps its working files (Scripts, episodes,
  TTS batches, temporary M4B parts). It is never synced and never scanned by
  the audiobook server.
- **Library root:** the configured folder that the audiobook server scans
  (today the Syncthing folder mirrored to `/sdcard/Audiobooks`). It only ever
  receives complete Library items.
- **Publish:** the final step of a run. It assembles the Library item under a
  hidden temporary name inside the Library root, then renames it into place
  (ADR-0003).
- **Re-publish:** Publish into a Library item that already exists. It replaces
  the M4B, `metadata.json` and `cover.jpg`, keeps the Source ebook, and keeps
  every file Audify did not create. The folder path does not change, so the
  server keeps the same book and every listener's progress.
- **Item manifest:** `.audify.json` inside a Library item (the dot prefix keeps
  the server from scanning it). It records the item's identity, the folder path
  fixed at first Publish, and every file Audify wrote together with its hash.
  Re-publish uses it in two ways. It finds the existing item, so a later change
  of title or author never moves the folder; only `metadata.json` changes. It
  also detects hand edits: a file whose hash no longer matches was edited by
  hand, so it is kept and the skip is logged.
- **Book key:** the identity of a book across runs. It is the hash of the
  book's text: the EPUB spine (chapter content, excluding the OPF metadata and
  the cover), or the extracted text of a PDF. A metadata fix made in Calibre
  keeps the key. A different edition, with different text, gets a new key.
  _Avoid_: whole-file hash (any metadata edit changes it). See ADR-0004.
- **Item identity:** what Re-publish matches on. It is the Book key plus the
  rendition kind (task, and translation target language if any). So a book's
  Book rendition, its Translated rendition and its Derived works are distinct
  Library items that share one Book key.
- **Part:** one M4B file of a long book. A book over 14 hours is split into
  parts. Every part carries the same title tags and a track number `n/N`, so
  the server plays them in order. The book's one continuous **chapter list**
  (offsets across all parts) lives in `metadata.json`, not only in each part's
  embedded chapters.
- **Adopt:** Publish applied to an output made before Library items existed
  (`audify publish --adopt`). It needs no new TTS. It pairs the old M4B with
  its Source ebook by Book key, writes tags, `metadata.json` and the Item
  manifest, and moves the result into the `Author/Title` layout. The old folder
  is removed only after the new item is in place. The server treats the new
  folder as a new book, so listening progress on the old item does not carry
  over.

## Quality report schema

The terminal `report` node of every graph mode aggregates per-chapter
**flags** raised by cyclic detectors into a single end-of-run report, written
as `quality_report.json` (machine-readable) and `quality_report.txt`
(human-readable Rich-rendered table) next to the M4B output.

### Verdicts

Per chapter, the aggregator emits one of three verdicts:

- **`clean`** — no flag was raised against the chapter.
- **`flagged`** — at least one cycle fired and resolved within its retry
  budget (no `FlagEntry` has `exhausted=True`).
- **`unrecoverable`** — at least one cycle ran out of retries
  (`exhausted=True`). The best-effort artifact is kept on disk and the
  chapter is surfaced in the report; the run does not abort. Mirrors the
  warn-only halting philosophy of commit `0bffc34`.

### `FlagEntry`

Each entry in `flags[chapter_id]` records:

- **`cycle`** — which remediation cycle fired the flag. Allowed values:
  `cycle_1_escalation`, `cycle_2_reroute`, `cycle_3_retry`.
- **`reason`** — short, human-readable diagnostic, e.g. `"WER 0.42 exceeds
  0.3 threshold"` or `"text-extraction: mojibake"`.
- **`exhausted`** — `True` when the cycle's retry budget reached zero;
  `False` when the detector fired but the cycle resolved within budget.

### JSON shape

```json
{
  "pipeline_status": "complete" | "partial" | "failed" | "no_chapters",
  "episodes_synthesised": int,
  "chapters_expected": int,
  "verdict_counts": { "clean": int, "flagged": int, "unrecoverable": int },
  "chapters": {
    "chapter_<n>": {
      "title": str,
      "verdict": "clean" | "flagged" | "unrecoverable",
      "attempts_used": { "<cycle_id>": int },
      "best_wer": float | null,
      "flags": [FlagEntry, ...]
    }
  }
}
```

`attempts_used` only contains keys for cycles that actually fired against
the chapter — cycles that never ran are omitted rather than reported as
`0/3`. `best_wer` is populated only when `cycle_3_retry` ran; otherwise
`null`. With stub data (no cyclic detector has shipped yet) every chapter
is `clean` with an empty `flags` list.
