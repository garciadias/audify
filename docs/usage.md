# Usage

## Basic TTS Conversion

Convert ebooks directly to audio without LLM processing:

```bash
# EPUB to audiobook (direct TTS, no LLM)
audify book.epub --task direct

# PDF to audio
audify document.pdf --task direct

# Specify language (affects voice selection and TTS)
audify book.epub --task direct --language pt

# Translate content (text is translated before TTS)
audify book.epub --task direct --translate es

# Combined: source language with translation
audify book.epub --task direct --language en --translate es

# Choose TTS provider
audify book.epub --task direct --tts-provider openai
```

### Understanding --language and --translate

- **`--language`** (or `-l`): Sets the language for TTS voice selection and audio output. Default: `en` (English).
- **`--translate`** (or `-t`): Translates the extracted text to a target language before TTS synthesis.
  - Can be used alone when source language is autodetected; otherwise provide `--language` for explicit source language
  - Example: `--language en --translate es` means "extract English text, translate to Spanish, then synthesize Spanish speech"
  - Uses your configured LLM (local Ollama or commercial API) to perform translation

#### Translation Examples

```bash
# Translate an English book to Spanish audio
audify english-book.epub --task direct --language en --translate es

# Portuguese book to French audio
audify book.epub --task direct --language pt --translate fr

# Without source language specified, translation uses the autodetected language
audify mixed-language-book.epub --task direct --translate es
```

## LLM-Powered Audiobook Generation

Use an LLM to transform text into engaging audiobook scripts before TTS:

```bash
# Default audiobook style (--task audiobook is the default)
audify book.epub

# Limit chapters
audify book.epub --max-chapters 5

# Custom voice and language
audify book.epub --voice af_bella --language en

# With translation (LLM processes source text, then script is translated for synthesis)
audify book.epub --translate pt

# Full workflow: extract as English, LLM processes English text, translate script to Spanish, synthesize
audify book.epub --language en --translate es -m "api:deepseek/deepseek-chat"
```

:::{note}
When using `--translate` with audiobook generation, the LLM processes the extracted text in the source language to generate the script, then that script is translated to the target language before synthesis. This ensures the LLM has access to the original text for best results.
:::

### Using commercial LLM APIs

```bash
# DeepSeek (cost-effective)
audify book.epub -m "api:deepseek/deepseek-chat"

# Claude (high quality)
audify book.epub -m "api:anthropic/claude-3-5-sonnet-20240620"

# GPT-4
audify book.epub -m "api:openai/gpt-4-turbo-preview"

# Gemini
audify book.epub -m "api:gemini/gemini-1.5-pro"
```

See [Commercial APIs](commercial-apis.md) for API key setup.

## Task System

Use the `--task` flag to control how the LLM transforms your text:

```bash
# Audiobook style (default)
audify book.epub --task audiobook

# Podcast/lecture style
audify book.epub --task podcast

# Summary
audify book.epub --task summary

# Guided meditation
audify book.epub --task meditation

# Classroom lecture
audify book.epub --task lecture

# Custom prompt file
audify book.epub --prompt-file my-prompt.txt
```

See [Tasks](tasks.md) for details on creating custom tasks.

## Two-Stage Workflow (Process / Synthesize)

For large books or when iterating on scripts, you can split audiobook creation
into two stages:

```bash
# Stage 1: Extract text and generate LLM scripts (no TTS)
audify book.epub --process-only

# Review or edit generated scripts in audiobooks/[book_name]/scripts/

# Stage 2: Synthesise audio from saved scripts
audify book.epub --synthesize-only
```

This is useful when:

- You want to review or manually edit scripts before synthesising
- TTS service is temporarily unavailable
- You want to re-synthesise with a different voice or provider without re-running the LLM

The `--process-only` flag also saves a `chapter_titles.json` alongside the scripts so that `--synthesize-only` can reconstruct chapter metadata.

### Resumability

If a run is interrupted, re-running the same command will skip episodes whose
MP3 files already exist. This applies to both full runs and the two-stage
workflow.

## Directory Processing

Process multiple files into a single audiobook:

```bash
# All supported files in a directory
audify path/to/articles/

# With translation
audify path/to/articles/ --translate es
```

**Supported file types**: EPUB, PDF, TXT, MD

Each file becomes a separate episode with a synthesized title, and all episodes are combined into a single M4B audiobook with chapter markers.

## Listing Available Options

```bash
# List available tasks
audify list-tasks

# Validate a custom prompt file
audify validate-prompt my-prompt.txt

# List languages
audify --list-languages

# List TTS voices
audify --list-voices

# List TTS models
audify --list-models

# List TTS providers
audify --list-tts-providers
```

## Output Structure

### Basic TTS (`audify book.epub --task direct`)

```text
data/output/[book_name]/
    chapters.txt           # Metadata
    cover.jpg              # Cover image (EPUB)
    chapters_001.mp3       # Chapter audio files
    chapters_002.mp3
    book_name.m4b          # Final audiobook
```

### LLM Audiobook (`audify book.epub`)

```text
audiobooks/[book_name]/
    episodes/
        episode_001.mp3
        episode_002.mp3
    scripts/
        episode_001_script.txt
        original_text_001.txt
        chapter_titles.json
    chapters.txt
    book_name.m4b
```
