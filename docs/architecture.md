# Architecture

## Overview

```text
                        CLI / REST API
                             |
                    +--------v--------+
                    |   Task System   |
                    | (PromptManager, |
                    |  TaskRegistry)  |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |   Direct TTS    |          |   LLM Pipeline  |
     | (EpubSynth,     |          | (AudiobookCreator|
     |  PdfSynth)      |          |  + LLMClient)   |
     +--------+--------+          +--------+--------+
              |                             |
              |                    +--------v--------+
              |                    |  LLM Providers  |
              |                    | Ollama | Cloud   |
              |                    +-----------------+
              |                             |
              +-------------+---------------+
                            |
                   +--------v--------+
                   |  TTS Providers  |
                   | Kokoro | OpenAI |
                   | AWS    | Google |
                   | Qwen             |
                   +--------+--------+
                            |
                   +--------v--------+
                   | Audio Assembly  |
                   | (M4B builder,  |
                   |  AudioProcessor)|
                   +-----------------+
```

## Pipeline

**CLI** -> **Creator** -> **Synthesizer** -> **Reader**

### Readers (`audify/readers/`)

Extract text from source files:

- `EpubReader` -- EPUB with chapter detection
- `PdfReader` -- PDF documents
- `TextReader` -- Plain text / Markdown

Abstract interface: `audify/domain/reader.py`

### Task System (`audify/prompts/`)

- `PromptManager` -- Loads built-in and custom prompts
- `TaskRegistry` -- Maps task names to `TaskConfig` (prompt + LLM params)
- Built-in prompts stored as text files in `audify/prompts/builtin/`

### AudiobookCreator (`audify/audiobook_creator.py`)

Orchestrates LLM script generation:

- `LLMClient` -- Routes to Ollama or commercial APIs (via LiteLLM)
  - `generate_script()` -- Accepts any prompt + LLM params
  - `generate_audiobook_script()` -- Backward-compatible wrapper
- `AudiobookCreator` -- Base class for LLM-powered audiobook creation
- `AudiobookEpubCreator` / `AudiobookPdfCreator` -- Format-specific subclasses
- `DirectoryAudiobookCreator` -- Multi-file processing

### Synthesizers (`audify/text_to_speech.py`)

Convert text to speech via TTS providers:

- `BaseSynthesizer` -- Manages TTS config, sentence splitting, WAV/MP3 conversion
- `EpubSynthesizer` / `PdfSynthesizer` -- Direct TTS (no LLM)
- `VoiceSamplesSynthesizer` -- Voice comparison tool

### Audio Processing (`audify/utils/`)

- `audio.py` -- Combines audio, format conversion
- `m4b_builder.py` -- FFmpeg metadata, chapter markers, M4B assembly

## External Services

Configured via `docker-compose.yml`:

- **Kokoro TTS** -- Port 8887 (GPU)
- **Ollama LLM** -- Port 11434 (GPU)

Commercial APIs (no Docker needed):

- LLM: DeepSeek, Claude, GPT-4, Gemini (via LiteLLM)
- TTS: OpenAI, AWS Polly, Google Cloud TTS
