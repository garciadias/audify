# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Audify

Audify converts ebooks (EPUB, PDF, TXT) into audiobooks using Kokoro TTS and LLM-powered script generation (local Ollama or commercial APIs via LiteLLM).

## Common Commands

All task commands are defined in `pyproject.toml` under `[tool.taskipy.tasks]` and run via `task`:

```bash
task test          # format + mypy + pytest with coverage + html report
task format        # ruff check ./audify ./tests --fix
task run book.epub # basic TTS conversion (python -m audify.start)
task audiobook book.epub  # LLM-powered audiobook (python -m audify.create_audiobook)
task up            # docker compose up
```

Run a single test file:

```bash
uv run pytest tests/test_audiobook_creator.py
```

Type checking alone:

```bash
mypy ./audify ./tests --ignore-missing-imports
```

## Architecture

**Pipeline:** CLI → Creator → Synthesizer → Reader

- **Readers** (`audify/readers/`): Extract text from EPUB, PDF, TXT via abstract `Reader` interface in `audify/domain/reader.py`
- **AudiobookCreator** (`audify/audiobook_creator.py`): Orchestrates LLM script generation. `LLMClient` routes to Ollama or commercial APIs based on `api:` model prefix (e.g., `api:deepseek/deepseek-chat`). Subclasses: `AudiobookEpubCreator`, `AudiobookPdfCreator`, `DirectoryAudiobookCreator`
- **Synthesizers** (`audify/text_to_speech.py`): Convert text to speech via Kokoro API. Base class `BaseSynthesizer` with format-specific subclasses
- **Audio processing** (`audify/utils/audio.py`): Combines episodes, adds chapter markers, embeds covers, produces M4B

**External services** (via `docker-compose.yml`):

- Kokoro TTS API on port 8887 (GPU)
- Ollama LLM on port 11434 (GPU)

**Commercial API support**: Keys loaded from `.keys` file by `audify/utils/api_keys.py`, configured through `CommercialAPIConfig` in `audify/utils/api_config.py`, routed via LiteLLM.

## Code Quality

- Linter: ruff (rules: I, F, E, W, PT)
- Type checker: mypy
- Python: 3.10–3.13
- Package manager: uv
- CI: GitHub Actions runs `uv run task test` on Python 3.11

## Modifying code

When modifying code, ensure to:

1. Follow existing code style and conventions
2. Add or update tests in `tests/` to cover new functionality
3. Run `task format` and `task test` to ensure code quality
4. Update documentation in `README.md` or `docs/` as needed
5. Commit changes with clear messages referencing related issues or features, NEVER mentioning to claude or AI.
