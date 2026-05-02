# Draft: Audiobook Conversion Plans

## Requirements (confirmed)

- **Project**: Audify - ebook to audiobook converter
- **Source formats**: EPUB, PDF, TXT
- **Output formats**: MP3 + M4B (confirmed)
- **Test strategy**: Setup TDD + Tests for pytest fixtures, mocking patterns
- **Agent-Executed QA**: ALWAYS (mandatory for all tasks regardless of test choice)

## Technical Decisions

### LLM Routing Strategy

- **Keep dual-mode approach**: Ollama (local) + Commercial APIs (LiteLLM)
- **Routing**: `llm_client.py` routes based on model prefix (`ollama:xxx`, `api:xxx`)
- **Auth**: Commercial API keys from `.keys` file via `audify/utils/api_keys.py`

## Test Infrastructure (assessed)

- **Framework**: pytest (via `uv run task test`)
- **Tools**: ruff, mypy, pytest-cov, pytest-html
- **Setup**: TDD-style with fixtures and mocking
- **Test commands**:
  - `task test` → format + mypy + pytest + coverage + html report
  - `task format` → ruff check --fix
  - `task run book.epub` → basic TTS conversion

## Test Evidence from Codebase

### From `audify/llm_client.py`

- Routes to Ollama: `OLLAMA_BASE_URL = "http://localhost:11434"`
- Commercial APIs: Uses LiteLLM, keys from `.keys` file
- Model prefix detection: `OLLAMA_MODEL_PREFIX = "ollama:"`

### From `audify/text_to_speech.py`

- Kokoro TTS integration on port 8887
- Supports MP3 and M4B formats (confirmed)

### Architecture

- **Pipeline**: CLI → `AudiobookCreator` → `Synthesizer` → `Reader`
- **Readers** (`audify/readers/`): EPUB, PDF, TXT extraction
- **AudiobookCreator** (`audify/audiobook_creator.py`): LLM script orchestration
- **Synthesizers** (`audify/text_to_speech.py`): Text-to-speech conversion
- **Audio processing** (`audify/utils/audio.py`): Combine episodes, add chapters, produce M4B

## Open Questions

1. **Error handling**: What if Kokoro API is unavailable - queue, retry, or skip?
2. **Error recovery**: What failure modes need handling (LLM timeout, API rate limits, etc.)?
3. **Model selection**: Which models should be available in Ollama?
4. **Voice characteristics**: Any specific voice requirements for different content types?

## Scope Boundaries

### IN SCOPE

- EPUB/PDF/TXT ebook to audiobook conversion
- LLM-powered script generation
- Kokoro TTS integration
- MP3 and M4B output formats
- TDD-style test infrastructure
- Agent-Executed QA scenarios

### OUT OF SCOPE

- Changing source file locations (unless necessary for feature)
- Docker orchestration (existing compose file)
- UI/dashboard features
- User authentication

## Research Findings Summary

### Codebase Structure

- 3 reader implementations following `Reader` interface
- LLM client with dual-mode routing (Ollama + Commercial APIs)
- Multiple synthesizer subclasses for different TTS formats
- Audio processing utilities for final assembly

### Testing Patterns

- Existing pytest structure in `tests/` directory
- Uses mocking, fixtures for external services (LLM, TTS APIs)
- CI runs full test suite with coverage

## Questions for User

> **Question**: Before generating the plan, I need clarity on:
>
> 1. **Error handling**: What if Kokoro API is unavailable - queue, retry, or skip?
> 2. **LLM features**: Any specific features to leverage (context window, tool calling for complex scripts)?
> 3. **Output requirements**: Any specific chapter markers, duration limits, or metadata requirements?

Once I have these answers, I'll generate the complete plan immediately.

---

*This draft serves as working memory during the planning session. Delete after plan generation.*
