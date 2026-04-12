# Contributing

We welcome contributions! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/garciadias/audify.git
cd audify
uv venv
source .venv/bin/activate
uv sync --group dev
```

## Development Workflow

### 1. Create a branch

```bash
git checkout -b feature/your-feature
```

### 2. Make changes

Follow existing code style and conventions. Add type hints and docstrings.

### 3. Run checks

```bash
task format    # Lint and auto-fix with ruff
task test      # Run pytest with coverage + mypy
```

### 4. Submit a PR

- Ensure all tests pass
- Update documentation if needed
- Write a clear PR description

## Task Commands

Defined in `pyproject.toml`:

```bash
task test      # Format + mypy + pytest with coverage
task format    # ruff check --fix
task run       # Basic TTS conversion
task audiobook # LLM-powered audiobook
task api       # Start REST API (dev mode)
task up        # docker compose up
```

Run a single test file:

```bash
uv run pytest tests/test_audiobook_creator.py
```

Type checking alone:

```bash
mypy ./audify ./tests --ignore-missing-imports
```

## Code Quality

- **Linter**: ruff (rules: I, F, E, W, PT)
- **Type checker**: mypy
- **Python**: 3.10--3.13
- **Package manager**: uv
- **CI**: GitHub Actions runs `uv run task test` on Python 3.11

## Project Structure

```text
audify/
    cli.py                  # CLI entry point
    start.py                # Basic TTS command
    create_audiobook.py     # Audiobook command
    convert.py              # Unified convert command
    audiobook_creator.py    # LLMClient + AudiobookCreator
    text_to_speech.py       # TTS synthesizers
    prompts/                # Task/prompt system
    readers/                # EPUB, PDF, text readers
    utils/                  # Audio, config, text utilities
    api/                    # REST API
tests/
    test_*.py               # Unit and integration tests
    fixtures/               # Test fixtures
docs/                       # Sphinx documentation
```
