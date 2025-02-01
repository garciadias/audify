# Contributing Guide

## Development Setup

### Install development dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync --group dev
```

### Install pre-commit hooks:

```bash
pre-commit install
```

## Development Workflow

### Create new branch:

```bash
git checkout -b feature/your-feature
```

### Run tests and linting:

```bash
task test      # Runs pytest with coverage
task format    # Formats code
```

## Task Commands

All commands are defined in pyproject.toml and can be run using `task`:

```bash
task test      # Run tests
task format    # Format code
task run       # Run main application
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests for new features

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Submit PR with clear description
4. Wait for review
