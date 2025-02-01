# Audify

Convert ebooks to audiobooks using AI text-to-speech.

## Prerequisites

- Python 3.9-3.12
- UV package manager
- CUDA-capable GPU
- CUDA toolkit

## Quick Start

### Install UV if you haven't already

Follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)

### Create virtual environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Convert an ebook

```bash
task run path/to/book.epub
```

## Available Tasks

```bash
task test      # Run tests with coverage
task format    # Format code with black, isort and ruff
task run       # Convert ebook to audiobook
```

## Contributing

- [Contributing Guide](docs/CONTRIBUTING.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project Structure

```bash
audify/
├── audify/
│   ├── ebook_read.py    # Ebook processing
│   ├── text_to_speech.py # TTS conversion
│   └── start.py         # Entry point
├── tests/               # Test suite
└── data/               # Output directory
```
