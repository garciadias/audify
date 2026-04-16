# Installation

## Requirements

- **Python 3.10--3.13**
- **UV package manager** ([installation guide](https://docs.astral.sh/uv/getting-started/installation/)) or pip

### Optional

- **Docker & Docker Compose** -- for local Kokoro TTS
- **CUDA GPU** -- recommended for local TTS providers
- **API keys** -- for cloud TTS/LLM providers

## Install from PyPI

The simplest way to install Audify:

```bash
pip install audify-cli
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install audify-cli
```

This installs the `audify` CLI command with all subcommands.

## Install from Source

For development or to get the latest features:

```bash
git clone https://github.com/garciadias/audify.git
cd audify
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

### Development dependencies

```bash
uv sync --group dev
```

## Verify Installation

```bash
audify --version
audify --help
```

## Docker Setup (Optional)

Docker is only needed for the local **Kokoro TTS** provider:

```bash
cd audify
docker compose up -d

# Wait for services (~2-3 minutes), then check:
docker compose ps
```

This starts:

| Service      | Port  | Description                    |
|--------------|-------|--------------------------------|
| Kokoro TTS   | 8887  | GPU-accelerated speech synthesis |
| Ollama       | 11434 | Local LLM for script generation  |
| Audify API   | 8000  | REST API server                  |

### Pull an Ollama model

```bash
docker compose exec ollama ollama pull qwen3:30b
# Or for testing:
docker compose exec ollama ollama pull llama3.2:3b
```
