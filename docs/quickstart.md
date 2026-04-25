# Quick Start

Choose a quick-start path based on your TTS provider preference.

## Option 1: Cloud TTS (Easiest)

No Docker required. Use OpenAI, AWS Polly, or Google Cloud TTS.

### 1. Install Audify

```bash
pip install audify-cli
```

### 2. Configure credentials

Create a `.keys` file in your working directory:

```bash
# OpenAI TTS (simplest cloud option)
OPENAI_API_KEY=sk-your-openai-api-key
TTS_PROVIDER=openai
```

### 3. Convert a book

```bash
audify book.epub --task direct --tts-provider openai
```

## Option 2: Kokoro TTS (Local, Free)

Requires Docker and a CUDA GPU. Fast and low-latency synthesis.

### 1. Clone and start services

```bash
git clone https://github.com/garciadias/audify.git
cd audify
docker compose up -d
# Wait ~2-3 minutes for services to start
```

### 2. Pull an Ollama model

```bash
docker compose exec ollama ollama pull qwen3:30b
```

### 3. Convert a book

```bash
uv sync
audify book.epub --task direct
```

## Next Steps

- Learn about all [usage options](usage.md) including translation
- Explore the [task system](tasks.md) for LLM-powered audio generation
- Set up [cloud LLM APIs](commercial-apis.md) for advanced audiobook creation
