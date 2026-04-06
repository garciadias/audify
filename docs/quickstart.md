# Quick Start

Choose a quick-start path based on your TTS provider preference.

## Option 1: Cloud TTS (Easiest)

No Docker required. Use OpenAI, AWS Polly, or Google Cloud TTS.

### 1. Install Audify

```bash
pip install audify
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
audify run book.epub --tts-provider openai
```

## Option 2: Kokoro TTS (Local, Free)

Requires Docker and a CUDA GPU.

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
audify run book.epub
```

## Option 3: Qwen-TTS (Local, Free, Multilingual)

Requires GPU and the Qwen-TTS API server.

### 1. Start Qwen-TTS

```bash
git clone https://github.com/QwenLM/Qwen3-TTS
cd Qwen3-TTS
make up
# API available at http://localhost:8890
```

### 2. Install and configure Audify

```bash
pip install audify
```

Create a `.keys` file:

```bash
TTS_PROVIDER=qwen
QWEN_API_URL=http://localhost:8890
QWEN_TTS_VOICE=Vivian
```

### 3. Convert a book

```bash
audify run book.epub --tts-provider qwen
```

## Next Steps

- Learn about all [usage options](usage.md)
- Explore the [task system](tasks.md) for LLM-powered audio
- Set up [cloud LLM APIs](commercial-apis.md) for audiobook generation
