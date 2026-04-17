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
audify run book.epub --tts-provider openai
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
audify run book.epub
```

## Option 3: Qwen-TTS (Local, Free)

Requires a local Qwen-TTS-compatible API and typically a GPU.

### Option 3A: Docker Compose Profile (Recommended)

Run the provided qwen-tts service in this repository:

1. Start Qwen-TTS and Ollama:

   ```bash
   docker compose --profile qwen up -d qwen-tts ollama
   ```

2. Confirm Qwen-TTS health:

   ```bash
   curl http://localhost:8890/health
   ```

3. Use Qwen in Audify:

   ```bash
   # Direct TTS
   audify book.epub --task direct --tts-provider qwen

   # Audiobook generation with Ollama LLM + Qwen-TTS
   audify book.epub --task audiobook --tts-provider qwen -m gemma4:31b
   ```

### Option 3B: Local API Wrapper Script

1. Install dependencies:

   ```bash
   pip install qwen-tts fastapi uvicorn soundfile numpy torch
   ```

2. Run the wrapper:

   ```bash
   python scripts/qwen_tts_api.py
   ```

3. Configure Audify:

   ```bash
   TTS_PROVIDER=qwen
   QWEN_API_URL=http://localhost:8890
   QWEN_TTS_VOICE=Vivian
   ```

### Task aliases (equivalent commands)

```bash
task --tts-provider qwen run "book.epub"
task --tts-provider qwen --llm-model gemma4:31b audiobook "book.epub"
```

## Next Steps

- Learn about all [usage options](usage.md) including translation
- Explore the [task system](tasks.md) for LLM-powered audio generation
- Set up [cloud LLM APIs](commercial-apis.md) for advanced audiobook creation
