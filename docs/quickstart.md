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

Requires GPU. Offers multilingual support and high-quality synthesis. Three approaches:

### Option 3A: DashScope Cloud API (Simplest)
**Note**: Cloud API may incur costs.

1. Get an API key from [DashScope](https://help.aliyun.com/zh/model-studio/qwen-tts-realtime)
2. Add to your `.keys` file:
   ```bash
   DASHSCOPE_API_KEY=your-api-key
   TTS_PROVIDER=qwen
   # Note: Cloud API uses different endpoints, not yet fully integrated
   # Currently only local API server is supported
   ```

### Option 3B: Local API Server (Recommended for programmatic use)
Run a local FastAPI server that wraps the Qwen3-TTS model:

1. Install dependencies:
   ```bash
   pip install qwen-tts fastapi uvicorn soundfile numpy
   ```

2. Download the API server script:
   ```bash
   curl -o qwen_tts_api.py https://raw.githubusercontent.com/garciadias/audify/main/scripts/qwen_tts_api.py
   ```

3. Run the server:
   ```bash
   python qwen_tts_api.py
   # API available at http://localhost:8890
   ```

4. Configure Audify in `.keys`:
   ```bash
   TTS_PROVIDER=qwen
   QWEN_API_URL=http://localhost:8890
   QWEN_TTS_VOICE=Vivian
   ```

### Option 3C: Gradio Demo (For testing)
Interactive web interface for manual testing:

1. Install and run:
   ```bash
   pip install qwen-tts
   qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8890
   ```
2. Open browser to http://localhost:8890

**Note**: The Gradio demo doesn't provide the API endpoints Audify needs. Use Option 3B for programmatic access.

### Convert a book
Once your Qwen-TTS server is running (Option 3B):

```bash
audify run book.epub --tts-provider qwen
```

## Next Steps

- Learn about all [usage options](usage.md) including translation
- Explore the [task system](tasks.md) for LLM-powered audio generation
- Set up [cloud LLM APIs](commercial-apis.md) for advanced audiobook creation
