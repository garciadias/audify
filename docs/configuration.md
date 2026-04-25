# Configuration

Audify is configured through environment variables or a `.keys` file.

**Priority**: environment variables > `.keys` file > defaults.

## The `.keys` File

Create a `.keys` file in the project root (or your working directory):

```bash
cp .keys.example .keys
```

```ini
# TTS provider (kokoro, qwen, openai, aws, google)
TTS_PROVIDER=kokoro

# Kokoro TTS (local)
KOKORO_API_URL=http://localhost:8887/v1

# Qwen-TTS (local)
QWEN_API_URL=http://localhost:8890
QWEN_TTS_VOICE=Vivian

# OpenAI TTS
OPENAI_API_KEY=sk-your-key
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=coral

# AWS Polly
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
AWS_POLLY_VOICE=Joanna
AWS_POLLY_ENGINE=neural

# Google Cloud TTS
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_TTS_VOICE=en-US-Neural2-F
GOOGLE_TTS_LANGUAGE_CODE=en-US

# Ollama (local LLM)
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=magistral:24b
OLLAMA_TRANSLATION_MODEL=qwen3:30b

# Commercial LLM API keys
DEEPSEEK=sk-your-deepseek-key
ANTHROPIC=sk-ant-your-anthropic-key
OPENAI=sk-your-openai-key
GEMINI=your-google-api-key
```

:::{warning}
Never commit `.keys` to version control. It is already in `.gitignore`.
:::

## TTS Provider Comparison

| Provider         | Local? | Free? | GPU needed? | Key Features                   |
|------------------|--------|-------|-------------|-------------------------------|
| **Kokoro**       | Yes    | Yes   | Recommended | Fast, low-latency synthesis   |
| **Qwen-TTS**     | Yes    | Yes   | Recommended | Multilingual, high quality    |
| **OpenAI**       | No     | No    | No          | High quality, easy setup      |
| **AWS Polly**    | No     | No    | No          | Enterprise, multiple engines  |
| **Google Cloud** | No     | No    | No          | Multilingual, neural voices   |

## Google Cloud TTS Configuration

When using Google Cloud TTS, the default voice is set to **`en-US-Neural2-F`** (English US, Female).

To use different languages or voices, set the environment variables:

```ini
GOOGLE_TTS_VOICE=<voice-id>          # e.g., es-ES-Neural2-A for Spanish
GOOGLE_TTS_LANGUAGE_CODE=<lang-code> # e.g., es-ES for Spanish
```

:::{warning}
**Important:** The voice locale must match the language. If you specify a voice like `en-US-Chirp-HD-F`, you should set `GOOGLE_TTS_LANGUAGE_CODE=en-US`. Mismatched values will cause errors.
:::

## Docker Services

The `docker-compose.yml` provides local services:

| Service    | Port  | Description                           |
|------------|-------|---------------------------------------|
| Kokoro TTS | 8887  | GPU-accelerated speech synthesis       |
| Ollama     | 11434 | Local LLM for script/translation       |
| Audify API | 8000  | REST API (starts after dependencies)   |
| Qwen-TTS   | 8890  | Local Qwen-TTS wrapper (qwen profile)  |

```bash
docker compose up -d      # Start all services
docker compose ps         # Check status
docker compose logs -f    # Follow logs
docker compose down       # Stop all services
```

Enable Qwen-TTS service:

```bash
docker compose --profile qwen up -d qwen-tts
```

If Audify runs on host machine, use:

```ini
QWEN_API_URL=http://localhost:8890
```

If Audify runs inside the `api` compose service, use:

```ini
QWEN_API_URL=http://qwen-tts:8890
```

## Supported Languages

English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Hungarian, Korean, Japanese, Hindi.

Translation supports any language pair available in your configured LLM.
