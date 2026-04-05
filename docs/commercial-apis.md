# Commercial APIs

Audify supports commercial LLM APIs for audiobook script generation and cloud TTS providers for speech synthesis.

## LLM API Setup

### 1. Create a `.keys` file

```bash
cp .keys.example .keys
```

### 2. Add your API keys

```ini
DEEPSEEK=sk-your-deepseek-api-key
ANTHROPIC=sk-ant-your-anthropic-api-key
OPENAI=sk-your-openai-api-key
GEMINI=your-google-api-key
```

### 3. Use with `api:` prefix

```bash
# DeepSeek
audify audiobook book.epub -m "api:deepseek/deepseek-chat"

# Claude
audify audiobook book.epub -m "api:anthropic/claude-3-5-sonnet-20240620"

# GPT-4
audify audiobook book.epub -m "api:openai/gpt-4-turbo-preview"

# Gemini
audify audiobook book.epub -m "api:gemini/gemini-1.5-pro"
```

## Supported LLM Providers

| Provider   | Model prefix                 | Key variable |
|------------|------------------------------|-------------|
| DeepSeek   | `api:deepseek/deepseek-chat` | `DEEPSEEK`  |
| Anthropic  | `api:anthropic/claude-*`     | `ANTHROPIC` |
| OpenAI     | `api:openai/gpt-*`          | `OPENAI`    |
| Google     | `api:gemini/gemini-*`       | `GEMINI`    |

All commercial APIs are routed through [LiteLLM](https://www.litellm.ai/) for a unified interface.

## Cloud TTS Providers

### OpenAI TTS

```ini
OPENAI_API_KEY=sk-your-key
OPENAI_TTS_MODEL=gpt-4o-mini-tts-2025-03-20
OPENAI_TTS_VOICE=coral
```

```bash
audify run book.epub --tts-provider openai
```

### AWS Polly

```ini
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
AWS_POLLY_VOICE=Joanna
AWS_POLLY_ENGINE=neural
```

```bash
audify run book.epub --tts-provider aws
```

### Google Cloud TTS

```ini
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_TTS_VOICE=en-US-Neural2-F
GOOGLE_TTS_LANGUAGE_CODE=en-US
```

```bash
audify run book.epub --tts-provider google
```

## Troubleshooting

**API key not found**: Ensure the key name in `.keys` matches exactly (e.g., `DEEPSEEK`, not `DEEPSEEK_API_KEY`).

**Connection errors**: Check your internet connection and verify the API endpoint is reachable.

**Rate limits**: Commercial APIs have rate limits. For large books, consider adding delays or using local Ollama.
