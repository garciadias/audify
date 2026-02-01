# Using Commercial APIs with Audify

Audify now supports commercial LLM APIs in addition to local Ollama models. This allows you to use powerful cloud-based models like DeepSeek, Claude, GPT-4, and Gemini for generating audiobook scripts.

## Supported APIs

- **DeepSeek** - Fast and cost-effective API
- **Anthropic Claude** - High-quality reasoning and long context
- **OpenAI GPT** - Industry-standard models
- **Google Gemini** - Google's latest AI models

## Setup

### 1. Create a `.keys` File

Create a `.keys` file in the project root directory with your API keys:

```bash
cp .keys.example .keys
```

Edit the `.keys` file and add your API keys:

```
DEEPSEEK=sk-your-deepseek-api-key-here
ANTHROPIC=sk-ant-your-anthropic-api-key-here
OPENAI=sk-your-openai-api-key-here
GEMINI=your-google-api-key-here
# Or use GOOGLE (both names are equivalent):
# GOOGLE=your-google-api-key-here
```

**Note:** For Google Gemini, you can use either `GEMINI` or `GOOGLE` as the key name in your `.keys` file. Both are supported and equivalent. Similarly, when setting environment variables, both `GEMINI_API_KEY` and `GOOGLE_API_KEY` will work.

**Important Security Notes:**

- The `.keys` file is already in `.gitignore` to prevent accidental commits
- Never commit API keys to version control
- Keep your API keys secure and private
- You can also set API keys as environment variables (e.g., `DEEPSEEK_API_KEY`)

### 2. Get API Keys

#### DeepSeek

1. Visit [https://platform.deepseek.com/](https://platform.deepseek.com/)
2. Sign up for an account
3. Generate an API key from your dashboard
4. DeepSeek offers competitive pricing and good performance

#### Anthropic Claude

1. Visit [https://console.anthropic.com/](https://console.anthropic.com/)
2. Create an account
3. Generate an API key
4. Claude excels at reasoning and has large context windows

#### OpenAI

1. Visit [https://platform.openai.com/](https://platform.openai.com/)
2. Sign up and add billing information
3. Generate an API key from the API section
4. Models include GPT-4, GPT-4-turbo, GPT-3.5-turbo

#### Google Gemini

1. Visit [https://ai.google.dev/](https://ai.google.dev/)
2. Get started with Gemini API
3. Create an API key
4. Gemini offers various models including Gemini Pro

## Usage

To use a commercial API, prefix the model name with `api:` when using the `-m` or `--llm-model` option:

### DeepSeek Examples

```bash
# Using DeepSeek Chat
python -m audify.create_audiobook mybook.epub -m "api:deepseek/deepseek-chat"

# Using DeepSeek R1 (reasoning model)
python -m audify.create_audiobook mybook.epub -m "api:deepseek/deepseek-reasoner"
```

### Claude Examples

```bash
# Using Claude 3 Sonnet
python -m audify.create_audiobook mybook.epub -m "api:anthropic/claude-3-sonnet-20240229"

# Using Claude 3.5 Sonnet (latest)
python -m audify.create_audiobook mybook.epub -m "api:anthropic/claude-3-5-sonnet-20240620"

# Using Claude 3 Opus (most capable)
python -m audify.create_audiobook mybook.epub -m "api:anthropic/claude-3-opus-20240229"
```

### OpenAI Examples

```bash
# Using GPT-4
python -m audify.create_audiobook mybook.epub -m "api:openai/gpt-4"

# Using GPT-4 Turbo
python -m audify.create_audiobook mybook.epub -m "api:openai/gpt-4-turbo-preview"

# Using GPT-3.5 Turbo (faster, cheaper)
python -m audify.create_audiobook mybook.epub -m "api:openai/gpt-3.5-turbo"
```

### Google Gemini Examples

```bash
# Using Gemini Pro
python -m audify.create_audiobook mybook.epub -m "api:gemini/gemini-pro"

# Using Gemini 1.5 Pro
python -m audify.create_audiobook mybook.epub -m "api:gemini/gemini-1.5-pro"
```

## Complete Example

```bash
# Create audiobook using DeepSeek with Spanish translation
python -m audify.create_audiobook mybook.epub \
  -m "api:deepseek/deepseek-chat" \
  -l en \
  -t es \
  -v af_bella \
  --save-scripts

# Create audiobook from PDF using Claude
python -m audify.create_audiobook document.pdf \
  -m "api:anthropic/claude-3-sonnet-20240229" \
  -l en \
  -v af_sarah

# Process directory of books using GPT-4
python -m audify.create_audiobook ./books/ \
  -m "api:openai/gpt-4-turbo-preview" \
  -l en \
  -o ./output
```

## Comparing Ollama vs Commercial APIs

| Feature | Ollama (Local) | Commercial APIs |
|---------|---------------|-----------------|
| **Cost** | Free | Pay per token |
| **Privacy** | Complete privacy | Data sent to cloud |
| **Speed** | Depends on hardware | Generally faster |
| **Quality** | Varies by model | State-of-the-art |
| **Internet** | Not required | Required |
| **Setup** | Install Ollama + models | Just API key |

## Model Recommendations

### For Best Quality

- Claude 3 Opus: Best reasoning and quality
- GPT-4: Reliable and well-tested
- Claude 3.5 Sonnet: Great balance of quality and speed

### For Best Value

- DeepSeek Chat: Very cost-effective
- GPT-3.5 Turbo: Fast and affordable
- Gemini Pro: Good balance

### For Reasoning Tasks

- DeepSeek R1: Specialized reasoning model
- Claude 3 Opus: Excellent reasoning
- GPT-4: Strong reasoning capabilities

## Troubleshooting

### API Key Not Found

```
Error: API key issue. Please ensure your API key is properly configured
```

**Solution:** Check that:

1. The `.keys` file exists in the project root
2. The API key is properly formatted (e.g., `DEEPSEEK=sk-...`)
3. No extra spaces around the `=` sign
4. The key name matches the service (DEEPSEEK, ANTHROPIC, OPENAI, GEMINI)

### Connection Error

```
Error: Could not connect to commercial API
```

**Solution:**

1. Check your internet connection
2. Verify the API key is valid
3. Check if the API service is experiencing downtime
4. Ensure you have API credits/billing enabled

### Invalid Model Name

```
Error: Failed to generate audiobook script
```

**Solution:**

1. Verify the model name is correct (check API documentation)
2. Ensure you're using the `api:` prefix
3. Some models may require specific API access tiers

## Environment Variables (Alternative to .keys file)

Instead of using a `.keys` file, you can set environment variables:

```bash
# Bash/Zsh
export DEEPSEEK_API_KEY="sk-your-key"
export ANTHROPIC_API_KEY="sk-ant-your-key"
export OPENAI_API_KEY="sk-your-key"
# For Google Gemini, use either GOOGLE_API_KEY or GEMINI_API_KEY (both work):
export GOOGLE_API_KEY="your-key"
# export GEMINI_API_KEY="your-key"  # Alternative, equivalent to above

# Then run normally
python -m audify.create_audiobook mybook.epub -m "api:deepseek/deepseek-chat"
```

## Cost Considerations

Commercial APIs charge based on token usage (input + output). Audiobook generation can use significant tokens due to:

- Processing entire chapters
- Generating narrative scripts
- Multiple chapters per book

**Tips to minimize costs:**

1. Use more cost-effective models like DeepSeek or GPT-3.5
2. Limit `--max-chapters` for testing
3. Monitor your API usage on the provider's dashboard
4. Consider using Ollama for testing, then commercial APIs for final output

## Technical Details

The implementation uses [LiteLLM](https://github.com/BerriAI/litellm) to provide a unified interface across different API providers. This means:

- Consistent API regardless of provider
- Easy to add new providers
- Automatic retry and error handling
- Unified token counting

Model names are passed directly to LiteLLM, so any model supported by LiteLLM can be used with the `api:` prefix.
