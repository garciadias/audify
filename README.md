# Audify

[![codecov](https://codecov.io/github/garciadias/audify/branch/main/graph/badge.svg)](https://codecov.io/github/garciadias/audify)
[![Tests](https://github.com/garciadias/audify/workflows/Run%20Tests/badge.svg)](https://github.com/garciadias/audify/actions)

Convert ebooks and PDFs to audiobooks using AI text-to-speech and translation services.

Audify is a pipeline and REST API that transforms written content into high-quality audio using:

- **Multiple TTS Providers** - Choose from Kokoro (local), Qwen-TTS (local), OpenAI, AWS Polly, or Google Cloud TTS
- **Ollama + LiteLLM** for intelligent translation
- **LLM-powered audiobook generation** for engaging audio content

## 🚀 Features

- **📚 Multiple Formats**: Convert EPUB ebooks, PDF documents, TXT, and MD files
- **📁 Directory Processing**: Create audiobooks from multiple files in a directory
- **🎙️ Audiobook Creation**: Generate audiobook-style content from books using LLM
- **🌐 REST API**: HTTP API for programmatic synthesis and audiobook creation
- **🔒 Multiple TTS Providers**: Choose from Kokoro (local), Qwen-TTS (local), OpenAI, AWS Polly, or Google Cloud TTS
- **🌍 Multi-language Support**: Translate content
- **🎵 High-Quality TTS**: Natural-sounding speech with multiple provider options
- **⚙️ Flexible Configuration**: Environment-based settings and `.keys` file support

## 📋 Prerequisites

### Core Requirements

- **Python 3.10-3.13**
- **UV package manager** ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### For Local TTS Providers (Optional)

#### Kokoro TTS

- **Docker & Docker Compose** (for API services)
- **CUDA-capable GPU** (recommended for optimal performance)

#### Qwen-TTS

- **Qwen-TTS API Server** running on port 8890 (see [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS))
- **CUDA-capable GPU** (recommended for optimal performance)

### For Cloud TTS Providers (Optional)

- **OpenAI TTS**: OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- **AWS Polly**: AWS account with access keys ([AWS setup](https://aws.amazon.com/polly/))
- **Google Cloud TTS**: Google Cloud project with credentials ([GCP setup](https://cloud.google.com/text-to-speech))
## 📦 Installation as a command-line tool

You can install Audify as a standalone command-line tool using pip or uv:

```bash
pip install audify
```

Or using uv (recommended):

```bash
uv pip install audify
```

This will install the `audify` command with subcommands:
- `audify run`: Basic TTS conversion of EPUB/PDF files
- `audify audiobook`: LLM-powered audiobook generation

Alternatively, you can use the direct commands:
- `audify-run`: Alias for `audify run`
- `audify-audiobook`: Alias for `audify audiobook`

After installation, you can run `audify --help` to see available options.

## 🐳 Quick Start with Docker (For Kokoro TTS)

> **Note**: Docker is only required if you want to use the local Kokoro TTS provider. For Qwen-TTS, you'll need to run the Qwen-TTS API separately (see Qwen-TTS Setup below). You can skip to "Quick Start with Cloud TTS" if you prefer using OpenAI, AWS Polly, or Google Cloud TTS.

### 1. Clone and Setup

```bash
git clone https://github.com/garciadias/audify.git
cd audify
```

### 2. Start API Services

```bash
# Start Kokoro TTS and Ollama services
docker compose up -d

# Wait for services to be ready (~2-3 minutes)
# Check status: docker compose ps
```

### 3. Install Python Dependencies

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 4. Setup Ollama Models

```bash
# Pull required models for translation and audiobook generation
docker compose exec ollama ollama pull qwen3:30b

# Or use lighter models for testing:
# docker compose exec ollama ollama pull llama3.2:3b
```

### 5. Convert Your First Book

```bash
# Convert EPUB to audiobook (using Kokoro TTS)
task run path/to/your/book.epub

# Convert PDF to audiobook
task run path/to/your/document.pdf

# Create audiobook from EPUB
task audiobook path/to/your/book.epub
```

## 🚀 Quick Start with Qwen-TTS (Local)

Qwen-TTS is a high-quality, free, and privacy-friendly local TTS solution with excellent multilingual support.

### 1. Setup Qwen-TTS API

First, set up the Qwen-TTS API server (requires GPU):

```bash
# Clone Qwen-TTS API repository
git clone https://github.com/QwenLM/Qwen3-TTS
cd Qwen3-TTS

# Start with Docker (recommended)
make up

# The API will be available at http://localhost:8890
```

For detailed setup instructions, see the [Qwen3-TTS documentation](https://github.com/QwenLM/Qwen3-TTS).

### 2. Install Audify

```bash
git clone https://github.com/garciadias/audify.git
cd audify
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 3. Configure Qwen-TTS

Create a `.keys` file:

```bash
TTS_PROVIDER=qwen
QWEN_API_URL=http://localhost:8890
QWEN_TTS_VOICE=Vivian
```

### 4. Convert Your First Book

```bash
# Convert using Qwen-TTS
task run path/to/your/book.epub

# Or specify provider explicitly
task --tts-provider qwen run path/to/your/book.epub
```

## 🚀 Quick Start with Cloud TTS

If you prefer to use cloud TTS providers without Docker:

### 1. Clone and Install

```bash
git clone https://github.com/garciadias/audify.git
cd audify
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 2. Configure Your TTS Provider

Create a `.keys` file with your credentials:

```bash
cp .keys.example .keys
# Edit .keys and add your provider credentials
# See Configuration section for details
```

### 3. Convert Books with Cloud TTS

```bash
# Using OpenAI TTS
task --tts-provider openai run "book.epub"

# Using AWS Polly
task --tts-provider aws run "book.epub"

# Using Google Cloud TTS
task --tts-provider google run "book.epub"
```

## 📖 Usage Examples

### Basic Audiobook Conversion

```bash
# English EPUB to audiobook
task run "book.epub"

# PDF with specific language
task --language pt run "document.pdf"

# With translation (English to Spanish)
task --language en --translate es run "book.epub"
```

### Audiobook Generation

```bash
# Create audiobook from EPUB
task audiobook "book.epub"

# Limit to first 5 chapters
task audiobook "book.epub" --max-chapters 5

# Custom voice and language
task audiobook "book.epub" --voice af_bella --language en

# With translation
task audiobook "book.epub" --translate pt
```

### Using Commercial APIs (DeepSeek, Claude, GPT-4, Gemini)

Instead of local Ollama models, you can use commercial APIs for better quality or faster processing:

```bash
# Using DeepSeek (cost-effective)
task audiobook "book.epub" -m "api:deepseek/deepseek-chat"

# Using Claude 3.5 Sonnet (high quality)
task audiobook "book.epub" -m "api:anthropic/claude-3-5-sonnet-20240620"

# Using GPT-4 (reliable)
task audiobook "book.epub" -m "api:openai/gpt-4-turbo-preview"

# Using Gemini Pro
task audiobook "book.epub" -m "api:gemini/gemini-1.5-pro"
```

**Setup Required**: Create a `.keys` file with your API keys for the provider(s) you intend to use. See [Commercial APIs Guide](docs/COMMERCIAL_APIS.md) for detailed instructions.

```bash
# Copy example file and add your keys
cp .keys.example .keys
# Edit .keys and add keys for your chosen provider(s):
# DEEPSEEK=your-deepseek-api-key-here
# ANTHROPIC=your-anthropic-api-key-here
# OPENAI=your-openai-api-key-here
# GEMINI=your-google-api-key-here
```

### Directory Input (Multi-file Processing)

Process multiple files from a directory into a single audiobook:

```bash
# Create audiobook from directory of files
task audiobook "path/to/directory/"

# Process directory with translation
task --translate es audiobook "path/to/articles/" 

# Directory with custom voice
task --voice af_bella --language en audiobook "path/to/papers/" 
```

**Supported file types in directory**: EPUB, PDF, TXT, MD

The directory mode will:

- Process each file as a separate episode
- Use the filename as the episode title
- Combine all episodes into a single M4B audiobook with chapter markers
- Synthesize the title audio for each episode

### Advanced Options

```bash
# List available languages
task run --list-languages

# List available TTS models
task --list-models run

# Save extracted text
task --save-text run "book.epub"

# Skip confirmation prompts
task -y run "book.epub"

# Use different TTS provider
task --tts-provider openai run "book.epub"    # OpenAI TTS
task --tts-provider aws run "book.epub"       # AWS Polly
task --tts-provider google run "book.epub"    # Google Cloud TTS
task --tts-provider qwen run "book.epub"      # Qwen-TTS (local)

# List available TTS providers
task --list-tts-providers run
```

## ⚙️ Configuration

### TTS Provider Configuration

Audify supports multiple TTS providers. Configure your preferred provider using environment variables or a `.keys` file:

#### Option 1: Using `.keys` File (Recommended)

Create a `.keys` file in the project root:

```bash
cp .keys.example .keys
```

Edit `.keys` and add your credentials:

```bash
# OpenAI TTS
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_TTS_MODEL=tts-1-hd
OPENAI_TTS_VOICE=alloy

# AWS Polly
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
AWS_POLLY_VOICE=Joanna
AWS_POLLY_ENGINE=neural

# Google Cloud TTS
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_TTS_VOICE=en-US-Chirp-HD-F
GOOGLE_TTS_LANGUAGE_CODE=en-US

# Qwen-TTS (Local)
QWEN_API_URL=http://localhost:8890
QWEN_TTS_VOICE=Vivian

# Default TTS Provider
TTS_PROVIDER=kokoro  # Options: kokoro, qwen, openai, aws, google
```

#### Option 2: Environment Variables

```bash
# Kokoro TTS API (Local)
export KOKORO_API_URL="http://localhost:8887/v1/audio"

# OpenAI TTS
export OPENAI_API_KEY="sk-your-key"
export OPENAI_TTS_MODEL="tts-1-hd"  # or "tts-1"
export OPENAI_TTS_VOICE="alloy"     # alloy, echo, fable, onyx, nova, shimmer

# AWS Polly
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
export AWS_POLLY_VOICE="Joanna"     # Neural voices recommended
export AWS_POLLY_ENGINE="neural"    # "standard" or "neural"

# Google Cloud TTS
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export GOOGLE_TTS_VOICE="en-US-Chirp-HD-F"
export GOOGLE_TTS_LANGUAGE_CODE="en-US"

# Qwen-TTS (Local)
export QWEN_API_URL="http://localhost:8890"
export QWEN_TTS_VOICE="Vivian"

# Default Provider
export TTS_PROVIDER="kokoro"  # Options: kokoro, qwen, openai, aws, google

# Ollama Configuration
export OLLAMA_API_BASE_URL="http://localhost:11434"
export OLLAMA_TRANSLATION_MODEL="qwen3:30b"
export OLLAMA_MODEL="magistral:24b"
```

### Choosing a TTS Provider

| Provider | Pros | Cons | Best For |
|----------|------|------|----------|
| **Kokoro** (Local) | Free, privacy-friendly, GPU-accelerated | Requires local setup | Development, privacy-sensitive projects |
| **Qwen-TTS** (Local) | Free, privacy-friendly, GPU-accelerated, multilingual | Requires separate API setup | Multilingual projects, privacy-sensitive content |
| **OpenAI** | High quality, easy setup | Pay per character | Production, high-quality output |
| **AWS Polly** | Neural voices, scalable | AWS account required | Enterprise, AWS-integrated projects |
| **Google Cloud TTS** | Natural voices, many languages | GCP account required | Multi-language projects |

### Docker Services

The `docker-compose.yml` configures (only needed for local/Kokoro TTS):

- **Kokoro TTS**: Port 8887 (GPU-accelerated speech synthesis, local)
- **Ollama**: Port 11434 (LLM for translation and audiobook generation, optional)
- **Audify API**: Port 8000 (REST API server, starts after Kokoro and Ollama are healthy)

The `api` service waits for Kokoro and Ollama to pass their healthchecks before starting, so services are always ready when the API accepts requests.

Note: Docker services are only required for Kokoro (local TTS). Commercial TTS providers (OpenAI, AWS, Google) and LLM APIs (DeepSeek, Claude, GPT-4, Gemini) work without Docker.

## 📁 Output Structure

```text
data/output/
├── [book_name]/
│   ├── chapters.txt           # Book metadata
│   ├── cover.jpg              # Book cover image
│   ├── chapters_001.mp3       # Individual chapter audio
│   ├── chapters_002.mp3
│   ├── chapters_003.mp3
│   ├── ...                    # More chapters
│   └── book_name.m4b          # Final audiobook
│
└── audiobooks/
    └── [book_name]/
        ├── episodes/
        │   ├── episode_001.mp3     # Audiobook episodes
        │   ├── episode_002.mp3
        │   └── ...
        ├── scripts/                # Generated scripts
        │   ├── episode_001_script.txt
        │   ├── original_text_001.txt
        │   └── ...
        ├── chapters.txt            # FFmpeg metadata
        └── [book_name].m4b         # Final M4B audiobook
```

**Directory audiobook output:**

```text
data/output/
└── [directory_name]/
    ├── episodes/
    │   ├── episode_001.mp3     # Episode from first file
    │   ├── episode_002.mp3     # Episode from second file
    │   └── ...
    ├── scripts/
    │   ├── episode_001_script.txt
    │   └── ...
    ├── chapters.txt            # Chapter metadata
    └── [directory_name].m4b    # Combined audiobook
```

## 🛠️ Development

### Available Tasks

```bash
task test      # Run tests with coverage
task format    # Format code with ruff
task run       # Convert ebook to audiobook
task audiobook # Create audiobook from content
task up        # Start Docker services
task api       # Start REST API server (dev mode, port 8000)
```

You can also use the installed CLI commands directly:
- `audify run` (or `audify-run`) - equivalent to `task run`
- `audify audiobook` (or `audify-audiobook`) - equivalent to `task audiobook`

### Local Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Run tests
task test

# Format code
task format

# Type checking (included in pre_test)
mypy ./audify ./tests --ignore-missing-imports
```

## 🌐 REST API

Audify exposes a FastAPI HTTP server for programmatic access to synthesis and audiobook creation.

### Starting the API

```bash
# Development mode (auto-reload)
task api

# Or via Docker (starts with Kokoro and Ollama)
docker compose up -d
```

The API runs on `http://localhost:8000` by default.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/providers` | List available TTS providers |
| `GET` | `/voices?provider=kokoro&language=en` | List voices for a provider |
| `POST` | `/synthesize` | Convert EPUB or PDF to MP3 |
| `POST` | `/audiobook` | Convert EPUB or PDF to M4B audiobook |

### Example: Synthesize an EPUB

```bash
curl -X POST http://localhost:8000/synthesize \
  -F "file=@book.epub" \
  -F "voice=af_bella" \
  -F "language=en" \
  --output book.mp3
```

### Example: Create an M4B Audiobook

```bash
curl -X POST http://localhost:8000/audiobook \
  -F "file=@book.epub" \
  -F "voice=af_bella" \
  -F "language=en" \
  --output book.m4b
```

### API Reference

Interactive docs are available at `http://localhost:8000/docs` (Swagger UI) once the server is running.

## 🏗️ Architecture

Audify uses a flexible multi-provider architecture supporting both local and cloud services:

```text
┌──────────────────────────────────┐
│   Audify REST API (port 8000)    │
│ • POST /synthesize               │
│ • POST /audiobook                │
│ • GET  /voices, /providers       │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│   Audify CLI / Python API        │
│ • EPUB/PDF/TXT Reader            │
│ • LLM Script Generation          │
│ • Audio Combine & M4B Assembly   │
└──────┬───────────────────────────┘
       │
       ├─── TTS Providers ───────────┐
       │    ├─ Kokoro (local)        │
       │    ├─ Qwen-TTS (local)      │
       │    ├─ OpenAI TTS            │
       │    ├─ AWS Polly             │
       │    └─ Google Cloud TTS      │
       │                             │
       └─── LLM APIs ───────────────┘
            ├─ Ollama (local)
            ├─ DeepSeek
            ├─ Claude
            ├─ GPT-4
            └─ Gemini
```

### Key Components

- **Text Extraction**: EPUB/PDF parsing with chapter detection
- **Translation**: LiteLLM + Commercial/Local LLMs for high-quality translation
- **TTS**: Multi-provider support (Kokoro, OpenAI, AWS Polly, Google Cloud TTS)
- **Audiobook Generation**: LLM-powered script creation with commercial API support
- **Audio Processing**: Pydub for format conversion and combining
- **API Management**: Unified API key management via .keys file or environment variables

## 🌍 Supported Languages

**Primary**: English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Hungarian, Korean, Japanese, Hindi

**Translation**: Any language pair supported by your Ollama model

## 🔧 Troubleshooting

### Common Issues

**Services not responding (Docker/Kokoro):**

```bash
# Check service status
docker compose ps

# Restart services
docker compose restart

# Check logs
docker compose logs kokoro
docker compose logs ollama
```

**Commercial API errors:**

```bash
# Verify API key configuration
cat .keys

# Test API connectivity
uv run audify translate test.txt --model api:deepseek-chat

# Check API key is loaded
# The system will show an error if the API key is missing or invalid
```

**TTS Provider issues:**

```bash
# List available TTS providers
uv run audify --list-tts-providers

# Test specific provider
uv run audify translate test.txt --tts-provider openai

# Check provider credentials in .keys file
# OpenAI: OPENAI_API_KEY
# AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Google: GOOGLE_APPLICATION_CREDENTIALS (path to JSON file)
```

**Ollama model not found:**

```bash
# List available models
docker compose exec ollama ollama list

# Pull required model
docker compose exec ollama ollama pull qwen3:30b
```

**GPU issues:**

```bash
# Check GPU availability
docker compose exec kokoro nvidia-smi

# If no GPU, services will run on CPU (slower)
```

### Performance Tips

- Use SSD storage for model caching
- Ensure adequate GPU memory (8GB+ recommended) for Kokoro
- Use lighter models for testing: `llama3.2:3b` instead of `magistral:24b`
- Commercial TTS providers (OpenAI, AWS, Google) are faster than local Kokoro
- Commercial LLM APIs often provide better latency than local Ollama
- Consider running local services on separate machines for large workloads
- Use cloud providers for production workloads requiring high reliability

## 📚 Examples

Check the `examples/` directory for sample usage patterns and configuration files.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `task test`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kokoro TTS](https://github.com/hexgrad/kokoro) for high-quality speech synthesis
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) accessible kokoro via FastAPI
- [Ollama](https://ollama.ai/) for local LLM inference
- [LiteLLM](https://www.litellm.ai/) for unified LLM API interface
- [OpenAI](https://openai.com/) for GPT and TTS APIs
- [Anthropic](https://www.anthropic.com/) for Claude API
- [DeepSeek](https://www.deepseek.com/) for DeepSeek API
- [Google](https://cloud.google.com/text-to-speech) for Gemini and Cloud TTS
- [AWS Polly](https://aws.amazon.com/polly/) for Text-to-Speech service
