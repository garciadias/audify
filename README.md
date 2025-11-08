# Audify

[![codecov](https://codecov.io/github/garciadias/audify/branch/master/graph/badge.svg)](https://codecov.io/github/garciadias/audify)
[![Tests](https://github.com/garciadias/audify/workflows/Run%20Tests/badge.svg)](https://github.com/garciadias/audify/actions)

Convert ebooks and PDFs to audiobooks and audiobooks using AI text-to-speech and translation services.

Audify is a API-based system that transforms written content into high-quality audio using:

- **Kokoro TTS API** for natural speech synthesis
- **Ollama + LiteLLM** for intelligent translation
- **LLM-powered audiobook generation** for engaging audio content

## 🚀 Features

- **📚 Multiple Formats**: Convert EPUB ebooks and PDF documents
- **🎙️ Audiobook Creation**: Generate audiobook-style content from books using LLM
- **🌍 Multi-language Support**: Translate content
- **🎵 High-Quality TTS**: Natural-sounding speech via Kokoro API
- **⚙️ Flexible Configuration**: Environment-based settings

## 📋 Prerequisites

- **Python 3.10-3.13**
- **UV package manager** ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker & Docker Compose** (for API services)
- **CUDA-capable GPU** (recommended for optimal performance)

## 🐳 Quick Start with Docker (Recommended)

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
docker compose exec ollama ollama pull qwen3:30b  # Translation
docker compose exec ollama ollama pull qwen3:30b         # Audiobook generation

# Or use lighter models for testing:
# docker compose exec ollama ollama pull llama3.2:3b
```

### 5. Convert Your First Book

```bash
# Convert EPUB to audiobook
task run path/to/your/book.epub

# Convert PDF to audiobook
task run path/to/your/document.pdf

# Create audiobook from EPUB
task audiobook path/to/your/book.epub
```

## 📖 Usage Examples

### Basic Audiobook Conversion

```bash
# English EPUB to audiobook
task run "book.epub"

# PDF with specific language
task run "document.pdf" --language pt

# With translation (English to Spanish)
task run "book.epub" --language en --translate es
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

### Advanced Options

```bash
# List available languages
task run --list-languages

# List available TTS models
task run --list-models

# Save extracted text
task run "book.epub" --save-text

# Skip confirmation prompts
task run "book.epub" -y
```

## ⚙️ Configuration

### Environment Variables

```bash
# Kokoro TTS API
export KOKORO_API_URL="http://localhost:8887/v1/audio"

# Ollama Configuration
export OLLAMA_API_BASE_URL="http://localhost:11434"
export OLLAMA_TRANSLATION_MODEL="qwen3:30b"
export OLLAMA_MODEL="qwen3:30b"
```

### Docker Services

The `docker-compose.yml` configures:

- **Kokoro TTS**: Port 8887 (GPU-accelerated speech synthesis)
- **Ollama**: Port 11434 (LLM for translation and audiobook generation)

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
        ├── episode_01.mp3     # Audiobook episodes
        ├── episode_02.mp3
        └── scripts/           # Generated scripts
```

## 🛠️ Development

### Available Tasks

```bash
task test      # Run tests with coverage
task format    # Format code with ruff
task run       # Convert ebook to audiobook
task audiobook   # Create audiobook from content
task up        # Start Docker services
```

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

## 🏗️ Architecture

Audify uses a modern microservices architecture:

```text
┌─────────────────┐    ┌──────────────┐    ┌──────────────────┐
   Audify CLI            Kokoro              Ollama    
                         TTS API             LLM API   
                                                       
 • EPUB/PDF Read         • Speech           • Translation
 • Text Process            Synthesis        • Audiobook scripts
 • Audio Combine         • Multi-voice      
└─────────────────┘    └──────────────┘    └──────────────────┘
```

### Key Components

- **Text Extraction**: EPUB/PDF parsing with chapter detection
- **Translation**: LiteLLM + Ollama for high-quality translation
- **TTS**: Kokoro API for natural speech synthesis
- **Audiobook Generation**: LLM-powered script creation
- **Audio Processing**: Pydub for format conversion and combining

## 🌍 Supported Languages

**Primary**: English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Hungarian, Korean, Japanese, Hindi

**Translation**: Any language pair supported by your Ollama model

## 🔧 Troubleshooting

### Common Issues

**Services not responding:**

```bash
# Check service status
docker compose ps

# Restart services
docker compose restart

# Check logs
docker compose logs kokoro
docker compose logs ollama
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
- Ensure adequate GPU memory (8GB+ recommended)
- Use lighter models for testing: `llama3.2:3b` instead of `qwen3:30b`
- Consider running services on separate machines for large workloads

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
