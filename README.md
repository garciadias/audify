# Audify

Convert ebooks to audiobooks using AI text-to-speech with job management and queue processing.

## Features

- ✅ **Job Management**: Create, monitor, and manage multiple text-to-speech jobs
- ✅ **Progress Tracking**: Real-time progress updates for running jobs
- ✅ **Queue System**: Handle multiple files concurrently with background processing
- ✅ **Job Persistence**: SQLite database stores job history and status
- ✅ **Web Interface**: User-friendly web UI for job management
- ✅ **REST API**: Complete API for programmatic access
- ✅ **File Support**: PDF and EPUB file formats
- ✅ **Multiple Engines**: Kokoro and TTS Models support
- ✅ **Translation**: Optional text translation before synthesis
- ✅ **Download Management**: Direct download of completed audiobooks

## Prerequisites

- Python 3.9-3.12
- UV package manager
- CUDA-capable GPU (recommended)
- CUDA toolkit and NVIDIA drivers (if using GPU)

## Quick Start

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/garciadias/audify.git
   cd audify
   ```

2. Run the setup script:

   ```bash
   ./setup.sh
   ```

### Running the Application

Start the FastAPI server:

```bash
uv run python main.py
```

The application will be available at:

- **Web Interface**: <http://localhost:8000>
- **API Documentation**: <http://localhost:8000/docs>
- **API Alternative Documentation**: <http://localhost:8000/redoc>

## API Usage

### Create a Job

```bash
curl -X POST "http://localhost:8000/jobs/" \
  -F "file=@your_book.pdf" \
  -F "language=en" \
  -F "engine=kokoro"
```

### Get All Jobs

```bash
curl "http://localhost:8000/jobs/"
```

### Get Job Status

```bash
curl "http://localhost:8000/jobs/1"
```

### Cancel a Job

```bash
curl -X POST "http://localhost:8000/jobs/1/cancel"
```

### Download Result

```bash
curl "http://localhost:8000/jobs/1/download" -o audiobook.mp3
```

## Job Status

Jobs can have the following statuses:

- **PENDING**: Job created and waiting to start
- **RUNNING**: Job is currently being processed
- **COMPLETED**: Job finished successfully
- **FAILED**: Job encountered an error
- **CANCELLED**: Job was cancelled by user

## Configuration

### Environment Variables

- `DATABASE_URL`: SQLite database URL (default: `sqlite:///./audify.db`)

### Supported Languages

- English (en), Spanish (es), French (fr), German (de)
- Italian (it), Portuguese (pt), Polish (pl), Turkish (tr)
- Russian (ru), Dutch (nl), Czech (cs), Arabic (ar)
- Chinese (zh), Hungarian (hu), Korean (ko), Japanese (ja), Hindi (hi)

### TTS Engines

1. **Kokoro** (default): Fast, lightweight TTS engine
2. **TTS Models**: More advanced models with customizable options

## Development

### Running with Auto-reload

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Database Management

The SQLite database (`audify.db`) is created automatically. To reset:

```bash
rm audify.db
```

### Testing

Run tests with:

```bash
uv run pytest
```

## Docker Support

For containerized deployment:

```bash
docker-compose up --build
```

## Architecture

- **FastAPI**: Web framework and API server
- **SQLModel**: Database ORM and data validation  
- **SQLite**: Job persistence and history
- **Background Tasks**: Threading-based job processing
- **Static Files**: HTML/CSS/JS frontend

```bash
task run path/to/book.epub
```

## Available Tasks

```bash
task test      # Run tests with coverage
task format    # Format code with black, isort and ruff
task run       # Convert ebook to audiobook
```

## Contributing

- [Contributing Guide](docs/CONTRIBUTING.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
