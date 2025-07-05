# Audify

Convert ebooks to audiobooks using AI text-to-speech.

## Prerequisites

- Python 3.9-3.12
- UV package manager
- CUDA-capable GPU
- CUDA toolkit
- NVIDIA drivers (if using GPU)
- Docker (optional, for containerized setup)
- Docker Compose (optional, for containerized setup)

## Quick Start

## Using Docker Compose (Recommended for quick use)

1. Clone the repository:

   ```bash
   git clone https://github.com/garciadias/audify.git
   cd audify
   ```

2. Build and run the Docker container:

   ```bash
   docker-compose up --build
   ```

3. Access the application:
    Open your web browser and go to `http://localhost:8501`.

## Using UV (Recommended for Local Development)

### Install UV if you haven't already

Follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)

### Create virtual environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Convert an ebook

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
