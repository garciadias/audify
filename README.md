# Audify

Convert ebooks to audiobooks using AI text-to-speech with **multi-file batch processing**, **real-time progress monitoring**, and **job cancellation** capabilities.

## 🚀 New Features

- **📚 Multi-file Processing**: Upload and process multiple PDF/EPUB files simultaneously
- **📊 Real-time Progress Monitoring**: Watch progress bars and detailed status for each file
- **⏸️ Job Cancellation**: Cancel individual jobs or all jobs at any point
- **🔄 Batch Operations**: Process multiple files with different settings
- **💾 Session Persistence**: Jobs persist across browser refreshes
- **📈 Job Statistics**: View comprehensive statistics about processing jobs

## Prerequisites

- Python 3.9-3.12
- UV package manager
- CUDA-capable GPU (optional, for faster processing)
- CUDA toolkit (if using GPU)
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

### Run the Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8503` to access the multi-file interface.

## 🎯 How to Use the Multi-File Interface

### 1. Upload Files

- Use the sidebar file uploader to select multiple PDF or EPUB files
- Files are stored in the session and persist across page refreshes
- Supported formats: `.pdf`, `.epub`

### 2. Configure Settings

- **Language**: Select the language of your text files
- **Translation**: Optionally translate text to another language before synthesis
- **TTS Engine**: Choose between:
  - **Kokoro** (fast, lightweight, recommended)
  - **TTS Models** (higher quality, slower, requires model specification)

### 3. Select Files to Process

- Check the files you want to process from the uploaded files list
- View the current status of each file (pending, running, completed, failed, cancelled)

### 4. Start Processing

- Click "🚀 Start Processing" to begin batch synthesis
- Each file will be queued and processed according to the configured settings

### 5. Monitor Progress

- **Real-time Updates**: Progress bars show current synthesis progress
- **Job Statistics**: View counts of pending, running, completed, failed, and cancelled jobs
- **Detailed Status**: Expand each job to see detailed information and progress

### 6. Manage Jobs

- **Cancel Individual Jobs**: Cancel specific jobs while others continue
- **Pause All Jobs**: Stop all running jobs at once
- **Clear Completed**: Remove completed jobs from the interface

### 7. Download Results

- Once synthesis is complete, download buttons appear for each successful job
- Files are available in MP3/M4B format depending on the source type

## 🏗️ Architecture

The enhanced multi-file system consists of several key components:

### Job Management (`audify/job_manager.py`)

- **JobManager**: Orchestrates multiple concurrent jobs
- **Job**: Represents individual file processing tasks with progress tracking
- **JobProgress**: Tracks detailed progress information
- **JobStatus**: Manages job states (pending, running, completed, failed, cancelled)

### Enhanced Synthesizers (`audify/enhanced_synthesizers.py`)

- **EnhancedEpubSynthesizer**: EPUB processing with progress callbacks and cancellation
- **EnhancedPdfSynthesizer**: PDF processing with progress callbacks and cancellation
- **EnhancedBaseSynthesizer**: Base class with common progress/cancellation functionality

### Job Processing (`audify/job_processor.py`)

- **process_file_job**: Main function that processes individual files with progress tracking
- Handles both PDF and EPUB files with appropriate synthesizers
- Manages temporary files and cleanup

## 🎛️ Advanced Features

### Concurrent Processing

- Process up to 2 files simultaneously (configurable)
- ThreadPoolExecutor manages concurrent synthesis tasks
- Resource management prevents system overload

### Progress Tracking

- **Sentence-level Progress**: Track synthesis progress at the sentence level
- **Chapter Progress**: For EPUBs, track progress through individual chapters
- **Overall Progress**: View total progress across all jobs

### Error Handling

- Robust error handling with detailed error messages
- Failed jobs don't affect other running jobs
- Automatic cleanup of temporary files

### Session Management

- Jobs persist across browser refreshes
- Uploaded files remain in session until cleared
- Job history and statistics maintained

## 🛠️ Command Line Usage (Original Functionality)

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
