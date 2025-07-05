# Use CUDA-enabled base image compatible with PyTorch 2.5.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    build-essential \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

RUN mkdir -p /app/src/audify
# Copy project files
COPY ./audify/ /app/src/audify/

# Add uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Sync the project into a new environment, using the frozen lockfile
COPY pyproject.toml README.md uv.lock ./

# Install Python dependencies using uv
RUN uv sync --no-cache-dir

# Expose the port for the application
EXPOSE 8501

# Set the entrypoint for the container
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]