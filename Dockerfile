# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Create app user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install system dependencies with fresh package lists and --fix-missing
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --fix-missing \
    git \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv for the app user
USER appuser
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set PATH to include uv
ENV PATH="/home/appuser/.cargo/bin:${PATH}"

# Copy all project files and set ownership
USER root
COPY --chown=appuser:appuser . .

# Switch back to app user
USER appuser

# Create virtual environment and follow your exact setup process
RUN uv venv

# Activate venv and ensure pip is installed
RUN . .venv/bin/activate && \
    python -m ensurepip && \
    python -m pip install --upgrade pip

# Sync dependencies with uv
RUN uv sync

# Expose the port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_TEMP_DIR=/tmp/gradio_cache

# Run the application exactly as you do locally
CMD ["uv", "run", "main.py"]