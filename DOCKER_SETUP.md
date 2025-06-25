# StoryDiffusion Docker Setup

This Docker setup **preserves your exact local workflow** inside containers for consistent, reproducible deployments.

## Workflow Preservation

Your local commands:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source .venv/bin/activate
python -m ensurepip
python -m pip install --upgrade pip
uv sync
uv run main.py
```

Are replicated **exactly** in the Dockerfile:
```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv
RUN . .venv/bin/activate && python -m ensurepip && python -m pip install --upgrade pip
RUN uv sync
CMD ["uv", "run", "main.py"]
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run
make build
make run

# Access the app at http://localhost:7860
```

### Option 2: CPU-Only Docker (Works on all platforms including Apple Silicon)

```bash
# Build and run CPU version
make docker-cpu

# Access the app at http://localhost:7860
```

### Option 3: Manual Docker Commands

```bash
# Build the image
docker build -f Dockerfile.cpu -t storydiffusion:latest .

# Run the container
docker run -d \
  --name storydiffusion \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  storydiffusion:latest
```

## Features

### Preserved in Docker:
- ✅ All refactored code structure
- ✅ Model configurations
- ✅ Persistent model storage (via volumes)
- ✅ Results saved to host machine
- ✅ HuggingFace cache shared with host
- ✅ No SentinelOne conflicts (frpc runs inside container)

### Volume Mounts:
- `./data`: PhotoMaker and other model data
- `./results`: Generated images
- `~/.cache/huggingface`: HuggingFace model cache (avoid re-downloads)

### Environment Variables:
- `GRADIO_SERVER_NAME=0.0.0.0`: Listen on all interfaces
- `GRADIO_SERVER_PORT=7860`: Default Gradio port
- `GRADIO_TEMP_DIR=/tmp/gradio_cache`: Gradio cache location
- `HF_HUB_DISABLE_TELEMETRY=1`: Disable telemetry
- `GRADIO_ANALYTICS_ENABLED=False`: Disable Gradio analytics

## Platform-Specific Notes

### macOS (Apple Silicon)
- Use `Dockerfile.cpu` as MPS device is not available in Docker
- Performance will be slower than native, but fully functional
- Consider running natively with `uv run main.py` for best performance

### Linux (NVIDIA GPU)
- Uncomment GPU sections in `docker-compose.yml`
- Install nvidia-docker2
- Use the standard `Dockerfile`

### Windows
- Use Docker Desktop with WSL2 backend
- CPU version recommended unless you have WSL2 GPU support

## Managing the Container

```bash
# View logs
make logs
# or
docker logs -f storydiffusion

# Stop container
make stop
# or
docker stop storydiffusion

# Remove everything
make clean
# or
docker rm -f storydiffusion
docker rmi storydiffusion:latest
```

## Customization

### Change Port
Edit `docker-compose.yml` or use:
```bash
docker run -p 8080:7860 storydiffusion:latest
```

### Add GPU Support (NVIDIA)
1. Install nvidia-docker2
2. Uncomment GPU sections in `docker-compose.yml`
3. Use `make docker-gpu`

### Production Deployment

For production, consider:
1. Using a reverse proxy (nginx/traefin)
2. Adding SSL/TLS
3. Setting up health checks
4. Using Kubernetes for orchestration
5. Implementing resource limits

Example production docker-compose:
```yaml
version: '3.8'
services:
  storydiffusion:
    image: storydiffusion:latest
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## Troubleshooting

### Container won't start
- Check logs: `docker logs storydiffusion`
- Ensure port 7860 is not in use: `lsof -i :7860`

### Out of memory
- Increase Docker memory allocation in Docker Desktop
- Use CPU-only version which uses less memory

### Slow performance
- CPU version is slower than GPU/MPS
- Ensure Docker has enough CPU/memory allocated
- Consider running natively on Apple Silicon

### Models re-downloading
- Ensure HuggingFace cache volume is mounted correctly
- Check permissions on ~/.cache/huggingface

## Development Workflow

1. Make changes to code
2. Rebuild image: `make build`
3. Restart container: `make restart`
4. View logs: `make logs`

Or use bind mounts for live development:
```bash
docker run -v $(pwd):/app storydiffusion:latest
```