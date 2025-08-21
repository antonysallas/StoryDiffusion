.PHONY: help build run stop clean docker-cpu docker-gpu dev local-setup

help:
	@echo "StoryDiffusion Commands:"
	@echo "  make local-setup - Install uv and setup local environment (mimics your workflow)"
	@echo "  make dev         - Run locally with uv"
	@echo "  make build       - Build the Docker image"
	@echo "  make run         - Run the container"
	@echo "  make stop        - Stop the container"
	@echo "  make clean       - Remove containers and images"
	@echo "  make docker-cpu  - Build and run CPU-only version"
	@echo "  make docker-gpu  - Build and run GPU version (NVIDIA only)"

# Default Docker build (auto-detects GPU)
build:
	docker-compose build

run:
	docker-compose up -d
	@echo "StoryDiffusion is running at http://localhost:7860"

stop:
	docker-compose down

clean:
	docker-compose down -v
	docker rmi storydiffusion:latest || true

# CPU-only version (works on all platforms)
docker-cpu:
	docker build -f Dockerfile.cpu -t storydiffusion:cpu .
	docker run -d --name storydiffusion-cpu \
		-p 7860:7860 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/results:/app/results \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		storydiffusion:cpu
	@echo "StoryDiffusion (CPU) is running at http://localhost:7860"

# GPU version (NVIDIA only)
docker-gpu:
	docker build -t storydiffusion:gpu .
	docker run -d --name storydiffusion-gpu \
		--gpus all \
		-p 7860:7860 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/results:/app/results \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		storydiffusion:gpu
	@echo "StoryDiffusion (GPU) is running at http://localhost:7860"

# Local setup (mimics your exact workflow)
local-setup:
	@echo "Setting up local environment (following your exact steps)..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "Creating virtual environment..."
	uv venv
	@echo "Activating venv and upgrading pip..."
	. .venv/bin/activate && python -m ensurepip && python -m pip install --upgrade pip
	@echo "Syncing dependencies..."
	uv sync
	@echo "Setup complete! Run 'make dev' to start the application."

# Local development (your exact command)
dev:
	@echo "Starting StoryDiffusion locally..."
	uv run main.py

# Alternative: if you prefer to activate venv first
dev-activated:
	@echo "Starting StoryDiffusion with activated environment..."
	. .venv/bin/activate && python main.py

# Container management
logs:
	docker-compose logs -f

shell:
	docker-compose exec storydiffusion bash

restart:
	docker-compose restart