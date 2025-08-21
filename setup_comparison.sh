#!/bin/bash

echo "StoryDiffusion Setup Comparison"
echo "==============================="
echo ""

echo "🏠 LOCAL SETUP (Your current workflow):"
echo "----------------------------------------"
echo "1. curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "2. source .venv/bin/activate"
echo "3. python -m ensurepip"
echo "4. python -m pip install --upgrade pip"
echo "5. uv sync"
echo "6. uv run main.py"
echo ""

echo "🐳 DOCKER SETUP (Containerized version):"
echo "----------------------------------------"
echo "1. docker build -t storydiffusion ."
echo "2. docker run -p 7860:7860 storydiffusion"
echo ""
echo "   OR using Make:"
echo "   make build && make run"
echo ""

echo "📋 DOCKERFILE STEPS (mirrors your workflow):"
echo "--------------------------------------------"
echo "1. RUN curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "2. RUN uv venv"
echo "3. RUN . .venv/bin/activate && python -m ensurepip"
echo "4. RUN . .venv/bin/activate && python -m pip install --upgrade pip"
echo "5. RUN uv sync"
echo "6. CMD [\"uv\", \"run\", \"main.py\"]"
echo ""

echo "✅ EXACT SAME WORKFLOW PRESERVED!"
echo "================================="
echo "The Docker container follows your exact local setup process."
echo "Only difference: runs in isolated container environment."
echo ""

echo "🚀 QUICK START:"
echo "--------------"
echo "Local:  make local-setup && make dev"
echo "Docker: make build && make run"

chmod +x setup_comparison.sh