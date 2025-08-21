#!/bin/bash

# Clear the screen for a clean start
clear

# RTX 5090 Optimization Settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export XFORMERS_ENABLE_FLASH_ATTN=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Enable TF32 for better performance on newer GPUs
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Activate virtual environment
source .venv/bin/activate

# Verify HuggingFace cache models
echo "Checking HuggingFace cache models..."
if [ -n "$HF_HOME" ]; then
    echo "Using HF_HOME: $HF_HOME"
    HF_SD35_PATH="$HF_HOME/hub/models--InstantX--SD3.5-Large-IP-Adapter"
else
    echo "Using default HuggingFace cache"
    HF_SD35_PATH="$HOME/.cache/huggingface/hub/models--InstantX--SD3.5-Large-IP-Adapter"
fi

if [ -d "$HF_SD35_PATH" ]; then
    echo "✓ SD 3.5 IP-Adapter models found"
else
    echo "⚠ SD 3.5 IP-Adapter not found - will download when needed"
fi

# Run the application with the low VRAM version optimized for RTX 5090
echo "Starting StoryDiffusion with RTX 5090 optimizations..."
echo "Available models:"
python -c "from utils.load_models_utils import get_models_dict; models = get_models_dict(); [print(f'  - {name} ({info.get(\"architecture\", \"unknown\")})') for name, info in models.items()]"
echo
python story_diffusion_app.py