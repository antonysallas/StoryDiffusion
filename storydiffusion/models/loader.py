"""
Model loading utilities for StoryDiffusion.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

from ..config import ModelConfig
from ..device import clear_memory, get_device_capabilities, optimize_for_device


def download_model(
    model_config: ModelConfig,
    local_dir: Union[str, Path] = "data",
    force_download: bool = False,
) -> str:
    """Download a model file if it doesn't exist locally.

    Args:
        model_config: Configuration for the model to download
        local_dir: Directory to store downloaded files
        force_download: Whether to force re-download even if file exists

    Returns:
        str: Path to the model file
    """
    # Ensure the data directory exists
    local_dir = Path(local_dir)
    local_dir.mkdir(exist_ok=True, parents=True)

    # Handle both single file and directory models
    if model_config.filename:
        # Single file model
        local_path = local_dir / model_config.filename
        if not force_download and local_path.exists():
            print(f"Using locally cached {model_config.name}: {local_path}")
            return str(local_path)

        print(f"Downloading {model_config.name} from {model_config.repo_id}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_config.repo_id,
                filename=model_config.filename,
                repo_type="model",
                local_dir=local_dir,
            )
            print(f"Successfully downloaded {model_config.name} to {downloaded_path}")
            return downloaded_path
        except Exception as e:
            print(f"Error downloading {model_config.name}: {e}")
            part_file = str(local_path) + ".part"
            if os.path.exists(part_file):
                print(f"Removing partial download: {part_file}")
                os.remove(part_file)
            raise
    else:
        # Directory model - just return the repo_id to use with from_pretrained
        return model_config.repo_id


def load_stable_diffusion_pipeline(
    model_config: ModelConfig, 
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[StableDiffusionXLPipeline, str]:
    """Load the Stable Diffusion XL pipeline with optimizations for the current device.

    Args:
        model_config: Configuration for the model to load
        device: Device to load the model on (cuda, mps, cpu)
        dtype: Data type for model weights

    Returns:
        tuple: (pipeline, model identifier)
    """
    # Get device capabilities
    device_caps = get_device_capabilities()

    # Determine device and dtype based on capabilities if not explicitly provided
    if device is None:
        device = "cuda" if device_caps["cuda"] else "mps" if device_caps["mps"] else "cpu"
    
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {model_config.name} from {model_config.path} on {device} with {dtype}")

    # Clear memory before loading model
    clear_memory(device_caps)

    try:
        # Determine loading method based on config
        if model_config.single_file:
            # Download single file if needed
            model_path = download_model(model_config)
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, 
                torch_dtype=dtype,
                safety_checker=None if device == "mps" else "default"
            )
        else:
            # Load from pretrained
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_config.path,
                torch_dtype=dtype,
                use_safetensors=model_config.use_safetensors,
                safety_checker=None if device == "mps" else "default"
            )

        # Move model to device
        pipe = pipe.to(device)

        # Apply device-specific optimizations
        optimize_for_device(device, pipe)

        # Common scheduler configuration
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(50)

        # Extract model identifier
        model_identifier = f"{model_config.name}-{device}-{dtype}"

        # Clear memory after setup
        clear_memory(device_caps)

        return pipe, model_identifier

    except Exception as e:
        print(f"Error loading model: {e}")
        # Provide a more specific error message based on the device
        if device == "mps" and "attention_processor" in str(e).lower():
            print("MPS-specific error. Try setting PYTORCH_ENABLE_MPS_FALLBACK=1")
        elif device == "cuda" and "out of memory" in str(e).lower():
            print("CUDA out of memory. Try a smaller model or reduce batch size.")
        raise


def load_photomaker_adapter(
    pipe: StableDiffusionXLPipeline,
    model_config: ModelConfig,
    trigger_word: str = "img",
) -> StableDiffusionXLPipeline:
    """Load the PhotoMaker adapter into a StableDiffusionXL pipeline.

    Args:
        pipe: The SDXL pipeline to adapt
        model_config: Configuration for the PhotoMaker model
        trigger_word: Trigger word to activate PhotoMaker

    Returns:
        StableDiffusionXLPipeline: Pipeline with PhotoMaker adapter
    """
    try:
        # Import the PhotoMaker pipeline only when needed
        from utils import PhotoMakerStableDiffusionXLPipeline
        
        # Download the model if needed
        photomaker_path = download_model(model_config)
        
        # Convert regular pipeline to PhotoMaker pipeline
        photomaker_pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            pipe.config.name_or_path,
            torch_dtype=pipe.dtype,
            use_safetensors=model_config.use_safetensors,
            safety_checker=None if pipe.device.type == "mps" else "default"
        )
        
        # Move to the same device as the original pipeline
        photomaker_pipe = photomaker_pipe.to(pipe.device)
        
        # Load the adapter
        photomaker_pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word=trigger_word
        )
        
        # Apply the same optimizations as the original pipeline
        optimize_for_device(pipe.device.type, photomaker_pipe)
        
        # Fuse LoRA weights
        photomaker_pipe.fuse_lora()
        
        return photomaker_pipe
        
    except Exception as e:
        print(f"Error loading PhotoMaker adapter: {e}")
        raise