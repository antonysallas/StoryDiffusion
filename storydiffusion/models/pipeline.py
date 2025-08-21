"""Pipeline management for StoryDiffusion models."""

import torch
import gc
import os
from typing import Dict, Any, Optional, Union
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from huggingface_hub import hf_hub_download
from utils.load_models_utils import load_models
from utils.gradio_utils import cal_attn_mask_xl
from .attention import set_attention_processor
from ..config import (
    DEVICE, 
    DTYPE,
    MODELS_DICT, 
    PHOTOMAKER_REPO_ID, 
    PHOTOMAKER_FILENAME,
    LOCAL_DATA_DIR,
    global_state,
    apply_device_optimizations,
    clear_device_cache
)


def download_photomaker_model() -> str:
    """
    Download and cache the PhotoMaker model for reference image-based generation.
    
    Returns:
        str: Path to the downloaded PhotoMaker model file
    """
    local_path = os.path.join(LOCAL_DATA_DIR, PHOTOMAKER_FILENAME)
    
    if not os.path.exists(local_path):
        # Download from Hugging Face if not present locally
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        photomaker_path = hf_hub_download(
            repo_id=PHOTOMAKER_REPO_ID,
            filename=PHOTOMAKER_FILENAME,
            repo_type="model",
            local_dir=LOCAL_DATA_DIR,
        )
    else:
        photomaker_path = local_path
    
    return photomaker_path


def initialize_pipeline(
    model_name: str = "Unstable",
    model_type: str = "original",
    device: str = DEVICE
) -> StableDiffusionXLPipeline:
    """
    Initialize the Stable Diffusion XL pipeline with the specified model.
    
    Args:
        model_name: Name of the model from MODELS_DICT
        model_type: Either "original" or "Photomaker"
        device: Compute device to use
        
    Returns:
        StableDiffusionXLPipeline: Initialized pipeline
    """
    model_info = MODELS_DICT[model_name]
    sd_model_path = model_info["path"]
    single_files = model_info.get("single_files", False)
    use_safetensors = model_info.get("use_safetensors", True)
    
    # Load pipeline based on model source
    if single_files:
        # Load from single checkpoint file
        pipe = StableDiffusionXLPipeline.from_single_file(
            sd_model_path, torch_dtype=DTYPE
        )
    else:
        # Load from Hugging Face model repository
        pipe = StableDiffusionXLPipeline.from_pretrained(
            sd_model_path, torch_dtype=DTYPE, use_safetensors=use_safetensors
        )
    
    # Configure pipeline scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(50)  # Set denoising steps
    
    # Apply device-specific optimizations
    pipe = apply_device_optimizations(pipe, device)
    
    return pipe


def load_pipeline(
    model_name: str,
    model_type: str,
    id_length: int = 4,
    height: int = 768,
    width: int = 768,
    sa32: float = 0.5,
    sa64: float = 0.5,
    photomaker_path: Optional[str] = None,
    force_reload: bool = False
) -> StableDiffusionXLPipeline:
    """
    Load or reload the pipeline with the specified configuration.
    
    Args:
        model_name: Name of the model from MODELS_DICT
        model_type: Either "original" or "Photomaker"
        id_length: Number of character reference images
        height: Generated image height
        width: Generated image width
        sa32: Paired attention strength at 32x32 resolution
        sa64: Paired attention strength at 64x64 resolution
        photomaker_path: Path to PhotoMaker model (required if model_type is "Photomaker")
        force_reload: Force reload even if same model type
        
    Returns:
        StableDiffusionXLPipeline: Configured pipeline
    """
    current_model_key = f"{model_name}-{model_type}"
    
    # Check if we need to reload the model
    if not force_reload and global_state.cur_model_type == current_model_key and global_state.pipe is not None:
        # Just update attention processors if needed
        if global_state.id_length != id_length:
            set_attention_processor(global_state.pipe.unet, id_length)
            global_state.id_length = id_length
            global_state.total_length = id_length + 1
        return global_state.pipe
    
    # Clean up existing pipeline
    if global_state.pipe is not None:
        del global_state.pipe
        clear_device_cache(DEVICE)
    
    # Load new pipeline
    if model_type == "Photomaker":
        if photomaker_path is None:
            photomaker_path = download_photomaker_model()
        
        model_info = MODELS_DICT[model_name].copy()
        model_info["model_type"] = "Photomaker"
        pipe = load_models(model_info, device=DEVICE, photomaker_path=photomaker_path)
    else:
        pipe = initialize_pipeline(model_name, model_type, DEVICE)
    
    # Configure attention processors
    set_attention_processor(pipe.unet, id_length)
    
    # Configure scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    pipe.enable_vae_slicing()
    
    if DEVICE != "mps":
        pipe.enable_model_cpu_offload()
    
    # Update global state
    global_state.pipe = pipe
    global_state.unet = pipe.unet
    global_state.cur_model_type = current_model_key
    global_state.id_length = id_length
    global_state.total_length = id_length + 1
    global_state.height = height
    global_state.width = width
    global_state.sa32 = sa32
    global_state.sa64 = sa64
    
    # Pre-calculate attention masks
    global_state.mask1024, global_state.mask4096 = cal_attn_mask_xl(
        global_state.total_length,
        id_length,
        sa32,
        sa64,
        height,
        width,
        device=DEVICE,
        dtype=torch.float16,
    )
    
    return pipe


def get_model_type(model_type_str: str) -> str:
    """
    Convert UI model type string to internal model type.
    
    Args:
        model_type_str: UI model type string
        
    Returns:
        str: Internal model type ("original" or "Photomaker")
    """
    if model_type_str == "Using Ref Images":
        return "Photomaker"
    else:
        return "original"


def clear_attention_banks() -> None:
    """Clear all stored attention features from the processors."""
    
    if global_state.pipe is None:
        return
    
    for attn_processor in global_state.pipe.unet.attn_processors.values():
        if hasattr(attn_processor, 'id_bank'):
            for character_key in list(attn_processor.id_bank.keys()):
                del attn_processor.id_bank[character_key]
            attn_processor.id_bank = {}
    
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()