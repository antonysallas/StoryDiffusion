"""
Device management utilities for StoryDiffusion.
Handles device detection, memory management, and optimization.
"""

import gc
import multiprocessing
from typing import Dict, Tuple

import torch

def get_device_capabilities() -> Dict[str, bool]:
    """Determine available compute devices and their capabilities.

    Returns:
        Dict[str, bool]: Dictionary with availability flags for different compute devices
    """
    capabilities = {"cuda": False, "mps": False, "cpu": True}  # CPU is always available

    # Check CUDA (NVIDIA GPUs)
    try:
        capabilities["cuda"] = torch.cuda.is_available()
    except:
        pass

    # Check MPS (Apple Silicon)
    try:
        capabilities["mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except:
        pass

    return capabilities


def clear_memory(device_caps: Dict[str, bool] = None) -> None:
    """Clear device memory safely and efficiently.
    
    Args:
        device_caps: Optional dictionary with device capabilities
    """
    if device_caps is None:
        device_caps = get_device_capabilities()

    try:
        # CUDA devices (NVIDIA GPUs)
        if device_caps["cuda"]:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # MPS devices (Apple Silicon)
        elif device_caps["mps"]:
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        # Always run garbage collection regardless of device
        gc.collect()
    except Exception as e:
        print(f"Warning: Error during memory clearing: {e}")


def set_random_seed(seed: int, device_caps: Dict[str, bool] = None) -> None:
    """Set up random seeds for reproducibility across all libraries.

    Args:
        seed: The seed value to use
        device_caps: Optional dictionary with device capabilities
    """
    import random
    import numpy as np
    
    # Get device capabilities if not provided
    if device_caps is None:
        device_caps = get_device_capabilities()

    # Base random seed setup (works on all platforms)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device-specific seeding
    if device_caps["cuda"]:
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device_caps["mps"] and hasattr(torch.mps, "manual_seed"):
        # Future-proofing for potential MPS seed API
        torch.mps.manual_seed(seed)


def initialize_device() -> Tuple[str, torch.dtype]:
    """Initialize the optimal device with appropriate precision settings.

    Returns:
        Tuple[str, torch.dtype]: Device name and optimal dtype for that device
    """
    # Get device capabilities once
    device_caps = get_device_capabilities()

    # Clear memory before initialization
    clear_memory(device_caps)

    # Device configuration with priorities
    if device_caps["cuda"]:
        device_name = "cuda"
        dtype = torch.float16
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        # CUDA-specific optimizations
        torch.backends.cudnn.benchmark = True

    elif device_caps["mps"]:
        device_name = "mps"
        dtype = torch.float32
        print("Using MPS (Metal Performance Shaders) for Apple Silicon")

    else:
        device_name = "cpu"
        dtype = torch.float32
        print("Using CPU (generation will be slow)")
        # CPU-specific optimizations
        if hasattr(torch, "set_num_threads"):
            threads = max(2, multiprocessing.cpu_count() // 2)
            torch.set_num_threads(threads)
            print(f"Set PyTorch to use {threads} CPU threads")

    # Clear memory after initialization
    clear_memory(device_caps)

    return device_name, dtype


def optimize_for_device(device_name: str, pipeline) -> None:
    """Apply device-specific optimizations to the pipeline.
    
    Args:
        device_name: Device name (cuda, mps, cpu)
        pipeline: The diffusion pipeline to optimize
    """
    # Common optimizations for all devices
    pipeline.enable_attention_slicing(slice_size="auto")
    pipeline.enable_vae_slicing()
    
    # Device-specific optimizations
    if device_name == "cuda":
        pipeline.enable_model_cpu_offload()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_vae_tiling()
    elif device_name == "mps":
        # MPS-specific optimizations (currently limited)
        pass
    elif device_name == "cpu":
        # CPU optimizations
        pipeline.enable_sequential_cpu_offload()
        
    # Common final optimizations
    pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)