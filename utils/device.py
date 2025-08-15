# utils/device.py
from typing import Tuple

import torch


def clear_memory() -> None:
    """Clear CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def initialize_device() -> Tuple[str, torch.dtype]:
    """Initialize compute device and dtype."""
    try:
        if torch.cuda.is_available():
            return "cuda", torch.float32
        elif torch.backends.mps.is_available():
            return "mps", torch.float32
        return "cpu", torch.float32
    except Exception as e:
        print(f"Error initializing device: {e}")
        return "cpu", torch.float32
    finally:
        clear_memory()
