"""
StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation

A package for generating consistent character images across multiple frames using
Stable Diffusion XL with custom attention mechanisms.
"""

__version__ = "0.2.0"
__author__ = "HVision-NKU"

from .generation.generator import process_generation
from .models.attention import SpatialAttnProcessor2_0
from .ui.app import create_demo

__all__ = ["process_generation", "SpatialAttnProcessor2_0", "create_demo"]