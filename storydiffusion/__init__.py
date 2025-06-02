"""
StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation.

StoryDiffusion is a text-to-image generation system that maintains consistency
of characters and elements across multiple generated images, allowing for
coherent visual storytelling.
"""

__version__ = "0.3.0"
__author__ = "Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, Qibin Hou"

# Import essential components for easy access
from .config import GenerationSettings, ModelConfig
from .device import initialize_device, clear_memory, set_random_seed

# Provide convenient access to main functionality
__all__ = [
    "GenerationSettings",
    "ModelConfig",
    "initialize_device", 
    "clear_memory",
    "set_random_seed",
]
