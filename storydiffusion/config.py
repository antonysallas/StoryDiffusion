"""
Configuration and settings for StoryDiffusion.
Contains constants, paths, and default configuration values.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

# Constants
MAX_SEED = 2147483647  # np.iinfo(np.int32).max
DEFAULT_STYLE_NAME = "Japanese Anime"
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 768

# Paths
DATA_DIR = Path("data")
FONTS_DIR = Path("fonts")
RESULTS_DIR = Path("results")
DEFAULT_FONT = "Inkfree.ttf"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model configuration
@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    repo_id: str
    filename: Optional[str] = None
    single_file: bool = False
    use_safetensors: bool = True
    path: Optional[str] = None
    
    def __post_init__(self):
        if not self.path and self.repo_id:
            self.path = self.repo_id

# Default models dictionary
MODELS = {
    "SDXL": ModelConfig(
        name="Stable Diffusion XL", 
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
        single_file=False
    ),
    "Unstable": ModelConfig(
        name="Unstable Diffusion XL",
        repo_id="stablediffusionapi/sdxl-unstable-diffusers-y",
        use_safetensors=True,
        single_file=False
    ),
    "Juggernaut": ModelConfig(
        name="Juggernaut XL",
        repo_id="RunDiffusion/Juggernaut-XL-v8",
        use_safetensors=True,
        single_file=False
    ),
    "RealVision": ModelConfig(
        name="RealVision XL",
        repo_id="SG161222/RealVisXL_V4.0",
        use_safetensors=True,
        single_file=False
    ),
}

PHOTOMAKER_CONFIG = ModelConfig(
    name="PhotoMaker-V2",
    repo_id="TencentARC/PhotoMaker-V2",
    filename="photomaker-v2.bin",
    single_file=True
)

# Generation settings
@dataclass
class GenerationSettings:
    """Settings for image generation."""
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    sa32_strength: float = 0.5  # Paired Attention strength at 32x32
    sa64_strength: float = 0.5  # Paired Attention strength at 64x64
    id_length: int = 2          # Number of ID images
    seed: int = 0
    
# Application settings
@dataclass
class AppConfig:
    """Application configuration."""
    device: Optional[str] = None
    models: Dict[str, ModelConfig] = None
    photomaker: ModelConfig = PHOTOMAKER_CONFIG
    default_model: str = "Unstable"
    generation: GenerationSettings = GenerationSettings()
    
    def __post_init__(self):
        if self.models is None:
            self.models = MODELS