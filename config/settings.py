# config/settings.py
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Model Settings
DEFAULT_DTYPE = "float32"
LOCAL_DIR = Path("data/")
PHOTOMAKER_FILENAME = "photomaker-v1.bin"
HF_REPO_ID = "TencentARC/PhotoMaker"


@dataclass
class ModelConfig:
    """Model configuration settings."""

    batch_size: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024


@dataclass
class AppConfig:
    """Application configuration settings."""

    model_config: ModelConfig = ModelConfig()
    device: Optional[str] = None
    log_level: int = logging.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[Path] = Path("logs/sdxl.log")


def configure_logging(config: AppConfig = AppConfig()) -> None:
    """Configure application logging."""
    # Create logs directory if it doesn't exist
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=config.log_level,
        format=config.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.log_file) if config.log_file else logging.NullHandler(),
        ],
    )


# Default configuration instance
default_config = AppConfig()
