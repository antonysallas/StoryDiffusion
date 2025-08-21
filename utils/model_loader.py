# models/model_loader.py
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from PIL import Image

from config.settings import AppConfig
from utils.device import initialize_device
from utils.file_utils import ensure_photomaker_model


class ModelLoader:
    def __init__(self, config: AppConfig = AppConfig()):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device, self.dtype = initialize_device()
        self.pipeline: Optional[StableDiffusionXLPipeline] = None
        self.model_path: Path = ensure_photomaker_model()

    def load_models(self) -> Dict:
        """Load and initialize models."""
        try:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None,
            ).to(self.device)

            # Set scheduler
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

            self.logger.info("Models loaded successfully")
            return {"pipeline": self.pipeline}

        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
    ) -> Tuple[List[Image.Image], Dict]:
        """Generate images based on prompt."""
        try:
            if not self.pipeline:
                raise RuntimeError("Pipeline not initialized")

            model_config = self.config.model_config

            with torch.no_grad():
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images,
                    num_inference_steps=model_config.num_inference_steps,
                    guidance_scale=model_config.guidance_scale,
                    height=model_config.height,
                    width=model_config.width,
                )

            self.logger.info(f"Generated {num_images} images successfully")
            return output.images, output.metadata

        except Exception as e:
            self.logger.error(f"Image generation failed: {str(e)}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

    def clear_models(self) -> None:
        """Clear loaded models from memory."""
        try:
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.logger.info("Models cleared from memory")
        except Exception as e:
            self.logger.error(f"Failed to clear models: {str(e)}")
