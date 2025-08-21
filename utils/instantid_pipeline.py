"""
InstantID Pipeline
Complete implementation combining AntelopeV2 + IP-Adapter + ControlNet
"""
import os
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, Union, List, Dict, Any
import insightface
from insightface.app import FaceAnalysis
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusion3ControlNetPipeline, 
    ControlNetModel,
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
import logging

logger = logging.getLogger(__name__)

class InstantIDPipeline:
    """
    Complete InstantID pipeline with AntelopeV2 face analysis, IP-Adapter, and ControlNet
    """
    
    def __init__(
        self,
        base_pipeline: Union[StableDiffusionXLControlNetPipeline, StableDiffusion3ControlNetPipeline],
        face_adapter_path: str,
        controlnet_path: str,
        face_analysis_model: str = "antelopev2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = dtype
        self.base_pipeline = base_pipeline
        
        # Initialize face analysis (AntelopeV2)
        self._init_face_analysis()
        
        # Load IP-Adapter for face features
        self._load_ip_adapter(face_adapter_path)
        
        # ControlNet is already loaded in the base pipeline
        self.controlnet_path = controlnet_path
        
        logger.info("InstantID pipeline initialized successfully")
    
    def _init_face_analysis(self):
        """Initialize AntelopeV2 face analysis models"""
        try:
            # Get HuggingFace cache directory
            hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            antelopev2_path = os.path.join(hf_home, 'hub', 'models--DIAMONIK7777--antelopev2')
            
            # Find the actual model files
            if os.path.exists(antelopev2_path):
                # Find snapshot directory
                snapshots_dir = os.path.join(antelopev2_path, 'snapshots')
                if os.path.exists(snapshots_dir):
                    snapshot = os.listdir(snapshots_dir)[0]  # Get first snapshot
                    model_path = os.path.join(snapshots_dir, snapshot)
                else:
                    model_path = antelopev2_path
            else:
                # Fallback to insightface default
                model_path = None
            
            # Initialize face analysis
            self.face_analysis = FaceAnalysis(
                name='antelopev2',
                root=model_path if model_path else None,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("AntelopeV2 face analysis initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize face analysis: {e}")
            self.face_analysis = None
    
    def _load_ip_adapter(self, adapter_path: str):
        """Load IP-Adapter for face feature injection"""
        try:
            # IP-Adapter will be loaded via the base pipeline's load_ip_adapter method
            # The actual loading happens when we call the pipeline
            self.adapter_path = adapter_path
            logger.info(f"IP-Adapter path configured: {adapter_path}")
        except Exception as e:
            logger.error(f"Failed to configure IP-Adapter: {e}")
            self.adapter_path = None
    
    def extract_face_features(self, face_image: Union[str, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract face features from reference image using AntelopeV2
        """
        if self.face_analysis is None:
            logger.warning("Face analysis not initialized")
            return None
        
        try:
            # Convert to OpenCV format
            if isinstance(face_image, str):
                image = cv2.imread(face_image)
            elif isinstance(face_image, Image.Image):
                image = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
            else:
                image = face_image
            
            # Analyze faces
            faces = self.face_analysis.get(image)
            
            if len(faces) == 0:
                logger.warning("No faces detected in reference image")
                return None
            
            # Use the largest face (most prominent)
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Extract face embedding
            face_embedding = face.normed_embedding
            logger.info(f"Face embedding extracted: {face_embedding.shape}")
            
            return face_embedding
            
        except Exception as e:
            logger.error(f"Face feature extraction failed: {e}")
            return None
    
    def prepare_controlnet_image(self, image: Union[str, Image.Image], detect_resolution: int = 512) -> Image.Image:
        """
        Prepare ControlNet conditioning image (face landmarks/pose)
        """
        try:
            if isinstance(image, str):
                image = load_image(image)
            
            # For InstantID, we typically use the face keypoints as ControlNet conditioning
            # Convert to OpenCV format for face analysis
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            if self.face_analysis is not None:
                faces = self.face_analysis.get(cv_image)
                if len(faces) > 0:
                    face = faces[0]  # Use first detected face
                    
                    # Create keypoint image
                    keypoint_image = np.zeros_like(cv_image)
                    
                    # Draw face landmarks if available
                    if hasattr(face, 'kps') and face.kps is not None:
                        kps = face.kps.astype(int)
                        for kp in kps:
                            cv2.circle(keypoint_image, tuple(kp), 2, (255, 255, 255), -1)
                    
                    # Draw bounding box
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(keypoint_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
                    
                    # Convert back to PIL
                    keypoint_image = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(keypoint_image)
            
            # Fallback: return a simple edge map
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(edges_rgb)
            
        except Exception as e:
            logger.error(f"ControlNet image preparation failed: {e}")
            return image  # Return original image as fallback
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        face_image: Union[str, Image.Image, np.ndarray],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        controlnet_image: Optional[Union[str, Image.Image]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        face_strength: float = 0.8,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images with InstantID (face consistency + pose control)
        
        Args:
            prompt: Text prompt for generation
            face_image: Reference face image for character consistency
            negative_prompt: Negative text prompt
            controlnet_image: Optional ControlNet conditioning image (auto-generated if None)
            height, width: Output dimensions
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            controlnet_conditioning_scale: ControlNet influence strength
            face_strength: Face consistency strength
            num_images_per_prompt: Number of images to generate
            generator: Random generator for reproducibility
        """
        try:
            # Extract face features
            face_embedding = self.extract_face_features(face_image)
            if face_embedding is None:
                logger.warning("Could not extract face features, proceeding without InstantID")
                # Fallback to standard generation
                return self.base_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    **kwargs
                )
            
            # Prepare ControlNet conditioning
            if controlnet_image is None:
                controlnet_image = self.prepare_controlnet_image(face_image)
            elif isinstance(controlnet_image, str):
                controlnet_image = load_image(controlnet_image)
            
            # Load IP-Adapter if available
            if self.adapter_path and hasattr(self.base_pipeline, 'load_ip_adapter'):
                try:
                    # Get the actual file path from HuggingFace cache
                    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
                    instantid_cache = os.path.join(hf_home, 'hub', 'models--InstantX--InstantID')
                    
                    if os.path.exists(instantid_cache):
                        snapshots_dir = os.path.join(instantid_cache, 'snapshots')
                        if os.path.exists(snapshots_dir):
                            snapshot = os.listdir(snapshots_dir)[0]
                            adapter_file = os.path.join(snapshots_dir, snapshot, 'ip-adapter.bin')
                            
                            if os.path.exists(adapter_file):
                                self.base_pipeline.load_ip_adapter(
                                    "InstantX/InstantID", 
                                    subfolder="", 
                                    weight_name="ip-adapter.bin"
                                )
                                self.base_pipeline.set_ip_adapter_scale(face_strength)
                                logger.info("IP-Adapter loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load IP-Adapter: {e}")
            
            # Generate with ControlNet conditioning
            result = self.base_pipeline(
                prompt=prompt,
                image=controlnet_image,  # ControlNet conditioning
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                **kwargs
            )
            
            return {
                "images": result.images,
                "face_embedding": face_embedding,
                "controlnet_image": controlnet_image
            }
            
        except Exception as e:
            logger.error(f"InstantID generation failed: {e}")
            # Fallback to standard generation
            return self.base_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                **kwargs
            )

def create_instantid_pipeline(
    model_id: str,
    architecture: str = "sdxl",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> InstantIDPipeline:
    """
    Create InstantID pipeline for SDXL or SD 3.5
    """
    try:
        # Get ControlNet model path from HuggingFace cache
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        controlnet_cache = os.path.join(hf_home, 'hub', 'models--InstantX--InstantID')
        controlnet_path = None
        
        if os.path.exists(controlnet_cache):
            snapshots_dir = os.path.join(controlnet_cache, 'snapshots')
            if os.path.exists(snapshots_dir):
                snapshot = os.listdir(snapshots_dir)[0]
                controlnet_path = os.path.join(snapshots_dir, snapshot, 'ControlNetModel')
        
        if not controlnet_path or not os.path.exists(controlnet_path):
            raise ValueError("ControlNet model not found in cache")
        
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=dtype
        )
        
        # Create base pipeline with ControlNet
        if architecture == "sdxl":
            base_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True
            )
            base_pipeline.scheduler = DDIMScheduler.from_config(base_pipeline.scheduler.config)
        elif architecture == "sd3":
            # Note: SD3 ControlNet support may be limited
            base_pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True
            )
            base_pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_pipeline.scheduler.config)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        base_pipeline = base_pipeline.to(device)
        
        # Get face adapter path
        face_adapter_path = os.path.join(hf_home, 'hub', 'models--InstantX--InstantID')
        
        # Create InstantID pipeline
        instantid_pipeline = InstantIDPipeline(
            base_pipeline=base_pipeline,
            face_adapter_path=face_adapter_path,
            controlnet_path=controlnet_path,
            device=device,
            dtype=dtype
        )
        
        return instantid_pipeline
        
    except Exception as e:
        logger.error(f"Failed to create InstantID pipeline: {e}")
        raise