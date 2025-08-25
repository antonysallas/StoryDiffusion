"""
TensorRT-optimized Stable Diffusion 3.5 Pipeline
Custom pipeline for NVIDIA TensorRT-optimized SD 3.5 models
"""
import os
import torch
import numpy as np
from typing import Optional, List, Union, Dict, Any
from PIL import Image
import onnxruntime as ort
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer, 
    T5EncoderModel, 
    T5TokenizerFast
)
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging

logger = logging.get_logger(__name__)

class TensorRTSD35Pipeline:
    """
    TensorRT-optimized Stable Diffusion 3.5 Pipeline
    
    This pipeline loads and runs NVIDIA TensorRT-optimized ONNX models
    for Stable Diffusion 3.5, providing significant performance improvements
    while maintaining compatibility with the diffusers interface.
    """
    
    def __init__(
        self,
        model_path: str,
        precision: str = "fp8",  # "bf16" or "fp8"
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_path = model_path
        self.precision = precision
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Detect RTX 5090 Blackwell architecture for optimal settings
        self.is_rtx_5090 = self._detect_rtx_5090()
        
        # ONNX session providers - optimize for RTX 5090 Blackwell architecture
        gpu_mem_limit = 24 * 1024 * 1024 * 1024 if self.is_rtx_5090 else 8 * 1024 * 1024 * 1024
        
        # RTX 5090 specific optimizations - use only compatible options
        cuda_provider_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': gpu_mem_limit,
            'cudnn_conv_algo_search': 'EXHAUSTIVE' if self.is_rtx_5090 else 'HEURISTIC',
            'do_copy_in_default_stream': True,
        }
        
        # For RTX 5090 and TensorRT models, use TensorRT provider first
        if self.is_rtx_5090:
            print("🚀 Using RTX 5090 compatible TensorRT + CUDA provider settings")
            self.providers = [
                # TensorRT provider for optimal performance
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': gpu_mem_limit,
                    'trt_fp16_enable': True if self.precision in ["fp8", "fp16"] else False,
                    'trt_engine_cache_enable': True,
                    'trt_timing_cache_enable': True,
                    'trt_dla_core': 0,
                }),
                # CUDA provider as fallback
                ('CUDAExecutionProvider', cuda_provider_options),
                'CPUExecutionProvider'
            ]
        else:
            self.providers = [
                ('CUDAExecutionProvider', cuda_provider_options),
                'CPUExecutionProvider'
            ]
        
        # Session options for better compatibility
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Component paths
        self.onnx_dir = os.path.join(model_path, "ONNX")
        
        # Initialize components
        self.tokenizer_one = None
        self.tokenizer_two = None  
        self.tokenizer_three = None
        self.text_encoder_one = None
        self.text_encoder_two = None
        self.text_encoder_three = None
        self.transformer_session = None
        self.vae_session = None
        self.scheduler = None
        
        self._load_components()
    
    def _detect_rtx_5090(self) -> bool:
        """Detect if we're running on RTX 5090 (Blackwell architecture)"""
        try:
            if not torch.cuda.is_available():
                return False
            
            # Get GPU name and compute capability
            gpu_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            
            # RTX 5090 has compute capability 9.0+ and Blackwell in the name, or compute 12.0
            is_5090 = ("5090" in gpu_name or 
                      (compute_cap[0] >= 12) or  # Blackwell architecture
                      (compute_cap[0] == 9 and compute_cap[1] >= 0 and "RTX 50" in gpu_name))
            
            if is_5090:
                logger.info(f"Detected RTX 5090 Blackwell GPU: {gpu_name}, compute {compute_cap}")
            
            return is_5090
            
        except Exception as e:
            logger.warning(f"Could not detect GPU type: {e}")
            return False
    
    def _get_optimal_precision(self) -> str:
        """Get optimal precision for current hardware"""
        if self.is_rtx_5090:
            # RTX 5090 supports FP8 and FP4, prefer FP8 for better quality/speed balance
            if self.precision in ["fp8", "fp4"]:
                return self.precision
            else:
                logger.info("RTX 5090 detected, using FP8 precision for optimal performance")
                return "fp8"
        else:
            # Fallback to FP16 for older hardware
            return "fp16"
    
    def _get_rtx_5090_providers(self):
        """Get RTX 5090 specific TensorRT providers with Blackwell optimizations"""
        return [
            # Use TensorRT provider first for TensorRT models
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 20 * 1024 * 1024 * 1024,  # 20GB for RTX 5090
                'trt_max_partition_iterations': 1000,
                'trt_min_subgraph_size': 1,
                'trt_fp16_enable': True if self.precision in ["fp8", "fp16"] else False,
                'trt_int8_enable': False,  # Disable INT8 for now
                'trt_engine_cache_enable': True,
                'trt_timing_cache_enable': True,
                # RTX 5090 Blackwell optimizations  
                'trt_dla_core': 0,  # Use GPU, not DLA
            }),
            # Fallback to CUDA provider with basic settings
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 20 * 1024 * 1024 * 1024,
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
    
    def _get_fp16_cuda_providers(self):
        """Get FP16 compatible CUDA providers for RTX 5090"""
        return [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo', 
                'gpu_mem_limit': 16 * 1024 * 1024 * 1024,  # Even more conservative
                'cudnn_conv_algo_search': 'HEURISTIC',  # Faster for fallback
                'do_copy_in_default_stream': True,
                # Remove all advanced options for maximum compatibility
            }),
            'CPUExecutionProvider'
        ]
    
    def _load_components(self):
        """Load all pipeline components"""
        logger.info(f"Loading TensorRT SD 3.5 components from {self.model_path}")
        
        # Load tokenizers and text encoders (PyTorch)
        self._load_text_components()
        
        # Load ONNX sessions for transformer and VAE
        self._load_onnx_components()
        
        # Initialize scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", 
            subfolder="scheduler"
        )
        
        logger.info("TensorRT SD 3.5 pipeline loaded successfully")
    
    def _load_text_components(self):
        """Load text encoders and tokenizers"""
        # CLIP-L
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="tokenizer"
        )
        
        # CLIP-G  
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="tokenizer_2"
        )
        
        # T5
        self.tokenizer_three = T5TokenizerFast.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="tokenizer_3"
        )
        
        # Create ONNX sessions for text encoders
        clip_l_path = os.path.join(self.onnx_dir, "clip_l", "model_optimized.onnx")
        clip_g_path = os.path.join(self.onnx_dir, "clip_g", "model_optimized.onnx")
        t5_path = os.path.join(self.onnx_dir, "t5", "model_optimized.onnx")
        
        # RTX 5090 specific CUDA provider with better bfloat16 handling
        rtx_5090_providers = self._get_rtx_5090_providers()
        
        try:
            # Try RTX 5090 optimized providers first
            if self.is_rtx_5090:
                self.clip_l_session = ort.InferenceSession(clip_l_path, sess_options=self.session_options, providers=rtx_5090_providers)
            else:
                self.clip_l_session = ort.InferenceSession(clip_l_path, sess_options=self.session_options, providers=self.providers)
            logger.info("✅ CLIP-L loaded successfully")
        except Exception as e:
            logger.error(f"CLIP-L failed to load: {e}")
            raise RuntimeError(f"TensorRT CLIP-L failed to load") from e
        
        try:
            if self.is_rtx_5090:
                self.clip_g_session = ort.InferenceSession(clip_g_path, sess_options=self.session_options, providers=rtx_5090_providers)
            else:
                self.clip_g_session = ort.InferenceSession(clip_g_path, sess_options=self.session_options, providers=self.providers)
            logger.info("✅ CLIP-G loaded successfully")
        except Exception as e:
            logger.error(f"CLIP-G failed to load: {e}")
            raise RuntimeError(f"TensorRT CLIP-G failed to load") from e
        
        try:
            if self.is_rtx_5090:
                self.t5_session = ort.InferenceSession(t5_path, sess_options=self.session_options, providers=rtx_5090_providers)
            else:
                self.t5_session = ort.InferenceSession(t5_path, sess_options=self.session_options, providers=self.providers)
            logger.info("✅ T5 loaded successfully")
        except Exception as e:
            logger.error(f"T5 failed to load: {e}")
            raise RuntimeError(f"TensorRT T5 failed to load") from e
    
    def _load_onnx_components(self):
        """Load ONNX sessions for transformer and VAE"""
        # Transformer (MMDiT)
        # Check available precision directories
        transformer_dir = os.path.join(self.onnx_dir, "transformer")
        available_precisions = []
        if os.path.exists(transformer_dir):
            available_precisions = [d for d in os.listdir(transformer_dir) 
                                   if os.path.isdir(os.path.join(transformer_dir, d))]
            print(f"Available transformer precisions: {available_precisions}")
        
        # Map requested precision to actual directory
        if self.precision == "fp8" and "fp8" in available_precisions:
            transformer_path = os.path.join(self.onnx_dir, "transformer", "fp8", "model_optimized.onnx")
        elif self.precision == "fp16" and "fp16" in available_precisions:
            transformer_path = os.path.join(self.onnx_dir, "transformer", "fp16", "model_optimized.onnx")
        elif "fp16" in available_precisions:
            # Default to fp16 if available
            transformer_path = os.path.join(self.onnx_dir, "transformer", "fp16", "model_optimized.onnx")
            print(f"Using fp16 transformer (requested {self.precision} not available)")
        elif "bf16" in available_precisions:
            # Only use bf16 as last resort
            transformer_path = os.path.join(self.onnx_dir, "transformer", "bf16", "model_optimized.onnx")
            print("WARNING: Only bf16 transformer available, may have compatibility issues")
        else:
            raise ValueError(f"No compatible transformer precision found in {transformer_dir}")
        
        # VAE
        vae_path = os.path.join(self.onnx_dir, "vae", "model_optimized.onnx")
        
        # Create sessions with fallback
        fallback_providers = ['CPUExecutionProvider']
        
        try:
            logger.info(f"Loading transformer from: {transformer_path}")
            self.transformer_session = ort.InferenceSession(transformer_path, sess_options=self.session_options, providers=self.providers)
            logger.info("✅ Transformer loaded successfully on GPU")
        except Exception as e:
            error_str = str(e)
            if "bfloat16" in error_str or "tensor(bfloat16)" in error_str:
                logger.error(f"❌ Transformer has bfloat16 compatibility issue: {e}")
                logger.error("This TensorRT model was compiled with bfloat16 which is not supported by your ONNX Runtime GPU provider")
                logger.error("The model needs to be recompiled with FP16 or FP32 inputs for GPU compatibility")
                raise RuntimeError("TensorRT model is incompatible - bfloat16 not supported by ONNX Runtime GPU provider") from e
            else:
                logger.error(f"Transformer failed to load: {e}")
                raise RuntimeError(f"TensorRT transformer failed to load") from e
        
        try:
            self.vae_session = ort.InferenceSession(vae_path, sess_options=self.session_options, providers=self.providers)
            logger.info("✅ VAE loaded successfully on GPU")
        except Exception as e:
            logger.error(f"VAE failed to load: {e}")
            raise RuntimeError(f"TensorRT VAE failed to load") from e
    
    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Dict[str, torch.Tensor]:
        """Encode text prompt using all three text encoders"""
        # Tokenize
        tokens_one = self.tokenizer_one(
            prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="np"
        )
        
        tokens_two = self.tokenizer_two(
            prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="np"
        )
        
        tokens_three = self.tokenizer_three(
            prompt, 
            padding="max_length", 
            max_length=256, 
            truncation=True, 
            return_tensors="np"
        )
        
        # Encode with ONNX sessions
        clip_l_output = self.clip_l_session.run(None, {"input_ids": tokens_one.input_ids})[0]
        clip_g_output = self.clip_g_session.run(None, {"input_ids": tokens_two.input_ids})[0]
        t5_output = self.t5_session.run(None, {"input_ids": tokens_three.input_ids})[0]
        
        # Convert to torch tensors
        prompt_embeds = torch.from_numpy(np.concatenate([clip_l_output, clip_g_output], axis=-1)).to(
            device=self.device, dtype=self.torch_dtype
        )
        
        t5_embeds = torch.from_numpy(t5_output).to(device=self.device, dtype=self.torch_dtype)
        
        # Handle negative prompt
        if negative_prompt:
            # Similar process for negative prompt
            neg_tokens_one = self.tokenizer_one(
                negative_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np"
            )
            neg_tokens_two = self.tokenizer_two(
                negative_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np"
            )
            neg_tokens_three = self.tokenizer_three(
                negative_prompt, padding="max_length", max_length=256, truncation=True, return_tensors="np"
            )
            
            neg_clip_l = self.clip_l_session.run(None, {"input_ids": neg_tokens_one.input_ids})[0]
            neg_clip_g = self.clip_g_session.run(None, {"input_ids": neg_tokens_two.input_ids})[0]
            neg_t5 = self.t5_session.run(None, {"input_ids": neg_tokens_three.input_ids})[0]
            
            negative_prompt_embeds = torch.from_numpy(np.concatenate([neg_clip_l, neg_clip_g], axis=-1)).to(
                device=self.device, dtype=self.torch_dtype
            )
            negative_t5_embeds = torch.from_numpy(neg_t5).to(device=self.device, dtype=self.torch_dtype)
        else:
            # Create zero embeddings for negative prompt
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_t5_embeds = torch.zeros_like(t5_embeds)
        
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "t5_embeds": t5_embeds,
            "negative_t5_embeds": negative_t5_embeds
        }
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs
    ):
        """
        Run inference with TensorRT-optimized SD 3.5
        """
        # Handle batch prompts
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
        
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        # Encode prompts
        prompt_embeds_dict = self.encode_prompt(prompt[0], negative_prompt[0])
        
        # Prepare latents
        shape = (batch_size * num_images_per_prompt, 16, height // 8, width // 8)
        latents = torch.randn(shape, device=self.device, dtype=self.torch_dtype, generator=generator)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Prepare timestep
            timestep = t.expand(latent_model_input.shape[0]).float().cpu().numpy()
            
            # Prepare embeddings
            encoder_hidden_states = torch.cat([
                prompt_embeds_dict["negative_prompt_embeds"], 
                prompt_embeds_dict["prompt_embeds"]
            ]) if guidance_scale > 1.0 else prompt_embeds_dict["prompt_embeds"]
            
            pooled_projections = torch.cat([
                prompt_embeds_dict["negative_t5_embeds"],
                prompt_embeds_dict["t5_embeds"]  
            ]) if guidance_scale > 1.0 else prompt_embeds_dict["t5_embeds"]
            
            # Run transformer inference
            noise_pred = self.transformer_session.run(
                None,
                {
                    "hidden_states": latent_model_input.cpu().numpy(),
                    "timestep": timestep,
                    "encoder_hidden_states": encoder_hidden_states.cpu().numpy(),
                    "pooled_projections": pooled_projections.cpu().numpy()
                }
            )[0]
            
            noise_pred = torch.from_numpy(noise_pred).to(device=self.device, dtype=self.torch_dtype)
            
            # Apply guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents
        latents_np = latents.cpu().numpy()
        image = self.vae_session.run(None, {"latent_sample": latents_np})[0]
        image = torch.from_numpy(image)
        
        # Convert to PIL
        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.permute(0, 2, 3, 1).float().cpu().numpy()
            image = (image * 255).round().astype("uint8")
            image = [Image.fromarray(img) for img in image]
        
        if not return_dict:
            return (image,)
        
        return {"images": image}
    
    def to(self, device):
        """Move pipeline to device (compatibility method)"""
        self.device = device
        return self
    
    def enable_model_cpu_offload(self):
        """Enable CPU offload (compatibility method)"""
        logger.warning("CPU offload not implemented for TensorRT pipeline")
        return self
    
    def enable_attention_slicing(self, slice_size=None):
        """Enable attention slicing (compatibility method)"""
        logger.warning("Attention slicing not needed for TensorRT pipeline")
        return self