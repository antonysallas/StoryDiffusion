import yaml
import torch
from typing import Optional
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusion3ControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler
)
from utils import PhotoMakerStableDiffusionXLPipeline
try:
    from utils.tensorrt_sd35_pipeline import TensorRTSD35Pipeline
    TENSORRT_AVAILABLE = True
except ImportError:
    TensorRTSD35Pipeline = None
    TENSORRT_AVAILABLE = False
import os
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

def _is_tensorrt_model(path: str) -> bool:
    """
    Check if a model path contains TensorRT-optimized ONNX models
    """
    if os.path.isdir(path):
        # Check for ONNX directory structure
        onnx_dir = os.path.join(path, "ONNX")
        if os.path.exists(onnx_dir):
            # Check for required ONNX components
            required_components = ["clip_g", "clip_l", "t5", "transformer", "vae"]
            for component in required_components:
                component_path = os.path.join(onnx_dir, component)
                if not os.path.exists(component_path):
                    return False
            return True
    return False

def _detect_rtx_5090() -> bool:
    """Detect if we're running on RTX 5090 (Blackwell architecture)"""
    try:
        if not torch.cuda.is_available():
            return False

        # Get GPU name and compute capability
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)

        # RTX 5090 has compute capability 12.0 (Blackwell) or contains "5090"
        is_5090 = ("5090" in gpu_name or
                  (compute_cap[0] >= 12) or  # Blackwell architecture
                  (compute_cap[0] == 9 and compute_cap[1] >= 0 and "RTX 50" in gpu_name))

        if is_5090:
            print(f"🚀 Detected RTX 5090 Blackwell GPU: {gpu_name}, compute {compute_cap}")

        return is_5090

    except Exception as e:
        print(f"Could not detect GPU type: {e}")
        return False

def _resolve_tensorrt_model_path(model_name_or_path: str, auto_download: bool = True) -> str:
    """
    Resolve the actual local path for TensorRT models using HF_HOME
    If not found locally and auto_download is True, attempt to download the model
    """
    # If it's already a local path, return it
    if os.path.isdir(model_name_or_path):
        if _is_tensorrt_model(model_name_or_path):
            return model_name_or_path
        else:
            raise ValueError(f"Path {model_name_or_path} exists but does not contain a valid TensorRT model")

    # Get HuggingFace cache directory from environment
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    hub_cache_dir = os.path.join(hf_home, 'hub')

    # Convert model name to cache directory format
    cache_name = f"models--{model_name_or_path.replace('/', '--')}"
    cache_model_dir = os.path.join(hub_cache_dir, cache_name)

    if os.path.exists(cache_model_dir):
        # Look for snapshots directory
        snapshots_dir = os.path.join(cache_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            # Get the latest snapshot
            snapshots = [d for d in os.listdir(snapshots_dir)
                       if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                # Use the first (and usually only) snapshot
                latest_snapshot = snapshots[0]
                snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
                if _is_tensorrt_model(snapshot_path):
                    print(f"Found TensorRT model at: {snapshot_path}")
                    return snapshot_path

    # If not found locally and auto_download is enabled, try to download
    if auto_download:
        print(f"TensorRT model not found in HF_HOME ({hub_cache_dir})")
        print(f"Attempting to download TensorRT model: {model_name_or_path}")
        try:
            downloaded_path = download_tensorrt_model(model_name_or_path, cache_dir=hf_home)
            return downloaded_path
        except Exception as e:
            print(f"Failed to download TensorRT model: {e}")
            raise RuntimeError(f"TensorRT model '{model_name_or_path}' not found locally and download failed") from e
    else:
        # If not found locally, return the original path
        print(f"TensorRT model not found in HF_HOME ({hub_cache_dir}), using: {model_name_or_path}")
        return model_name_or_path

def download_tensorrt_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    """
    Download TensorRT model from HuggingFace Hub
    
    Args:
        model_name: HuggingFace model name (e.g., "stabilityai/stable-diffusion-3.5-large-tensorrt")
        cache_dir: Optional cache directory path
    
    Returns:
        Local path to the downloaded model
    """
    try:
        print(f"Downloading TensorRT model: {model_name}")
        print("This may take several minutes depending on your internet connection...")
        
        # Use snapshot_download to get the entire model
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        
        # Validate that this is actually a TensorRT model
        if not _is_tensorrt_model(local_path):
            raise ValueError(f"Downloaded model at {local_path} does not appear to be a valid TensorRT model")
        
        print(f"✅ TensorRT model downloaded successfully to: {local_path}")
        return local_path
        
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print("❌ Authentication required to download this model.")
            print("Please set up your HuggingFace token:")
            print("1. Create a token at https://huggingface.co/settings/tokens")
            print("2. Run: huggingface-cli login")
            print("3. Or set HF_TOKEN environment variable")
            raise RuntimeError("HuggingFace authentication required") from e
        elif e.response.status_code == 403:
            print("❌ Access denied. Please request access to the model:")
            print(f"Visit https://huggingface.co/{model_name} and click 'Request access'")
            raise RuntimeError(f"Access denied for model {model_name}") from e
        else:
            print(f"❌ HTTP error downloading model: {e}")
            raise RuntimeError(f"Failed to download model {model_name}") from e
    except Exception as e:
        print(f"❌ Unexpected error downloading model: {e}")
        raise RuntimeError(f"Failed to download model {model_name}") from e

def get_models_dict():
    # 打开并读取YAML文件
    with open('config/models.yaml', 'r') as stream:
        try:
            # 解析YAML文件内容
            data = yaml.safe_load(stream)

            # 此时 'data' 是一个Python字典，里面包含了YAML文件的所有数据
            return data

        except yaml.YAMLError as exc:
            # 如果在解析过程中发生了错误，打印异常信息
            print(exc)

def load_models(model_info, device, photomaker_path=None, enable_controlnet=False):
    path = model_info["path"]
    single_files = model_info["single_files"]
    use_safetensors = model_info["use_safetensors"]
    model_type = model_info["model_type"]
    architecture = model_info.get("architecture", "sdxl")  # Default to SDXL for backward compatibility

    # Determine dtype based on GPU capability and architecture
    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_capability()
        is_rtx_5090 = _detect_rtx_5090()

        if is_rtx_5090:
            # RTX 5090 Blackwell: Use FP16 for better ONNX compatibility
            dtype = torch.float16
            print("🚀 RTX 5090: Using FP16 for optimal compatibility")
        elif compute_cap[0] >= 8:
            # RTX 30/40 series: Use bfloat16
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float16

    print(f"Loading {architecture} model with dtype {dtype}")

    # Load SD 3.5 models
    if architecture == "sd3":
        # Check if this is a TensorRT-optimized model
        is_tensorrt = "tensorrt" in path.lower() or _is_tensorrt_model(path)

        if model_type == "original":
            if is_tensorrt and TENSORRT_AVAILABLE:
                print(f"Loading TensorRT-optimized SD 3.5 model from {path}")
                # Start with bf16 precision for better compatibility
                precision = "bf16"

                # Resolve the actual local path for TensorRT models (with auto-download)
                try:
                    actual_model_path = _resolve_tensorrt_model_path(path, auto_download=True)
                except Exception as e:
                    print(f"Failed to resolve/download TensorRT model: {e}")
                    print("Falling back to standard SD 3.5 pipeline")
                    is_tensorrt = False
                    actual_model_path = None

                # Try different precisions optimized for RTX 5090
                is_rtx_5090 = _detect_rtx_5090()
                if is_rtx_5090:
                    # RTX 5090 Blackwell: try FP8 first, then FP16 (skip problematic BF16)
                    precisions_to_try = ["fp8", "fp16"]
                    print("🚀 RTX 5090: Trying FP8 then FP16 precision (skipping problematic BF16)")
                elif torch.cuda.get_device_capability()[0] >= 9:
                    # RTX 4090: try FP8 then BF16
                    precisions_to_try = ["fp8", "bf16"]
                else:
                    # Older GPUs: BF16 only
                    precisions_to_try = ["bf16"]

                pipe = None
                if actual_model_path and is_tensorrt:  # Only try TensorRT if we have a valid model path
                    for prec in precisions_to_try:
                        try:
                            print(f"Attempting to load with {prec} precision...")
                            pipe = TensorRTSD35Pipeline(
                                model_path=actual_model_path,
                                precision=prec,
                                device=device,
                                torch_dtype=torch.float16 if prec == "fp8" else dtype  # Use fp16 for fp8 models
                            )
                            print(f"✓ TensorRT SD 3.5 pipeline loaded with {prec} precision")
                            break
                        except Exception as e:
                            print(f"✗ Failed with {prec} precision: {e}")
                            continue

                if pipe is None:
                    print("Failed to load TensorRT pipeline with any precision")
                    print("Falling back to standard SD 3.5 pipeline")
                    is_tensorrt = False

            if not is_tensorrt or not TENSORRT_AVAILABLE:
                # If TensorRT was requested but failed to load, provide helpful guidance
                if "tensorrt" in path.lower():
                    if not TENSORRT_AVAILABLE:
                        print("❌ TensorRT pipeline not available. Missing dependencies or import failed.")
                        print("Fallback to standard SD 3.5 pipeline will be used.")
                        is_tensorrt = False  # Allow fallback
                    else:
                        print("❌ TensorRT model failed to load - this usually means:")
                        print("1. Model not downloaded yet (will be attempted automatically)")
                        print("2. HuggingFace authentication required")
                        print("3. Model access not granted")
                        print("Fallback to standard SD 3.5 pipeline will be used.")
                        is_tensorrt = False  # Allow fallback instead of raising exception

                # Check if ControlNet should be enabled for SD 3.5
                controlnet_sd3 = None
                if enable_controlnet:
                    controlnet_sd3 = _load_sd3_controlnet_pose()

                if controlnet_sd3 is not None:
                    # Load SD 3.5 with ControlNet support
                    print(f"Loading SD 3.5 model with ControlPose from {path}")
                    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                        path,
                        controlnet=controlnet_sd3,
                        torch_dtype=dtype,
                        use_safetensors=use_safetensors
                    )
                    print("✅ SD 3.5 pipeline loaded with ControlPose support")
                else:
                    # Standard SD 3.5 pipeline with RTX 5090 optimizations
                    print(f"Loading standard SD 3.5 model from {path}")

                    # Detect RTX 5090 for optimal settings
                    is_rtx_5090 = _detect_rtx_5090()

                    # RTX 5090 optimization: use variant for FP8 if available
                    variant = None
                    if is_rtx_5090:
                        print("🚀 RTX 5090 detected - applying Blackwell optimizations")
                        # Try to use FP8 variant if available
                        try:
                            pipe = StableDiffusion3Pipeline.from_pretrained(
                                path,
                                variant="fp8",  # Try FP8 variant for RTX 5090
                                torch_dtype=torch.float16,  # Use FP16 for FP8 models
                                use_safetensors=use_safetensors,
                                low_cpu_mem_usage=True
                            )
                            print("✅ Loaded with FP8 optimization for RTX 5090")
                        except:
                            # Fallback to standard loading with RTX 5090 optimizations
                            pipe = StableDiffusion3Pipeline.from_pretrained(
                                path,
                                torch_dtype=dtype,
                                use_safetensors=use_safetensors,
                                low_cpu_mem_usage=True
                            )
                            print("✅ Loaded with RTX 5090 memory optimizations")
                    else:
                        pipe = StableDiffusion3Pipeline.from_pretrained(
                            path,
                            torch_dtype=dtype,
                            use_safetensors=use_safetensors
                        )

                pipe = pipe.to(device)
                # Configure SD 3.5 scheduler
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        else:
            # SD 3.5 doesn't support PhotoMaker yet
            raise ValueError("PhotoMaker is not yet supported for SD 3.5 models. Please use InstantID or IP-Adapter instead.")

    # Load SDXL models
    elif architecture == "sdxl":
        if model_type == "original":
            # Check if ControlNet should be enabled
            controlnet = None
            if enable_controlnet:
                controlnet = _load_instantid_controlnet()

            if controlnet is not None:
                # Load with ControlNet support
                if single_files:
                    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                        path,
                        controlnet=controlnet,
                        torch_dtype=dtype
                    )
                else:
                    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                        path,
                        controlnet=controlnet,
                        torch_dtype=dtype,
                        use_safetensors=use_safetensors
                    )
                print("✅ SDXL pipeline loaded with ControlNet support")
            else:
                # Standard SDXL pipeline
                if single_files:
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        path,
                        torch_dtype=dtype
                    )
                else:
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        path,
                        torch_dtype=dtype,
                        use_safetensors=use_safetensors
                    )

            pipe = pipe.to(device)
            # Configure SDXL scheduler
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        elif model_type == "Photomaker":
            if photomaker_path is None:
                raise ValueError("PhotoMaker requires photomaker_path to be provided")
            if single_files:
                print("loading from a single_files")
                pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
                    path,
                    torch_dtype=dtype
                )
            else:
                pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                    path,
                    torch_dtype=dtype,
                    use_safetensors=use_safetensors
                )
            pipe = pipe.to(device)
            pipe.load_photomaker_adapter(
                os.path.dirname(photomaker_path),
                subfolder="",
                weight_name=os.path.basename(photomaker_path),
                trigger_word="img"  # define the trigger word
            )
            pipe.fuse_lora()
            # Configure SDXL scheduler
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Apply RTX 5090-specific optimizations
    is_rtx_5090 = _detect_rtx_5090()
    skip_xformers = TENSORRT_AVAILABLE and TensorRTSD35Pipeline and isinstance(pipe, TensorRTSD35Pipeline)

    if is_rtx_5090 and not skip_xformers:
        print("🚀 Applying RTX 5090 Blackwell optimizations...")

        # Enable memory efficient attention with RTX 5090 optimizations
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ Enabled optimized attention for RTX 5090")
            except Exception as e:
                print(f"Could not enable xformers: {e}")

        # RTX 5090 has 32GB VRAM, use sequential CPU offload instead of full offload
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload()
                print("✅ Enabled sequential CPU offload for RTX 5090")
            except Exception as e:
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                    print("✅ Enabled fallback CPU offload")

        # Enable FreeU for better quality (RTX 5090 can handle it)
        if hasattr(pipe, "enable_freeu") and architecture == "sdxl":
            try:
                pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
                print("✅ Enabled FreeU optimization for RTX 5090")
            except Exception as e:
                print(f"Could not enable FreeU: {e}")

        # Enable VAE slicing for better memory efficiency
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
            print("✅ Enabled VAE slicing for RTX 5090")

    else:
        # Standard optimizations for non-RTX 5090 GPUs
        if hasattr(pipe, "enable_xformers_memory_efficient_attention") and not skip_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except Exception as e:
                print(f"Could not enable xformers: {e}")

        # Enable model CPU offload for better memory management (skip for TensorRT)
        if hasattr(pipe, "enable_model_cpu_offload") and not skip_xformers:
            pipe.enable_model_cpu_offload()
            print("Enabled model CPU offload")

    return pipe

def download_instantid_models():
    """Verify InstantID models are available (uses HF_HOME environment variable)"""
    # Get HuggingFace cache directory from environment
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    hub_cache_dir = os.path.join(hf_home, 'hub')

    # Check if models are already cached
    instantid_cache_dir = os.path.join(hub_cache_dir, "models--InstantX--InstantID")
    antelopev2_cache_dir = os.path.join(hub_cache_dir, "models--DIAMONIK7777--antelopev2")

    print(f"Using HuggingFace cache: {hub_cache_dir}")

    models_found = False
    if os.path.exists(instantid_cache_dir):
        print(f"✓ InstantID models found at: {instantid_cache_dir}")
        models_found = True

    if not models_found:
        print("Downloading InstantID models to HuggingFace cache...")
        try:
            # Download InstantID models - will use HF_HOME automatically
            hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin")
            hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json")
            hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors")

            # Download antelopev2 models - will use HF_HOME automatically
            for model in ["1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx"]:
                hf_hub_download(repo_id="DIAMONIK7777/antelopev2", filename=model)

        except Exception as e:
            print(f"Error downloading InstantID models: {e}")

    print("InstantID models ready")

def _load_instantid_controlnet() -> Optional[ControlNetModel]:
    """Load InstantID ControlNet model from HuggingFace cache"""
    try:
        # Get HuggingFace cache directory
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        instantid_cache = os.path.join(hf_home, 'hub', 'models--InstantX--InstantID')

        if not os.path.exists(instantid_cache):
            print("InstantID cache not found")
            return None

        # Find snapshot directory
        snapshots_dir = os.path.join(instantid_cache, 'snapshots')
        if not os.path.exists(snapshots_dir):
            print("InstantID snapshots directory not found")
            return None

        snapshots = os.listdir(snapshots_dir)
        if not snapshots:
            print("No InstantID snapshots found")
            return None

        # Get ControlNet path
        snapshot_path = os.path.join(snapshots_dir, snapshots[0])
        controlnet_path = os.path.join(snapshot_path, "ControlNetModel")

        if not os.path.exists(controlnet_path):
            print(f"ControlNet model not found at {controlnet_path}")
            return None

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16
        )

        print(f"✅ InstantID ControlNet loaded from {controlnet_path}")
        return controlnet

    except Exception as e:
        print(f"Failed to load InstantID ControlNet: {e}")
        return None

def _load_sd3_controlnet_pose() -> Optional[ControlNetModel]:
    """Load SD3 ControlPose model from HuggingFace cache"""
    try:
        # Get HuggingFace cache directory
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        controlnet_cache = os.path.join(hf_home, 'hub', 'models--InstantX--SD3-Controlnet-Pose')

        if not os.path.exists(controlnet_cache):
            print("SD3-Controlnet-Pose cache not found")
            return None

        # Find snapshot directory
        snapshots_dir = os.path.join(controlnet_cache, 'snapshots')
        if not os.path.exists(snapshots_dir):
            print("SD3-Controlnet-Pose snapshots directory not found")
            return None

        snapshots = os.listdir(snapshots_dir)
        if not snapshots:
            print("No SD3-Controlnet-Pose snapshots found")
            return None

        # Get the model path (use first snapshot)
        snapshot_path = os.path.join(snapshots_dir, snapshots[0])

        # Load ControlNet for SD3
        controlnet = ControlNetModel.from_pretrained(
            snapshot_path,
            torch_dtype=torch.float16
        )

        print(f"✅ SD3-Controlnet-Pose loaded from {snapshot_path}")
        return controlnet

    except Exception as e:
        print(f"Failed to load SD3-Controlnet-Pose: {e}")
        return None