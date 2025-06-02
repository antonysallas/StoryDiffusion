# Suppress future warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Standard library imports
import copy
import datetime
import gc
import os
import random

# Third-party imports
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

# Diffusers imports
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils.loading_utils import load_image
from huggingface_hub import hf_hub_download
from PIL import ImageFont

# Local imports
from utils.gradio_utils import (
    cal_attn_indice_xl_effcient_memory,
    cal_attn_mask_xl,
    character_to_dict,
    get_ref_character,
    is_torch2_available,
    process_original_prompt,
)
from utils.load_models_utils import get_models_dict, load_models
from utils.style_template import styles
from utils.utils import get_comic

# Choose the appropriate AttnProcessor based on PyTorch version
if is_torch2_available():
    from utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils import AttnProcessor

# Constants
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Japanese Anime"
dtype = torch.float16

# Globals
models_dict = get_models_dict()


def get_device_capabilities():
    """Determine available compute devices and their capabilities.

    Returns:
        dict: Dictionary containing availability flags for different compute devices
    """
    capabilities = {"cuda": False, "mps": False, "cpu": True}  # CPU is always available

    # Check CUDA (NVIDIA GPUs)
    try:
        capabilities["cuda"] = torch.cuda.is_available()
    except:
        pass

    # Check MPS (Apple Silicon)
    try:
        capabilities["mps"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except:
        pass

    return capabilities


def clear_memory(device_caps=None):
    """Clear device memory safely and efficiently."""
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


def initialize_device():
    """Initialize the optimal device with appropriate precision settings."""
    global dtype

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
            import multiprocessing

            threads = max(2, multiprocessing.cpu_count() // 2)
            torch.set_num_threads(threads)
            print(f"Set PyTorch to use {threads} CPU threads")

    # Clear memory after initialization
    clear_memory(device_caps)

    return device_name


# Select and configure device
device = initialize_device()
print(f"Device selected: {device} with precision {dtype}")


# check if the file exists locally at a specified path before downloading it.
# if the file doesn't exist, it uses `hf_hub_download` to download the file
# and optionally move it to a specific directory. If the file already
# exists, it simply uses the local path.
def download_model(model_name, repo_id, filename, local_dir="data/"):
    """Download a model file if it doesn't exist locally.

    Args:
        model_name (str): Human-readable name of the model
        repo_id (str): Hugging Face repository ID
        filename (str): Target filename to download
        local_dir (str): Directory to store downloaded files

    Returns:
        str: Path to the model file
    """
    # Ensure the data directory exists
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        print(f"Using locally cached {model_name} model: {local_path}")
        return local_path

    print(f"Downloading {model_name} model from {repo_id}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=local_dir,
        )
        print(f"Successfully downloaded {model_name} to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        if os.path.exists(local_path + ".part"):
            print(f"Removing partial download: {local_path}.part")
            os.remove(local_path + ".part")
        raise


# Model configuration
MODEL_CONFIG = {
    "photomaker": {
        "name": "PhotoMaker-V2",
        "repo_id": "TencentARC/PhotoMaker-V2",
        "filename": "photomaker-v2.bin",
    }
}

# Download PhotoMaker model
photomaker_path = download_model(
    model_name=MODEL_CONFIG["photomaker"]["name"],
    repo_id=MODEL_CONFIG["photomaker"]["repo_id"],
    filename=MODEL_CONFIG["photomaker"]["filename"],
)

# Maximum seed value for random number generation
MAX_SEED = np.iinfo(np.int32).max


def setup_seed(seed, device_caps=None):
    """Set up random seeds for reproducibility across all libraries.

    Args:
        seed (int): The seed value to use
        device_caps (dict, optional): Device capabilities dictionary. If None, will be determined.
    """
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


def get_status_update(is_finished=False):
    """Get a status update message for the UI.

    Args:
        is_finished (bool): Whether the generation is finished

    Returns:
        gr.update: A Gradio update object with the status message
    """
    if is_finished:
        return gr.update(visible=True, value="<h3>Generation Finished ✅</h3>")
    else:
        return gr.update(
            visible=True,
            value="<h3><span style='color:#FF6700'>⚙️ Generating...</span> The intermediate results will be shown.</h3>",
        )


def get_image_paths(folder_path, extensions=(".jpg", ".jpeg", ".png", ".webp", ".bmp")):
    """Get a sorted list of image paths from a folder.

    Args:
        folder_path (str): Path to the folder containing images
        extensions (tuple): File extensions to include

    Returns:
        list: Sorted list of full paths to images
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    # Get all files and filter by extension
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            image_paths.append(os.path.join(folder_path, filename))

    # Sort paths naturally (so 1.jpg comes before 10.jpg)
    return sorted(image_paths, key=lambda x: os.path.basename(x))


# Function aliases for backward compatibility
set_text_unfinished = lambda: get_status_update(is_finished=False)
set_text_finished = lambda: get_status_update(is_finished=True)
get_image_path_list = get_image_paths


#################################################
class SpatialAttnProcessor2_0(torch.nn.Module):
    """
    Attention processor for spatial self-attention with PyTorch 2.0 optimizations.

    This processor implements paired self-attention for consistent image generation
    across multiple frames or scenes.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        id_length=4,
        device_caps=None,
        dtype=None,
    ):
        super().__init__()

        # Verify PyTorch version compatibility
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "SpatialAttnProcessor2_0 requires PyTorch 2.0 or later for scaled_dot_product_attention. "
                "Please upgrade PyTorch to version 2.0+."
            )

        # Get device capabilities if not provided
        if device_caps is None:
            device_caps = get_device_capabilities()

        # Determine the appropriate device based on capabilities
        if device_caps["cuda"]:
            self.device = "cuda"
        elif device_caps["mps"]:
            self.device = "mps"
        else:
            self.device = "cpu"

        # Default to fp16 on GPU and fp32 on CPU/MPS if not specified
        if dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype

        # Store attention parameters
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.id_length = id_length
        self.total_length = id_length + 1

        # Initialize id_bank as an empty dictionary to store character embeddings
        self.id_bank = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """Process attention for spatial self-attention with character consistency.

        This method implements the paired self-attention mechanism with careful memory management
        to avoid OOM errors, especially on MPS devices.

        Args:
            attn: The attention module
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states
            attention_mask: Optional attention mask
            temb: Optional time embedding

        Returns:
            torch.Tensor: Processed hidden states
        """
        # Access necessary globals
        global total_count, attn_count, cur_step, indices1024, indices4096
        global sa32, sa64, write, height, width, cur_character

        # Determine resolution type once
        is_1024_resolution = hidden_states.shape[1] == (height // 32) * (width // 32)

        # Clear memory before operations
        clear_memory()

        # Initialize attention indices if needed - only at the beginning of generation
        if attn_count == 0 and cur_step == 0:
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )
            clear_memory()  # Clear memory after creating indices

        # Choose appropriate indices based on input dimensions
        indices = indices1024 if is_1024_resolution else indices4096

        # Process based on write or read mode with careful memory management
        if write:
            # Validate character count in write mode
            if not cur_character or len(cur_character) != 1:
                raise ValueError(
                    f"Expected exactly 1 character in write mode, got {len(cur_character)}"
                )

            # Extract character for easier access
            character = cur_character[0]

            # Extract dimensions for easier reference
            batch_size, num_tokens, channels = hidden_states.shape
            img_count = batch_size // 2

            # Initialize character entry in id_bank if needed
            if character not in self.id_bank:
                self.id_bank[character] = {}

            # Process each image position separately to reduce peak memory usage
            self.id_bank[character][cur_step] = []

            # Reshape once for efficiency
            hidden_states_reshaped = hidden_states.reshape(
                -1, img_count, num_tokens, channels
            )

            for img_idx in range(img_count):
                # Extract only the needed embeddings for this position
                position_embedding = (
                    hidden_states_reshaped[:, img_idx, indices[img_idx], :]
                    .reshape(2, -1, channels)
                    .clone()
                )
                self.id_bank[character][cur_step].append(position_embedding)

                # Move to CPU immediately if using CUDA/MPS to free GPU memory
                if str(self.device) != "cpu":
                    self.id_bank[character][cur_step][-1] = self.id_bank[character][
                        cur_step
                    ][-1].cpu()

                # Clear intermediates
                del position_embedding
                clear_memory()

            # Use original hidden_states to avoid storing the reshaped version
            del hidden_states_reshaped
            clear_memory()
        else:
            # Reading mode: collect character embeddings
            encoder_arr = []
            for character in cur_character:
                if character in self.id_bank and cur_step in self.id_bank[character]:
                    # Load tensors to device one by one to avoid OOM
                    for tensor in self.id_bank[character][cur_step]:
                        encoder_arr.append(tensor.to(self.device))
                        clear_memory()  # Clear after each transfer

        # Apply attention based on current step
        if cur_step < 1:
            # Always use standard attention for initial steps
            hidden_states = self.__call2__(
                attn, hidden_states, None, attention_mask, temb
            )
        else:
            # Decide between paired and standard attention
            attention_threshold = 0.3 if cur_step < 20 else 0.1
            use_paired_attention = random.random() > attention_threshold

            if use_paired_attention:
                if write:
                    # Write mode paired attention with incremental processing
                    batch_size, num_tokens, channels = hidden_states.shape
                    img_count = batch_size // 2

                    # Process each batch position separately to reduce memory usage
                    hidden_states_result = []

                    for img_idx in range(img_count):
                        # Reshape only the portion we need
                        current_states = hidden_states.reshape(
                            -1, img_count, num_tokens, channels
                        )[:, img_idx, :, :]

                        # Extract position embeddings
                        position_embeddings = []
                        for pos_idx in range(img_count):
                            if pos_idx != img_idx:  # Skip current position
                                pos_embed = hidden_states.reshape(
                                    -1, img_count, num_tokens, channels
                                )[:, pos_idx, indices[pos_idx], :].reshape(
                                    2, -1, channels
                                )
                                position_embeddings.append(pos_embed)

                        # Create encoder states with other positions + current full states
                        encoder_states = torch.cat(
                            position_embeddings + [current_states], dim=1
                        )

                        # Apply paired attention
                        processed_states = self.__call2__(
                            attn, current_states, encoder_states, None, temb
                        )
                        hidden_states_result.append(processed_states)

                        # Clean up intermediates
                        del position_embeddings, encoder_states, current_states
                        clear_memory()

                    # Reconstruct hidden_states from processed parts
                    hidden_states_new = torch.zeros_like(hidden_states)
                    for img_idx, processed in enumerate(hidden_states_result):
                        hidden_states_new.reshape(-1, img_count, num_tokens, channels)[
                            :, img_idx, :, :
                        ] = processed

                    hidden_states = hidden_states_new
                    del hidden_states_result, hidden_states_new
                    clear_memory()
                else:
                    # Read mode paired attention
                    _, num_tokens, channels = hidden_states.shape

                    # Process without creating unnecessary copies
                    part_0 = hidden_states.reshape(2, -1, num_tokens, channels)[
                        :, 0, :, :
                    ]

                    # Create encoder states with saved embeddings + current full states
                    encoder_states = torch.cat(encoder_arr + [part_0], dim=1)

                    # Apply paired attention and update in-place
                    processed_part = self.__call2__(
                        attn, part_0, encoder_states, None, temb
                    )
                    hidden_states.reshape(2, -1, num_tokens, channels)[:, 0, :, :] = (
                        processed_part
                    )

                    # Clean up
                    del part_0, encoder_states, processed_part, encoder_arr
                    clear_memory()
            else:
                # Use standard attention
                hidden_states = self.__call2__(
                    attn, hidden_states, None, attention_mask, temb
                )
                clear_memory()

        # Update step counters
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1

            # Recalculate indices for the next step
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )
            clear_memory()

        return hidden_states

    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """Optimized attention processing using PyTorch 2.0 scaled_dot_product_attention.

        Args:
            attn: The attention module
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask
            temb: Optional time embedding for conditioning

        Returns:
            torch.Tensor: Processed hidden states
        """
        # Save residual connection for later
        residual = hidden_states

        # Apply spatial normalization if available
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # Remember input dimensions
        input_ndim = hidden_states.ndim
        original_shape = hidden_states.shape

        # Handle 4D inputs (common for image data)
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            # Flatten spatial dimensions
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        # Extract dimensions after possible reshaping
        batch_size, sequence_length, channel = hidden_states.shape

        # Prepare attention mask if provided
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # Reshape for scaled_dot_product_attention
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        # Apply group normalization if available
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # Project query from hidden states
        query = attn.to_q(hidden_states)

        # Use input hidden states as encoder states if none provided (self-attention)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Project key and value
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Free memory after projections
        clear_memory()

        # Calculate inner dimensions
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Free memory before the expensive attention calculation
        clear_memory()

        try:
            # Use efficient scaled dot product attention (PyTorch 2.0+)
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        except RuntimeError as e:
            # Handle out of memory error by offloading to CPU, processing in chunks
            if "out of memory" in str(e).lower():
                print("Warning: OOM in attention - trying to recover...")

                # Move tensors to CPU
                query_cpu = query.cpu()
                key_cpu = key.cpu()
                value_cpu = value.cpu()
                attention_mask_cpu = (
                    attention_mask.cpu() if attention_mask is not None else None
                )

                # Clear GPU memory
                del query, key, value, attention_mask
                clear_memory()

                # Process in smaller chunks if tensors are large
                if sequence_length > 1024:
                    # Process in chunks of 512 tokens
                    chunk_size = 512
                    result_chunks = []

                    for i in range(0, sequence_length, chunk_size):
                        end_idx = min(i + chunk_size, sequence_length)
                        # Process subset of query (sequence dimension is at index 2)
                        query_chunk = query_cpu[:, :, i:end_idx, :].to(self.device)

                        # Compute attention for this chunk
                        if attention_mask_cpu is not None:
                            mask_chunk = attention_mask_cpu[:, :, i:end_idx, :].to(
                                self.device
                            )
                        else:
                            mask_chunk = None

                        # Compute chunk result
                        chunk_result = F.scaled_dot_product_attention(
                            query_chunk,
                            key_cpu.to(self.device),
                            value_cpu.to(self.device),
                            attn_mask=mask_chunk,
                            dropout_p=0.0,
                            is_causal=False,
                        )

                        # Save result and free memory
                        result_chunks.append(chunk_result.cpu())
                        del query_chunk, mask_chunk, chunk_result
                        clear_memory()

                    # Combine chunks
                    hidden_states = torch.cat(
                        [chunk.to(self.device) for chunk in result_chunks], dim=2
                    )
                    del result_chunks
                else:
                    # Process the entire tensor at once on CPU if small enough
                    with torch.no_grad():
                        hidden_states_cpu = F.scaled_dot_product_attention(
                            query_cpu,
                            key_cpu,
                            value_cpu,
                            attn_mask=attention_mask_cpu,
                            dropout_p=0.0,
                            is_causal=False,
                        )
                    hidden_states = hidden_states_cpu.to(self.device)
                    del hidden_states_cpu

                # Clean up CPU tensors
                del query_cpu, key_cpu, value_cpu, attention_mask_cpu
            else:
                # If error is not OOM, re-raise it
                raise

        # Reshape back to original format
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        # Ensure proper dtype (important for mixed precision)
        hidden_states = hidden_states.to(query.dtype)

        # Apply output projections
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Reshape back to 4D if input was 4D
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # Apply residual connection if specified
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # Apply rescaling
        hidden_states = hidden_states / attn.rescale_output_factor

        # Clear memory before returning
        clear_memory()

        return hidden_states


def set_attention_processor(unet, id_length, device_caps=None, is_ipadapter=False):
    """Configure the UNet attention processors for different layers.

    Args:
        unet: The UNet model to configure
        id_length: Number of ID images to process
        device_caps: Device capabilities dictionary (optional)
        is_ipadapter: Whether to use IP-Adapter attention

    Returns:
        dict: The configured attention processors
    """
    # Get device capabilities if not provided
    if device_caps is None:
        device_caps = get_device_capabilities()

    # Clear memory before creating new processors
    clear_memory()

    # Prepare attention processor mapping
    attn_procs = {}
    # Track counts for logging
    total_processors = 0
    spatial_processors = 0

    try:
        # Iterate through all attention blocks in the UNet
        for name in unet.attn_processors.keys():
            # Determine if this is a cross-attention layer
            is_cross_attn = not name.endswith("attn1.processor")
            cross_attention_dim = (
                None if not is_cross_attn else unet.config.cross_attention_dim
            )

            # Extract block location and determine hidden size
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(
                    name[len("up_blocks.") :].split(".")[0]
                )  # Extract block number safely
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(
                    name[len("down_blocks.") :].split(".")[0]
                )  # Extract block number safely
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                # Unknown block type - use default attention
                attn_procs[name] = AttnProcessor()
                total_processors += 1
                continue

            # Assign appropriate processor based on layer type
            if not is_cross_attn:
                # Self-attention (attn1) layers
                if name.startswith("up_blocks"):
                    # Use spatial attention for upblocks (self-attention)
                    attn_procs[name] = SpatialAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        id_length=id_length,
                        device_caps=device_caps,
                    )
                    spatial_processors += 1
                else:
                    # Use regular attention for other self-attention layers
                    attn_procs[name] = AttnProcessor()
            else:
                # Cross-attention layers
                if is_ipadapter:
                    # Use IP adapter for cross-attention if specified
                    # Determine device from capabilities
                    dtype = torch.float16 if device_caps["cuda"] else torch.float32
                    device = (
                        "cuda"
                        if device_caps["cuda"]
                        else "mps"
                        if device_caps["mps"]
                        else "cpu"
                    )

                    attn_procs[name] = IPAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1,
                        num_tokens=4,
                    ).to(device=device, dtype=dtype)
                else:
                    # Use regular attention for cross-attention
                    attn_procs[name] = AttnProcessor()

            total_processors += 1

            # Periodically clear memory during initialization
            if total_processors % 10 == 0:
                clear_memory()

        # Log information about created processors
        print(
            f"Created {total_processors} attention processors ({spatial_processors} spatial)"
        )

        # Apply processors to the UNet
        unet.set_attn_processor(copy.deepcopy(attn_procs))

        # Clear memory after setting processors
        clear_memory()

        return attn_procs

    except Exception as e:
        print(f"Error setting attention processors: {e}")
        # In case of error, ensure UNet has basic attention processors
        unet.set_attn_processor(AttnProcessor())
        raise


#################################################
#################################################
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

css = """
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
"""


def save_single_character_weights(unet, character, description, filepath):
    """
    保存 attention_processor 类中的 id_bank GPU Tensor 列表到指定文件中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重要保存到的文件路径。
    """
    weights_to_save = {}
    weights_to_save["description"] = description
    weights_to_save["character"] = character
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 将每个 Tensor 转到 CPU 并转为列表，以确保它可以被序列化
            weights_to_save[attn_name] = {}
            for step_key in attn_processor.id_bank[character].keys():
                weights_to_save[attn_name][step_key] = [
                    tensor.cpu()
                    for tensor in attn_processor.id_bank[character][step_key]
                ]
    # 使用torch.save保存权重
    torch.save(weights_to_save, filepath)


def load_single_character_weights(unet, filepath):
    """
    从指定文件中加载权重到 attention_processor 类的 id_bank 中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重文件的路径。
    """
    # 使用torch.load来读取权重
    weights_to_load = torch.load(filepath, map_location=torch.device("cpu"))
    character = weights_to_load["character"]
    description = weights_to_load["description"]
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 转移权重到GPU（如果GPU可用的话）并赋值给id_bank
            attn_processor.id_bank[character] = {}
            for step_key in weights_to_load[attn_name].keys():
                attn_processor.id_bank[character][step_key] = [
                    tensor.to(unet.device)
                    for tensor in weights_to_load[attn_name][step_key]
                ]


def save_results(unet, img_list):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"results/{timestamp}"
    weight_folder_name = f"{folder_name}/weights"
    # 创建文件夹
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(weight_folder_name)

    for idx, img in enumerate(img_list):
        file_path = os.path.join(folder_name, f"image_{idx}.png")  # 图片文件名
        img.save(file_path)
    global character_dict
    # for char in character_dict:
    #     description = character_dict[char]
    #     save_single_character_weights(unet,char,description,os.path.join(weight_folder_name, f'{char}.pt'))


#################################################
title = r"""
<h1 align="center">StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'><b>StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</b></a>.<br>
❗️❗️❗️[<b>Important</b>] Personalization steps:<br>
1️⃣ Enter a Textual Description for Character, if you add the Ref-Image, making sure to <b>follow the class word</b> you want to customize with the <b>trigger word</b>: `img`, such as: `man img` or `woman img` or `girl img`.<br>
2️⃣ Enter the prompt array, each line corrsponds to one generated image.<br>
3️⃣ Choose your preferred style template.<br>
4️⃣ Click the <b>Submit</b> button to start customizing.
"""

article = r"""

If StoryDiffusion is helpful, please help to ⭐ the <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/HVision-NKU/StoryDiffusion?style=social)](https://github.com/HVision-NKU/StoryDiffusion)
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:

```bibtex
@article{Zhou2024storydiffusion,
  title={StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation},
  author={Zhou, Yupeng and Zhou, Daquan and Cheng, Ming-Ming and Feng, Jiashi and Hou, Qibin},
  year={2024}
}
```
📋 **License**
<br>
Apache-2.0 LICENSE.

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>ypzhousdu@gmail.com</b>.
"""
version = r"""
<h3 align="center">StoryDiffusion Version 0.02 (test version)</h3>

<h5 >1. Support image ref image. (Cartoon Ref image is not support now)</h5>
<h5 >2. Support Typesetting Style and Captioning.(By default, the prompt is used as the caption for each image. If you need to change the caption, add a # at the end of each line. Only the part after the # will be added as a caption to the image.)</h5>
<h5 >3. [NC]symbol (The [NC] symbol is used as a flag to indicate that no characters should be present in the generated scene images. If you want do that, prepend the "[NC]" at the beginning of the line. For example, to generate a scene of falling leaves without any character, write: "[NC] The leaves are falling.")</h5>
<h5 align="center">Tips: </h4>
"""
#################################################
global attn_count, total_count, id_length, total_length, cur_step, cur_model_type
global write
global sa32, sa64
global height, width
attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
global attn_procs, unet
attn_procs = {}
###
write = False
###
sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
###
global pipe
global sd_model_path
pipe = None
# Initialize model with correct parameters


### LOAD Stable Diffusion Pipeline
def load_stable_diffusion_pipeline(
    model_path, model_type, device_caps=None, dtype=None
):
    """Load the Stable Diffusion XL pipeline with optimizations for the current device.

    Args:
        model_path (str): Path or HF repo ID for the model
        model_type (str): Model type (e.g., "original", "Photomaker")
        device_caps (dict, optional): Device capabilities
        dtype (torch.dtype, optional): Data type for model weights

    Returns:
        tuple: (pipeline, UNet model, model identifier)
    """
    # Get device capabilities if not provided
    if device_caps is None:
        device_caps = get_device_capabilities()

    # Determine device and dtype based on capabilities
    device = "cuda" if device_caps["cuda"] else "mps" if device_caps["mps"] else "cpu"
    if dtype is None:
        dtype = torch.float16 if device_caps["cuda"] else torch.float32

    print(f"Loading {model_type} model from {model_path} on {device} with {dtype}")

    # Clear memory before loading model
    clear_memory(device_caps)

    try:
        # Determine loading method based on file type
        is_single_file = model_path.endswith(".safetensors") or model_path.endswith(
            ".ckpt"
        )
        use_safetensors = (
            model_path.endswith(".safetensors") if is_single_file else True
        )

        # Load the model
        if is_single_file:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=dtype
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                use_safetensors=use_safetensors,
                # Add safety settings
                safety_checker=None if device == "mps" else "default",
            )

        # Move model to device
        pipe = pipe.to(device)

        # Apply device-specific optimizations
        if device == "mps":
            # Optimizations for Apple Silicon
            pipe.enable_attention_slicing(slice_size="auto")
            pipe.enable_vae_slicing()
        elif device == "cuda":
            # Optimizations for NVIDIA GPUs
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing(slice_size=1)
            pipe.enable_vae_tiling()
        else:
            # CPU optimizations
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

        # Common optimizations
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        pipe.scheduler.set_timesteps(50)

        # Extract UNet for further processing
        unet = pipe.unet
        model_identifier = f"{os.path.basename(model_path)}-{model_type}"

        # Clear memory after setup
        clear_memory(device_caps)

        return pipe, unet, model_identifier

    except Exception as e:
        print(f"Error loading model: {e}")
        # Provide a more specific error message based on the device
        if device == "mps" and "attention_processor" in str(e):
            print("MPS-specific error. Try setting PYTORCH_ENABLE_MPS_FALLBACK=1")
        elif device == "cuda" and "out of memory" in str(e):
            print("CUDA out of memory. Try a smaller model or reduce batch size.")
        raise


sd_model_path = models_dict["SDXL"]["path"]  # "SG161222/RealVisXL_V4.0"
# sd_model_path = models_dict["Unstable"]["path"]  # "SG161222/RealVisXL_V4.0"
single_files = models_dict["SDXL"]["single_files"]
use_safetensors = models_dict["SDXL"]["use_safetensors"]

# Load the pipeline with optimizations
pipe, unet, cur_model_type = load_stable_diffusion_pipeline(
    model_path=sd_model_path,
    model_type="original",
    device_caps=get_device_capabilities(),
    dtype=dtype,
)

# if single_files:
#     pipe = StableDiffusionXLPipeline.from_single_file(sd_model_path, torch_dtype=dtype)
# else:
#     pipe = StableDiffusionXLPipeline.from_pretrained(
#         sd_model_path, torch_dtype=dtype, use_safetensors=use_safetensors
#     )
# pipe = pipe.to(device)
# pipe.enable_attention_slicing()

# Device-specific optimizations
# if device == "mps":
# pipe.enable_attention_slicing(slice_size="auto")
# elif device == "cuda":
# pipe.enable_sequential_cpu_offload()
# pipe.enable_model_cpu_offload()

# pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler.set_timesteps(50)
# pipe.enable_vae_slicing()

# unet = pipe.unet
# cur_model_type = "Unstable" + "-" + "original"

# Initialize the attention processors
total_count = 0
for name in unet.attn_processors.keys():
    cross_attention_dim = (
        None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    )
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None and (name.startswith("up_blocks")):
        # Pass device_caps instead of device directly
        device_caps = get_device_capabilities()
        attn_procs[name] = SpatialAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            id_length=id_length,
            device_caps=device_caps,
        )
        total_count += 1
    else:
        attn_procs[name] = AttnProcessor()

print(f"Successfully loaded paired self-attention with {total_count} processors")
unet.set_attn_processor(copy.deepcopy(attn_procs))


global mask1024, mask4096
mask1024, mask4096 = cal_attn_mask_xl(
    total_length,
    id_length,
    sa32,
    sa64,
    height,
    width,
    device=device,
    dtype=torch.float16,
)


######### Gradio Fuction #############
def swap_to_gallery(images):
    return (
        gr.update(value=images, visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def upload_example_to_gallery(images, prompt, style, negative_prompt):
    return (
        gr.update(value=images, visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def remove_tips():
    return gr.update(visible=False)


def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [
        p.replace("{prompt}", positive) for positive in positives
    ], n + " " + negative


def change_visiale_by_model_type(_model_type):
    if _model_type == "Only Using Textual Description":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif _model_type == "Using Ref Images":
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    else:
        raise ValueError("Invalid model type", _model_type)


def load_character_files(character_files: str):
    if character_files == "":
        raise gr.Error("Please set a character file!")
    character_files_arr = character_files.splitlines()
    primarytext = []
    for character_file_name in character_files_arr:
        character_file = torch.load(
            character_file_name, map_location=torch.device("cpu")
        )
        primarytext.append(character_file["character"] + character_file["description"])
    return array2string(primarytext)


def load_character_files_on_running(unet, character_files: str):
    if character_files == "":
        return False
    character_files_arr = character_files.splitlines()
    for character_file in character_files_arr:
        load_single_character_weights(unet, character_file)
    return True


######### Image Generation ##############
def process_generation(
    _sd_type,
    _model_type,
    _upload_images,
    _num_steps,
    style_name,
    _Ip_Adapter_Strength,
    _style_strength_ratio,
    guidance_scale,
    seed_,
    sa32_,
    sa64_,
    id_length_,
    general_prompt,
    negative_prompt,
    prompt_array,
    G_height,
    G_width,
    _comic_type,
    font_choice,
    _char_files,
):  # Corrected font_choice usage
    if len(general_prompt.splitlines()) >= 3:
        raise gr.Error(
            "Support for more than three characters is temporarily unavailable due to VRAM limitations, but this issue will be resolved soon."
        )
    _model_type = "Photomaker" if _model_type == "Using Ref Images" else "original"
    if _model_type == "Photomaker" and "img" not in general_prompt:
        raise gr.Error(
            'Please add the triger word " img "  behind the class word you want to customize, such as: man img or woman img'
        )
    if _upload_images is None and _model_type != "original":
        raise gr.Error("Cannot find any input face image!")
    global sa32, sa64, id_length, total_length, attn_procs, unet, cur_model_type
    global write
    global cur_step, attn_count
    global height, width
    height = G_height
    width = G_width
    global pipe
    global sd_model_path, models_dict
    sd_model_path = models_dict[_sd_type]
    use_safe_tensor = True
    for attn_processor in pipe.unet.attn_processors.values():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            for values in attn_processor.id_bank.values():
                del values
            attn_processor.id_bank = {}
            attn_processor.id_length = id_length
            attn_processor.total_length = id_length + 1
    # gc.collect()
    # torch.cuda.empty_cache()
    clear_memory()
    if cur_model_type != _sd_type + "-" + _model_type:
        # apply the style template
        ##### load pipe
        del pipe
        # gc.collect()
        # if device == "cuda":
        #     torch.cuda.empty_cache()
        clear_memory()
        model_info = models_dict[_sd_type]
        model_info["model_type"] = _model_type
        pipe = load_models(model_info, device=device, photomaker_path=photomaker_path)
        clear_memory()
        set_attention_processor(pipe.unet, id_length_, is_ipadapter=False)
        ##### ########################
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        cur_model_type = _sd_type + "-" + _model_type
        pipe.enable_vae_slicing()
        if device != "mps":
            pipe.enable_model_cpu_offload()
    else:
        unet = pipe.unet
        # unet.set_attn_processor(copy.deepcopy(attn_procs))

    load_chars = load_character_files_on_running(unet, character_files=_char_files)

    prompts = prompt_array.splitlines()
    global \
        character_dict, \
        character_index_dict, \
        invert_character_index_dict, \
        ref_indexs_dict, \
        ref_totals
    character_dict, character_list = character_to_dict(general_prompt)

    start_merge_step = int(float(_style_strength_ratio) / 100 * _num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"start_merge_step:{start_merge_step}")
    generator = torch.Generator(device=device).manual_seed(seed_)
    sa32, sa64 = sa32_, sa64_
    id_length = id_length_
    clipped_prompts = prompts[:]
    nc_indexs = []
    for ind, prompt in enumerate(clipped_prompts):
        if "[NC]" in prompt:
            nc_indexs.append(ind)
            if ind < id_length:
                raise gr.Error(
                    f"The first {id_length} row is id prompts, cannot use [NC]!"
                )
    prompts = [
        prompt if "[NC]" not in prompt else prompt.replace("[NC]", "")
        for prompt in clipped_prompts
    ]

    prompts = [
        prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts
    ]
    print(prompts)
    # id_prompts = prompts[:id_length]
    (
        character_index_dict,
        invert_character_index_dict,
        replace_prompts,
        ref_indexs_dict,
        ref_totals,
    ) = process_original_prompt(character_dict, prompts.copy(), id_length)
    if _model_type != "original":
        input_id_images_dict = {}
        if len(_upload_images) != len(character_dict.keys()):
            raise gr.Error(
                f"You upload images({len(_upload_images)}) is not equal to the number of characters({len(character_dict.keys())})!"
            )
        for ind, img in enumerate(_upload_images):
            input_id_images_dict[character_list[ind]] = [load_image(img)]
    print(character_dict)
    print(character_index_dict)
    print(invert_character_index_dict)
    # real_prompts = prompts[id_length:]
    # if device == "cuda":
    #     torch.cuda.empty_cache()
    clear_memory()
    write = True
    cur_step = 0

    attn_count = 0
    # id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
    # print(id_prompts)
    setup_seed(seed_)
    total_results = []
    id_images = []
    results_dict = {}
    global cur_character
    if not load_chars:
        for character_key in character_dict.keys():
            cur_character = [character_key]
            ref_indexs = ref_indexs_dict[character_key]
            print(character_key, ref_indexs)
            current_prompts = [replace_prompts[ref_ind] for ref_ind in ref_indexs]
            print(current_prompts)
            setup_seed(seed_)
            generator = torch.Generator(device=device).manual_seed(seed_)
            cur_step = 0
            cur_positive_prompts, negative_prompt = apply_style(
                style_name, current_prompts, negative_prompt
            )
            if _model_type == "original":
                id_images = pipe(
                    cur_positive_prompts,
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=generator,
                ).images
            elif _model_type == "Photomaker":
                id_images = pipe(
                    cur_positive_prompts,
                    input_id_images=input_id_images_dict[character_key],
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    start_merge_step=start_merge_step,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=generator,
                ).images
            else:
                raise NotImplementedError(
                    "You should choice between original and Photomaker!",
                    f"But you choice {_model_type}",
                )

            # total_results = id_images + total_results
            # yield total_results
            print(id_images)
            for ind, img in enumerate(id_images):
                print(ref_indexs[ind])
                results_dict[ref_indexs[ind]] = img
            # real_images = []
            yield [results_dict[ind] for ind in results_dict.keys()]
    write = False
    if not load_chars:
        real_prompts_inds = [
            ind for ind in range(len(prompts)) if ind not in ref_totals
        ]
    else:
        real_prompts_inds = [ind for ind in range(len(prompts))]
    print(real_prompts_inds)

    for real_prompts_ind in real_prompts_inds:
        real_prompt = replace_prompts[real_prompts_ind]
        cur_character = get_ref_character(prompts[real_prompts_ind], character_dict)
        print(cur_character, real_prompt)
        setup_seed(seed_)
        if len(cur_character) > 1 and _model_type == "Photomaker":
            raise gr.Error(
                "Temporarily Not Support Multiple character in Ref Image Mode!"
            )
        generator = torch.Generator(device=device).manual_seed(seed_)
        cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        if _model_type == "original":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        elif _model_type == "Photomaker":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                input_id_images=(
                    input_id_images_dict[cur_character[0]]
                    if real_prompts_ind not in nc_indexs
                    else input_id_images_dict[character_list[0]]
                ),
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                start_merge_step=start_merge_step,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
                nc_flag=True if real_prompts_ind in nc_indexs else False,
            ).images[0]
        else:
            raise NotImplementedError(
                "You should choice between original and Photomaker!",
                f"But you choice {_model_type}",
            )
        yield [results_dict[ind] for ind in results_dict.keys()]
    total_results = [results_dict[ind] for ind in range(len(prompts))]
    if _comic_type != "No typesetting (default)":
        captions = prompt_array.splitlines()
        captions = [caption.replace("[NC]", "") for caption in captions]
        captions = [
            caption.split("#")[-1] if "#" in caption else caption
            for caption in captions
        ]
        font_path = os.path.join("fonts", font_choice)
        font = ImageFont.truetype(font_path, int(45))
        total_results = (
            get_comic(total_results, _comic_type, captions=captions, font=font)
            + total_results
        )
    save_results(pipe.unet, total_results)

    yield total_results


def array2string(arr):
    stringtmp = ""
    for i, part in enumerate(arr):
        if i != len(arr) - 1:
            stringtmp += part + "\n"
        else:
            stringtmp += part

    return stringtmp


#################################################
#################################################
### define the interface

with gr.Blocks(css=css) as demo:
    binary_matrixes = gr.State([])
    color_layout = gr.State([])

    # gr.Markdown(logo)
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Group(elem_id="main-image"):
            prompts = []
            colors = []

            with gr.Column(visible=True) as gen_prompt_vis:
                sd_type = gr.Dropdown(
                    choices=list(models_dict.keys()),
                    value="Unstable",
                    label="sd_type",
                    info="Select pretrained model",
                )
                model_type = gr.Radio(
                    ["Only Using Textual Description", "Using Ref Images"],
                    label="model_type",
                    value="Only Using Textual Description",
                    info="Control type of the Character",
                )
                with gr.Group(visible=False) as control_image_input:
                    files = gr.Files(
                        label="Drag (Select) 1 or more photos of your face",
                        file_types=["image"],
                    )
                    uploaded_files = gr.Gallery(
                        label="Your images",
                        visible=False,
                        columns=5,
                        rows=1,
                        height=200,
                    )
                    with gr.Column(visible=False) as clear_button:
                        remove_and_reupload = gr.ClearButton(
                            value="Remove and upload new ones",
                            components=files,
                            size="sm",
                        )
                general_prompt = gr.Textbox(
                    value="",
                    lines=2,
                    label="(1) Textual Description for Character",
                    interactive=True,
                )
                negative_prompt = gr.Textbox(
                    value="", label="(2) Negative_prompt", interactive=True
                )
                style = gr.Dropdown(
                    label="Style template",
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                )
                prompt_array = gr.Textbox(
                    lines=3,
                    value="",
                    label="(3) Comic Description (each line corresponds to a frame).",
                    interactive=True,
                )
                char_path = gr.Textbox(
                    lines=2,
                    value="",
                    visible=False,
                    label="(Optional) Character files",
                    interactive=True,
                )
                char_btn = gr.Button("Load Character files", visible=False)
                with gr.Accordion("(4) Tune the hyperparameters", open=True):
                    font_choice = gr.Dropdown(
                        label="Select Font",
                        choices=[
                            f for f in os.listdir("./fonts") if f.endswith(".ttf")
                        ],
                        value="Inkfree.ttf",
                        info="Select font for the final slide.",
                        interactive=True,
                    )
                    sa32_ = gr.Slider(
                        label=" (The degree of Paired Attention at 32 x 32 self-attention layers) ",
                        minimum=0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                    )
                    sa64_ = gr.Slider(
                        label=" (The degree of Paired Attention at 64 x 64 self-attention layers) ",
                        minimum=0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                    )
                    id_length_ = gr.Slider(
                        label="Number of id images in total images",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                    )
                    with gr.Row():
                        seed_ = gr.Slider(
                            label="Seed", minimum=-1, maximum=MAX_SEED, value=0, step=1
                        )
                        randomize_seed_btn = gr.Button("🎲", size="sm")
                    num_steps = gr.Slider(
                        label="Number of sample steps",
                        minimum=20,
                        maximum=100,
                        step=1,
                        value=35,
                    )
                    G_height = gr.Slider(
                        label="height",
                        minimum=256,
                        maximum=1024,
                        step=32,
                        value=768,
                    )
                    G_width = gr.Slider(
                        label="width",
                        minimum=256,
                        maximum=1024,
                        step=32,
                        value=768,
                    )
                    comic_type = gr.Radio(
                        [
                            "No typesetting (default)",
                            "Four Pannel",
                            "Classic Comic Style",
                        ],
                        value="Classic Comic Style",
                        label="Typesetting Style",
                        info="Select the typesetting style ",
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=5,
                    )
                    style_strength_ratio = gr.Slider(
                        label="Style strength of Ref Image (%)",
                        minimum=15,
                        maximum=50,
                        step=1,
                        value=20,
                        visible=False,
                    )
                    Ip_Adapter_Strength = gr.Slider(
                        label="Ip_Adapter_Strength",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.5,
                        visible=False,
                    )
                final_run_btn = gr.Button("Generate ! 😺")

        with gr.Column():
            out_image = gr.Gallery(label="Result", columns=2, height="auto")
            generated_information = gr.Markdown(
                label="Generation Details", value="", visible=False
            )
            gr.Markdown(version)
    model_type.change(
        fn=change_visiale_by_model_type,
        inputs=model_type,
        outputs=[control_image_input, style_strength_ratio, Ip_Adapter_Strength],
    )
    files.upload(
        fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files]
    )
    remove_and_reupload.click(
        fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files]
    )
    char_btn.click(fn=load_character_files, inputs=char_path, outputs=[general_prompt])

    randomize_seed_btn.click(
        fn=lambda: random.randint(-1, MAX_SEED),
        inputs=[],
        outputs=seed_,
    )

    final_run_btn.click(fn=set_text_unfinished, outputs=generated_information).then(
        process_generation,
        inputs=[
            sd_type,
            model_type,
            files,
            num_steps,
            style,
            Ip_Adapter_Strength,
            style_strength_ratio,
            guidance_scale,
            seed_,
            sa32_,
            sa64_,
            id_length_,
            general_prompt,
            negative_prompt,
            prompt_array,
            G_height,
            G_width,
            comic_type,
            font_choice,
            char_path,
        ],
        outputs=out_image,
    ).then(fn=set_text_finished, outputs=generated_information)

    gr.Examples(
        examples=[
            [
                0,
                0.5,
                0.5,
                2,
                "[Bob] A man, wearing a black suit\n[Alice]a woman, wearing a white shirt",
                "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
                array2string(
                    [
                        "[Bob] at home, read new paper #at home, The newspaper says there is a treasure house in the forest.",
                        "[Bob] on the road, near the forest",
                        "[Alice] is make a call at home # [Bob] invited [Alice] to join him on an adventure.",
                        "[NC]A tiger appeared in the forest, at night ",
                        "[NC] The car on the road, near the forest #They drives to the forest in search of treasure.",
                        "[Bob] very frightened, open mouth, in the forest, at night",
                        "[Alice] very frightened, open mouth, in the forest, at night",
                        "[Bob]  and [Alice] running very fast, in the forest, at night",
                        "[NC] A house in the forest, at night #Suddenly, They discovers the treasure house!",
                        "[Bob]  and [Alice]  in the house filled with  treasure, laughing, at night #He is overjoyed inside the house.",
                    ]
                ),
                "Comic book",
                "Only Using Textual Description",
                get_image_path_list("./examples/taylor"),
                768,
                768,
            ],
            [
                0,
                0.5,
                0.5,
                2,
                "[Bob] A man img, wearing a black suit\n[Alice]a woman img, wearing a white shirt",
                "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
                array2string(
                    [
                        "[Bob] at home, read new paper #at home, The newspaper says there is a treasure house in the forest.",
                        "[Bob] on the road, near the forest",
                        "[Alice] is make a call at home # [Bob] invited [Alice] to join him on an adventure.",
                        "[NC] The car on the road, near the forest #They drives to the forest in search of treasure.",
                        "[NC]A tiger appeared in the forest, at night ",
                        "[Bob] very frightened, open mouth, in the forest, at night",
                        "[Alice] very frightened, open mouth, in the forest, at night",
                        "[Bob]  running very fast, in the forest, at night",
                        "[NC] A house in the forest, at night #Suddenly, They discovers the treasure house!",
                        "[Bob]  in the house filled with  treasure, laughing, at night #They are overjoyed inside the house.",
                    ]
                ),
                "Comic book",
                "Using Ref Images",
                get_image_path_list("./examples/twoperson"),
                1024,
                1024,
            ],
            [
                1,
                0.5,
                0.5,
                3,
                "[Taylor]a woman img, wearing a white T-shirt, blue loose hair",
                "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
                array2string(
                    [
                        "[Taylor]wake up in the bed",
                        "[Taylor]have breakfast",
                        "[Taylor]is on the road, go to company",
                        "[Taylor]work in the company",
                        "[Taylor]Take a walk next to the company at noon",
                        "[Taylor]lying in bed at night",
                    ]
                ),
                "Japanese Anime",
                "Using Ref Images",
                get_image_path_list("./examples/taylor"),
                768,
                768,
            ],
            [
                0,
                0.5,
                0.5,
                3,
                "[Bob]a man, wearing black jacket",
                "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
                array2string(
                    [
                        "[Bob]wake up in the bed",
                        "[Bob]have breakfast",
                        "[Bob]is on the road, go to the company,  close look",
                        "[Bob]work in the company",
                        "[Bob]laughing happily",
                        "[Bob]lying in bed at night",
                    ]
                ),
                "Japanese Anime",
                "Only Using Textual Description",
                get_image_path_list("./examples/taylor"),
                768,
                768,
            ],
            [
                0,
                0.3,
                0.5,
                3,
                "[Kitty]a girl, wearing white shirt, black skirt, black tie, yellow hair",
                "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
                array2string(
                    [
                        "[Kitty]at home #at home, began to go to drawing",
                        "[Kitty]sitting alone on a park bench.",
                        "[Kitty]reading a book on a park bench.",
                        "[NC]A squirrel approaches, peeking over the bench. ",
                        "[Kitty]look around in the park. # She looks around and enjoys the beauty of nature.",
                        "[NC]leaf falls from the tree, landing on the sketchbook.",
                        "[Kitty]picks up the leaf, examining its details closely.",
                        "[NC]The brown squirrel appear.",
                        "[Kitty]is very happy # She is very happy to see the squirrel again",
                        "[NC]The brown squirrel takes the cracker and scampers up a tree. # She gives the squirrel cracker",
                    ]
                ),
                "Japanese Anime",
                "Only Using Textual Description",
                get_image_path_list("./examples/taylor"),
                768,
                768,
            ],
        ],
        inputs=[
            seed_,
            sa32_,
            sa64_,
            id_length_,
            general_prompt,
            negative_prompt,
            prompt_array,
            style,
            model_type,
            files,
            G_height,
            G_width,
        ],
        # outputs=[post_sketch, binary_matrixes, *color_row, *colors, *prompts, gen_prompt_vis, general_prompt, seed_],
        # run_on_click=True,
        label="😺 Examples 😺",
    )
    gr.Markdown(article)

if __name__ == "__main__":
    try:
        demo.queue(concurrency_count=1, max_size=10)  # Enable queue
        demo.launch(
            server_name="0.0.0.0", server_port=7860, share=False, enable_queue=True
        )
    except Exception as e:
        print(f"Launch error: {e}")
        # Fallback to basic launch
        demo.launch()

# demo.launch(server_name="0.0.0.0", share=True)
