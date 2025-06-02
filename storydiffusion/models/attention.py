"""
Attention processors for StoryDiffusion.
This includes the specialized SpatialAttnProcessor for consistent character generation.
"""

import copy
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..device import clear_memory, get_device_capabilities

# Import standard attention processor based on PyTorch version
if hasattr(F, "scaled_dot_product_attention"):
    # PyTorch 2.0+
    from utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
    TORCH2_AVAILABLE = True
else:
    # PyTorch < 2.0
    from utils.gradio_utils import AttnProcessor
    TORCH2_AVAILABLE = False


class SpatialAttnProcessor2_0(torch.nn.Module):
    """
    Attention processor for spatial self-attention with PyTorch 2.0 optimizations.

    This processor implements paired self-attention for consistent image generation
    across multiple frames or scenes.
    """

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        id_length: int = 4,
        device_caps: Optional[Dict[str, bool]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Verify PyTorch version compatibility
        if not TORCH2_AVAILABLE:
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
            from utils.gradio_utils import cal_attn_indice_xl_effcient_memory
            
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
                raise ValueError(f"Expected exactly 1 character in write mode, got {len(cur_character)}")

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
            hidden_states_reshaped = hidden_states.reshape(-1, img_count, num_tokens, channels)

            for img_idx in range(img_count):
                # Extract only the needed embeddings for this position
                position_embedding = (
                    hidden_states_reshaped[:, img_idx, indices[img_idx], :].reshape(2, -1, channels).clone()
                )
                self.id_bank[character][cur_step].append(position_embedding)

                # Move to CPU immediately if using CUDA/MPS to free GPU memory
                if str(self.device) != "cpu":
                    self.id_bank[character][cur_step][-1] = self.id_bank[character][cur_step][-1].cpu()

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
            hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
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
                        current_states = hidden_states.reshape(-1, img_count, num_tokens, channels)[
                            :, img_idx, :, :
                        ]

                        # Extract position embeddings
                        position_embeddings = []
                        for pos_idx in range(img_count):
                            if pos_idx != img_idx:  # Skip current position
                                pos_embed = hidden_states.reshape(-1, img_count, num_tokens, channels)[
                                    :, pos_idx, indices[pos_idx], :
                                ].reshape(2, -1, channels)
                                position_embeddings.append(pos_embed)

                        # Create encoder states with other positions + current full states
                        encoder_states = torch.cat(position_embeddings + [current_states], dim=1)

                        # Apply paired attention
                        processed_states = self.__call2__(attn, current_states, encoder_states, None, temb)
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
                    part_0 = hidden_states.reshape(2, -1, num_tokens, channels)[:, 0, :, :]

                    # Create encoder states with saved embeddings + current full states
                    encoder_states = torch.cat(encoder_arr + [part_0], dim=1)

                    # Apply paired attention and update in-place
                    processed_part = self.__call2__(attn, part_0, encoder_states, None, temb)
                    hidden_states.reshape(2, -1, num_tokens, channels)[:, 0, :, :] = processed_part

                    # Clean up
                    del part_0, encoder_states, processed_part, encoder_arr
                    clear_memory()
            else:
                # Use standard attention
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
                clear_memory()

        # Update step counters
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1

            # Recalculate indices for the next step
            from utils.gradio_utils import cal_attn_indice_xl_effcient_memory
            
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
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # Extract dimensions after possible reshaping
        batch_size, sequence_length, channel = hidden_states.shape

        # Prepare attention mask if provided
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # Reshape for scaled_dot_product_attention
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # Apply group normalization if available
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

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

        # Use efficient scaled dot product attention with OOM handling
        try:
            # Use efficient scaled dot product attention (PyTorch 2.0+)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        except RuntimeError as e:
            # Handle out of memory error by offloading to CPU, processing in chunks
            if "out of memory" in str(e).lower():
                print("Warning: OOM in attention - trying to recover...")

                # Move tensors to CPU
                query_cpu = query.cpu()
                key_cpu = key.cpu()
                value_cpu = value.cpu()
                attention_mask_cpu = attention_mask.cpu() if attention_mask is not None else None

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
                            mask_chunk = attention_mask_cpu[:, :, i:end_idx, :].to(self.device)
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
                    hidden_states = torch.cat([chunk.to(self.device) for chunk in result_chunks], dim=2)
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
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # Ensure proper dtype (important for mixed precision)
        hidden_states = hidden_states.to(query.dtype)

        # Apply output projections
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Reshape back to 4D if input was 4D
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

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
            cross_attention_dim = None if not is_cross_attn else unet.config.cross_attention_dim

            # Extract block location and determine hidden size
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0])  # Extract block number safely
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])  # Extract block number safely
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
                    # Import only when needed
                    from utils.gradio_utils import IPAttnProcessor2_0
                    
                    # Determine device from capabilities
                    dtype = torch.float16 if device_caps["cuda"] else torch.float32
                    device = "cuda" if device_caps["cuda"] else "mps" if device_caps["mps"] else "cpu"

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
        print(f"Created {total_processors} attention processors ({spatial_processors} spatial)")

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