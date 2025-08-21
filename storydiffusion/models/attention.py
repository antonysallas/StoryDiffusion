"""
Custom attention processors for maintaining character consistency.

This module implements the Paired Attention mechanism that allows StoryDiffusion
to generate consistent characters across multiple images.
"""

import torch
import torch.nn.functional as F
import random
import copy
from typing import Optional, Dict, Any, List
from utils.gradio_utils import (
    cal_attn_indice_xl_effcient_memory,
    is_torch2_available,
    AttnProcessor
)

if is_torch2_available():
    from utils.gradio_utils import AttnProcessor2_0 as BaseAttnProcessor
else:
    BaseAttnProcessor = AttnProcessor


class SpatialAttnProcessor2_0(torch.nn.Module):
    """
    Custom spatial attention processor for maintaining character consistency across generated images.

    This processor implements the Paired Attention mechanism that allows StoryDiffusion to generate
    consistent characters across multiple images. It works by:

    1. Storing character-specific attention features during initial generation (write mode)
    2. Retrieving and applying these features during subsequent generations (read mode)
    3. Using efficient memory indexing to handle different attention resolutions (32x32, 64x64)

    The processor maintains an id_bank that stores attention features for each character at each
    denoising step, enabling consistent character representation across the image sequence.

    Args:
        hidden_size (int, optional): The hidden size of the attention layer
        cross_attention_dim (int, optional): The number of channels in the encoder_hidden_states
        id_length (int): Number of character reference images to process
        device (torch.device): Compute device (cuda, mps, or cpu)
        dtype (torch.dtype): Data type for tensors (default: float16 for efficiency)
    """

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        id_length: int = 4,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        # Ensure PyTorch 2.0+ is available for scaled_dot_product_attention
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1  # Total attention length including self-attention
        self.id_length = id_length  # Number of character reference images
        self.id_bank: Dict[str, Dict[int, List[torch.Tensor]]] = {}  # Storage for character-specific attention features

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Main attention processing method that handles both storing (write mode) and retrieving (read mode)
        character-specific attention features.

        Args:
            attn: The attention module being processed
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask
            temb: Optional time embedding

        Returns:
            torch.Tensor: Processed hidden states with character consistency applied
        """
        # Access global state
        from ..config import global_state
        
        # Initialize attention indices for different resolutions on first step
        if global_state.attn_count == 0 and global_state.cur_step == 0:
            global_state.indices1024, global_state.indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                global_state.sa32,
                global_state.sa64,
                global_state.height,
                global_state.width,
                device=self.device,
                dtype=self.dtype,
            )

        # Write mode: Store character-specific attention features
        if global_state.write:
            assert len(global_state.cur_character) == 1  # Currently processing one character at a time

            # Select appropriate indices based on attention resolution
            if hidden_states.shape[1] == (global_state.height // 32) * (global_state.width // 32):
                indices = global_state.indices1024  # 32x32 resolution
            else:
                indices = global_state.indices4096  # 64x64 resolution

            # Reshape hidden states to separate batch and image dimensions
            total_batch_size, nums_token, channel = hidden_states.shape
            img_nums = total_batch_size // 2  # Unconditional + conditional
            hidden_states = hidden_states.reshape(-1, img_nums, nums_token, channel)

            # Initialize character storage if needed
            if global_state.cur_character[0] not in self.id_bank:
                self.id_bank[global_state.cur_character[0]] = {}

            # Store attention features for current character and step
            self.id_bank[global_state.cur_character[0]][global_state.cur_step] = [
                hidden_states[:, img_ind, indices[img_ind], :]
                .reshape(2, -1, channel)
                .clone()
                for img_ind in range(img_nums)
            ]
            hidden_states = hidden_states.reshape(-1, nums_token, channel)
        else:
            # Read mode: Retrieve stored character features for consistency
            encoder_arr = []
            for character in global_state.cur_character:
                encoder_arr = encoder_arr + [
                    tensor.to(self.device)
                    for tensor in self.id_bank[character][global_state.cur_step]
                ]
        
        # Process attention based on denoising step
        if global_state.cur_step < 1:
            # First step: standard attention without character features
            hidden_states = self.__call2__(
                attn, hidden_states, None, attention_mask, temb
            )
        else:
            # Later steps: apply character consistency with probability
            random_number = random.random()
            # Higher probability of applying consistency in early steps
            if global_state.cur_step < 20:
                rand_num = 0.3  # 70% chance of applying consistency
            else:
                rand_num = 0.1  # 90% chance of applying consistency

            if random_number > rand_num:
                if hidden_states.shape[1] == (global_state.height // 32) * (global_state.width // 32):
                    indices = global_state.indices1024
                else:
                    indices = global_state.indices4096
                
                if global_state.write:
                    total_batch_size, nums_token, channel = hidden_states.shape
                    img_nums = total_batch_size // 2
                    hidden_states = hidden_states.reshape(
                        -1, img_nums, nums_token, channel
                    )
                    encoder_arr = [
                        hidden_states[:, img_ind, indices[img_ind], :].reshape(
                            2, -1, channel
                        )
                        for img_ind in range(img_nums)
                    ]
                    for img_ind in range(img_nums):
                        img_ind_list = [i for i in range(img_nums)]
                        img_ind_list.remove(img_ind)
                        encoder_hidden_states_tmp = torch.cat(
                            [encoder_arr[img_ind] for img_ind in img_ind_list]
                            + [hidden_states[:, img_ind, :, :]],
                            dim=1,
                        )

                        hidden_states[:, img_ind, :, :] = self.__call2__(
                            attn,
                            hidden_states[:, img_ind, :, :],
                            encoder_hidden_states_tmp,
                            None,
                            temb,
                        )
                else:
                    _, nums_token, channel = hidden_states.shape
                    hidden_states = hidden_states.reshape(2, -1, nums_token, channel)
                    encoder_hidden_states_tmp = torch.cat(
                        encoder_arr + [hidden_states[:, 0, :, :]], dim=1
                    )
                    hidden_states[:, 0, :, :] = self.__call2__(
                        attn,
                        hidden_states[:, 0, :, :],
                        encoder_hidden_states_tmp,
                        None,
                        temb,
                    )
                hidden_states = hidden_states.reshape(-1, nums_token, channel)
            else:
                hidden_states = self.__call2__(
                    attn, hidden_states, None, attention_mask, temb
                )
        
        global_state.attn_count += 1
        if global_state.attn_count == global_state.total_count:
            global_state.attn_count = 0
            global_state.cur_step += 1
            global_state.indices1024, global_state.indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                global_state.sa32,
                global_state.sa64,
                global_state.height,
                global_state.width,
                device=self.device,
                dtype=self.dtype,
            )

        return hidden_states

    def __call2__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Core attention computation using PyTorch 2.0's scaled_dot_product_attention.

        This method performs the actual attention computation, handling:
        - Query, key, value projections
        - Multi-head attention with scaled dot product
        - Optional spatial normalization and group normalization
        - Residual connections

        Args:
            attn: The attention module containing projection layers
            hidden_states: Input hidden states
            encoder_hidden_states: Optional encoder states for cross-attention
            attention_mask: Optional attention mask
            temb: Optional time embedding

        Returns:
            torch.Tensor: Attention output with residual connection
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_attention_processor(unet, id_length: int, is_ipadapter: bool = False) -> None:
    """
    Configure attention processors for the UNet model to enable character consistency.

    This function sets up custom attention processors for specific layers in the UNet:
    - SpatialAttnProcessor2_0 for up_blocks self-attention layers (character consistency)
    - Standard AttnProcessor for other layers
    - Optional IPAttnProcessor2_0 for IP-Adapter support

    Args:
        unet: The UNet model to configure
        id_length (int): Number of character reference images
        is_ipadapter (bool): Whether to use IP-Adapter processors for cross-attention
    """
    from ..config import global_state
    
    attn_procs = {}

    for name in unet.attn_processors.keys():
        # Determine if this is self-attention or cross-attention
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )

        # Calculate hidden size based on block location
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        # Assign appropriate processor based on layer type
        if cross_attention_dim is None:
            # Self-attention layers
            if name.startswith("up_blocks"):
                # Use custom processor for character consistency in up_blocks
                attn_procs[name] = SpatialAttnProcessor2_0(
                    id_length=id_length,
                    device=unet.device,
                    dtype=torch.float16
                )
                global_state.total_count += 1
            else:
                attn_procs[name] = BaseAttnProcessor()
        else:
            # Cross-attention layers
            if is_ipadapter:
                # Note: IPAttnProcessor2_0 would need to be imported/implemented
                # For now, use standard processor
                attn_procs[name] = BaseAttnProcessor()
            else:
                attn_procs[name] = BaseAttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))
    global_state.attn_procs = attn_procs