"""
Gradio Utilities for StoryDiffusion

This module provides essential utilities for the StoryDiffusion Gradio application, including:

1. Custom Attention Processors:
   - SpatialAttnProcessor2_0: Implements Paired Attention for character consistency
   - AttnProcessor: Standard attention processor for non-character layers
   - AttnProcessor2_0: PyTorch 2.0 optimized attention with scaled_dot_product_attention

2. Attention Mask Utilities:
   - Functions to calculate and apply attention masks for different resolutions
   - Memory-efficient indexing for attention operations

3. Character Processing Utilities:
   - Parse character descriptions from prompts
   - Map characters to their appearances in the story
   - Handle character references and replacements

The module is designed to work with both standard Stable Diffusion and SDXL models,
providing the core functionality for maintaining character consistency across generated images.
"""

from calendar import c
from operator import invert
from webbrowser import get
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

class SpatialAttnProcessor2_0(torch.nn.Module):
    """
    Spatial Attention Processor for implementing Paired Attention in StoryDiffusion.
    
    This processor is the core component that enables consistent character generation across
    multiple images. It works by storing attention features from character reference images
    and applying them during story image generation.
    
    Key features:
    - Stores character-specific attention patterns in id_bank
    - Applies stored patterns with configurable probability
    - Supports different attention resolutions (16x16, 32x32, 64x64)
    - Uses PyTorch 2.0's efficient scaled_dot_product_attention
    
    Args:
        hidden_size (int, optional): Hidden dimension of attention layer
        cross_attention_dim (int, optional): Dimension for cross-attention operations
        id_length (int): Number of character reference images (default: 4)
        device (str): Compute device (default: "cuda")
        dtype (torch.dtype): Data type for tensors (default: float16)
    """

    def __init__(self, hidden_size = None, cross_attention_dim=None,id_length = 4,device = "cuda",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1  # Total attention length (refs + current)
        self.id_length = id_length  # Number of reference images
        self.id_bank = {}  # Storage for character attention features

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        """
        Process attention with character consistency logic.
        
        This method handles both storing character features (write mode) and applying
        them for consistency (read mode). It uses probabilistic application based on
        the denoising step to balance consistency with generation quality.
        
        Args:
            attn: Attention module containing projection layers
            hidden_states: Input hidden states [batch, seq_len, dim]
            encoder_hidden_states: Optional encoder states for cross-attention
            attention_mask: Optional attention mask
            temb: Time embedding for diffusion process
            
        Returns:
            torch.Tensor: Processed hidden states with character consistency
        """
        # Access global control variables
        global total_count,attn_count,cur_step,mask256,mask1024,mask4096
        global sa16, sa32, sa64
        global write
        
        # Write mode: Store character reference features
        if write:
            # Split hidden states into reference and non-reference parts
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            # Read mode: Prepare encoder states with stored character features
            encoder_hidden_states = torch.cat(self.id_bank[cur_step][0],hidden_states[:1],self.id_bank[cur_step][1],hidden_states[1:])
        
        # Apply different attention strategies based on denoising step
        if cur_step < 5:
            # Early steps: Use standard attention without character injection
            hidden_states = self.__call2__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
        else:
            # Later steps: Probabilistically apply character consistency
            random_number = random.random()
            # Higher probability of consistency in middle steps (0.7) vs late steps (0.9)
            if cur_step < 20:
                rand_num = 0.3  # 70% chance of applying consistency
            else:
                rand_num = 0.1  # 90% chance of applying consistency
                
            if random_number > rand_num:
                # Apply character consistency with appropriate mask
                if not write:
                    # Read mode: Use mask for non-reference tokens
                    if hidden_states.shape[1] == 32 * 32:
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    elif hidden_states.shape[1] == 16 * 16:
                        attention_mask = mask256[mask256.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    # Write mode: Use mask for reference tokens
                    if hidden_states.shape[1] == 32 * 32:
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length]
                    elif hidden_states.shape[1] == 16 * 16:
                        attention_mask = mask256[:mask256.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length]
                # Use specialized attention with character injection
                hidden_states = self.__call1__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
            else:
                # Standard attention without character injection
                hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        
        # Update attention counter and step
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            # Recalculate attention masks for next step
            mask256,mask1024,mask4096 = cal_attn_mask(self.total_length,self.id_length,sa16,sa32,sa64, device=self.device, dtype= self.dtype)

        return hidden_states
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """
        Specialized attention computation for character consistency injection.
        
        This method reshapes hidden states to process multiple images together,
        allowing character features to be shared across the batch. It uses
        PyTorch 2.0's scaled_dot_product_attention for efficiency.
        
        Args:
            attn: Attention module
            hidden_states: Input features to process
            encoder_hidden_states: Character features to inject (unused in this version)
            attention_mask: Mask for controlling attention patterns
            temb: Time embedding
            
        Returns:
            torch.Tensor: Attention output with character features applied
        """
        residual = hidden_states
        if encoder_hidden_states is not None:
            raise Exception("not implement")
            
        # Apply spatial normalization if available
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        # Reshape 4D input to 3D for attention computation
        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
            
        # Reshape to process multiple images together
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2  # Unconditional + conditional
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(-1, img_nums * nums_token, channel)

        batch_size, sequence_length, _ = hidden_states.shape

        # Apply group normalization if available
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

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

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # if input_ndim == 4:
        #     tile_hidden_states = tile_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     tile_hidden_states = tile_hidden_states + residual

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        """
        Standard attention computation without character injection.
        
        This method performs regular self-attention or cross-attention without
        any character consistency modifications. Used for early denoising steps
        or when character injection is not applied.
        
        Args:
            attn: Attention module with projection layers
            hidden_states: Input features [batch, seq_len, dim]
            encoder_hidden_states: Optional encoder features for cross-attention
            attention_mask: Optional mask for attention weights
            temb: Time embedding for conditional generation
            
        Returns:
            torch.Tensor: Standard attention output
        """
        residual = hidden_states

        # Apply spatial normalization if available
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # Handle 4D input (batch, channel, height, width)
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # Determine sequence length from appropriate source
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # Apply group normalization
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Compute query projection
        query = attn.to_q(hidden_states)

        # Handle encoder hidden states for cross-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # Self-attention
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Compute key and value projections
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def cal_attn_mask(total_length,id_length,sa16,sa32,sa64,device="cuda",dtype= torch.float16):
    """
    Calculate attention masks for different resolutions in standard Stable Diffusion.
    
    This function creates boolean masks that control which tokens can attend to each other
    during the attention computation. The masks ensure that:
    1. Each token always attends to itself
    2. Reference tokens (first id_length) are masked appropriately
    3. Random dropout is applied based on sa (self-attention) parameters
    
    Args:
        total_length (int): Total number of images (references + generated)
        id_length (int): Number of reference images
        sa16 (float): Self-attention probability for 16x16 resolution (256 tokens)
        sa32 (float): Self-attention probability for 32x32 resolution (1024 tokens)
        sa64 (float): Self-attention probability for 64x64 resolution (4096 tokens)
        device (str): Compute device (default: "cuda")
        dtype (torch.dtype): Data type for masks (default: float16)
        
    Returns:
        tuple: (mask256, mask1024, mask4096) - Attention masks for each resolution
    """
    # Create random boolean matrices based on attention probabilities
    bool_matrix256 = torch.rand((1, total_length * 256),device = device,dtype = dtype) < sa16
    bool_matrix1024 = torch.rand((1, total_length * 1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * 4096),device = device,dtype = dtype) < sa64
    
    # Repeat for each position in sequence
    bool_matrix256 = bool_matrix256.repeat(total_length,1)
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    
    # Configure attention patterns
    for i in range(total_length):
        # Mask out tokens after reference images
        bool_matrix256[i:i+1,id_length*256:] = False
        bool_matrix1024[i:i+1,id_length*1024:] = False
        bool_matrix4096[i:i+1,id_length*4096:] = False
        
        # Ensure each position attends to itself
        bool_matrix256[i:i+1,i*256:(i+1)*256] = True
        bool_matrix1024[i:i+1,i*1024:(i+1)*1024] = True
        bool_matrix4096[i:i+1,i*4096:(i+1)*4096] = True
    
    # Reshape masks for attention computation
    mask256 = bool_matrix256.unsqueeze(1).repeat(1,256,1).reshape(-1,total_length * 256)
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,1024,1).reshape(-1,total_length * 1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,4096,1).reshape(-1,total_length * 4096)
    
    return mask256,mask1024,mask4096

def cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width,device="cuda",dtype= torch.float16):
    """
    Calculate attention masks for SDXL models with variable image dimensions.
    
    Unlike standard SD which has fixed token counts, SDXL's token count varies with
    image dimensions. This function adapts the mask calculation to handle arbitrary
    height and width while maintaining the same attention pattern logic.
    
    Args:
        total_length (int): Total number of images (references + generated)
        id_length (int): Number of reference images
        sa32 (float): Self-attention probability for 32x32 blocks
        sa64 (float): Self-attention probability for 64x64 blocks (higher resolution)
        height (int): Image height in pixels
        width (int): Image width in pixels
        device (str): Compute device (default: "cuda")
        dtype (torch.dtype): Data type for masks (default: float16)
        
    Returns:
        tuple: (mask1024, mask4096) - Attention masks for each resolution level
    """
    # Calculate number of tokens based on image dimensions
    nums_1024 = (height // 32) * (width // 32)  # Tokens at 32x32 resolution
    nums_4096 = (height // 16) * (width // 16)  # Tokens at 16x16 resolution
    
    # Create random boolean matrices
    bool_matrix1024 = torch.rand((1, total_length * nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * nums_4096),device = device,dtype = dtype) < sa64
    
    # Repeat for each position
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    
    # Configure attention patterns
    for i in range(total_length):
        # Mask out tokens after reference images
        bool_matrix1024[i:i+1,id_length*nums_1024:] = False
        bool_matrix4096[i:i+1,id_length*nums_4096:] = False
        
        # Ensure self-attention for current position
        bool_matrix1024[i:i+1,i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1,i*nums_4096:(i+1)*nums_4096] = True
    
    # Reshape for attention computation
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,nums_1024,1).reshape(-1,total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,nums_4096,1).reshape(-1,total_length * nums_4096)
    
    return mask1024,mask4096


def cal_attn_indice_xl_effcient_memory(total_length,id_length,sa32,sa64,height,width,device="cuda",dtype= torch.float16):
    """
    Memory-efficient attention index calculation for SDXL models.
    
    Instead of creating large attention masks, this function generates indices of
    positions that should be attended to. This significantly reduces memory usage
    for high-resolution images by storing only the active attention positions.
    
    Args:
        total_length (int): Total number of images
        id_length (int): Number of reference images
        sa32 (float): Self-attention probability for lower resolution
        sa64 (float): Self-attention probability for higher resolution
        height (int): Image height in pixels
        width (int): Image width in pixels
        device (str): Compute device
        dtype (torch.dtype): Data type
        
    Returns:
        tuple: (indices1024, indices4096) - Lists of indices for each resolution
               Each element is a tensor of positions to attend to
    """
    # Calculate token counts for each resolution
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    
    # Generate random attention patterns
    bool_matrix1024 = torch.rand((total_length,nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((total_length,nums_4096),device = device,dtype = dtype) < sa64
    
    # Extract indices of True values (positions to attend to)
    # This is more memory efficient than storing full masks
    indices1024 = [torch.nonzero(bool_matrix1024[i], as_tuple=True)[0] for i in range(total_length)]
    indices4096 = [torch.nonzero(bool_matrix4096[i], as_tuple=True)[0] for i in range(total_length)]

    return indices1024,indices4096


class AttnProcessor(nn.Module):
    """
    Default attention processor for standard attention computations.
    
    This is the basic attention processor used for layers that don't require
    character consistency modifications. It implements standard multi-head
    attention with support for both self-attention and cross-attention.
    
    Compatible with PyTorch < 2.0 using traditional attention computation
    instead of scaled_dot_product_attention.
    
    Args:
        hidden_size (int, optional): Hidden dimension size
        cross_attention_dim (int, optional): Cross-attention dimension
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    """
    Optimized attention processor using PyTorch 2.0's scaled_dot_product_attention.
    
    This processor leverages PyTorch 2.0's efficient attention implementation which
    includes optimizations like Flash Attention when available. It provides the same
    functionality as AttnProcessor but with better performance.
    
    Used for standard attention layers that don't require character consistency.
    
    Args:
        hidden_size (int, optional): Hidden dimension size
        cross_attention_dim (int, optional): Cross-attention dimension
        
    Raises:
        ImportError: If PyTorch version < 2.0
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

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

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def is_torch2_available():
    """
    Check if PyTorch 2.0+ is available.
    
    Returns:
        bool: True if scaled_dot_product_attention is available, False otherwise
    """
    return hasattr(F, "scaled_dot_product_attention")


def character_to_dict(general_prompt):
    """
    Parse character descriptions from the general prompt text.
    
    Extracts character identifiers (enclosed in square brackets) and their
    descriptions from a multi-line prompt. Format expected:
    [Character1] Description of character 1
    [Character2] Description of character 2
    
    Args:
        general_prompt (str): Multi-line string with character definitions
        
    Returns:
        tuple: (character_dict, character_list)
            - character_dict: Maps character tags to descriptions
            - character_list: Ordered list of character tags
            
    Raises:
        gr.Error: If duplicate character tags are found
    """
    character_dict = {}    
    generate_prompt_arr = general_prompt.splitlines()
    character_index_dict = {}
    invert_character_index_dict = {}
    character_list = []
    
    for ind, string in enumerate(generate_prompt_arr):
        # Find character tag enclosed in square brackets
        start = string.find('[')
        end = string.find(']')
        if start != -1 and end != -1:
            key = string[start:end+1]  # Include brackets in key
            value = string[end+1:]  # Everything after closing bracket
            
            # Remove caption part after # if present
            if "#" in value:
                value = value.rpartition('#')[0] 
                
            # Check for duplicate character definitions
            if key in character_dict:
                raise gr.Error("duplicate character descirption: " + key)
                
            character_dict[key] = value
            character_list.append(key)
        
    return character_dict, character_list 

def get_id_prompt_index(character_dict,id_prompts):
    """
    Map character references in ID prompts to their indices.
    
    This function processes the reference prompts to:
    1. Find which prompts contain which characters
    2. Replace character tags with their descriptions
    3. Create bidirectional mappings for lookup
    
    Args:
        character_dict (dict): Maps character tags to descriptions
        id_prompts (list): List of reference prompt strings
        
    Returns:
        tuple: (character_index_dict, invert_character_index_dict, replace_id_prompts)
            - character_index_dict: Maps characters to prompt indices
            - invert_character_index_dict: Maps prompt indices to characters
            - replace_id_prompts: Prompts with character tags replaced
    """
    replace_id_prompts = []
    character_index_dict = {}
    invert_character_index_dict = {}
    
    for ind, id_prompt in enumerate(id_prompts):
        for key in character_dict.keys():
            if key in id_prompt:
                # Track which prompts contain this character
                if key not in character_index_dict:
                    character_index_dict[key] = []
                character_index_dict[key].append(ind)
                
                # Reverse mapping for quick lookup
                invert_character_index_dict[ind] = key
                
                # Replace character tag with description
                replace_id_prompts.append(id_prompt.replace(key, character_dict[key]))

    return character_index_dict, invert_character_index_dict, replace_id_prompts

def get_cur_id_list(real_prompt,character_dict,character_index_dict):
    """
    Get list of character indices present in a prompt and replace character tags.
    
    Args:
        real_prompt (str): Prompt text that may contain character tags
        character_dict (dict): Maps character tags to descriptions
        character_index_dict (dict): Maps characters to their reference indices
        
    Returns:
        tuple: (list_arr, real_prompt)
            - list_arr: Indices of characters found in the prompt
            - real_prompt: Prompt with character tags replaced by descriptions
    """
    list_arr = []
    for keys in character_index_dict.keys():
        if keys in real_prompt:
            # Add all indices associated with this character
            list_arr = list_arr + character_index_dict[keys]
            # Replace character tag with description
            real_prompt = real_prompt.replace(keys, character_dict[keys])
    return list_arr, real_prompt

def process_original_prompt(character_dict,prompts,id_length):
    """
    Process all prompts to map characters and select reference images.
    
    This comprehensive function:
    1. Maps which characters appear in which prompts
    2. Replaces character tags with descriptions
    3. Selects reference prompts for each character
    4. Validates that each character has enough unique prompts
    
    Args:
        character_dict (dict): Maps character tags to descriptions
        prompts (list): All prompt strings (references + story)
        id_length (int): Number of reference images needed per character
        
    Returns:
        tuple: (character_index_dict, invert_character_index_dict, replace_prompts, 
                ref_index_dict, ref_totals)
            - character_index_dict: Maps characters to all their prompt indices
            - invert_character_index_dict: Maps indices to characters in that prompt
            - replace_prompts: All prompts with character tags replaced
            - ref_index_dict: Maps characters to their selected reference indices
            - ref_totals: All reference indices combined
            
    Raises:
        gr.Error: If a character doesn't have enough unique prompts for references
    """
    replace_prompts = []
    character_index_dict = {}
    invert_character_index_dict = {}
    
    # First pass: Find all character occurrences and replace tags
    for ind, prompt in enumerate(prompts):
        for key in character_dict.keys():
            if key in prompt:
                # Track character appearances
                if key not in character_index_dict:
                    character_index_dict[key] = []
                character_index_dict[key].append(ind)
                
                # Track multiple characters per prompt
                if ind not in invert_character_index_dict:
                    invert_character_index_dict[ind] = []
                invert_character_index_dict[ind].append(key)
        
        # Replace all character tags in this prompt
        cur_prompt = prompt
        if ind in invert_character_index_dict:
            for key in invert_character_index_dict[ind]:
                cur_prompt = cur_prompt.replace(key, character_dict[key] + " ")
        replace_prompts.append(cur_prompt)
    
    # Second pass: Select reference prompts for each character
    ref_index_dict = {}
    ref_totals = []
    print(character_index_dict)
    
    for character_key in character_index_dict.keys():
        # Get prompts where this character appears alone (for clean references)
        index_list = character_index_dict[character_key]
        index_list = [index for index in index_list if len(invert_character_index_dict[index]) == 1]
        
        # Validate sufficient reference prompts
        if len(index_list) < id_length:
            raise gr.Error(f"{character_key} not have enough prompt description, need no less than {id_length}, but you give {len(index_list)}")
        
        # Select first id_length prompts as references
        ref_index_dict[character_key] = index_list[:id_length]
        ref_totals = ref_totals + index_list[:id_length]
        
    return character_index_dict, invert_character_index_dict, replace_prompts, ref_index_dict, ref_totals


def get_ref_character(real_prompt,character_dict):
    """
    Find all characters referenced in a given prompt.
    
    Args:
        real_prompt (str): Prompt text to search for character tags
        character_dict (dict): Dictionary of known character tags
        
    Returns:
        list: Character tags found in the prompt
    """
    list_arr = []
    for keys in character_dict.keys():
        if keys in real_prompt:
            list_arr = list_arr + [keys]
    return list_arr