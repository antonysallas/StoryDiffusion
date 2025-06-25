"""Character weight management utilities."""

import torch
from typing import Dict, Any
from ..models.attention import SpatialAttnProcessor2_0


def save_single_character_weights(
    unet, 
    character: str, 
    description: str, 
    filepath: str
) -> None:
    """
    Save character-specific attention weights from the UNet's attention processors.

    This function extracts and saves the id_bank tensors that store character-specific
    attention features, allowing them to be reused in future generations for consistent
    character representation.

    Args:
        unet: The UNet model containing attention processors with character data
        character (str): Character identifier to save
        description (str): Character description for reference
        filepath (str): Path where the weights file will be saved
    """
    weights_to_save: Dict[str, Any] = {}
    weights_to_save["description"] = description
    weights_to_save["character"] = character

    # Extract attention features from each SpatialAttnProcessor2_0
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # Move tensors to CPU for serialization
            weights_to_save[attn_name] = {}
            if character in attn_processor.id_bank:
                for step_key in attn_processor.id_bank[character].keys():
                    weights_to_save[attn_name][step_key] = [
                        tensor.cpu()
                        for tensor in attn_processor.id_bank[character][step_key]
                    ]

    # Save weights using PyTorch's serialization
    torch.save(weights_to_save, filepath)


def load_single_character_weights(unet, filepath: str) -> None:
    """
    Load saved character-specific attention weights into the UNet's attention processors.

    This function restores previously saved character attention features, enabling
    consistent character generation without needing to regenerate reference images.

    Args:
        unet: The UNet model to load weights into
        filepath (str): Path to the saved weights file

    Returns:
        None (modifies UNet attention processors in-place)
    """
    # Load weights from file
    weights_to_load = torch.load(filepath, map_location=torch.device("cpu"))
    character = weights_to_load["character"]
    description = weights_to_load["description"]

    # Restore weights to each SpatialAttnProcessor2_0
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # Transfer weights to appropriate device and restore to id_bank
            if attn_name in weights_to_load:
                attn_processor.id_bank[character] = {}
                for step_key in weights_to_load[attn_name].keys():
                    attn_processor.id_bank[character][step_key] = [
                        tensor.to(unet.device)
                        for tensor in weights_to_load[attn_name][step_key]
                    ]


def load_character_files_on_running(unet, character_files: str) -> bool:
    """
    Load saved character weights into the UNet during generation.

    Args:
        unet: The UNet model to load weights into
        character_files (str): Newline-separated paths to character weight files

    Returns:
        bool: True if weights were loaded, False if no files provided
    """
    if not character_files or character_files.strip() == "":
        return False
    
    character_files_arr = character_files.strip().splitlines()

    # Load each character's weights into the UNet
    for character_file in character_files_arr:
        if character_file.strip():  # Skip empty lines
            load_single_character_weights(unet, character_file.strip())
    
    return True