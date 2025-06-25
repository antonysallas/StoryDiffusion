"""Image processing and saving utilities."""

import os
import datetime
from typing import List
from PIL import Image


def get_image_path_list(folder_name: str) -> List[str]:
    """
    Get a sorted list of image file paths from a folder.

    Args:
        folder_name (str): Path to the folder containing images

    Returns:
        list: Sorted list of full paths to image files in the folder
    """
    if not os.path.exists(folder_name):
        return []
    
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted(
        [os.path.join(folder_name, basename) for basename in image_basename_list]
    )
    return image_path_list


def save_results(unet, img_list: List[Image.Image]) -> str:
    """
    Save generated images to a timestamped folder.

    Creates a results folder with the current timestamp and saves all generated
    images. Also creates a weights subfolder for potential character weight saving.

    Args:
        unet: The UNet model (for potential weight saving)
        img_list (list): List of PIL images to save
        
    Returns:
        str: Path to the created results folder
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"results/{timestamp}"
    weight_folder_name = f"{folder_name}/weights"

    # Create output directories
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(weight_folder_name)

    # Save each generated image
    for idx, img in enumerate(img_list):
        file_path = os.path.join(folder_name, f"image_{idx}.png")
        img.save(file_path)

    # Optional: Save character weights for future use
    # This is commented out in the original code but could be enabled if needed
    # from ..config import GlobalState
    # from .character import save_single_character_weights
    # global_state = GlobalState()
    # for char in global_state.character_dict:
    #     description = global_state.character_dict[char]
    #     save_single_character_weights(
    #         unet, char, description, 
    #         os.path.join(weight_folder_name, f'{char}.pt')
    #     )
    
    return folder_name


def array2string(arr: List[str]) -> str:
    """
    Convert a list of strings to a single newline-separated string.

    Args:
        arr (list): List of strings to join

    Returns:
        str: Newline-separated string
    """
    return "\n".join(arr)