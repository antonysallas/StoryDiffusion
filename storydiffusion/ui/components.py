"""UI component handlers and utilities."""

import gradio as gr
import torch
import random
import os
from typing import List, Tuple, Optional, Any
from ..utils.image import array2string
from ..config import MAX_SEED


def set_text_unfinished() -> gr.update:
    """
    Update the UI to show generation is in progress.

    Returns:
        gr.update: Gradio update object with visible generation progress message
    """
    return gr.update(
        visible=True,
        value="<h3>(Not Finished) Generating ···  The intermediate results will be shown.</h3>",
    )


def set_text_finished() -> gr.update:
    """
    Update the UI to show generation is complete.

    Returns:
        gr.update: Gradio update object with visible completion message
    """
    return gr.update(visible=True, value="<h3>Generation Finished</h3>")


def swap_to_gallery(images: List[Any]) -> Tuple[gr.update, gr.update, gr.update]:
    """
    Switch UI view to show uploaded images in gallery format.

    Args:
        images: Uploaded image files

    Returns:
        tuple: Gradio update objects to show gallery and hide file upload
    """
    return (
        gr.update(value=images, visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def upload_example_to_gallery(
    images: List[Any], 
    prompt: str, 
    style: str, 
    negative_prompt: str
) -> Tuple[gr.update, gr.update, gr.update]:
    """
    Load example images into the gallery view.

    Args:
        images: Example images to display
        prompt: Associated prompt text
        style: Style template name
        negative_prompt: Negative prompt text

    Returns:
        tuple: Gradio update objects to show gallery
    """
    return (
        gr.update(value=images, visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def remove_back_to_files() -> Tuple[gr.update, gr.update, gr.update]:
    """
    Return UI to file upload mode from gallery view.

    Returns:
        tuple: Gradio update objects to hide gallery and show file upload
    """
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def remove_tips() -> gr.update:
    """
    Hide tips/information display.

    Returns:
        gr.update: Hidden tips element
    """
    return gr.update(visible=False)


def change_visiale_by_model_type(
    _model_type: str
) -> Tuple[gr.update, gr.update, gr.update]:
    """
    Update UI visibility based on selected model type.

    Args:
        _model_type (str): Either "Only Using Textual Description" or "Using Ref Images"

    Returns:
        tuple: Gradio update objects for conditional UI elements
    """
    if _model_type == "Only Using Textual Description":
        return (
            gr.update(visible=False),  # Hide image upload
            gr.update(visible=False),  # Hide style strength
            gr.update(visible=False),  # Hide IP adapter strength
        )
    elif _model_type == "Using Ref Images":
        return (
            gr.update(visible=True),   # Show image upload
            gr.update(visible=True),   # Show style strength
            gr.update(visible=False),  # Keep IP adapter hidden
        )
    else:
        raise ValueError("Invalid model type", _model_type)


def load_character_files(character_files: str) -> str:
    """
    Load character descriptions from saved weight files.

    Args:
        character_files (str): Newline-separated paths to character weight files

    Returns:
        str: Combined character descriptions from loaded files

    Raises:
        gr.Error: If no character files provided
    """
    if not character_files or character_files.strip() == "":
        raise gr.Error("Please set a character file!")
    
    character_files_arr = character_files.strip().splitlines()
    primarytext = []

    # Load each character file and extract descriptions
    for character_file_name in character_files_arr:
        if character_file_name.strip():  # Skip empty lines
            character_file = torch.load(
                character_file_name.strip(), map_location=torch.device("cpu")
            )
            primarytext.append(
                character_file["character"] + " " + character_file["description"]
            )
    
    return array2string(primarytext)


def randomize_seed() -> int:
    """Generate a random seed value."""
    return random.randint(-1, MAX_SEED)


def get_available_fonts(font_dir: str = "./fonts") -> List[str]:
    """Get list of available font files."""
    if not os.path.exists(font_dir):
        return []
    return [f for f in os.listdir(font_dir) if f.endswith(".ttf")]