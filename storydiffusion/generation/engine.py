"""
Core generation engine for StoryDiffusion.
Implements the main image generation logic with character consistency.
"""

import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.utils.loading_utils import load_image
from PIL import Image

from ..config import GenerationSettings, ModelConfig
from ..device import clear_memory, set_random_seed
from ..models.attention import set_attention_processor

# Global state variables for the generation process
# These are needed for the attention processors to work correctly
global attn_count, total_count, cur_step, cur_model_type
global write, sa32, sa64, height, width
global character_dict, cur_character

# Initialize global variables
attn_count = 0
total_count = 0
cur_step = 0
cur_model_type = ""
write = False
sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
character_dict = {}
cur_character = []


def prepare_character_prompt(
    character_prompt: str,
    reference_images: Optional[List[str]] = None,
    model_type: str = "original",
) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse character descriptions and prepare for generation.
    
    Args:
        character_prompt: Text description of the characters
        reference_images: Optional reference images for character appearance
        model_type: Model type ("original" or "Photomaker")
        
    Returns:
        Tuple containing character dictionary and character list
    """
    from utils.gradio_utils import character_to_dict
    
    # Parse character descriptions
    character_dict, character_list = character_to_dict(character_prompt)
    
    # Validate reference images if using PhotoMaker
    if model_type == "Photomaker":
        if not reference_images:
            raise ValueError("Reference images are required when using PhotoMaker")
            
        if len(reference_images) != len(character_dict):
            raise ValueError(
                f"Number of reference images ({len(reference_images)}) must match "
                f"number of characters ({len(character_dict)})"
            )
            
        # Verify img trigger word is present in prompts when using PhotoMaker
        for character, description in character_dict.items():
            if "img" not in description:
                raise ValueError(
                    f'Trigger word "img" missing for character "{character}". '
                    'Add "img" after the class word, e.g., "woman img"'
                )
                
    return character_dict, character_list


def process_prompts(
    prompt_array: str, 
    character_dict: Dict[str, str],
    id_length: int
) -> Tuple[List[str], Dict[str, List[int]], List[int]]:
    """
    Process the array of prompts for the comic.
    
    Args:
        prompt_array: Multi-line string with prompts, one per line
        character_dict: Dictionary mapping character names to descriptions
        id_length: Number of ID images
        
    Returns:
        List of processed prompts, dictionary of character reference indices,
        list of indices for frames without characters
    """
    from utils.gradio_utils import process_original_prompt
    
    # Split lines and clean up
    prompts = [line.strip() for line in prompt_array.splitlines() if line.strip()]
    
    # Check for NC (No Character) tags
    nc_indices = []
    for idx, prompt in enumerate(prompts):
        if "[NC]" in prompt:
            if idx < id_length:
                raise ValueError(f"The first {id_length} prompts are for ID generation and cannot use [NC]")
            nc_indices.append(idx)
            # Remove [NC] tag from prompt
            prompts[idx] = prompt.replace("[NC]", "").strip()
    
    # Extract captions (text after #)
    captions = []
    clean_prompts = []
    for prompt in prompts:
        if "#" in prompt:
            clean_part, _, caption = prompt.partition("#")
            clean_prompts.append(clean_part.strip())
            captions.append(caption.strip())
        else:
            clean_prompts.append(prompt)
            captions.append(prompt)  # Use prompt as caption if no explicit caption
    
    # Process prompts to identify character references
    (
        character_index_dict,
        invert_character_index_dict,
        replacement_prompts,
        reference_indices_dict,
        reference_totals
    ) = process_original_prompt(character_dict, clean_prompts.copy(), id_length)
    
    return clean_prompts, captions, reference_indices_dict, reference_totals, nc_indices


def setup_generation_environment(
    settings: GenerationSettings,
    character_dict: Dict[str, str],
) -> None:
    """
    Set up the global environment for generation.
    
    Args:
        settings: Generation settings
        character_dict: Dictionary mapping character names to descriptions
    """
    global sa32, sa64, height, width, id_length
    
    # Set global parameters from settings
    sa32 = settings.sa32_strength
    sa64 = settings.sa64_strength
    height = settings.height
    width = settings.width
    id_length = settings.id_length
    
    # Set random seed for reproducibility
    set_random_seed(settings.seed)
    
    # Store character dictionary globally
    globals()["character_dict"] = character_dict


def generate_character_references(
    pipe,
    character_key: str,
    reference_indices: List[int],
    prompts: List[str],
    style_name: str,
    negative_prompt: str,
    settings: GenerationSettings,
    reference_images: Optional[List[str]] = None,
) -> List[Image.Image]:
    """
    Generate reference images for a single character.
    
    Args:
        pipe: Diffusion pipeline (regular or PhotoMaker)
        character_key: The character identifier
        reference_indices: List of indices for prompts to use for this character
        prompts: List of all prompts
        style_name: Style template name
        negative_prompt: Negative prompt text
        settings: Generation settings
        reference_images: Optional list of reference images (for PhotoMaker)
        
    Returns:
        List of generated images
    """
    from utils.style_template import styles
    from utils.gradio_utils import apply_style
    
    global cur_character, cur_step, write
    
    # Set global state for character reference generation
    cur_character = [character_key]
    cur_step = 0
    write = True
    
    # Get prompts for this character's reference images
    current_prompts = [prompts[ref_idx] for ref_idx in reference_indices]
    
    # Apply style template
    styled_prompts, styled_negative = apply_style(style_name, current_prompts, negative_prompt)
    
    # Set up random seed
    set_random_seed(settings.seed)
    generator = torch.Generator(device=pipe.device.type).manual_seed(settings.seed)
    
    # Generate images
    if reference_images is None:
        # Text-only generation
        images = pipe(
            styled_prompts,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            height=settings.height,
            width=settings.width,
            negative_prompt=styled_negative,
            generator=generator,
        ).images
    else:
        # PhotoMaker generation with reference image
        from utils.gradio_utils import apply_style
        
        # Load reference image
        input_images = [load_image(img) for img in reference_images]
        
        # Generate images
        images = pipe(
            styled_prompts,
            input_id_images=input_images,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            start_merge_step=int(min(30, settings.num_inference_steps * 0.2)),
            height=settings.height,
            width=settings.width,
            negative_prompt=styled_negative,
            generator=generator,
        ).images
    
    return images


def generate_story_frames(
    pipe,
    prompts: List[str],
    captions: List[str],
    settings: GenerationSettings,
    style_name: str,
    negative_prompt: str,
    character_reference_dict: Dict[str, List[int]],
    reference_totals: List[int],
    nc_indices: List[int],
    reference_images_dict: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[int, Image.Image], List[Image.Image]]:
    """
    Generate story frames using character references.
    
    Args:
        pipe: Diffusion pipeline
        prompts: List of processed prompts
        captions: List of captions for each prompt
        settings: Generation settings
        style_name: Style template name
        negative_prompt: Negative prompt text
        character_reference_dict: Dictionary mapping characters to reference indices
        reference_totals: List of indices used for references
        nc_indices: List of indices for frames without characters
        reference_images_dict: Optional dict of reference images by character
        
    Returns:
        Tuple containing dictionary of results by index and ordered list of images
    """
    from utils.gradio_utils import apply_style_positive, get_ref_character
    from utils.style_template import styles
    
    global cur_character, cur_step, write
    
    # Initialize results dictionary
    results_dict = {}
    
    # Set global state for story frame generation
    write = False
    
    # First, generate character references for each character
    for character_key, reference_indices in character_reference_dict.items():
        ref_images = None
        if reference_images_dict:
            ref_images = reference_images_dict.get(character_key)
            
        # Generate reference images for this character
        id_images = generate_character_references(
            pipe,
            character_key,
            reference_indices,
            prompts,
            style_name,
            negative_prompt,
            settings,
            ref_images
        )
        
        # Store generated reference images in results
        for idx, img in enumerate(id_images):
            results_dict[reference_indices[idx]] = img
    
    # Generate remaining story frames
    frame_indices = [idx for idx in range(len(prompts)) if idx not in reference_totals]
    
    for frame_idx in frame_indices:
        prompt = prompts[frame_idx]
        
        # Get characters referenced in this prompt
        cur_character = get_ref_character(prompt, character_dict)
        
        # Set up random seed and generator for consistent results
        set_random_seed(settings.seed)
        generator = torch.Generator(device=pipe.device.type).manual_seed(settings.seed)
        
        # Reset step counter for each frame
        cur_step = 0
        
        # Apply style to prompt
        styled_prompt = apply_style_positive(style_name, prompt)
        
        # Check if this is a No Character frame
        is_nc_frame = frame_idx in nc_indices
        
        # Generate the frame
        if reference_images_dict is None or is_nc_frame:
            # Text-only generation
            results_dict[frame_idx] = pipe(
                styled_prompt,
                num_inference_steps=settings.num_inference_steps,
                guidance_scale=settings.guidance_scale,
                height=settings.height,
                width=settings.width,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        else:
            # PhotoMaker generation
            # Get reference image for the first character (or default to first character)
            character = cur_character[0] if cur_character else next(iter(character_dict.keys()))
            reference_image = reference_images_dict.get(character, [reference_images_dict[next(iter(reference_images_dict.keys()))][0]])
            
            # Generate with reference
            results_dict[frame_idx] = pipe(
                styled_prompt,
                input_id_images=reference_image,
                num_inference_steps=settings.num_inference_steps,
                guidance_scale=settings.guidance_scale,
                start_merge_step=int(min(30, settings.num_inference_steps * 0.2)),
                height=settings.height,
                width=settings.width,
                negative_prompt=negative_prompt,
                generator=generator,
                nc_flag=is_nc_frame,
            ).images[0]
    
    # Create ordered list of results
    ordered_results = [results_dict[idx] for idx in range(len(prompts))]
    
    return results_dict, ordered_results


def create_comic_layout(
    images: List[Image.Image],
    captions: List[str],
    comic_type: str,
    font_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Create a comic layout from the generated images.
    
    Args:
        images: List of generated images
        captions: List of captions for each image
        comic_type: Type of comic layout
        font_path: Path to font file
        
    Returns:
        Comic layout image or None if no layout requested
    """
    if comic_type == "No typesetting (default)":
        return None
        
    from PIL import ImageFont
    from utils.utils import get_comic
    
    # Set default font path if not provided
    if font_path is None:
        font_path = os.path.join("fonts", "Inkfree.ttf")
    
    # Load font
    font = ImageFont.truetype(font_path, size=45)
    
    # Generate comic layout
    comic = get_comic(images, comic_type, captions=captions, font=font)
    
    return comic


def save_results(
    images: List[Image.Image], 
    unet,
    save_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Save generated images and character weights.
    
    Args:
        images: List of images to save
        unet: UNet model with character weights
        save_dir: Optional directory to save results
        
    Returns:
        Path to saved results directory
    """
    # Create timestamped folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if save_dir is None:
        save_dir = Path("results")
    else:
        save_dir = Path(save_dir)
    
    folder_name = save_dir / timestamp
    weight_folder = folder_name / "weights"
    
    # Create directories
    folder_name.mkdir(parents=True, exist_ok=True)
    weight_folder.mkdir(exist_ok=True)
    
    # Save images
    for idx, img in enumerate(images):
        file_path = folder_name / f"image_{idx}.png"
        img.save(file_path)
    
    # TODO: Add weight saving functionality
    # Currently commented out in the original code
    
    return str(folder_name)


def generate_story(
    model_config: ModelConfig,
    character_prompt: str,
    prompt_array: str,
    settings: GenerationSettings,
    style_name: str,
    negative_prompt: str = "",
    comic_type: str = "No typesetting (default)",
    font_path: Optional[str] = None,
    reference_images: Optional[List[str]] = None,
    model_type: str = "original",
    pipe=None,
) -> Tuple[List[Image.Image], Dict]:
    """
    Generate a complete visual story with consistent characters.
    
    Args:
        model_config: Configuration for the model
        character_prompt: Text description of the characters
        prompt_array: Multi-line string with prompts, one per line
        settings: Generation settings
        style_name: Style template name
        negative_prompt: Negative prompt text
        comic_type: Type of comic layout
        font_path: Path to font file
        reference_images: Optional list of reference images
        model_type: Model type ("original" or "Photomaker")
        pipe: Optional pre-loaded pipeline
        
    Returns:
        Tuple containing list of generated images and metadata
    """
    global height, width, sa32, sa64, id_length
    
    # Update global parameters
    height = settings.height
    width = settings.width
    sa32 = settings.sa32_strength
    sa64 = settings.sa64_strength
    id_length = settings.id_length
    
    # Prepare character information
    character_dict, character_list = prepare_character_prompt(
        character_prompt, reference_images, model_type
    )
    
    # Process prompts
    prompts, captions, reference_indices_dict, reference_totals, nc_indices = process_prompts(
        prompt_array, character_dict, settings.id_length
    )
    
    # Set up reference images dictionary if provided
    reference_images_dict = None
    if reference_images and model_type == "Photomaker":
        reference_images_dict = {}
        for idx, character in enumerate(character_list):
            if idx < len(reference_images):
                reference_images_dict[character] = [reference_images[idx]]
    
    # Setup generation environment
    setup_generation_environment(settings, character_dict)
    
    # Clear memory
    clear_memory()
    
    # Generate all frames
    results_dict, ordered_results = generate_story_frames(
        pipe,
        prompts,
        captions,
        settings,
        style_name,
        negative_prompt,
        reference_indices_dict,
        reference_totals,
        nc_indices,
        reference_images_dict
    )
    
    # Create comic layout if requested
    if comic_type != "No typesetting (default)":
        comic = create_comic_layout(ordered_results, captions, comic_type, font_path)
        if comic:
            ordered_results = [comic] + ordered_results
    
    # Save results
    output_dir = save_results(ordered_results, pipe.unet)
    
    # Return results and metadata
    metadata = {
        "character_count": len(character_dict),
        "frame_count": len(prompts),
        "settings": settings,
        "output_dir": output_dir,
    }
    
    return ordered_results, metadata