"""Main generation logic for StoryDiffusion."""

import torch
import gc
import gradio as gr
from PIL import ImageFont
from typing import List, Optional, Dict, Any, Generator
from diffusers.utils.loading_utils import load_image

from utils.gradio_utils import (
    character_to_dict,
    process_original_prompt,
    get_ref_character,
)
from utils.utils import get_comic
from .utils import (
    setup_seed,
    apply_style,
    apply_style_positive,
    process_prompts_for_generation
)
from ..models.pipeline import load_pipeline, get_model_type, clear_attention_banks
from ..utils.character import load_character_files_on_running
from ..utils.image import save_results
from ..config import (
    DEVICE,
    DEFAULT_NUM_STEPS,
    clear_device_cache,
)


def process_generation(
    _sd_type: str,
    _model_type: str,
    _upload_images: Optional[List[Any]],
    _num_steps: int,
    style_name: str,
    _Ip_Adapter_Strength: float,
    _style_strength_ratio: float,
    guidance_scale: float,
    seed_: int,
    sa32_: float,
    sa64_: float,
    id_length_: int,
    general_prompt: str,
    negative_prompt: str,
    prompt_array: str,
    G_height: int,
    G_width: int,
    _comic_type: str,
    font_choice: str,
    _char_files: str,
) -> Generator[List[Any], None, None]:
    """
    Main image generation function that orchestrates the entire StoryDiffusion pipeline.

    This function handles:
    1. Model loading and configuration based on selected options
    2. Character reference image generation (if using PhotoMaker)
    3. Sequential story image generation with character consistency
    4. Optional comic-style typesetting and captioning

    Args:
        _sd_type (str): Stable Diffusion model type to use
        _model_type (str): "Using Ref Images" for PhotoMaker or "Only Using Textual Description"
        _upload_images: List of uploaded reference images for characters
        _num_steps (int): Number of denoising steps
        style_name (str): Style template to apply
        _Ip_Adapter_Strength (float): IP-Adapter strength (unused in current version)
        _style_strength_ratio (float): Style strength percentage for PhotoMaker
        guidance_scale (float): Classifier-free guidance scale
        seed_ (int): Random seed for generation
        sa32_ (float): Paired attention strength at 32x32 resolution
        sa64_ (float): Paired attention strength at 64x64 resolution
        id_length_ (int): Number of character reference images
        general_prompt (str): Character descriptions (one per line)
        negative_prompt (str): Negative prompt for all generations
        prompt_array (str): Story prompts (one per line, one per image)
        G_height (int): Generated image height
        G_width (int): Generated image width
        _comic_type (str): Comic typesetting style
        font_choice (str): Font file for captions
        _char_files (str): Optional character weight files to load

    Yields:
        list: List of generated PIL images (updates progressively)
    """
    # Access global state
    from ..config import global_state
    
    # Validate character limit due to VRAM constraints
    if len(general_prompt.splitlines()) >= 3:
        raise gr.Error(
            "Support for more than three characters is temporarily unavailable due to VRAM limitations, but this issue will be resolved soon."
        )

    # Convert model type selection to internal format
    _model_type = get_model_type(_model_type)

    # Validate PhotoMaker requirements
    if _model_type == "Photomaker" and "img" not in general_prompt:
        raise gr.Error(
            'Please add the trigger word " img " behind the class word you want to customize, such as: man img or woman img'
        )
    if _upload_images is None and _model_type != "original":
        raise gr.Error(f"Cannot find any input face image!")

    # Clear any existing attention banks
    clear_attention_banks()
    
    # Update global configuration
    global_state.sa32 = sa32_
    global_state.sa64 = sa64_
    global_state.id_length = id_length_
    global_state.total_length = id_length_ + 1
    global_state.height = G_height
    global_state.width = G_width
    global_state.write = True
    global_state.cur_step = 0
    global_state.attn_count = 0
    
    # Load or reload pipeline as needed
    pipe = load_pipeline(
        model_name=_sd_type,
        model_type=_model_type,
        id_length=id_length_,
        height=G_height,
        width=G_width,
        sa32=sa32_,
        sa64=sa64_
    )
    
    # Load character weights if provided
    load_chars = load_character_files_on_running(pipe.unet, character_files=_char_files)
    
    # Process prompts and character information
    prompts, nc_indices, captions = process_prompts_for_generation(prompt_array, id_length_)
    character_dict, character_list = character_to_dict(general_prompt)
    
    # Calculate PhotoMaker start merge step
    start_merge_step = int(float(_style_strength_ratio) / 100 * _num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"start_merge_step:{start_merge_step}")
    
    # Process prompts and create character mappings
    (
        character_index_dict,
        invert_character_index_dict,
        replace_prompts,
        ref_indexs_dict,
        ref_totals,
    ) = process_original_prompt(character_dict, prompts.copy(), id_length_)
    
    # Update global state with character information
    global_state.character_dict = character_dict
    global_state.character_index_dict = character_index_dict
    global_state.invert_character_index_dict = invert_character_index_dict
    global_state.ref_indexs_dict = ref_indexs_dict
    global_state.ref_totals = ref_totals
    
    # Prepare input images for PhotoMaker
    input_id_images_dict = {}
    if _model_type == "Photomaker":
        if len(_upload_images) != len(character_dict.keys()):
            raise gr.Error(
                f"You upload images({len(_upload_images)}) is not equal to the number of characters({len(character_dict.keys())})!"
            )
        for ind, img in enumerate(_upload_images):
            input_id_images_dict[character_list[ind]] = [load_image(img)]
    
    # Initialize generation
    setup_seed(seed_, DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(seed_)
    results_dict: Dict[int, Any] = {}
    
    # Generate character reference images (if not loading from weights)
    if not load_chars:
        for character_key in character_dict.keys():
            global_state.cur_character = [character_key]
            ref_indexs = ref_indexs_dict[character_key]
            print(f"Generating reference images for {character_key}: {ref_indexs}")
            
            current_prompts = [replace_prompts[ref_ind] for ref_ind in ref_indexs]
            setup_seed(seed_, DEVICE)
            generator = torch.Generator(device=DEVICE).manual_seed(seed_)
            global_state.cur_step = 0
            
            cur_positive_prompts, styled_negative = apply_style(
                style_name, current_prompts, negative_prompt
            )
            
            # Generate reference images
            if _model_type == "original":
                id_images = pipe(
                    cur_positive_prompts,
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    height=G_height,
                    width=G_width,
                    negative_prompt=styled_negative,
                    generator=generator,
                ).images
            elif _model_type == "Photomaker":
                id_images = pipe(
                    cur_positive_prompts,
                    input_id_images=input_id_images_dict[character_key],
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    start_merge_step=start_merge_step,
                    height=G_height,
                    width=G_width,
                    negative_prompt=styled_negative,
                    generator=generator,
                ).images
            
            # Store results
            for ind, img in enumerate(id_images):
                results_dict[ref_indexs[ind]] = img
            
            yield [results_dict[ind] for ind in sorted(results_dict.keys())]
    
    # Switch to read mode for story generation
    global_state.write = False
    
    # Determine which prompts to generate
    if not load_chars:
        real_prompts_inds = [
            ind for ind in range(len(prompts)) if ind not in ref_totals
        ]
    else:
        real_prompts_inds = list(range(len(prompts)))
    
    print(f"Generating story images for indices: {real_prompts_inds}")
    
    # Generate story images
    for real_prompts_ind in real_prompts_inds:
        real_prompt = replace_prompts[real_prompts_ind]
        global_state.cur_character = get_ref_character(prompts[real_prompts_ind], character_dict)
        print(f"Generating image {real_prompts_ind} with characters: {global_state.cur_character}")
        
        setup_seed(seed_, DEVICE)
        
        if len(global_state.cur_character) > 1 and _model_type == "Photomaker":
            raise gr.Error(
                "Temporarily Not Support Multiple character in Ref Image Mode!"
            )
        
        generator = torch.Generator(device=DEVICE).manual_seed(seed_)
        global_state.cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        
        # Generate image
        if _model_type == "original":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                height=G_height,
                width=G_width,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        elif _model_type == "Photomaker":
            # Select appropriate character images (or use first character for NC prompts)
            if real_prompts_ind in nc_indices:
                input_images = input_id_images_dict[character_list[0]]
            else:
                input_images = input_id_images_dict[global_state.cur_character[0]]
            
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                input_id_images=input_images,
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                start_merge_step=start_merge_step,
                height=G_height,
                width=G_width,
                negative_prompt=negative_prompt,
                generator=generator,
                nc_flag=real_prompts_ind in nc_indices,
            ).images[0]
        
        yield [results_dict[ind] for ind in sorted(results_dict.keys())]
    
    # Compile final results
    total_results = [results_dict[ind] for ind in range(len(prompts))]
    
    # Apply comic typesetting if requested
    if _comic_type != "No typesetting (default)":
        font_path = f"fonts/{font_choice}"
        font = ImageFont.truetype(font_path, 45)
        comic_results = get_comic(
            total_results, 
            _comic_type, 
            captions=captions[:len(total_results)], 
            font=font
        )
        total_results = comic_results + total_results
    
    # Save results
    save_results(pipe.unet, total_results)
    
    # Memory cleanup only at the end
    clear_device_cache(DEVICE)
    
    yield total_results