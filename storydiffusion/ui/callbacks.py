"""
Gradio UI callbacks for StoryDiffusion.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
import torch
from diffusers.utils.loading_utils import load_image
from PIL import Image

from ..config import GenerationSettings, ModelConfig
from ..device import clear_memory
from ..generation.engine import generate_story
from ..models.loader import load_photomaker_adapter, load_stable_diffusion_pipeline


def process_generation(
    sd_type: str,
    model_type: str,
    upload_images: List[str],
    num_steps: int,
    style_name: str,
    ip_adapter_strength: float,
    style_strength_ratio: float,
    guidance_scale: float,
    seed: int,
    sa32_strength: float,
    sa64_strength: float,
    id_length: int,
    general_prompt: str,
    negative_prompt: str,
    prompt_array: str,
    height: int,
    width: int,
    comic_type: str,
    font_choice: str,
    char_path: str = "",
    pipe = None,
    model_config: Optional[ModelConfig] = None,
) -> List[Image.Image]:
    """
    Main callback function for generating a story from the UI inputs.
    
    Args:
        sd_type: Selected model type from dropdown
        model_type: Model type ("Only Using Textual Description" or "Using Ref Images")
        upload_images: List of uploaded image paths
        num_steps: Number of inference steps
        style_name: Selected style template
        ip_adapter_strength: Strength of IP adapter
        style_strength_ratio: Strength of reference image style
        guidance_scale: Guidance scale for generation
        seed: Random seed
        sa32_strength: Self-attention strength at 32x32
        sa64_strength: Self-attention strength at 64x64
        id_length: Number of ID images per character
        general_prompt: Character descriptions
        negative_prompt: Negative prompt text
        prompt_array: Story prompts, one per line
        height: Image height
        width: Image width
        comic_type: Type of comic layout
        font_choice: Selected font
        char_path: Optional character file paths
        pipe: Optional pre-loaded pipeline
        model_config: Optional model configuration
        
    Returns:
        List of generated images
    """
    # Convert UI model type to internal string
    model_type_internal = "Photomaker" if model_type == "Using Ref Images" else "original"
    
    # Validate inputs
    if model_type_internal == "Photomaker" and "img" not in general_prompt:
        raise gr.Error(
            'Please add the trigger word "img" behind the class word you want to customize, '
            'such as: man img or woman img'
        )
    
    if not upload_images and model_type_internal == "Photomaker":
        raise gr.Error("Cannot find any input face image!")
    
    # Create settings object
    settings = GenerationSettings(
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        sa32_strength=sa32_strength,
        sa64_strength=sa64_strength,
        id_length=id_length,
        seed=seed if seed >= 0 else torch.randint(0, 2**32 - 1, (1,)).item(),
    )
    
    # Get model configuration - normally this would be passed in
    if model_config is None:
        from utils.load_models_utils import get_models_dict
        models_dict = get_models_dict()
        model_info = models_dict[sd_type]
    else:
        model_info = model_config
    
    # Load or reuse pipeline
    if pipe is None:
        # Import dynamically to avoid circular imports
        from utils.load_models_utils import load_models
        
        # Get device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        # Load the appropriate pipeline
        if model_type_internal == "original":
            pipe, _ = load_stable_diffusion_pipeline(model_info, device)
        else:
            # Load base pipeline
            base_pipe, _ = load_stable_diffusion_pipeline(model_info, device)
            
            # Load PhotoMaker adapter
            from ..config import PHOTOMAKER_CONFIG
            pipe = load_photomaker_adapter(base_pipe, PHOTOMAKER_CONFIG, trigger_word="img")
    
    # Set font path
    font_path = os.path.join("fonts", font_choice)
    
    # Generate the story
    try:
        # Clear memory before generation
        clear_memory()
        
        # Call generation function
        images, metadata = generate_story(
            model_config=model_info,
            character_prompt=general_prompt,
            prompt_array=prompt_array,
            settings=settings,
            style_name=style_name,
            negative_prompt=negative_prompt,
            comic_type=comic_type,
            font_path=font_path,
            reference_images=upload_images,
            model_type=model_type_internal,
            pipe=pipe,
        )
        
        # Return images
        return images
    
    except Exception as e:
        # Log and re-raise error
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error during generation: {str(e)}")
    

def set_text_unfinished() -> gr.update:
    """Update status to show generation is in progress.
    
    Returns:
        gr.update: A Gradio update object with the status message
    """
    return gr.update(
        visible=True,
        value="<h3><span style='color:#FF6700'>⚙️ Generating...</span> The intermediate results will be shown.</h3>",
    )


def set_text_finished() -> gr.update:
    """Update status to show generation is complete.
    
    Returns:
        gr.update: A Gradio update object with the status message
    """
    return gr.update(visible=True, value="<h3>Generation Finished ✅</h3>")


def setup_callbacks(demo: gr.Blocks) -> None:
    """Set up callbacks for the Gradio interface.
    
    Args:
        demo: The Gradio Blocks interface
    """
    # Get the elements we need
    final_run_btn = None
    generated_information = None
    out_image = None
    
    for component in demo.blocks.values():
        if isinstance(component, gr.Button) and component.value == "Generate! 😺":
            final_run_btn = component
        elif isinstance(component, gr.Markdown) and component.label == "Generation Details":
            generated_information = component
        elif isinstance(component, gr.Gallery) and component.label == "Results":
            out_image = component
    
    if not all([final_run_btn, generated_information, out_image]):
        raise ValueError("Could not find all required UI components")
    
    # Set up the main generation callback
    final_run_btn.click(
        fn=set_text_unfinished, 
        outputs=generated_information
    ).then(
        process_generation,
        inputs=[
            demo.sd_type,
            demo.model_type,
            demo.files,
            demo.num_steps,
            demo.style,
            demo.Ip_Adapter_Strength,
            demo.style_strength_ratio,
            demo.guidance_scale,
            demo.seed_,
            demo.sa32_,
            demo.sa64_,
            demo.id_length_,
            demo.general_prompt,
            demo.negative_prompt,
            demo.prompt_array,
            demo.G_height,
            demo.G_width,
            demo.comic_type,
            demo.font_choice,
            demo.char_path,
        ],
        outputs=out_image,
    ).then(
        fn=set_text_finished, 
        outputs=generated_information
    )