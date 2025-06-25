"""Main Gradio application for StoryDiffusion."""

import os
import gradio as gr
from typing import List
from ..config import (
    CSS,
    TITLE,
    DESCRIPTION,
    ARTICLE,
    VERSION,
    STYLE_NAMES,
    DEFAULT_STYLE_NAME,
    DEFAULT_NUM_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_SA32,
    DEFAULT_SA64,
    DEFAULT_ID_LENGTH,
    DEFAULT_STYLE_STRENGTH,
    MAX_SEED,
    MODELS_DICT
)
from ..generation.generator import process_generation
from .components import (
    set_text_unfinished,
    set_text_finished,
    swap_to_gallery,
    upload_example_to_gallery,
    remove_back_to_files,
    change_visiale_by_model_type,
    load_character_files,
    randomize_seed,
    get_available_fonts
)
from .examples import get_examples


def create_demo() -> gr.Blocks:
    """
    Create the main Gradio demo interface for StoryDiffusion.
    
    Returns:
        gr.Blocks: The configured Gradio interface
    """
    with gr.Blocks(css=CSS) as demo:
        # State variables for storing UI data
        binary_matrixes = gr.State([])
        color_layout = gr.State([])

        # Header
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Group(elem_id="main-image"):
                with gr.Column(visible=True) as gen_prompt_vis:
                    # Model selection
                    sd_type = gr.Dropdown(
                        choices=list(MODELS_DICT.keys()),
                        value="Unstable",
                        label="SD Model",
                        info="Select pretrained model",
                    )
                    model_type = gr.Radio(
                        ["Only Using Textual Description", "Using Ref Images"],
                        label="Model Type",
                        value="Only Using Textual Description",
                        info="Control type of the Character",
                    )
                    
                    # Reference image upload (conditional)
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
                    
                    # Text inputs
                    general_prompt = gr.Textbox(
                        value="",
                        lines=2,
                        label="(1) Textual Description for Character",
                        interactive=True,
                    )
                    negative_prompt = gr.Textbox(
                        value="", 
                        label="(2) Negative Prompt", 
                        interactive=True
                    )
                    style = gr.Dropdown(
                        label="Style Template",
                        choices=STYLE_NAMES,
                        value=DEFAULT_STYLE_NAME,
                    )
                    prompt_array = gr.Textbox(
                        lines=3,
                        value="",
                        label="(3) Comic Description (each line corresponds to a frame).",
                        interactive=True,
                    )
                    
                    # Character file loading (optional)
                    char_path = gr.Textbox(
                        lines=2,
                        value="",
                        visible=False,
                        label="(Optional) Character files",
                        interactive=True,
                    )
                    char_btn = gr.Button("Load Character files", visible=False)
                    
                    # Hyperparameter controls
                    with gr.Accordion("(4) Tune the hyperparameters", open=True):
                        # Font selection for comic captions
                        font_choice = gr.Dropdown(
                            label="Select Font",
                            choices=get_available_fonts(),
                            value="Inkfree.ttf",
                            info="Select font for the final slide.",
                            interactive=True,
                        )
                        
                        # Paired Attention strength controls
                        sa32_ = gr.Slider(
                            label="Paired Attention at 32x32",
                            minimum=0,
                            maximum=1.0,
                            value=DEFAULT_SA32,
                            step=0.1,
                        )
                        sa64_ = gr.Slider(
                            label="Paired Attention at 64x64",
                            minimum=0,
                            maximum=1.0,
                            value=DEFAULT_SA64,
                            step=0.1,
                        )
                        id_length_ = gr.Slider(
                            label="Number of ID Images",
                            minimum=1,
                            maximum=4,
                            value=DEFAULT_ID_LENGTH,
                            step=1,
                        )
                        
                        with gr.Row():
                            seed_ = gr.Slider(
                                label="Seed", 
                                minimum=-1, 
                                maximum=MAX_SEED, 
                                value=0, 
                                step=1
                            )
                            randomize_seed_btn = gr.Button("🎲", size="sm")
                        
                        num_steps = gr.Slider(
                            label="Number of Sample Steps",
                            minimum=20,
                            maximum=100,
                            step=1,
                            value=DEFAULT_NUM_STEPS,
                        )
                        G_height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=1024,
                            step=32,
                            value=DEFAULT_HEIGHT,
                        )
                        G_width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=1024,
                            step=32,
                            value=DEFAULT_WIDTH,
                        )
                        comic_type = gr.Radio(
                            [
                                "No typesetting (default)",
                                "Four Pannel",
                                "Classic Comic Style",
                            ],
                            value="Classic Comic Style",
                            label="Typesetting Style",
                            info="Select the typesetting style",
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.1,
                            maximum=10.0,
                            step=0.1,
                            value=DEFAULT_GUIDANCE_SCALE,
                        )
                        style_strength_ratio = gr.Slider(
                            label="Style Strength of Ref Image (%)",
                            minimum=15,
                            maximum=50,
                            step=1,
                            value=DEFAULT_STYLE_STRENGTH,
                            visible=False,
                        )
                        Ip_Adapter_Strength = gr.Slider(
                            label="IP Adapter Strength",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.5,
                            visible=False,
                        )
                    
                    final_run_btn = gr.Button("Generate! 😺")

            # Results column
            with gr.Column():
                out_image = gr.Gallery(label="Result", columns=2, height="auto")
                generated_information = gr.Markdown(
                    label="Generation Details", value="", visible=False
                )
                gr.Markdown(VERSION)
        
        # Event handlers
        model_type.change(
            fn=change_visiale_by_model_type,
            inputs=model_type,
            outputs=[control_image_input, style_strength_ratio, Ip_Adapter_Strength],
        )
        
        files.upload(
            fn=swap_to_gallery, 
            inputs=files, 
            outputs=[uploaded_files, clear_button, files]
        )
        
        remove_and_reupload.click(
            fn=remove_back_to_files, 
            outputs=[uploaded_files, clear_button, files]
        )
        
        char_btn.click(
            fn=load_character_files, 
            inputs=char_path, 
            outputs=[general_prompt]
        )

        randomize_seed_btn.click(
            fn=randomize_seed,
            inputs=[],
            outputs=seed_,
        )

        final_run_btn.click(
            fn=set_text_unfinished, 
            outputs=generated_information
        ).then(
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
        ).then(
            fn=set_text_finished, 
            outputs=generated_information
        )

        # Examples
        gr.Examples(
            examples=get_examples(),
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
            label="😺 Examples 😺",
        )
        
        gr.Markdown(ARTICLE)

    return demo