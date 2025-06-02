"""
Gradio UI interface for StoryDiffusion.
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
from PIL import Image

from ..config import DEFAULT_STYLE_NAME, MAX_SEED
from ..utils.style_template import styles

# Constants
STYLE_NAMES = list(styles.keys())


def get_status_update(is_finished: bool = False) -> gr.update:
    """Get a status update message for the UI.

    Args:
        is_finished: Whether the generation is finished

    Returns:
        gr.update: A Gradio update object with the status message
    """
    if is_finished:
        return gr.update(visible=True, value="<h3>Generation Finished ✅</h3>")
    else:
        return gr.update(
            visible=True,
            value="<h3><span style='color:#FF6700'>⚙️ Generating...</span> The intermediate results will be shown.</h3>",
        )


def change_visibility_by_model_type(model_type: str) -> Tuple[gr.update, gr.update, gr.update]:
    """Change UI element visibility based on model type.

    Args:
        model_type: Model type selection

    Returns:
        Tuple of Gradio update objects for different UI components
    """
    if model_type == "Only Using Textual Description":
        return (
            gr.update(visible=False),  # control_image_input
            gr.update(visible=False),  # style_strength_ratio
            gr.update(visible=False),  # ip_adapter_strength
        )
    elif model_type == "Using Ref Images":
        return (
            gr.update(visible=True),  # control_image_input
            gr.update(visible=True),  # style_strength_ratio
            gr.update(visible=False),  # ip_adapter_strength
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def apply_style_positive(style_name: str, positive: str) -> str:
    """Apply style template to a single positive prompt.

    Args:
        style_name: Name of style from templates
        positive: Positive prompt text

    Returns:
        String with style applied to prompt
    """
    p, _ = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = "") -> Tuple[List[str], str]:
    """Apply style template to a list of prompts and a negative prompt.

    Args:
        style_name: Name of style from templates
        positives: List of positive prompts
        negative: Negative prompt text

    Returns:
        Tuple containing styled positive prompts and negative prompt
    """
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + " " + negative


def array2string(arr: List[str]) -> str:
    """Convert an array of strings to a single string with newlines.

    Args:
        arr: List of strings

    Returns:
        Single string with elements joined by newlines
    """
    return "\n".join(arr)


def string2array(s: str) -> List[str]:
    """Convert a newline-separated string to a list of strings.

    Args:
        s: String with newlines

    Returns:
        List of strings split by newlines
    """
    return [line.strip() for line in s.splitlines() if line.strip()]


def swap_to_gallery(images):
    """Update UI to show gallery view with uploaded images.

    Args:
        images: List of uploaded images

    Returns:
        Tuple of updates for UI components
    """
    return (
        gr.update(value=images, visible=True),  # uploaded_files
        gr.update(visible=True),                # clear_button
        gr.update(visible=False),               # files
    )


def remove_back_to_files():
    """Update UI to go back to file upload view.

    Returns:
        Tuple of updates for UI components
    """
    return (
        gr.update(visible=False),  # uploaded_files
        gr.update(visible=False),  # clear_button
        gr.update(visible=True),   # files
    )


def create_ui() -> gr.Blocks:
    """Create the main Gradio UI interface.

    Returns:
        gr.Blocks: The Gradio blocks interface
    """
    # Get list of available fonts
    available_fonts = [f for f in os.listdir("./fonts") if f.endswith(".ttf")]
    
    # Get list of models
    from utils.load_models_utils import get_models_dict
    models_dict = get_models_dict()
    
    # CSS for styling
    css = """
    #color-bg{display:flex;justify-content: center;align-items: center;}
    .color-bg-item{width: 100%; height: 32px}
    #main_button{width:100%}
    <style>
    """

    # Title and header content
    title = r"""
    <h1 align="center">StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</h1>
    """

    description = r"""
    <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'><b>StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</b></a>.<br>
    ❗️❗️❗️[<b>Important</b>] Personalization steps:<br>
    1️⃣ Enter a Textual Description for Character, if you add the Ref-Image, making sure to <b>follow the class word</b> you want to customize with the <b>trigger word</b>: `img`, such as: `man img` or `woman img` or `girl img`.<br>
    2️⃣ Enter the prompt array, each line corrsponds to one generated image.<br>
    3️⃣ Choose your preferred style template.<br>
    4️⃣ Click the <b>Submit</b> button to start customizing.
    """

    article = r"""
    If StoryDiffusion is helpful, please help to ⭐ the <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'>Github Repo</a>. Thanks!
    [![GitHub Stars](https://img.shields.io/github/stars/HVision-NKU/StoryDiffusion?style=social)](https://github.com/HVision-NKU/StoryDiffusion)
    ---
    📝 **Citation**
    <br>
    If our work is useful for your research, please consider citing:

    ```bibtex
    @article{Zhou2024storydiffusion,
      title={StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation},
      author={Zhou, Yupeng and Zhou, Daquan and Cheng, Ming-Ming and Feng, Jiashi and Hou, Qibin},
      year={2024}
    }
    ```
    📋 **License**
    <br>
    Apache-2.0 LICENSE.

    📧 **Contact**
    <br>
    If you have any questions, please feel free to reach me out at <b>ypzhousdu@gmail.com</b>.
    """

    version = r"""
    <h3 align="center">StoryDiffusion Version 0.03</h3>

    <h5>1. Support image reference input. (Cartoon reference images are not supported yet)</h5>
    <h5>2. Support Typesetting Style and Captioning (By default, the prompt is used as the caption for each image. If you need to change the caption, add a # at the end of each line. Only the part after the # will be added as a caption to the image.)</h5>
    <h5>3. [NC] symbol (The [NC] symbol is used as a flag to indicate that no characters should be present in the generated scene images. If you want to do that, prepend the "[NC]" at the beginning of the line. For example, to generate a scene of falling leaves without any character, write: "[NC] The leaves are falling.")</h5>
    <h5 align="center">Tips: For optimal consistency, enter one character per line in the character description box using the [CharacterName] prefix.</h4>
    """

    # Create the Gradio interface
    with gr.Blocks(css=css) as demo:
        # State variables 
        binary_matrixes = gr.State([])
        color_layout = gr.State([])

        # Header
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Group(elem_id="main-image"):
                prompts = []
                colors = []

                with gr.Column(visible=True) as gen_prompt_vis:
                    sd_type = gr.Dropdown(
                        choices=list(models_dict.keys()),
                        value="Unstable",
                        label="SD Model",
                        info="Select pretrained SDXL model",
                    )
                    model_type = gr.Radio(
                        ["Only Using Textual Description", "Using Ref Images"],
                        label="Character Type",
                        value="Only Using Textual Description",
                        info="Control type of the Character",
                    )
                    with gr.Group(visible=False) as control_image_input:
                        files = gr.Files(
                            label="Drag or select photos of your characters",
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
                    general_prompt = gr.Textbox(
                        value="",
                        lines=2,
                        label="(1) Character Description",
                        info="Enter one character per line using [CharacterName] prefix, e.g. [Bob] A man with black hair",
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
                        lines=5,
                        value="",
                        label="(3) Comic Frames (each line corresponds to a frame)",
                        info="Use character names from the Character Description, e.g., [Bob] at home, reading a book",
                        interactive=True,
                    )
                    char_path = gr.Textbox(
                        lines=2,
                        value="",
                        visible=False,
                        label="(Optional) Character files",
                        interactive=True,
                    )
                    with gr.Accordion("(4) Advanced Settings", open=True):
                        font_choice = gr.Dropdown(
                            label="Font",
                            choices=available_fonts,
                            value="Inkfree.ttf" if "Inkfree.ttf" in available_fonts else available_fonts[0] if available_fonts else None,
                            info="Select font for captions",
                            interactive=True,
                        )
                        with gr.Row():
                            sa32_ = gr.Slider(
                                label="Paired Attention (32x32 layers)",
                                minimum=0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                info="Higher values increase character consistency but may reduce creativity",
                            )
                            sa64_ = gr.Slider(
                                label="Paired Attention (64x64 layers)",
                                minimum=0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                info="Higher values increase character consistency but may reduce creativity",
                            )
                        with gr.Row():
                            id_length_ = gr.Slider(
                                label="ID images per character",
                                minimum=1,
                                maximum=4,
                                value=2,
                                step=1,
                                info="Number of reference images per character. More images can improve consistency but requires more GPU memory.",
                            )
                            guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=0.1,
                                maximum=10.0,
                                step=0.1,
                                value=5,
                                info="Higher values make the generation follow the prompt more closely",
                            )
                        with gr.Row():
                            seed_ = gr.Slider(
                                label="Seed", 
                                minimum=-1, 
                                maximum=MAX_SEED, 
                                value=0, 
                                step=1,
                                info="Set to -1 for random seed each time"
                            )
                            randomize_seed_btn = gr.Button("🎲", size="sm")
                        with gr.Row():
                            num_steps = gr.Slider(
                                label="Sampling Steps",
                                minimum=20,
                                maximum=100,
                                step=1,
                                value=35,
                                info="More steps can improve quality but take longer",
                            )
                            comic_type = gr.Radio(
                                [
                                    "No typesetting (default)",
                                    "Four Panel",
                                    "Classic Comic Style",
                                ],
                                value="Classic Comic Style",
                                label="Comic Layout",
                                info="Select the typesetting style",
                            )
                        with gr.Row():
                            G_height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=1024,
                                step=32,
                                value=768,
                                info="Image height (larger sizes require more GPU memory)",
                            )
                            G_width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=1024,
                                step=32,
                                value=768,
                                info="Image width (larger sizes require more GPU memory)",
                            )
                        with gr.Row(visible=False) as advanced_photomaker:
                            style_strength_ratio = gr.Slider(
                                label="Style strength of Ref Image (%)",
                                minimum=15,
                                maximum=50,
                                step=1,
                                value=20,
                                info="Controls how strongly the reference image influences the result",
                                visible=False,
                            )
                            Ip_Adapter_Strength = gr.Slider(
                                label="IP-Adapter Strength",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.5,
                                info="Controls the strength of the IP-Adapter (image conditioning)",
                                visible=False,
                            )
                    final_run_btn = gr.Button("Generate! 😺", variant="primary")

            with gr.Column():
                out_image = gr.Gallery(label="Results", columns=2, height="auto")
                generated_information = gr.Markdown(label="Generation Details", value="", visible=False)
                gr.Markdown(version)
                
        # Set up event handlers
        model_type.change(
            fn=change_visibility_by_model_type,
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

        randomize_seed_btn.click(
            fn=lambda: random.randint(0, MAX_SEED),
            inputs=[],
            outputs=seed_,
        )

        # Define the generation examples
        examples = [
            [
                0,  # seed
                0.5,  # sa32
                0.5,  # sa64
                2,  # id_length
                "[Bob] A man, wearing a black suit\n[Alice] A woman, wearing a white shirt",  # general_prompt
                "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",  # negative_prompt
                array2string([
                    "[Bob] at home, read new paper #at home, The newspaper says there is a treasure house in the forest.",
                    "[Bob] on the road, near the forest",
                    "[Alice] is make a call at home # [Bob] invited [Alice] to join him on an adventure.",
                    "[NC]A tiger appeared in the forest, at night ",
                    "[NC] The car on the road, near the forest #They drives to the forest in search of treasure.",
                    "[Bob] very frightened, open mouth, in the forest, at night",
                    "[Alice] very frightened, open mouth, in the forest, at night",
                    "[Bob]  and [Alice] running very fast, in the forest, at night",
                    "[NC] A house in the forest, at night #Suddenly, They discovers the treasure house!",
                    "[Bob]  and [Alice]  in the house filled with  treasure, laughing, at night #He is overjoyed inside the house.",
                ]),  # prompt_array
                "Comic book",  # style
                "Only Using Textual Description",  # model_type
                "",  # files
                768,  # G_height
                768,  # G_width
            ],
            [
                0,  # seed
                0.5,  # sa32
                0.5,  # sa64
                2,  # id_length
                "[Bob] A man img, wearing a black suit\n[Alice] A woman img, wearing a white shirt",  # general_prompt
                "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",  # negative_prompt
                array2string([
                    "[Bob] at home, read new paper #at home, The newspaper says there is a treasure house in the forest.",
                    "[Bob] on the road, near the forest",
                    "[Alice] is make a call at home # [Bob] invited [Alice] to join him on an adventure.",
                    "[NC] The car on the road, near the forest #They drives to the forest in search of treasure.",
                    "[NC]A tiger appeared in the forest, at night ",
                    "[Bob] very frightened, open mouth, in the forest, at night",
                    "[Alice] very frightened, open mouth, in the forest, at night",
                    "[Bob]  running very fast, in the forest, at night",
                    "[NC] A house in the forest, at night #Suddenly, They discovers the treasure house!",
                    "[Bob]  in the house filled with  treasure, laughing, at night #They are overjoyed inside the house.",
                ]),  # prompt_array
                "Comic book",  # style
                "Using Ref Images",  # model_type
                ["./examples/twoperson/1.jpeg", "./examples/twoperson/2.png"],  # files
                1024,  # G_height
                1024,  # G_width
            ],
        ]

        # Add examples to the interface
        gr.Examples(
            examples=examples,
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
        
        # Add footer
        gr.Markdown(article)

    return demo