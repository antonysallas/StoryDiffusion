"""Configuration and constants for StoryDiffusion."""

import torch
import numpy as np
import gc
from typing import Dict, Any, Tuple
from utils.style_template import styles
from utils.load_models_utils import get_models_dict

# Utility Functions
def get_device_settings() -> Tuple[str, torch.dtype]:
    """Determine appropriate device and dtype based on system capabilities."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

def clear_device_cache(device: str) -> None:
    """Clear memory cache for the appropriate device."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        gc.collect()
        # Create and delete a dummy tensor to trigger memory cleanup
        dummy = torch.ones(1, device=device)
        del dummy
    # CPU doesn't need explicit cache clearing
    gc.collect()  # General garbage collection for all devices

def apply_device_optimizations(pipe, device: str):
    """Apply device-specific optimizations to the pipeline."""
    pipe = pipe.to(device)

    # Apply optimizations based on device capabilities
    if device == "cuda":
        try:
            pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        except AttributeError:
            pass  # FreeU not available in this pipeline version

    # Common optimizations across all devices
    try:
        pipe.enable_vae_slicing()
    except AttributeError:
        pass  # VAE slicing not available

    # Device-specific model offloading - skip for better performance
    # Note: CPU offloading trades speed for memory, keeping disabled for performance
    # if device != "mps":
    #     try:
    #         pipe.enable_model_cpu_offload()
    #     except AttributeError:
    #         pass

    return pipe

# Device configuration
DEVICE, DTYPE = get_device_settings()
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# Model configuration
MODELS_DICT = get_models_dict()
DEFAULT_MODEL = "Unstable"
PHOTOMAKER_REPO_ID = "TencentARC/PhotoMaker"
PHOTOMAKER_FILENAME = "photomaker-v1.bin"
LOCAL_DATA_DIR = "data/"

# Generation configuration
MAX_SEED = np.iinfo(np.int32).max
DEFAULT_NUM_STEPS = 35
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 768
DEFAULT_SA32 = 0.5
DEFAULT_SA64 = 0.5
DEFAULT_ID_LENGTH = 1
DEFAULT_STYLE_STRENGTH = 20

# Style configuration
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Japanese Anime"

# UI configuration
CANVAS_HTML = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
LOAD_JS = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

GET_JS_COLORS = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

CSS = """
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
"""

# Title and description
TITLE = r"""
<h1 align="center">StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</h1>
"""

DESCRIPTION = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'><b>StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</b></a>.<br>
❗️❗️❗️[<b>Important</b>] Personalization steps:<br>
1️⃣ Enter a Textual Description for Character, if you add the Ref-Image, making sure to <b>follow the class word</b> you want to customize with the <b>trigger word</b>: `img`, such as: `man img` or `woman img` or `girl img`.<br>
2️⃣ Enter the prompt array, each line corrsponds to one generated image.<br>
3️⃣ Choose your preferred style template.<br>
4️⃣ Click the <b>Submit</b> button to start customizing.
"""

ARTICLE = r"""

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

VERSION = r"""
<h3 align="center">StoryDiffusion Version 0.02 (test version)</h3>

<h5 >1. Support image ref image. (Cartoon Ref image is not support now)</h5>
<h5 >2. Support Typesetting Style and Captioning.(By default, the prompt is used as the caption for each image. If you need to change the caption, add a # at the end of each line. Only the part after the # will be added as a caption to the image.)</h5>
<h5 >3. [NC]symbol (The [NC] symbol is used as a flag to indicate that no characters should be present in the generated scene images. If you want do that, prepend the "[NC]" at the beginning of the line. For example, to generate a scene of falling leaves without any character, write: "[NC] The leaves are falling.")</h5>
<h5 align="center">Tips: </h4>
"""

# Global state configuration
class GlobalStateSingleton:
    """Container for global state variables (Singleton pattern)."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalStateSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Attention tracking
        self.attn_count = 0
        self.total_count = 0
        self.cur_step = 0
        self.id_length = DEFAULT_ID_LENGTH
        self.total_length = self.id_length + 1
        self.cur_model_type = ""
        
        # Attention processors
        self.attn_procs = {}
        
        # Generation mode
        self.write = False
        
        # Paired attention strengths
        self.sa32 = DEFAULT_SA32
        self.sa64 = DEFAULT_SA64
        
        # Image dimensions
        self.height = DEFAULT_HEIGHT
        self.width = DEFAULT_WIDTH
        
        # Character tracking
        self.character_dict = {}
        self.character_index_dict = {}
        self.invert_character_index_dict = {}
        self.cur_character = []
        self.ref_indexs_dict = {}
        self.ref_totals = []
        
        # Attention indices
        self.indices1024 = None
        self.indices4096 = None
        self.mask1024 = None
        self.mask4096 = None
        
        # Pipeline and model
        self.pipe = None
        self.unet = None
        self.sd_model_path = MODELS_DICT[DEFAULT_MODEL]["path"]
        self.single_files = MODELS_DICT[DEFAULT_MODEL]["single_files"]
        
        self._initialized = True

# Create global instance
global_state = GlobalStateSingleton()