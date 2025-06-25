#!/usr/bin/env python3
"""
StoryDiffusion Low VRAM Gradio Application

This is the main entry point for the StoryDiffusion application.
It launches a Gradio interface for consistent character generation
across multiple images using Stable Diffusion XL.
"""

from storydiffusion.ui.app import create_demo
from storydiffusion.config import DEVICE
from storydiffusion.models.pipeline import download_photomaker_model, initialize_pipeline
from storydiffusion.config import global_state

# Print device information
print(f"Using device: {DEVICE}")

# Download PhotoMaker model if needed
print("Ensuring PhotoMaker model is available...")
photomaker_path = download_photomaker_model()
print(f"PhotoMaker model path: {photomaker_path}")

# Initialize default pipeline
print("Initializing default pipeline...")
global_state.pipe = initialize_pipeline()
global_state.unet = global_state.pipe.unet
global_state.cur_model_type = "Unstable-original"
print("Pipeline initialized successfully")

# Create and launch the Gradio demo
if __name__ == "__main__":
    print("Creating Gradio interface...")
    demo = create_demo()
    
    print("Launching StoryDiffusion...")
    demo.launch(server_name="0.0.0.0", share=False)