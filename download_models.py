#!/usr/bin/env python3
"""
Download required models for StoryDiffusion with modern adapters
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.load_models_utils import download_instantid_models
from huggingface_hub import hf_hub_download

def download_sd35_ip_adapter():
    """Download SD 3.5 IP-Adapter models"""
    print("Downloading SD 3.5 IP-Adapter...")
    try:
        # Download SD 3.5 IP-Adapter
        hf_hub_download(
            repo_id="InstantX/SD3.5-Large-IP-Adapter",
            filename="ip-adapter_sd3.5.safetensors",
            local_dir="models/ip-adapter-sd35"
        )
        hf_hub_download(
            repo_id="InstantX/SD3.5-Large-IP-Adapter",
            filename="image_encoder_config.json",
            local_dir="models/ip-adapter-sd35"
        )
        print("SD 3.5 IP-Adapter downloaded successfully")
    except Exception as e:
        print(f"Error downloading SD 3.5 IP-Adapter: {e}")

def main():
    print("=== Downloading models for StoryDiffusion ===")
    
    # Create model directories
    os.makedirs("models/antelopev2", exist_ok=True)
    os.makedirs("models/instantid", exist_ok=True)
    os.makedirs("models/ip-adapter-sd35", exist_ok=True)
    
    # Download InstantID models
    print("\n1. Downloading InstantID models...")
    download_instantid_models()
    
    # Download SD 3.5 IP-Adapter
    print("\n2. Downloading SD 3.5 IP-Adapter...")
    download_sd35_ip_adapter()
    
    print("\n=== All models downloaded successfully! ===")
    print("\nYou can now use:")
    print("- InstantID for SDXL models (better than PhotoMaker)")
    print("- IP-Adapter for SD 3.5 models")

if __name__ == "__main__":
    main()