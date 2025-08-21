#!/usr/bin/env python3
"""
Test script to verify model configuration and architecture detection
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.load_models_utils import get_models_dict
import torch

def test_model_config():
    print("=== Model Configuration Test ===")
    
    models_dict = get_models_dict()
    
    for model_name, model_info in models_dict.items():
        architecture = model_info.get("architecture", "unknown")
        print(f"\n{model_name}:")
        print(f"  Architecture: {architecture}")
        print(f"  Path: {model_info['path']}")
        print(f"  Single Files: {model_info.get('single_files', False)}")
        print(f"  Use Safetensors: {model_info.get('use_safetensors', True)}")

def test_gpu_capabilities():
    print("\n=== GPU Test ===")
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        print(f"GPU Capability: {torch.cuda.get_device_capability()}")
        
        # Test dtype selection
        if torch.cuda.get_device_capability()[0] >= 8:
            print("Recommended dtype: bfloat16 (RTX 30/40/50 series)")
        else:
            print("Recommended dtype: float16")
    else:
        print("CUDA Not Available")

def test_directories():
    print("\n=== Directory Test ===")
    dirs_to_check = [
        "models/antelopev2",
        "models/instantid", 
        "models/ip-adapter-sd35",
        "data"
    ]
    
    for dir_path in dirs_to_check:
        exists = os.path.exists(dir_path)
        print(f"{dir_path}: {'✓' if exists else '✗'}")
        if exists:
            files = os.listdir(dir_path)
            if files:
                print(f"  Files: {len(files)} files")
            else:
                print("  (empty)")

if __name__ == "__main__":
    test_model_config()
    test_gpu_capabilities()
    test_directories()
    print("\n=== Test Complete ===")