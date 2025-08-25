#!/usr/bin/env python3
"""
Test script to verify TensorRT model loading fix
"""
import sys
import os
sys.path.append('/home/asallas/workarea/projects/personal/StoryDiffusion')

from utils.load_models_utils import get_models_dict, load_models

def test_tensorrt_model_loading():
    """Test the TensorRT model loading functionality"""
    print("🧪 Testing TensorRT model loading fix...")
    
    # Get models config
    models = get_models_dict()
    
    # Test with SD3.5-Large-RTX5090 model
    if 'SD3.5-Large-RTX5090' not in models:
        print("❌ SD3.5-Large-RTX5090 model not found in config")
        return False
    
    model_info = models['SD3.5-Large-RTX5090']
    print(f"📋 Testing model: {model_info}")
    
    try:
        # Test model loading
        print("🚀 Attempting to load TensorRT model...")
        pipe = load_models(
            model_info,
            device='cuda',
            enable_controlnet=False
        )
        
        if pipe is not None:
            print("✅ Model loaded successfully!")
            print(f"Pipeline type: {type(pipe)}")
            return True
        else:
            print("❌ Model loading returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tensorrt_model_loading()
    sys.exit(0 if success else 1)