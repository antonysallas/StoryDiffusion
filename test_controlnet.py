#!/usr/bin/env python3
"""
Test script to verify ControlNet pose model support for SD 3.5
"""
import os
import sys
import torch
from utils.load_models_utils import _load_sd3_controlnet_pose, _load_instantid_controlnet

def test_sd3_controlnet_pose():
    """Test SD3 ControlPose model loading"""
    print("Testing SD3-Controlnet-Pose model loading...")
    
    controlnet = _load_sd3_controlnet_pose()
    
    if controlnet is not None:
        print(f"✅ SD3-Controlnet-Pose loaded successfully!")
        print(f"   Model type: {type(controlnet)}")
        print(f"   Device: {controlnet.device if hasattr(controlnet, 'device') else 'Unknown'}")
        print(f"   Dtype: {controlnet.dtype if hasattr(controlnet, 'dtype') else 'Unknown'}")
        return True
    else:
        print("❌ Failed to load SD3-Controlnet-Pose model")
        return False

def test_instantid_controlnet():
    """Test InstantID ControlNet model loading"""
    print("\nTesting InstantID ControlNet model loading...")
    
    controlnet = _load_instantid_controlnet()
    
    if controlnet is not None:
        print(f"✅ InstantID ControlNet loaded successfully!")
        print(f"   Model type: {type(controlnet)}")
        print(f"   Device: {controlnet.device if hasattr(controlnet, 'device') else 'Unknown'}")
        print(f"   Dtype: {controlnet.dtype if hasattr(controlnet, 'dtype') else 'Unknown'}")
        return True
    else:
        print("❌ Failed to load InstantID ControlNet model")
        return False

def check_model_paths():
    """Check if model paths exist in HF_HOME"""
    print("\nChecking model paths...")
    
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    print(f"HF_HOME: {hf_home}")
    
    # Check SD3-Controlnet-Pose
    sd3_path = os.path.join(hf_home, 'hub', 'models--InstantX--SD3-Controlnet-Pose')
    print(f"SD3-Controlnet-Pose path exists: {os.path.exists(sd3_path)}")
    if os.path.exists(sd3_path):
        snapshots_dir = os.path.join(sd3_path, 'snapshots')
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            print(f"   Snapshots: {snapshots}")
    
    # Check InstantID
    instantid_path = os.path.join(hf_home, 'hub', 'models--InstantX--InstantID')
    print(f"InstantID path exists: {os.path.exists(instantid_path)}")
    if os.path.exists(instantid_path):
        snapshots_dir = os.path.join(instantid_path, 'snapshots')
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            print(f"   Snapshots: {snapshots}")

def main():
    print("ControlNet Pose Model Support Verification")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    
    check_model_paths()
    
    # Test models
    sd3_success = test_sd3_controlnet_pose()
    instantid_success = test_instantid_controlnet()
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY:")
    print(f"SD3-Controlnet-Pose: {'✅ WORKING' if sd3_success else '❌ FAILED'}")
    print(f"InstantID ControlNet: {'✅ WORKING' if instantid_success else '❌ FAILED'}")
    
    if sd3_success and instantid_success:
        print("\n🎉 All ControlNet models verified successfully!")
        print("   - SD 3.5 models can use SD3-Controlnet-Pose for pose control")
        print("   - SDXL models can use InstantID ControlNet for face pose control")
        return True
    else:
        print("\n⚠️  Some ControlNet models failed to load")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)