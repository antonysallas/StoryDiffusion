#!/usr/bin/env python3
"""
Convert .bin model files to .safetensors format
"""
import torch
from safetensors.torch import save_file
import os
import argparse

def convert_bin_to_safetensors(bin_path, safetensors_path=None):
    """Convert a .bin file to .safetensors format"""
    
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")
    
    if safetensors_path is None:
        safetensors_path = bin_path.replace('.bin', '.safetensors')
    
    print(f"Converting {bin_path} to {safetensors_path}...")
    
    # Load the .bin file
    try:
        state_dict = torch.load(bin_path, map_location='cpu')
        print(f"Loaded {len(state_dict)} tensors from .bin file")
        
        # Display some info about the tensors
        total_params = 0
        flat_dict = {}
        
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        # Flatten the dictionary if it contains nested structures
        flat_dict = flatten_dict(state_dict)
        
        for key, tensor in flat_dict.items():
            if hasattr(tensor, 'numel'):
                total_params += tensor.numel()
                print(f"  {key}: {tensor.shape} ({tensor.dtype})")
            else:
                print(f"  {key}: {type(tensor)} (not a tensor)")
        
        print(f"Total parameters: {total_params:,}")
        
        # Save as safetensors (using flattened dict)
        save_file(flat_dict, safetensors_path)
        print(f"✓ Successfully converted to {safetensors_path}")
        
        # Verify the conversion
        from safetensors.torch import load_file
        loaded_dict = load_file(safetensors_path)
        
        if len(loaded_dict) == len(flat_dict):
            print("✓ Verification passed - same number of tensors")
        else:
            print("⚠ Warning - tensor count mismatch after conversion")
            
        # Check file sizes
        original_size = os.path.getsize(bin_path) / (1024**2)
        new_size = os.path.getsize(safetensors_path) / (1024**2)
        print(f"File size: {original_size:.1f}MB → {new_size:.1f}MB")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert .bin model files to .safetensors")
    parser.add_argument("bin_file", help="Path to the .bin file to convert")
    parser.add_argument("-o", "--output", help="Output .safetensors file path (optional)")
    
    args = parser.parse_args()
    
    convert_bin_to_safetensors(args.bin_file, args.output)

if __name__ == "__main__":
    # Example usage for common IP-Adapter files
    common_paths = [
        "models/ip-adapter-sd35/ip-adapter.bin",
        "models/instantid/ip-adapter.bin"
    ]
    
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("IP-Adapter .bin to .safetensors converter")
        print("Usage:")
        print("  python convert_bin_to_safetensors.py <path_to_bin_file>")
        print("  python convert_bin_to_safetensors.py <path_to_bin_file> -o <output_path>")
        print("\nCommon files to convert:")
        
        for path in common_paths:
            if os.path.exists(path):
                print(f"  ✓ Found: {path}")
                try:
                    convert_bin_to_safetensors(path)
                    print()
                except Exception as e:
                    print(f"  ✗ Error converting {path}: {e}")
                    print()
            else:
                print(f"  - Not found: {path}")
        
        print("Done!")