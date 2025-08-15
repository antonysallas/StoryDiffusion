import torch
from transformers import AutoTokenizer, AutoModel

def verify_installations():
    print("=== Version Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Current Python Path: {__import__('sys').executable}")

    print("\n=== Device Information ===")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS backend built: {torch.backends.mps.is_built()}")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Current device: {device}")

    print("\n=== Basic Model Load Test ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        print("✓ Model loading successful")
    except Exception as e:
        print(f"× Model test failed: {str(e)}")

if __name__ == "__main__":
    verify_installations()