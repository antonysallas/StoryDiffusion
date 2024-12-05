import yaml
import torch
from diffusers import StableDiffusionXLPipeline
from utils import PhotoMakerStableDiffusionXLPipeline
import os

def get_models_dict():
    with open('config/models.yaml', 'r') as stream:
        try:
            data = yaml.safe_load(stream)

            print(data)
            return data

        except yaml.YAMLError as exc:
          print(f"Error loading config: {exc}")
          return None

def load_models(model_info,device,photomaker_path):
    # Set MPS memory limit if using MPS
    if device == "mps":
        dtype = torch.float32  # MPS needs float32
    else:
        dtype = torch.float16

    path =  model_info["path"]
    single_files =  model_info["single_files"]
    use_safetensors = model_info["use_safetensors"]
    model_type = model_info["model_type"]

    try:
        if model_type == "original":
            if single_files:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    path,
                    torch_dtype=dtype
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    path,
                    torch_dtype=dtype,
                    use_safetensors=use_safetensors
                )

            pipe = pipe.to(device)
            pipe.enable_attention_slicing()

            # Device-specific optimizations
            if device == "mps":
                pipe.enable_vae_slicing()
                pipe.enable_attention_slicing(slice_size="auto")
            elif device == "cuda":
                pipe.enable_sequential_cpu_offload()
                pipe.enable_model_cpu_offload()

        elif model_type == "Photomaker":
            if single_files:
                print("loading from a single_files")
                pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
                    path,
                    torch_dtype=torch.float16
                )
            else:
                pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                    path, torch_dtype=torch.float16, use_safetensors=use_safetensors)
            pipe = pipe.to(device)
            pipe.load_photomaker_adapter(
                os.path.dirname(photomaker_path),
                subfolder="",
                weight_name=os.path.basename(photomaker_path),
                trigger_word="img"  # define the trigger word
            )
            pipe.fuse_lora()
        else:
            raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {model_type}")
        return pipe
    except Exception as e:
      print(f"Error loading model: {e}")
      return None