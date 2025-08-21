# SD 3.5 TensorRT Support Notes

## Current Status
StoryDiffusion is built for SDXL (Stable Diffusion XL) models, which use a different architecture than SD 3.5 models.

### Key Differences:
1. **SDXL**: Uses UNet architecture, compatible with `StableDiffusionXLPipeline`
2. **SD 3.5**: Uses MMDiT (Multimodal Diffusion Transformer) architecture
3. **TensorRT models**: Require special ONNX runtime and TensorRT optimizations

## Why SD 3.5 TensorRT doesn't work currently:
- The `stabilityai/stable-diffusion-3.5-large-tensorrt` model contains ONNX exports optimized for TensorRT
- StoryDiffusion expects standard diffusers format models
- SD 3.5 requires `StableDiffusion3Pipeline` instead of `StableDiffusionXLPipeline`

## Available SDXL Models:
The following models are compatible and configured:
- `stabilityai/stable-diffusion-xl-base-1.0` - Standard SDXL
- `stabilityai/sdxl-turbo` - Faster SDXL variant
- `RunDiffusion/Juggernaut-XL-v9` - High quality SDXL variant
- `stablediffusionapi/sdxl-unstable-diffusers-y` - Artistic SDXL variant

## Future SD 3.5 Support:
To add SD 3.5 support would require:
1. Adding `StableDiffusion3Pipeline` support in `load_models_utils.py`
2. Updating the PhotoMaker adapter for SD 3.5 architecture
3. Modifying attention processors for MMDiT
4. For TensorRT: Adding ONNX runtime and TensorRT inference code

## RTX 5090 Optimization:
Your RTX 5090 is already optimized for SDXL models with:
- BF16 precision support
- TF32 enabled for better performance
- Flash Attention support
- 24GB VRAM fully utilized