# StoryDiffusion Code Refactoring

This document outlines the refactoring that was done to the StoryDiffusion codebase to improve its modularity, maintainability, and reusability.

## Refactoring Goals

1. **Modularity**: Split the monolithic application into logical modules
2. **Reusability**: Make components reusable across different contexts
3. **Maintainability**: Improve code organization and documentation
4. **Flexibility**: Easier adaptation for different use cases

## Project Structure

The refactored codebase is organized as follows:

```
StoryDiffusion/
├── run_app.py             # Main entry point script
├── storydiffusion/        # Core package
│   ├── __init__.py        # Package initialization
│   ├── __main__.py        # Entry point for running as a module
│   ├── config.py          # Configuration settings and constants
│   ├── device.py          # Device management utilities
│   ├── generation/        # Image generation engine
│   │   ├── __init__.py
│   │   └── engine.py      # Core generation logic
│   ├── models/            # Model management
│   │   ├── __init__.py
│   │   ├── attention.py   # Attention processors
│   │   └── loader.py      # Model loading utilities
│   ├── ui/                # User interface
│   │   ├── __init__.py
│   │   ├── callbacks.py   # Gradio callbacks
│   │   └── interface.py   # Gradio interface definition
│   └── utils/             # Utility functions
│       └── __init__.py
└── CLAUDE.md              # Documentation for agentic assistants
```

## Key Improvements

### Configuration Management
- Introduced dataclasses for strong typing and better configuration
- Centralized constants and default values
- Clear separation of model configuration from runtime settings

### Device Management
- Enhanced device detection and optimization
- Better memory management with explicit clearing
- Consistent handling across CUDA, MPS, and CPU

### Model Management
- Clear separation of model loading from usage
- Better error handling and reporting
- Support for different loading strategies (pre-trained, from disk)

### Generation Logic
- Character consistency mechanism as a separate module
- Clear phases for reference generation and story frame generation
- Improved comic layout creation

### UI Interface
- Separation of interface definition from callbacks
- More organized component structure
- Better error handling and user feedback

## Usage

### Running the Application

The refactored application can be run in two ways:

1. Direct script execution:
```bash
python run_app.py [options]
```

2. As a Python module:
```bash
python -m storydiffusion [options]
```

Available options:
- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: 7860)
- `--share`: Create a public URL
- `--debug`: Enable debug mode
- `--quiet`: Hide verbose output
- `--auth`: Username and password in the format 'username:password'
- `--concurrency`: Number of concurrent tasks (default: 1)

### API Usage

The refactored code can also be used programmatically:

```python
from storydiffusion.config import GenerationSettings, ModelConfig
from storydiffusion.device import initialize_device
from storydiffusion.models.loader import load_stable_diffusion_pipeline
from storydiffusion.generation.engine import generate_story

# Initialize device
device, dtype = initialize_device()

# Configure model and generation settings
model_config = ModelConfig(
    name="SDXL", 
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True
)

settings = GenerationSettings(
    num_inference_steps=50,
    guidance_scale=7.5,
    height=768,
    width=768,
    sa32_strength=0.5,
    sa64_strength=0.5,
    id_length=2,
    seed=42
)

# Load model
pipe, _ = load_stable_diffusion_pipeline(model_config, device, dtype)

# Generate story
images, metadata = generate_story(
    model_config=model_config,
    character_prompt="[Bob] A man with black hair\n[Alice] A woman with blonde hair",
    prompt_array="[Bob] at home\n[Alice] in the garden\n[Bob] and [Alice] at the beach",
    settings=settings,
    style_name="Comic book",
    negative_prompt="bad anatomy, blurry",
    pipe=pipe
)
```

## Future Improvements

1. Implement unit tests for core functionality
2. Add type hints for all functions and methods
3. Improve error handling and recovery
4. Create a dedicated logging system
5. Add support for different UI frameworks beyond Gradio