"""
Main module entry point for StoryDiffusion package.
Allows running with `python -m storydiffusion`
"""

import sys
import importlib.util

def main():
    """Import and run the main application."""
    try:
        # Try to dynamically import the run_app module
        spec = importlib.util.spec_from_file_location(
            "run_app", 
            "run_app.py"
        )
        run_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_app)
        
        # Run the main function
        run_app.main()
    except Exception as e:
        print(f"Error loading StoryDiffusion application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()