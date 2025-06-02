#!/usr/bin/env python
"""
Main entry point for StoryDiffusion application.
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

from storydiffusion.device import initialize_device
from storydiffusion.ui.callbacks import setup_callbacks
from storydiffusion.ui.interface import create_ui


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run StoryDiffusion UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--quiet", action="store_true", help="Hide verbose output")
    parser.add_argument("--auth", type=str, help="Username and password in the format 'username:password'")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent tasks")
    return parser.parse_args()


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_args()
    
    # Print welcome message
    print("=" * 80)
    print("StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation")
    print("=" * 80)
    
    # Initialize device and detect capabilities
    device, dtype = initialize_device()
    print(f"Using device: {device} with {dtype}")
    
    # Create UI
    demo = create_ui()
    
    # Set up callback functions
    setup_callbacks(demo)
    
    # Configure auth
    auth = None
    if args.auth:
        if ":" in args.auth:
            username, password = args.auth.split(":", 1)
            auth = (username, password)
        else:
            print("Warning: Invalid auth format. Should be 'username:password'")
    
    # Launch the app
    print(f"Starting StoryDiffusion UI on {args.host}:{args.port}")
    demo.queue(concurrency_count=args.concurrency, max_size=10)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        debug=args.debug,
        quiet=args.quiet
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping StoryDiffusion gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Error running StoryDiffusion: {e}")
        sys.exit(1)