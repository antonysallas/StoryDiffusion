#!/usr/bin/env python3
"""
StoryDiffusion Launcher
Simple launcher script for the modernized StoryDiffusion application
"""
import subprocess
import sys
import os

def main():
    """Launch StoryDiffusion with optimizations"""
    print("🚀 Starting StoryDiffusion...")
    print("📱 Modern UI with SDXL + SD 3.5 support")
    print("🎮 RTX 5090 optimized")
    print()
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Use the optimized launch script
        subprocess.run(["bash", "run_optimized.sh"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down StoryDiffusion...")
        sys.exit(0)

if __name__ == "__main__":
    main()