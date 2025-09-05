#!/usr/bin/env python3
"""
Quick Video Generator - Wrapper for the complete pipeline
Usage: python3 generate_video.py "Topic Name"
"""

import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_video.py 'Video Topic'")
        print("Example: python3 generate_video.py 'Pythagorean Theorem'")
        sys.exit(1)
    
    topic = sys.argv[1]
    print(f"🎬 Generating video for: {topic}")
    
    # Run the pipeline
    cmd = [
        "python3", "video_pipeline.py", 
        topic,
        "--quality", "480p15"  # Fast rendering for testing
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Video generation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Video generation failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()