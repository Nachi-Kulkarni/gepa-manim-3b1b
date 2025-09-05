#!/usr/bin/env python3
"""
Filter dataset to keep only the top 30 examples based on quality score.
"""

import os
import shutil
from pathlib import Path

def extract_top_video_ids():
    """Extract the top 30 video IDs from summary.txt"""
    summary_file = Path('dataset/examples/summary.txt')
    
    if not summary_file.exists():
        print("❌ summary.txt not found!")
        return []
    
    with open(summary_file, 'r') as f:
        content = f.read()
    
    # Extract video IDs from the top 30 entries
    # Pattern: 📁 videoXX
    import re
    pattern = r'📁 video(\d+)'
    matches = re.findall(pattern, content)
    
    # Get first 30 matches
    top_videos = [f"video{match}" for match in matches[:30]]
    
    return top_videos

def filter_dataset():
    """Remove all examples except top 30"""
    print("🔍 Filtering dataset to keep only top 30 examples...")
    
    dataset_dir = Path('dataset/examples')
    
    if not dataset_dir.exists():
        print("❌ dataset/examples directory not found!")
        return
    
    # Get top 30 video IDs
    top_videos = extract_top_video_ids()
    
    if not top_videos:
        print("❌ Could not extract top video IDs!")
        return
    
    print(f"📊 Top 30 videos to keep: {top_videos[:5]}...{top_videos[-5:]}")
    
    # Get all existing video directories
    existing_dirs = [d for d in dataset_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('video')]
    
    videos_to_remove = []
    videos_to_keep = []
    
    for video_dir in existing_dirs:
        if video_dir.name in top_videos:
            videos_to_keep.append(video_dir.name)
        else:
            videos_to_remove.append(video_dir.name)
    
    print(f"📈 Keeping: {len(videos_to_keep)} videos")
    print(f"🗑️  Removing: {len(videos_to_remove)} videos")
    
    # Remove non-top videos
    removed_count = 0
    for video_dir in videos_to_remove:
        dir_path = dataset_dir / video_dir
        try:
            shutil.rmtree(dir_path)
            removed_count += 1
            print(f"  🗑️  Removed {video_dir}")
        except Exception as e:
            print(f"  ❌ Error removing {video_dir}: {e}")
    
    # Update top_examples symlinks
    top_examples_dir = dataset_dir / "top_examples"
    if top_examples_dir.exists():
        print(f"\n🔄 Updating top_examples symlinks...")
        
        # Remove old symlinks
        for item in top_examples_dir.iterdir():
            if item.is_symlink():
                try:
                    item.unlink()
                except Exception as e:
                    print(f"  ⚠️  Could not unlink {item.name}: {e}")
        
        # Create new symlinks for top 10
        for i, video_name in enumerate(top_videos[:10], 1):
            source_path = dataset_dir / video_name
            target_path = top_examples_dir / f"top{i}_{video_name.replace('video', '')}"
            
            if source_path.exists():
                try:
                    target_path.symlink_to(source_path, target_is_directory=True)
                    print(f"  🔗 Created symlink: top{i}_{video_name.replace('video', '')}")
                except Exception as e:
                    print(f"  ⚠️  Could not create symlink for {video_name}: {e}")
    
    print(f"\n✅ Dataset filtering complete!")
    print(f"📊 Final dataset: {len(videos_to_keep)} high-quality examples")
    print(f"🗑️  Removed {removed_count} lower-quality examples")

if __name__ == "__main__":
    filter_dataset()