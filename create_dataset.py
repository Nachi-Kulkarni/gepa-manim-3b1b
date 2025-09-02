#!/usr/bin/env python3
"""
Script to create 30 examples from 3Blue1Brown videos with code, titles, and transcripts.
Each example will be organized as: dataset/examples/video1/code.py, transcript.py
"""

import re
import os
import shutil
from pathlib import Path

def extract_video_info():
    """Extract video IDs and titles from video_list.txt"""
    with open('docsa/video_list.txt', 'r') as f:
        content = f.read()
    
    # Pattern to match video entries
    pattern = r'(\d+)\.\s+(.*?)\s+ID:\s+([a-zA-Z0-9_-]+)'
    matches = re.findall(pattern, content)
    
    videos = []
    for num, title, video_id in matches:
        videos.append({
            'number': int(num),
            'title': title.strip(),
            'id': video_id
        })
    
    return videos

def extract_transcript(video_id):
    """Extract transcript for a specific video ID from channel_transcripts.txt"""
    try:
        with open('docsa/channel_transcripts.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the transcript section for this video
        pattern = f'VIDEO ID: {video_id}\\nTITLE: (.*?)\\nURL: (.*?)\\nPUBLISHED: (.*?)\\nSTATUS: SUCCESS\\n================================================================================\\n(.*?)(?=\\n================================================================================\\nVIDEO ID:|\\n================================================================================\\n$|$)'
        
        match = re.search(pattern, content, re.DOTALL)
        if match:
            title, url, published, transcript = match.groups()
            return {
                'title': title.strip(),
                'url': url.strip(),
                'published': published.strip(),
                'transcript': transcript.strip()
            }
    except Exception as e:
        print(f"Error extracting transcript for {video_id}: {e}")
    
    return None

def find_matching_code_files(video_title, video_id):
    """Try to find matching code files based on video title and ID"""
    code_files = []
    
    # Search in manim_codes_vids directory
    manim_dir = Path('manim_codes_vids')
    
    if not manim_dir.exists():
        return code_files
    
    # Convert title to filename-friendly format
    title_keywords = re.sub(r'[^\w\s]', '', video_title.lower()).split()
    
    # Search for relevant Python files
    for py_file in manim_dir.rglob('*.py'):
        # Check if filename contains keywords from title
        filename_lower = py_file.stem.lower()
        
        # Score based on keyword matches
        score = 0
        for keyword in title_keywords[:3]:  # Use first 3 keywords
            if keyword in filename_lower and len(keyword) > 2:
                score += 1
        
        if score > 0:
            code_files.append({
                'path': py_file,
                'score': score,
                'name': py_file.name
            })
    
    # Sort by score and return top matches
    code_files.sort(key=lambda x: x['score'], reverse=True)
    return code_files[:3]  # Return top 3 matches

def create_dataset_examples():
    """Create 30 dataset examples"""
    print("Creating dataset examples...")
    
    # Extract video information
    videos = extract_video_info()
    print(f"Found {len(videos)} videos in the list")
    
    # Create dataset directory
    dataset_dir = Path('dataset/examples')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    examples_created = 0
    examples_skipped = 0
    
    # Process all videos
    for i, video in enumerate(videos):
            
        print(f"\nProcessing video {i+1}: {video['title'][:50]}...")
        
        # Extract transcript
        transcript_data = extract_transcript(video['id'])
        if not transcript_data:
            print(f"  ⚠️  No transcript found for video {video['id']}")
            examples_skipped += 1
            continue
        
        # Find matching code files
        code_files = find_matching_code_files(video['title'], video['id'])
        
        if not code_files:
            print(f"  ⚠️  No matching code files found for: {video['title'][:30]}...")
            examples_skipped += 1
            continue
        
        # Create example directory
        example_dir = dataset_dir / f"video{examples_created + 1}"
        example_dir.mkdir(exist_ok=True)
        
        # Copy code files
        for j, code_file in enumerate(code_files[:2]):  # Max 2 code files per example
            dest_path = example_dir / f"code{j+1}.py"
            try:
                shutil.copy2(code_file['path'], dest_path)
                print(f"  ✅ Copied {code_file['name']} -> code{j+1}.py")
            except Exception as e:
                print(f"  ❌ Error copying {code_file['name']}: {e}")
        
        # Create transcript file
        transcript_path = example_dir / "transcript.py"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write(f"Video Title: {transcript_data['title']}\n")
            f.write(f"Video ID: {video['id']}\n")
            f.write(f"URL: {transcript_data['url']}\n")
            f.write(f"Published: {transcript_data['published']}\n")
            f.write('"""\n\n')
            f.write('# Transcript:\n')
            f.write('transcript = """\n')
            f.write(transcript_data['transcript'])
            f.write('\n"""\n')
        
        print(f"  ✅ Created transcript.py")
        
        # Create metadata file
        metadata_path = example_dir / "metadata.txt"
        code_files_count = len([f for f in example_dir.glob('code*.py')])
        transcript_length = len(transcript_data['transcript'])
        
        # Calculate date score (newer videos get higher scores)
        published_date = transcript_data['published'][:10]  # Extract YYYY-MM-DD
        from datetime import datetime
        date_obj = datetime.strptime(published_date, '%Y-%m-%d')
        days_since_2020 = (date_obj - datetime(2020, 1, 1)).days
        date_score = max(0, days_since_2020 // 10)  # 1 point per 10 days since 2020
        
        # Calculate quality score with date preference
        quality_score = code_files_count * 1000 + transcript_length + date_score
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Video Number: {video['number']}\n")
            f.write(f"Video ID: {video['id']}\n")
            f.write(f"Title: {transcript_data['title']}\n")
            f.write(f"URL: {transcript_data['url']}\n")
            f.write(f"Published: {transcript_data['published']}\n")
            f.write(f"Code Files: {code_files_count}\n")
            f.write(f"Transcript Length: {transcript_length} characters\n")
            f.write(f"Date Score: {date_score}\n")
            f.write(f"Quality Score: {quality_score}\n")  # Enhanced score for ranking
        
        print(f"  ✅ Created example {examples_created + 1} in {example_dir}")
        examples_created += 1
    
    print(f"\n🎉 Dataset creation complete!")
    print(f"✅ Created {examples_created} examples")
    print(f"⚠️  Skipped {examples_skipped} videos (no transcript or code)")
    print(f"📁 Examples saved to: {dataset_dir}")
    
    # Create ranking and summary
    create_ranking_summary(dataset_dir, examples_created)

def create_ranking_summary(dataset_dir, total_examples):
    """Create a ranking of examples by quality score"""
    examples_data = []
    
    # Collect data from all examples
    for i in range(1, total_examples + 1):
        example_dir = dataset_dir / f"video{i}"
        if example_dir.exists():
            metadata_file = example_dir / "metadata.txt"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_content = f.read()
                    
                # Parse metadata
                data = {}
                for line in metadata_content.split('\n'):
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        data[key] = value
                
                if data:
                    examples_data.append({
                        'index': i,
                        'title': data.get('Title', 'Unknown'),
                        'code_files': int(data.get('Code Files', 0)),
                        'transcript_length': int(data.get('Transcript Length', '0').split()[0]),
                        'quality_score': int(data.get('Quality Score', 0))
                    })
    
    # Sort by quality score (descending)
    examples_data.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Create summary file
    summary_path = dataset_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"3Blue1Brown Dataset Summary\n")
        f.write(f"===========================\n\n")
        f.write(f"Total Examples Created: {total_examples}\n")
        f.write(f"Videos Processed: 214\n")
        f.write(f"Success Rate: {total_examples/214*100:.1f}%\n\n")
        
        f.write("TOP 30 EXAMPLES BY QUALITY SCORE:\n")
        f.write("(Quality Score = Code Files × 1000 + Transcript Length)\n\n")
        
        for i, example in enumerate(examples_data[:30]):
            f.write(f"{i+1}. [Score: {example['quality_score']}] {example['title']}\n")
            f.write(f"   Code Files: {example['code_files']}, Transcript: {example['transcript_length']} chars\n")
            f.write(f"   📁 video{example['index']}\n\n")
        
        f.write(f"\nALL EXAMPLES (sorted by quality):\n")
        for example in examples_data:
            f.write(f"video{example['index']}: {example['quality_score']} ({example['code_files']} code, {example['transcript_length']} chars)\n")
    
    # Create top examples directory with symlinks/shortcuts
    top_dir = dataset_dir / "top_examples"
    top_dir.mkdir(exist_ok=True)
    
    print(f"\n🏆 TOP 10 EXAMPLES BY QUALITY:")
    for i, example in enumerate(examples_data[:10]):
        print(f"{i+1}. Score: {example['quality_score']} | {example['title'][:60]}...")
        print(f"   Code Files: {example['code_files']} | Transcript: {example['transcript_length']} chars")
        print(f"   📁 video{example['index']}")
        
        # Create symlink to top example
        source_dir = dataset_dir / f"video{example['index']}"
        target_dir = top_dir / f"top{i+1}_{example['index']}"
        try:
            target_dir.symlink_to(source_dir, target_is_directory=True)
        except:
            # If symlink fails, copy the metadata
            import shutil
            shutil.copy2(source_dir / "metadata.txt", target_dir / "metadata.txt")
    
    print(f"\n📊 Top examples linked/copied to: {top_dir}")
    print(f"📋 Full ranking saved to: {summary_path}")

if __name__ == "__main__":
    create_dataset_examples()