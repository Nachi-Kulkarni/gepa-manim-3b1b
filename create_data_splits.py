#!/usr/bin/env python3
"""
Create training/validation splits from the filtered dataset for GEPA training.
"""

import os
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple

def load_dataset_examples() -> List[Dict]:
    """Load all examples from the filtered dataset."""
    dataset_dir = Path('dataset/examples')
    examples = []
    
    # Get all video directories
    video_dirs = [d for d in dataset_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('video')]
    
    for video_dir in video_dirs:
        # Load metadata
        metadata_file = video_dir / 'metadata.txt'
        if not metadata_file.exists():
            continue
            
        with open(metadata_file, 'r') as f:
            metadata_content = f.read()
        
        # Parse metadata
        metadata = {}
        for line in metadata_content.split('\n'):
            if ': ' in line:
                key, value = line.split(': ', 1)
                metadata[key] = value
        
        # Load transcript
        transcript_file = video_dir / 'transcript.py'
        transcript_content = ""
        if transcript_file.exists():
            with open(transcript_file, 'r') as f:
                # Extract transcript content
                content = f.read()
                # Find transcript between triple quotes
                import re
                transcript_match = re.search(r'transcript = """(.*?)"""', content, re.DOTALL)
                if transcript_match:
                    transcript_content = transcript_match.group(1).strip()
        
        # Load code files
        code_files = []
        for code_file in video_dir.glob('code*.py'):
            with open(code_file, 'r') as f:
                code_content = f.read()
            code_files.append({
                'filename': code_file.name,
                'content': code_content
            })
        
        example = {
            'video_id': metadata.get('Video ID', ''),
            'title': metadata.get('Title', ''),
            'url': metadata.get('URL', ''),
            'published': metadata.get('Published', ''),
            'quality_score': int(metadata.get('Quality Score', 0)),
            'code_files': code_files,
            'transcript': transcript_content,
            'metadata': metadata,
            'directory': video_dir.name
        }
        
        examples.append(example)
    
    return examples

def create_splits(examples: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create train/validation/test splits."""
    # Sort by quality score to ensure good distribution
    examples_sorted = sorted(examples, key=lambda x: x['quality_score'], reverse=True)
    
    total_examples = len(examples_sorted)
    train_size = int(total_examples * train_ratio)
    val_size = int(total_examples * val_ratio)
    test_size = total_examples - train_size - val_size
    
    # Stratified sampling: distribute high-quality examples across splits
    train_set = []
    val_set = []
    test_set = []
    
    # Distribute examples in rounds to ensure quality distribution
    for i, example in enumerate(examples_sorted):
        if i % 5 < 3:  # 60% to train
            train_set.append(example)
        elif i % 5 < 4:  # 20% to val
            val_set.append(example)
        else:  # 20% to test
            test_set.append(example)
    
    # Adjust sizes to match desired ratios
    while len(train_set) > train_size:
        val_set.append(train_set.pop())
    while len(val_set) > val_size:
        test_set.append(val_set.pop())
    
    return train_set, val_set, test_set

def create_dspy_examples(examples: List[Dict], task_type: str) -> List:
    """Convert examples to DSPy Example format."""
    dspy_examples = []
    
    for example in examples:
        if task_type == 'code':
            # For code generation: input=title+transcript, output=code
            input_text = f"Title: {example['title']}\n\nTranscript: {example['transcript'][:500]}..."
            output_code = example['code_files'][0]['content'] if example['code_files'] else ""
            
            dspy_example = {
                'video_title': example['title'],
                'transcript_excerpt': example['transcript'][:1000],
                'target_code': output_code,
                'quality_score': example['quality_score']
            }
            
        elif task_type == 'transcript':
            # For transcript generation: input=title+code, output=transcript
            input_text = f"Title: {example['title']}\n\nCode: {example['code_files'][0]['content'][:500] if example['code_files'] else ''}"
            
            dspy_example = {
                'video_title': example['title'],
                'code_excerpt': example['code_files'][0]['content'][:1000] if example['code_files'] else '',
                'target_transcript': example['transcript'],
                'quality_score': example['quality_score']
            }
        
        dspy_examples.append(dspy_example)
    
    return dspy_examples

def save_splits(train_set: List[Dict], val_set: List[Dict], test_set: List[Dict]):
    """Save the data splits to files."""
    splits_dir = Path('dataset/splits')
    splits_dir.mkdir(exist_ok=True)
    
    # Save raw splits
    with open(splits_dir / 'train.json', 'w') as f:
        json.dump(train_set, f, indent=2)
    
    with open(splits_dir / 'val.json', 'w') as f:
        json.dump(val_set, f, indent=2)
    
    with open(splits_dir / 'test.json', 'w') as f:
        json.dump(test_set, f, indent=2)
    
    # Create DSPy examples
    train_code = create_dspy_examples(train_set, 'code')
    val_code = create_dspy_examples(val_set, 'code')
    test_code = create_dspy_examples(test_set, 'code')
    
    train_transcript = create_dspy_examples(train_set, 'transcript')
    val_transcript = create_dspy_examples(val_set, 'transcript')
    test_transcript = create_dspy_examples(test_set, 'transcript')
    
    # Save DSPy examples
    with open(splits_dir / 'train_code.json', 'w') as f:
        json.dump(train_code, f, indent=2)
    
    with open(splits_dir / 'val_code.json', 'w') as f:
        json.dump(val_code, f, indent=2)
    
    with open(splits_dir / 'test_code.json', 'w') as f:
        json.dump(test_code, f, indent=2)
    
    with open(splits_dir / 'train_transcript.json', 'w') as f:
        json.dump(train_transcript, f, indent=2)
    
    with open(splits_dir / 'val_transcript.json', 'w') as f:
        json.dump(val_transcript, f, indent=2)
    
    with open(splits_dir / 'test_transcript.json', 'w') as f:
        json.dump(test_transcript, f, indent=2)

def print_statistics(train_set: List[Dict], val_set: List[Dict], test_set: List[Dict]):
    """Print dataset statistics."""
    print("📊 Dataset Split Statistics")
    print("=" * 40)
    print(f"Total examples: {len(train_set) + len(val_set) + len(test_set)}")
    print(f"Training set: {len(train_set)} examples")
    print(f"Validation set: {len(val_set)} examples")
    print(f"Test set: {len(test_set)} examples")
    
    # Quality score distribution
    train_scores = [ex['quality_score'] for ex in train_set]
    val_scores = [ex['quality_score'] for ex in val_set]
    test_scores = [ex['quality_score'] for ex in test_set]
    
    print(f"\n📈 Quality Score Distribution:")
    print(f"Training: avg={sum(train_scores)/len(train_scores):.0f}, min={min(train_scores)}, max={max(train_scores)}")
    print(f"Validation: avg={sum(val_scores)/len(val_scores):.0f}, min={min(val_scores)}, max={max(val_scores)}")
    print(f"Test: avg={sum(test_scores)/len(test_scores):.0f}, min={min(test_scores)}, max={max(test_scores)}")
    
    # Code files distribution
    train_code_counts = [len(ex['code_files']) for ex in train_set]
    val_code_counts = [len(ex['code_files']) for ex in val_set]
    test_code_counts = [len(ex['code_files']) for ex in test_set]
    
    print(f"\n💻 Code Files per Example:")
    print(f"Training: avg={sum(train_code_counts)/len(train_code_counts):.1f}")
    print(f"Validation: avg={sum(val_code_counts)/len(val_code_counts):.1f}")
    print(f"Test: avg={sum(test_code_counts)/len(test_code_counts):.1f}")

def main():
    """Main function to create dataset splits."""
    print("🔧 Creating training/validation splits for GEPA...")
    
    # Load all examples
    examples = load_dataset_examples()
    print(f"📁 Loaded {len(examples)} examples from dataset")
    
    if not examples:
        print("❌ No examples found in dataset!")
        return
    
    # Create splits
    train_set, val_set, test_set = create_splits(examples)
    
    # Print statistics
    print_statistics(train_set, val_set, test_set)
    
    # Save splits
    save_splits(train_set, val_set, test_set)
    
    print(f"\n✅ Dataset splits created successfully!")
    print(f"📁 Saved to: dataset/splits/")
    
    # Show top examples from each split
    print(f"\n🏆 Top Training Examples:")
    for i, ex in enumerate(sorted(train_set, key=lambda x: x['quality_score'], reverse=True)[:3]):
        print(f"  {i+1}. Score: {ex['quality_score']} | {ex['title'][:50]}...")
    
    print(f"\n🥈 Top Validation Examples:")
    for i, ex in enumerate(sorted(val_set, key=lambda x: x['quality_score'], reverse=True)[:3]):
        print(f"  {i+1}. Score: {ex['quality_score']} | {ex['title'][:50]}...")

if __name__ == "__main__":
    main()