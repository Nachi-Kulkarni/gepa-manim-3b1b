#!/usr/bin/env python3
"""Regenerate subtitles to match the animation timing structure"""

import re
from pathlib import Path

def extract_narration_content(transcript_file):
    """Extract clean narration content from transcript"""
    with open(transcript_file, 'r') as f:
        content = f.read()
    
    # Remove stage directions and cleanup
    lines = content.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and stage directions
        if line and not line.startswith('[') and not line.startswith('('):
            # Remove any remaining stage direction markers
            clean_line = re.sub(r'\[.*?\]|\(.*?\)', '', line).strip()
            if clean_line:
                clean_lines.append(clean_line)
    
    return ' '.join(clean_lines)

def split_into_phrases(text, target_phrases=50):
    """Split text into approximately target_phrases phrases"""
    # Split into sentences first
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Calculate approximate words per phrase
    total_words = sum(len(s.split()) for s in sentences)
    words_per_phrase = total_words / target_phrases
    
    phrases = []
    current_phrase = ""
    current_words = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        if current_words + sentence_words <= words_per_phrase * 1.5 or not current_phrase:
            current_phrase += sentence + ". "
            current_words += sentence_words
        else:
            if current_phrase.strip():
                phrases.append(current_phrase.strip())
            current_phrase = sentence + ". "
            current_words = sentence_words
    
    if current_phrase.strip():
        phrases.append(current_phrase.strip())
    
    return phrases

def generate_synchronized_subtitles(transcript_file, audio_duration, output_file):
    """Generate subtitles that match the animation timing structure"""
    
    print(f"🎝 Regenerating subtitles for {audio_duration}s audio...")
    
    # Extract clean narration
    narration = extract_narration_content(transcript_file)
    print(f"📝 Extracted narration: {len(narration)} characters")
    
    # Split into phrases that match the 50-cue structure in the animation
    phrases = split_into_phrases(narration, target_phrases=50)
    print(f"🎯 Split into {len(phrases)} phrases")
    
    # Calculate timing based on animation structure
    # The animation uses roughly equal timing per cue
    cue_duration = audio_duration / len(phrases)
    
    # Generate SRT content
    srt_content = ""
    for i, phrase in enumerate(phrases):
        start_time = i * cue_duration
        end_time = (i + 1) * cue_duration
        
        # Convert to SRT time format
        start_srt = format_srt_time(start_time)
        end_srt = format_srt_time(end_time)
        
        srt_content += f"{i + 1}\n"
        srt_content += f"{start_srt} --> {end_srt}\n"
        srt_content += f"{phrase}\n\n"
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    print(f"✅ Generated {len(phrases)} subtitle entries")
    print(f"📁 Saved to: {output_file}")
    
    return output_file

def format_srt_time(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

if __name__ == "__main__":
    # File paths
    transcript_file = "explain_complex_numbers_in_detail_transcript.txt"
    audio_file = "explain_complex_numbers_in_detail_narration.wav"
    output_file = "explain_complex_numbers_in_detail_subtitles_synced.srt"
    
    # Get audio duration
    import subprocess
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', audio_file
        ], capture_output=True, text=True)
        audio_duration = float(result.stdout.strip())
    except:
        audio_duration = 448.25  # Fallback to known duration
    
    # Generate synchronized subtitles
    generate_synchronized_subtitles(transcript_file, audio_duration, output_file)