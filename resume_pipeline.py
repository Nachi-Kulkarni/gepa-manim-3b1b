#!/usr/bin/env python3
"""Resume pipeline from Step 5: Code generation with timing context"""

import os
import json
from pathlib import Path
from video_pipeline import VideoGenerationPipeline

def resume_from_step_5(video_title, render_quality="480p15"):
    """Resume pipeline from Step 5 using existing files"""
    
    print("🚀 Resuming pipeline from Step 5: Animation Code Generation")
    print("=" * 60)
    
    # Create pipeline
    pipeline = VideoGenerationPipeline()
    
    # Find existing files
    safe_title = pipeline._sanitize_filename(video_title)
    
    # Check for required files
    transcript_file = f"{safe_title}_transcript.txt"
    existing_audio = f"{safe_title}_narration.wav"
    srt_file = f"{safe_title}_subtitles.srt"
    
    # Verify required files exist
    if not Path(transcript_file).exists():
        print(f"❌ Required transcript file not found: {transcript_file}")
        return None
    
    # Read existing data
    print(f"📋 Loading existing files...")
    
    # Read transcript
    with open(transcript_file, 'r') as f:
        transcript = f.read()
    
    # Read narration script (extract from transcript or use cleaned version)
    narration_script = transcript  # For now, use transcript as narration script
    
    # Get audio duration if audio exists
    audio_duration = None
    if Path(existing_audio).exists():
        audio_duration = pipeline.get_audio_duration(existing_audio)
        print(f"🎵 Found existing audio: {existing_audio} ({audio_duration}s")
    
    # Read subtitle file if it exists
    if Path(srt_file).exists():
        print(f"📝 Found existing subtitles: {srt_file}")
    else:
        print(f"⚠️  No subtitle file found: {srt_file}")
        srt_file = None
    
    # Continue from Step 5
    print(f"\n💻 Step 5: Generating synchronized animation code with timing context...")
    
    # Create timing context
    estimated_duration = audio_duration if audio_duration else 60
    timing_context = (
        f"TIMING CONTEXT FOR SYNCHRONIZATION:\n"
        f"- Audio duration: {audio_duration or 'unknown'} seconds\n"
        f"- Estimated video duration: {estimated_duration} seconds\n"
        f"- Subtitle entries: {pipeline._count_subtitle_entries(srt_file) if srt_file else 0} timing cues\n"
        f"- Narration script length: {len(narration_script)} characters\n\n"
        f"SYNCHRONIZATION REQUIREMENTS:\n"
        f"- Animation timing should match narration flow\n"
        f"- Key visual elements should align with spoken explanations\n"
        f"- Use subtitle timing as reference for animation pacing\n"
        f"- Ensure smooth transitions between topics\n"
        f"- Match animation complexity to available time budget\n"
    )
    
    # Generate animation code
    code, reasoning = pipeline.generate_animation_code_with_timing(
        video_title, 
        transcript, 
        narration_script, 
        timing_context,
        srt_file
    )
    
    if code is None:
        print(f"❌ Animation code generation failed: {reasoning}")
        return None
    
    # Step 6: Save outputs
    code_file, transcript_file = pipeline.save_outputs(video_title, transcript, code, reasoning, existing_audio if Path(existing_audio).exists() else None)
    
    # Step 6.5: Perform dry run compilation check
    dry_run_success = pipeline.dry_run_check(code_file)
    
    # If dry run fails, try to fix common issues automatically
    if not dry_run_success and pipeline.refinement_system:
        print(f"\n🔧 Dry run failed - attempting automatic fix...")
        
        # Read the current code
        with open(code_file, 'r') as f:
            current_code = f.read()
        
        # Try to fix common issues automatically
        fixed_code = pipeline._auto_fix_common_issues(current_code)
        
        if fixed_code != current_code:
            print(f"🔧 Applied automatic fixes to code")
            with open(code_file, 'w') as f:
                f.write(fixed_code)
            
            # Retry dry run with fixed code
            dry_run_success = pipeline.dry_run_check(code_file)
            
            if dry_run_success:
                print(f"✅ Automatic fixes successful - dry run passed")
            else:
                print(f"⚠️  Automatic fixes didn't resolve all issues")
        else:
            print(f"⚠️  No automatic fixes applicable")
    
    if not dry_run_success:
        print(f"⚠️  Dry run compilation failed - proceeding anyway but expect rendering issues")
    
    # Step 8: Debug render with error capture and refinement
    print(f"\n🎬 Step 8: Debug rendering with error capture...")
    
    debug_result, error_info = pipeline.render_video(code_file, render_quality, debug_mode=True)
    
    # If rendering failed, use error info for refinement
    if debug_result is None and error_info and pipeline.refinement_system:
        print(f"\n🔧 Rendering failed - starting error-driven refinement...")
        
        # Read the current code
        with open(code_file, 'r') as f:
            current_code = f.read()
        
        # Refine based on actual rendering errors
        refinement_context = f"Generate Manim animation for: {video_title}"
        refinement_result = pipeline.refinement_system.self_refine_loop(
            initial_code=current_code,
            context=refinement_context,
            max_iterations=2,
            error_info=error_info
        )
        
        refined_code = refinement_result['final_code']
        quality_score = refinement_result['quality_score']
        iterations = refinement_result['iterations']
        is_valid = refinement_result['is_valid']
        
        print(f"🎯 Error-driven refinement completed:")
        print(f"   Quality score: {quality_score}/10")
        print(f"   Iterations: {iterations}")
        print(f"   Status: {'✅ Valid' if is_valid else '⚠️  Has issues'}")
        
        # Save refined code and retry rendering only if valid
        if is_valid:
            print(f"🌟 Retrying with refined code...")
            with open(code_file, 'w') as f:
                f.write(refined_code)
            
            # Retry rendering with refined code
        else:
            print(f"⚠️  Refined code still has issues - cannot retry rendering")
            print(f"   Issues found: {refinement_result.get('errors', 'Unknown')}")
            print(f"   Quality score: {quality_score}/10 - too low for retry")
            debug_result, _ = pipeline.render_video(code_file, render_quality, debug_mode=True)
            
            if debug_result:
                print(f"✅ Rendering successful after refinement!")
            else:
                print(f"⚠️  Still failing after refinement")
                
    video_file = debug_result
    
    # Step 9: Merge audio, video, and subtitles into final output
    final_video = video_file
    if Path(existing_audio).exists() and video_file:
        # Create complete video with audio and subtitles
        final_video = pipeline.merge_audio_video(video_file, existing_audio, srt_file)
    elif srt_file:
        # Only add subtitles (no audio available)
        subtitled_video = pipeline.add_subtitles_to_video(video_file, srt_file)
        if subtitled_video and subtitled_video != srt_file:
            try:
                if Path(video_file).exists():
                    Path(video_file).unlink()
                final_video = subtitled_video
            except Exception as e:
                print(f"⚠️  Could not remove intermediate video: {e}")
                final_video = subtitled_video
    
    # Summary
    print(f"\n🎉 Pipeline Complete!")
    print(f"📋 Topic: {video_title}")
    print(f"📄 Transcript: {transcript_file}")
    print(f"🎬 Animation: {code_file}")
    if srt_file:
        print(f"📝 Subtitles: {srt_file}")
    if Path(existing_audio).exists():
        print(f"🎵 Audio: {existing_audio}")
    if final_video:
        print(f"🎥 Final Video: {final_video}")
    else:
        print(f"⚠️  Video rendering failed - check {code_file} manually")
    
    return {
        'transcript_file': transcript_file,
        'code_file': code_file,
        'final_video': final_video,
        'srt_file': srt_file,
        'audio_file': existing_audio if Path(existing_audio).exists() else None,
        'success': final_video is not None
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume video generation pipeline from Step 5')
    parser.add_argument('topic', help='Video topic/title')
    parser.add_argument('--quality', choices=['480p15', '720p30', '1080p60'],
                       default='480p15', help='Render quality')
    
    args = parser.parse_args()
    
    result = resume_from_step_5(args.topic, args.quality)
    
    if result and result['success']:
        print("\n✅ Pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed")
        exit(1)