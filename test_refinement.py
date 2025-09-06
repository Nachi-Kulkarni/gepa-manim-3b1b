#!/usr/bin/env python3
"""Test script to verify self-refinement system works on problematic animation code"""

from video_pipeline import VideoGenerationPipeline

def test_self_refinement():
    print("🧪 Testing self-refinement system on problematic animation code...")
    
    # Initialize pipeline with refinement
    vp = VideoGenerationPipeline()
    vp.setup_refinement()
    
    if not vp.refinement_system:
        print("❌ Refinement system failed to initialize")
        return False
    
    # Read the problematic animation file
    try:
        with open('make_a_detailed_video_on_quadratic_equations_animation.py', 'r') as f:
            current_code = f.read()
        print(f"📄 Loaded animation code ({len(current_code)} characters)")
    except Exception as e:
        print(f"❌ Failed to read animation file: {e}")
        return False
    
    # Create error info from the actual error we saw
    error_info = {
        'error_type': 'AttributeError',
        'error_message': "VGroup object has no attribute 'clear'",
        'traceback': '''AttributeError: VGroup object has no attribute 'clear'
File "make_a_detailed_video_on_quadratic_equations_animation.py", line 218, in update_roots
    mob.clear()''',
        'stdout': '',
        'stderr': 'Animation failed during rendering'
    }
    
    print("🔧 Starting error-driven refinement...")
    
    # Run the refinement loop
    refinement_result = vp.refinement_system.self_refine_loop(
        initial_code=current_code,
        context="Generate Manim animation for quadratic equations visualization",
        max_iterations=2,
        error_info=error_info
    )
    
    print(f"🎯 Refinement completed:")
    print(f"   Quality score: {refinement_result['quality_score']}/10")
    print(f"   Iterations: {refinement_result['iterations']}")
    print(f"   Status: {'✅ Valid' if refinement_result['is_valid'] else '⚠️  Has issues'}")
    
    # Save the refined code
    refined_code = refinement_result['final_code']
    with open('make_a_detailed_video_on_quadratic_equations_animation_refined.py', 'w') as f:
        f.write(refined_code)
    
    print("💾 Refined code saved to 'make_a_detailed_video_on_quadratic_equations_animation_refined.py'")
    
    # Check if the fix was applied
    if 'mob.clear()' not in refined_code and 'mob.become(VGroup())' in refined_code:
        print("✅ Self-refinement successfully fixed the VGroup.clear() error!")
        return True
    else:
        print("⚠️  Self-refinement may not have fixed the error")
        return False

if __name__ == "__main__":
    test_self_refinement()