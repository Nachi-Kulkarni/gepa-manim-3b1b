#!/usr/bin/env python3
"""Debug script to test the refinement process"""

from video_pipeline import SelfRefinementSystem

def debug_refinement():
    print("🔍 Debugging refinement process...")
    
    # Initialize refinement system
    refinement = SelfRefinementSystem()
    
    # Test code with the error
    test_code = '''from manim import *

class TestScene(Scene):
    def construct(self):
        title = Title("Complex numbers = arrows in the plane", match_underline=True, font_size=42)
        self.play(FadeIn(title))
        self.wait(1)
'''
    
    print(f"📝 Original code:")
    print(test_code)
    
    # Test validation
    is_valid, errors = refinement.validate_code(test_code)
    print(f"✅ Validation result: is_valid={is_valid}, errors={errors}")
    
    # Test refinement
    print(f"\n🔧 Testing refinement...")
    result = refinement.refine_code(
        original_code=test_code,
        context="Test complex numbers animation",
        errors=errors,
        error_info=None,
        iteration=1
    )
    
    print(f"\n📋 Refined code:")
    print(result)
    
    # Test validation of refined code
    is_valid_refined, errors_refined = refinement.validate_code(result)
    print(f"\n✅ Refined validation: is_valid={is_valid_refined}, errors={errors_refined}")

if __name__ == "__main__":
    debug_refinement()