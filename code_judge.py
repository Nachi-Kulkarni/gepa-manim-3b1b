#!/usr/bin/env python3
"""
Code Judge System - Comprehensive evaluation system for generated Manim code using LLM-as-a-judge approach.
"""

import dspy
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
import os

class CodeQualitySignature(dspy.Signature):
    """
    Evaluate the quality of generated Manim code based on multiple criteria.
    """
    video_title = dspy.InputField(desc="Title of the educational video")
    generated_code = dspy.InputField(desc="Generated Manim code to evaluate")
    reference_code = dspy.InputField(desc="Reference/ground truth Manim code for comparison")
    transcript_excerpt = dspy.InputField(desc="Original transcript excerpt for context")
    
    # Output fields
    syntax_correctness: float = dspy.OutputField(desc="Score 0-1 for Python syntax correctness")
    manim_api_usage: float = dspy.OutputField(desc="Score 0-1 for proper Manim API usage")
    mathematical_accuracy: float = dspy.OutputField(desc="Score 0-1 for mathematical concept accuracy")
    visual_effectiveness: float = dspy.OutputField(desc="Score 0-1 for visual animation effectiveness")
    code_style_consistency: float = dspy.OutputField(desc="Score 0-1 for 3Blue1Brown style consistency")
    overall_quality: float = dspy.OutputField(desc="Overall quality score 0-1")
    detailed_feedback: str = dspy.OutputField(desc="Detailed feedback on code quality and improvements")

class EnhancedCodeJudge(dspy.Module):
    """
    Enhanced code judging system using multiple evaluation criteria.
    """
    def __init__(self):
        super().__init__()
        self.quality_judge = dspy.ChainOfThought(CodeQualitySignature)
    
    def forward(self, video_title: str, generated_code: str, reference_code: str, transcript_excerpt: str) -> dspy.Prediction:
        """Evaluate generated code quality comprehensively."""
        evaluation = self.quality_judge(
            video_title=video_title,
            generated_code=generated_code,
            reference_code=reference_code,
            transcript_excerpt=transcript_excerpt
        )
        
        # Safe conversion with defaults
        def safe_float(value, default=0.0):
            try:
                if value is None:
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Calculate weighted overall score
        scores = [
            safe_float(evaluation.syntax_correctness),
            safe_float(evaluation.manim_api_usage),
            safe_float(evaluation.mathematical_accuracy),
            safe_float(evaluation.visual_effectiveness),
            safe_float(evaluation.code_style_consistency)
        ]
        
        overall_score = sum(scores) / len(scores)
        
        return dspy.Prediction(
            syntax_correctness=safe_float(evaluation.syntax_correctness),
            manim_api_usage=safe_float(evaluation.manim_api_usage),
            mathematical_accuracy=safe_float(evaluation.mathematical_accuracy),
            visual_effectiveness=safe_float(evaluation.visual_effectiveness),
            code_style_consistency=safe_float(evaluation.code_style_consistency),
            overall_quality=overall_score,
            detailed_feedback=evaluation.detailed_feedback or "No feedback provided"
        )

class CodeComparisonSignature(dspy.Signature):
    """
    Compare two pieces of Manim code and determine which is better.
    """
    video_title = dspy.InputField(desc="Title of the educational video")
    code_a = dspy.InputField(desc="First Manim code to compare")
    code_b = dspy.InputField(desc="Second Manim code to compare")
    reference_code = dspy.InputField(desc="Reference/ground truth Manim code")
    transcript_excerpt = dspy.InputField(desc="Original transcript excerpt for context")
    
    # Output fields
    preferred_code: str = dspy.OutputField(desc="Which code is preferred ('A' or 'B')")
    reasoning: str = dspy.OutputField(desc="Detailed reasoning for the preference")
    improvement_suggestions: str = dspy.OutputField(desc="Suggestions for improving the preferred code")

class CodeComparisonJudge(dspy.Module):
    """
    Judge for comparing two pieces of code and determining which is better.
    """
    def __init__(self):
        super().__init__()
        self.comparison_judge = dspy.ChainOfThought(CodeComparisonSignature)
    
    def forward(self, video_title: str, code_a: str, code_b: str, reference_code: str, transcript_excerpt: str) -> dspy.Prediction:
        """Compare two code snippets and determine preference."""
        comparison = self.comparison_judge(
            video_title=video_title,
            code_a=code_a,
            code_b=code_b,
            reference_code=reference_code,
            transcript_excerpt=transcript_excerpt
        )
        
        return dspy.Prediction(
            preferred_code=comparison.preferred_code,
            reasoning=comparison.reasoning,
            improvement_suggestions=comparison.improvement_suggestions
        )

class CodeErrorAnalyzerSignature(dspy.Signature):
    """
    Analyze errors in generated Manim code and provide specific feedback.
    """
    video_title = dspy.InputField(desc="Title of the educational video")
    generated_code = dspy.InputField(desc="Generated Manim code with potential errors")
    error_message = dspy.InputField(desc="Error message if code execution failed (empty if no error)")
    reference_code = dspy.InputField(desc="Reference/ground truth Manim code")
    transcript_excerpt = dspy.InputField(desc="Original transcript excerpt for context")
    
    # Output fields
    has_syntax_error: bool = dspy.OutputField(desc="Whether the code has syntax errors")
    has_logic_error: bool = dspy.OutputField(desc="Whether the code has logic errors")
    error_type: str = dspy.OutputField(desc="Type of error (syntax, logic, runtime, style)")
    error_severity: float = dspy.OutputField(desc="Severity of error 0-1")
    fix_suggestions: str = dspy.OutputField(desc="Specific suggestions to fix the errors")
    corrected_code: str = dspy.OutputField(desc="Corrected version of the code")

class CodeErrorAnalyzer(dspy.Module):
    """
    Specialized judge for analyzing and fixing code errors.
    """
    def __init__(self):
        super().__init__()
        self.error_analyzer = dspy.ChainOfThought(CodeErrorAnalyzerSignature)
    
    def forward(self, video_title: str, generated_code: str, error_message: str, reference_code: str, transcript_excerpt: str) -> dspy.Prediction:
        """Analyze errors in generated code."""
        analysis = self.error_analyzer(
            video_title=video_title,
            generated_code=generated_code,
            error_message=error_message,
            reference_code=reference_code,
            transcript_excerpt=transcript_excerpt
        )
        
        def safe_bool(value, default=False):
            try:
                if value is None:
                    return default
                return str(value).lower() == 'true'
            except:
                return default
        
        def safe_float(value, default=0.0):
            try:
                if value is None:
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        return dspy.Prediction(
            has_syntax_error=safe_bool(analysis.has_syntax_error),
            has_logic_error=safe_bool(analysis.has_logic_error),
            error_type=analysis.error_type or "unknown",
            error_severity=safe_float(analysis.error_severity),
            fix_suggestions=analysis.fix_suggestions or "No suggestions provided",
            corrected_code=analysis.corrected_code or "No correction provided"
        )

def load_judge_test_data() -> List[Dict]:
    """Load test data for evaluating the judge system."""
    test_data = []
    
    # Load some examples from the validation set
    val_file = Path('dataset/splits/val_code.json')
    if val_file.exists():
        with open(val_file, 'r') as f:
            val_data = json.load(f)
        
        # Use first 5 examples for testing
        for item in val_data[:5]:
            test_data.append({
                'video_title': item['video_title'],
                'generated_code': item['target_code'],  # Using reference as generated for testing
                'reference_code': item['target_code'],
                'transcript_excerpt': item['transcript_excerpt']
            })
    
    return test_data

def test_code_judge_system():
    """Test the code judge system with sample data."""
    print("🧪 Testing Code Judge System")
    print("=" * 40)
    
    try:
        # Configure main LM using OpenRouter
        import os
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        main_lm = dspy.LM(
            model="openrouter/google/gemini-2.5-flash",
            api_key=api_key,
            temperature=0.3,  # Lower temperature for more consistent evaluations
            max_tokens=40000,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Configure DSPy with the main LM
        dspy.configure(lm=main_lm)
        print("✅ Main LM configured for Code Judge")
        
        # Initialize judges
        quality_judge = EnhancedCodeJudge()
        comparison_judge = CodeComparisonJudge()
        error_analyzer = CodeErrorAnalyzer()
        
        # Load test data
        test_data = load_judge_test_data()
        
        if not test_data:
            print("❌ No test data found")
            return
        
        print(f"📊 Loaded {len(test_data)} test examples")
        
        # Test quality judge
        print(f"\n🎯 Testing Quality Judge...")
        for i, example in enumerate(test_data[:2]):  # Test first 2 examples
            print(f"\nExample {i+1}: {example['video_title'][:50]}...")
            
            evaluation = quality_judge(
                video_title=example['video_title'],
                generated_code=example['generated_code'],
                reference_code=example['reference_code'],
                transcript_excerpt=example['transcript_excerpt'][:1000]  # Limit transcript
            )
            
            print(f"  Overall Quality: {evaluation.overall_quality:.3f}")
            print(f"  Syntax: {evaluation.syntax_correctness:.3f}")
            print(f"  Manim API: {evaluation.manim_api_usage:.3f}")
            print(f"  Math Accuracy: {evaluation.mathematical_accuracy:.3f}")
            print(f"  Visual Effectiveness: {evaluation.visual_effectiveness:.3f}")
            print(f"  Style Consistency: {evaluation.code_style_consistency:.3f}")
        
        # Test comparison judge
        if len(test_data) >= 2:
            print(f"\n⚖️ Testing Comparison Judge...")
            example_a = test_data[0]
            example_b = test_data[1]
            
            comparison = comparison_judge(
                video_title="Comparison Test",
                code_a=example_a['generated_code'],
                code_b=example_b['generated_code'],
                reference_code=example_a['reference_code'],
                transcript_excerpt=example_a['transcript_excerpt'][:500]
            )
            
            print(f"  Preferred: Code {comparison.preferred_code}")
            print(f"  Reasoning: {comparison.reasoning[:200]}...")
        
        # Test error analyzer
        print(f"\n🔍 Testing Error Analyzer...")
        # Introduce a syntax error
        buggy_code = example['generated_code'] + "\ninvalid_syntax_here"
        
        error_analysis = error_analyzer(
            video_title=example['video_title'],
            generated_code=buggy_code,
            error_message="Syntax error: invalid syntax",
            reference_code=example['reference_code'],
            transcript_excerpt=example['transcript_excerpt'][:500]
        )
        
        print(f"  Has Syntax Error: {error_analysis.has_syntax_error}")
        print(f"  Error Type: {error_analysis.error_type}")
        print(f"  Error Severity: {error_analysis.error_severity:.3f}")
        print(f"  Fix Suggestions: {error_analysis.fix_suggestions[:200]}...")
        
        print(f"\n🎉 Code Judge System tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to demonstrate the Code Judge system."""
    print("⚖️ Code Judge System")
    print("=" * 50)
    
    # Test the system
    test_code_judge_system()

if __name__ == "__main__":
    main()