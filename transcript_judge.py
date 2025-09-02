#!/usr/bin/env python3
"""
Transcript Judge System - Comprehensive evaluation system for generated educational video transcripts using LLM-as-a-judge approach.
"""

import dspy
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
import os

class TranscriptQualitySignature(dspy.Signature):
    """
    Evaluate the quality of generated educational transcripts based on multiple criteria.
    """
    video_title = dspy.InputField(desc="Title of the educational video")
    generated_transcript = dspy.InputField(desc="Generated transcript to evaluate")
    reference_transcript = dspy.InputField(desc="Reference/ground truth transcript for comparison")
    code_excerpt = dspy.InputField(desc="Original Manim code excerpt for context")
    
    # Output fields
    educational_clarity: float = dspy.OutputField(desc="Score 0-1 for educational clarity and explanation quality")
    mathematical_accuracy: float = dspy.OutputField(desc="Score 0-1 for mathematical concept accuracy")
    narrative_flow: float = dspy.OutputField(desc="Score 0-1 for narrative flow and storytelling")
    engagement_level: float = dspy.OutputField(desc="Score 0-1 for audience engagement and interest")
    style_consistency: float = dspy.OutputField(desc="Score 0-1 for 3Blue1Brown style consistency")
    overall_quality: float = dspy.OutputField(desc="Overall quality score 0-1")
    detailed_feedback: str = dspy.OutputField(desc="Detailed feedback on transcript quality and improvements")

class EnhancedTranscriptJudge(dspy.Module):
    """
    Enhanced transcript judging system using multiple evaluation criteria.
    """
    def __init__(self):
        super().__init__()
        self.quality_judge = dspy.ChainOfThought(TranscriptQualitySignature)
    
    def forward(self, video_title: str, generated_transcript: str, reference_transcript: str, code_excerpt: str) -> dspy.Prediction:
        """Evaluate generated transcript quality comprehensively."""
        evaluation = self.quality_judge(
            video_title=video_title,
            generated_transcript=generated_transcript,
            reference_transcript=reference_transcript,
            code_excerpt=code_excerpt
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
            safe_float(evaluation.educational_clarity),
            safe_float(evaluation.mathematical_accuracy),
            safe_float(evaluation.narrative_flow),
            safe_float(evaluation.engagement_level),
            safe_float(evaluation.style_consistency)
        ]
        
        overall_score = sum(scores) / len(scores)
        
        return dspy.Prediction(
            educational_clarity=safe_float(evaluation.educational_clarity),
            mathematical_accuracy=safe_float(evaluation.mathematical_accuracy),
            narrative_flow=safe_float(evaluation.narrative_flow),
            engagement_level=safe_float(evaluation.engagement_level),
            style_consistency=safe_float(evaluation.style_consistency),
            overall_quality=overall_score,
            detailed_feedback=evaluation.detailed_feedback or "No feedback provided"
        )

class TranscriptComparisonSignature(dspy.Signature):
    """
    Compare two transcripts and determine which is better.
    """
    video_title = dspy.InputField(desc="Title of the educational video")
    transcript_a = dspy.InputField(desc="First transcript to compare")
    transcript_b = dspy.InputField(desc="Second transcript to compare")
    reference_transcript = dspy.InputField(desc="Reference/ground truth transcript")
    code_excerpt = dspy.InputField(desc="Original Manim code excerpt for context")
    
    # Output fields
    preferred_transcript: str = dspy.OutputField(desc="Which transcript is preferred ('A' or 'B')")
    reasoning: str = dspy.OutputField(desc="Detailed reasoning for the preference")
    improvement_suggestions: str = dspy.OutputField(desc="Suggestions for improving the preferred transcript")

class TranscriptComparisonJudge(dspy.Module):
    """
    Judge for comparing two transcripts and determining which is better.
    """
    def __init__(self):
        super().__init__()
        self.comparison_judge = dspy.ChainOfThought(TranscriptComparisonSignature)
    
    def forward(self, video_title: str, transcript_a: str, transcript_b: str, reference_transcript: str, code_excerpt: str) -> dspy.Prediction:
        """Compare two transcripts and determine preference."""
        comparison = self.comparison_judge(
            video_title=video_title,
            transcript_a=transcript_a,
            transcript_b=transcript_b,
            reference_transcript=reference_transcript,
            code_excerpt=code_excerpt
        )
        
        return dspy.Prediction(
            preferred_transcript=comparison.preferred_transcript,
            reasoning=comparison.reasoning,
            improvement_suggestions=comparison.improvement_suggestions
        )

class TranscriptContentAnalyzerSignature(dspy.Signature):
    """
    Analyze the content structure and educational value of a transcript.
    """
    video_title = dspy.InputField(desc="Title of the educational video")
    transcript = dspy.InputField(desc="Transcript to analyze")
    code_excerpt = dspy.InputField(desc="Original Manim code excerpt for context")
    
    # Output fields
    has_introduction: bool = dspy.OutputField(desc="Whether transcript has proper introduction")
    has_conclusion: bool = dspy.OutputField(desc="Whether transcript has proper conclusion")
    mathematical_concepts_covered: str = dspy.OutputField(desc="List of mathematical concepts covered")
    explanation_quality: float = dspy.OutputField(desc="Score 0-1 for explanation quality")
    analogies_used: str = dspy.OutputField(desc="List of analogies and metaphors used")
    visual_references: str = dspy.OutputField(desc="List of visual references and descriptions")
    pacing_analysis: str = dspy.OutputField(desc="Analysis of pacing and flow")
    target_audience: str = dspy.OutputField(desc="Identified target audience level")

class TranscriptContentAnalyzer(dspy.Module):
    """
    Specialized judge for analyzing transcript content structure and educational value.
    """
    def __init__(self):
        super().__init__()
        self.content_analyzer = dspy.ChainOfThought(TranscriptContentAnalyzerSignature)
    
    def forward(self, video_title: str, transcript: str, code_excerpt: str) -> dspy.Prediction:
        """Analyze transcript content structure."""
        analysis = self.content_analyzer(
            video_title=video_title,
            transcript=transcript,
            code_excerpt=code_excerpt
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
            has_introduction=safe_bool(analysis.has_introduction),
            has_conclusion=safe_bool(analysis.has_conclusion),
            mathematical_concepts_covered=analysis.mathematical_concepts_covered or "No concepts identified",
            explanation_quality=safe_float(analysis.explanation_quality),
            analogies_used=analysis.analogies_used or "No analogies identified",
            visual_references=analysis.visual_references or "No visual references identified",
            pacing_analysis=analysis.pacing_analysis or "No pacing analysis provided",
            target_audience=analysis.target_audience or "Unknown audience"
        )

def load_transcript_test_data() -> List[Dict]:
    """Load test data for evaluating the transcript judge system."""
    test_data = []
    
    # Load some examples from the validation set
    val_file = Path('dataset/splits/val_transcript.json')
    if val_file.exists():
        with open(val_file, 'r') as f:
            val_data = json.load(f)
        
        # Use first 5 examples for testing
        for item in val_data[:5]:
            test_data.append({
                'video_title': item['video_title'],
                'generated_transcript': item['target_transcript'][:2000],  # Limit length for testing
                'reference_transcript': item['target_transcript'][:2000],
                'code_excerpt': item['code_excerpt'][:1000]  # Limit code excerpt
            })
    
    return test_data

def test_transcript_judge_system():
    """Test the transcript judge system with sample data."""
    print("🧪 Testing Transcript Judge System")
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
            max_tokens=0,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Configure DSPy with the main LM
        dspy.configure(lm=main_lm)
        print("✅ Main LM configured for Transcript Judge")
        
        # Initialize judges
        quality_judge = EnhancedTranscriptJudge()
        comparison_judge = TranscriptComparisonJudge()
        content_analyzer = TranscriptContentAnalyzer()
        
        # Load test data
        test_data = load_transcript_test_data()
        
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
                generated_transcript=example['generated_transcript'],
                reference_transcript=example['reference_transcript'],
                code_excerpt=example['code_excerpt']
            )
            
            print(f"  Overall Quality: {evaluation.overall_quality:.3f}")
            print(f"  Educational Clarity: {evaluation.educational_clarity:.3f}")
            print(f"  Math Accuracy: {evaluation.mathematical_accuracy:.3f}")
            print(f"  Narrative Flow: {evaluation.narrative_flow:.3f}")
            print(f"  Engagement Level: {evaluation.engagement_level:.3f}")
            print(f"  Style Consistency: {evaluation.style_consistency:.3f}")
        
        # Test comparison judge
        if len(test_data) >= 2:
            print(f"\n⚖️ Testing Comparison Judge...")
            example_a = test_data[0]
            example_b = test_data[1]
            
            comparison = comparison_judge(
                video_title="Comparison Test",
                transcript_a=example_a['generated_transcript'],
                transcript_b=example_b['generated_transcript'],
                reference_transcript=example_a['reference_transcript'],
                code_excerpt=example_a['code_excerpt']
            )
            
            print(f"  Preferred: Transcript {comparison.preferred_transcript}")
            print(f"  Reasoning: {comparison.reasoning[:200]}...")
        
        # Test content analyzer
        print(f"\n🔍 Testing Content Analyzer...")
        example = test_data[0]
        
        content_analysis = content_analyzer(
            video_title=example['video_title'],
            transcript=example['generated_transcript'],
            code_excerpt=example['code_excerpt']
        )
        
        print(f"  Has Introduction: {content_analysis.has_introduction}")
        print(f"  Has Conclusion: {content_analysis.has_conclusion}")
        print(f"  Explanation Quality: {content_analysis.explanation_quality:.3f}")
        print(f"  Target Audience: {content_analysis.target_audience}")
        print(f"  Concepts Covered: {content_analysis.mathematical_concepts_covered[:100]}...")
        
        print(f"\n🎉 Transcript Judge System tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to demonstrate the Transcript Judge system."""
    print("🎙️ Transcript Judge System")
    print("=" * 50)
    
    # Test the system
    test_transcript_judge_system()

if __name__ == "__main__":
    main()