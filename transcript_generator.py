#!/usr/bin/env python3
"""
Transcript GEPA System - Generates educational video transcripts in 3Blue1Brown style using GEPA optimization.
"""

import dspy
from dspy import GEPA
from typing import List, Dict, Any
import json
from pathlib import Path
import os

# Ultimate GEPA-optimized prompt for transcript generation (Iteration 2: 90.0% score)
ULTIMATE_TRANSCRIPT_PROMPT_V1 = """{
  "name": "3Blue1Brown Educational Video Transcript Generator",
  "description": "Generate an educational video transcript in the style of 3Blue1Brown, based on a video title and an excerpt of Manim code. The transcript should explain mathematical concepts visually and intuitively, mirroring 3Blue1Brown's characteristic approach.",
  "parameters": {
    "type": "object",
    "properties": {
      "video_title": {
        "type": "string",
        "description": "The title of the educational video, which provides context for the content."
      },
      "code_excerpt": {
        "type": "string",
        "description": "An excerpt of Python code written for Manim, a mathematical animation engine. This code describes the visual elements and animations to be used in the video. The assistant should analyze the Manim code to understand the visual representations and transformations that will be shown on screen."
      }
    },
    "required": [
      "video_title",
      "code_excerpt"
    ]
  },
  "visual_style": "The transcript should emulate the visual and narrative style of 3Blue1Brown. This includes:\\n- **Visual Intuition:** Emphasize geometric interpretations of abstract mathematical concepts.\\n- **Progressive Disclosure:** Introduce concepts gradually, building from simple ideas to more complex ones.\\n- **Engaging Narration:** Use a conversational yet precise tone, posing questions and guiding the viewer's understanding.\\n- **Manim-centric Descriptions:** Directly reference or imply the visual elements and animations that the Manim code would generate (e.g., 'watch what happens to our grid', 'the plane rotates').\\n- **Mathematical Notation:** Incorporate mathematical notation naturally into the text when explaining formulas or concepts (e.g., $a + bi$, $z \\\\cdot i$).\\n- **Problem-Solving Context:** Frame the explanation within a broader mathematical problem or question to be explored.\\n- **Color and Motion Cues:** While not explicitly stating Manim colors, the narrative should imply dynamic changes and highlight key elements through motion, similar to how 3Blue1Brown uses color and animation to draw attention.\\n- **Clarity and Precision:** Maintain mathematical accuracy while making complex topics accessible.",
  "manim_code_analysis_guidelines": "When analyzing the `code_excerpt`, pay attention to:\\n- **Class Names:** `ComplexTransformationScene` indicates the primary focus of the animation.\\n- **`CONFIG` Dictionary:** This often reveals customizable parameters for the scene, such as `plane_config`, `use_multicolored_plane`, color settings (`vert_start_color`, `horiz_start_color`), `num_anchors_to_add_per_line`, `post_transformation_stroke_width`, and `default_apply_complex_function_kwargs`. These parameters suggest the type of visual elements (e.g., a grid, its colors, density, and how it transforms).\\n- **Imports:** `ComplexHomotopy`, `MoveToTarget`, `ComplexPlane`, `VGroup` are strong indicators of what kind of mathematical objects and animations are involved. `ComplexHomotopy` specifically points to continuous, deforming transformations of the complex plane.\\n- **Method Signatures (`setup`)**: These can hint at the initial state and setup of the scene.\\n- **Comments (`TODO`)**: While not directly dictating content, they can sometimes reveal the intended direction or complexity of the scene.\\n\\nThe goal is to infer the visual story the Manim code is designed to tell and translate that into a narrative that explains the underlying mathematical concepts.",
  "transcript_structure": "The transcript should generally follow a structure that introduces a concept, sets up the visual, demonstrates an operation/transformation, and explains the intuition:\\n1.  **Introduction:** Briefly recap previous concepts (if applicable, suggested by 'Ep. 3' in title) or introduce the core problem/question.\\n2.  **Visual Setup:** Describe the initial Manim visualization (e.g., setting up the complex plane, drawing a grid, labeling axes).\\n3.  **Concept Demonstration:** Introduce a mathematical operation or concept and visually demonstrate its effect on the Manim scene (e.g., multiplication by 'i' rotating the plane).\\n4.  **Intuitive Explanation:** Explain *why* the visual transformation occurs, connecting it back to the mathematical definition and building geometric intuition.\\n5.  **Conclusion/Bridge (Optional):** Briefly summarize or hint at what might come next."
}"""

# Alternative GEPA-optimized prompt for transcript generation (Iteration 5: 82.0% score)
ULTIMATE_TRANSCRIPT_PROMPT_V2 = """You are an expert in mathematics and Manim animations, specializing in creating educational content in the style of 3Blue1Brown. Your task is to generate an educational video transcript based on a given video title and an excerpt of Manim code.

The core of your task is to synthesize a narrative that bridges the mathematical concept presented in the video title with the visual representations suggested by the Manim code. The 3Blue1Brown style is characterized by:
1.  **Intuitive Explanations:** Breaking down complex mathematical ideas into understandable, visual components.
2.  **Visual Storytelling:** Using animations to illustrate concepts, rather than just decorate them.
3.  **Progression of Ideas:** Starting with a simple idea and gradually building up to more complex ones.
4.  **Engaging Tone:** Conversational, curious, and inviting the viewer to explore alongside the narrator.
5.  **Manim-specific Language:** Referencing the visual elements that Manim code typically generates (e.g., "we can see this vector rotating," "a grid deforms," "these points trace out a path").

**Input:**
*   `video_title`: The title of the educational video. This clearly states the mathematical concept to be explained.
*   `code_excerpt`: A snippet of Python code written for the Manim animation library. This code describes the visual elements, animations, and mathematical objects that will be rendered.

**Output:**
*   `target_transcript`: A detailed video transcript, formatted as if it were to be spoken by a narrator.

**Constraints and Guidelines:**

1.  **Relevance is Paramount:** The transcript *must* directly relate the `video_title` to the `code_excerpt`. If the `code_excerpt` is completely unrelated to the `video_title`, you *must* identify this mismatch. In such a case, instead of generating a full transcript for the `video_title`, you should explain *why* the code is irrelevant to the title and suggest what kind of code *would* be relevant given the title, or what topic the provided code *does* relate to.

2.  **Interpret Manim Code:** Analyze the `code_excerpt` to understand what mathematical objects (e.g., vectors, circles, graphs, numbers, blocks), transformations (e.g., rotations, translations, scaling), and animations are being defined.

3.  **Connect Code to Concepts:** Explain how the visual elements from the Manim code illustrate the mathematical ideas in the `video_title`.

4.  **Structure of the Transcript:**
    *   **Introduction:** Briefly introduce the topic from the `video_title`.
    *   **Setup:** Describe the initial state of the animation, referencing elements from the `code_excerpt`.
    *   **Development:** Walk through the animation, explaining the mathematical concepts as they unfold visually.
    *   **Intuition Building:** Focus on building an intuitive understanding, not just stating facts.
    *   **Conclusion/Summary:** Briefly summarize the key takeaway or hint at further explorations.

5.  **Formatting:**
    *   Use clear, concise language.
    *   Paragraphs should be relatively short, mimicking spoken delivery.
    *   Avoid overly technical jargon unless immediately explained visually.

6.  **Example of Mismatch Handling:** If there's a mismatch between title and code, clearly explain the discrepancy and what would be expected instead."""

class GenerateTranscript(dspy.Signature):
    """Generate high-quality educational video transcript in 3Blue1Brown style from video titles and Manim code."""
    
    video_title = dspy.InputField(desc="Title of the educational video")
    code_excerpt = dspy.InputField(desc="Excerpt from the Manim code showing the animation structure")
    target_transcript = dspy.OutputField(desc="Generated transcript in 3Blue1Brown educational style")

class TranscriptGenerator(dspy.Module):
    """
    Ultimate GEPA-optimized transcript generator using Chain of Thought with comprehensive instructions.
    """
    def __init__(self, version="v1"):
        super().__init__()
        # Choose which optimized prompt to use
        if version == "v1":
            prompt = ULTIMATE_TRANSCRIPT_PROMPT_V1
        elif version == "v2":
            prompt = ULTIMATE_TRANSCRIPT_PROMPT_V2
        else:
            raise ValueError("Version must be 'v1' or 'v2'")
        
        # Create the signature with comprehensive instructions
        enhanced_signature = GenerateTranscript.with_instructions(prompt)
        self.generate_transcript = dspy.ChainOfThought(enhanced_signature)
    
    def forward(self, video_title: str, code_excerpt: str) -> dspy.Prediction:
        """Generate transcript based on video title and code."""
        prediction = self.generate_transcript(
            video_title=video_title,
            code_excerpt=code_excerpt
        )
        return dspy.Prediction(
            generated_transcript=prediction.target_transcript,
            reasoning=prediction.reasoning
        )

class TranscriptQualityJudge(dspy.Signature):
    """
    Judge the quality of generated transcripts based on multiple criteria.
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

class TranscriptJudgeSystem(dspy.Module):
    """
    Comprehensive transcript judging system using LLM-as-a-judge approach.
    """
    def __init__(self):
        super().__init__()
        self.quality_judge = dspy.ChainOfThought(TranscriptQualityJudge)
    
    def forward(self, video_title: str, generated_transcript: str, reference_transcript: str, code_excerpt: str) -> dspy.Prediction:
        """Evaluate generated transcript quality."""
        evaluation = self.quality_judge(
            video_title=video_title,
            generated_transcript=generated_transcript,
            reference_transcript=reference_transcript,
            code_excerpt=code_excerpt
        )
        
        # Calculate weighted overall score
        scores = [
            float(evaluation.educational_clarity),
            float(evaluation.mathematical_accuracy),
            float(evaluation.narrative_flow),
            float(evaluation.engagement_level),
            float(evaluation.style_consistency)
        ]
        
        overall_score = sum(scores) / len(scores)
        
        return dspy.Prediction(
            educational_clarity=float(evaluation.educational_clarity),
            mathematical_accuracy=float(evaluation.mathematical_accuracy),
            narrative_flow=float(evaluation.narrative_flow),
            engagement_level=float(evaluation.engagement_level),
            style_consistency=float(evaluation.style_consistency),
            overall_quality=overall_score,
            detailed_feedback=evaluation.detailed_feedback
        )

def transcript_quality_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Evaluation metric for transcript quality with detailed feedback for GEPA.
    """
    try:
        # Extract reference transcript from example
        reference_transcript = example.get('target_transcript', '')
        generated_transcript = prediction.generated_transcript if hasattr(prediction, 'generated_transcript') else ''
        
        if not generated_transcript or not reference_transcript:
            feedback = "Error: Missing generated or reference transcript."
            return dspy.Prediction(score=0.0, feedback=feedback)
        
        # Create judge system
        judge = TranscriptJudgeSystem()
        
        # Evaluate the generated transcript
        evaluation = judge(
            video_title=example.get('video_title', ''),
            generated_transcript=generated_transcript,
            reference_transcript=reference_transcript,
            code_excerpt=example.get('code_excerpt', '')
        )
        
        # Calculate overall score (0-1 scale)
        overall_score = evaluation.overall_quality
        
        # Create detailed feedback for GEPA optimization
        feedback = f"""Transcript Quality Evaluation - Score: {overall_score:.3f}

Detailed Scores:
- Educational Clarity: {evaluation.educational_clarity:.3f}
- Mathematical Accuracy: {evaluation.mathematical_accuracy:.3f}
- Narrative Flow: {evaluation.narrative_flow:.3f}
- Engagement Level: {evaluation.engagement_level:.3f}
- Style Consistency: {evaluation.style_consistency:.3f}

Detailed Feedback:
{evaluation.detailed_feedback}

Reference Transcript (first 500 chars):
{reference_transcript[:500]}...

Generated Transcript (first 500 chars):
{generated_transcript[:500]}...

Improvement Suggestions:
- Focus on areas with lowest scores
- Study 3Blue1Brown's narrative patterns and analogies
- Ensure mathematical concepts are explained clearly
- Maintain engaging storytelling throughout
- Use visual language that matches the Manim animations
"""
        
        return dspy.Prediction(score=overall_score, feedback=feedback)
        
    except Exception as e:
        feedback = f"Error during evaluation: {str(e)}"
        return dspy.Prediction(score=0.0, feedback=feedback)

def load_dspy_transcript_data(split_name: str = 'train', task_type: str = 'transcript') -> List[dspy.Example]:
    """Load DSPy examples from saved splits for transcript generation."""
    splits_dir = Path('dataset/splits')
    file_path = splits_dir / f'{split_name}_{task_type}.json'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        example = dspy.Example(
            video_title=item['video_title'],
            code_excerpt=item['code_excerpt'],
            target_transcript=item['target_transcript'],
            quality_score=item.get('quality_score', 0)
        ).with_inputs('video_title', 'code_excerpt')
        examples.append(example)
    
    return examples

def create_transcript_gepa_optimizer():
    """Create and configure the GEPA optimizer for transcript generation."""
    from dspy import GEPA
    
    # Load training data
    train_set = load_dspy_transcript_data('train', 'transcript')
    val_set = load_dspy_transcript_data('val', 'transcript')
    
    if not train_set:
        raise ValueError("No training data found for transcript generation!")
    
    print(f"📊 Loaded {len(train_set)} training examples, {len(val_set)} validation examples for transcript generation")
    
    # Initialize base program
    base_program = TranscriptGenerator()
    
    # Create reflection LM using OpenRouter with Gemini 2.5 Pro
    try:
        # Get API key from environment or config
        import os
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        reflection_lm = dspy.LM(
            model="openrouter/google/gemini-2.5-flash",
            api_key=api_key,
            temperature=0.7,
            max_tokens=32000,
            base_url="https://openrouter.ai/api/v1"
        )
        print("✅ Reflection LM configured with OpenRouter Gemini 2.5 Pro for transcript generation")
    except Exception as e:
        print(f"⚠️  Could not configure reflection LM: {e}")
        print("💡 Make sure OPENROUTER_API_KEY is set in your environment")
        return None, None, None, None
    
    # Create GEPA optimizer
    optimizer = GEPA(
        metric=transcript_quality_metric,
        max_metric_calls=50,  # Start with light budget for testing
        num_threads=4,
        track_stats=True,
        reflection_minibatch_size=2,
        reflection_lm=reflection_lm
    )
    
    return optimizer, base_program, train_set, val_set

def evaluate_transcript_program(program, test_set):
    """Evaluate transcript generation program on test set."""
    from dspy.evaluate import Evaluate
    
    def simple_metric(example, prediction, trace=None):
        """Simple metric for evaluation."""
        if hasattr(prediction, 'generated_transcript') and prediction.generated_transcript:
            # Basic check: if transcript was generated, score 1.0
            return 1.0
        return 0.0
    
    evaluate = Evaluate(
        devset=test_set,
        metric=simple_metric,
        num_threads=1,
        display_table=False,
        display_progress=True
    )
    
    results = evaluate(program)
    return results

def main():
    """Main function to demonstrate the Transcript GEPA system."""
    print("🎙️ Transcript GEPA System")
    print("=" * 50)
    
    try:
        # Configure main LM using OpenRouter
        import os
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        main_lm = dspy.LM(
            model="openrouter/google/gemini-2.5-flash",
            api_key=api_key,
            temperature=0.7,
            max_tokens=40000,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Configure DSPy with the main LM
        dspy.configure(lm=main_lm)
        print("✅ Main LM configured with OpenRouter Gemini 2.5 Pro for transcript generation")
        
        # Create optimizer
        optimizer, base_program, train_set, val_set = create_transcript_gepa_optimizer()
        
        if optimizer is None:
            raise ValueError("Failed to create GEPA optimizer for transcript generation")
        
        print(f"🎯 Base program created")
        print(f"📚 Training examples: {len(train_set)}")
        print(f"🔍 Validation examples: {len(val_set)}")
        
        # Quick test on a single example
        if train_set:
            test_example = train_set[0]
            print(f"\n🧪 Testing on: {test_example.video_title[:50]}...")
            
            # Generate transcript
            prediction = base_program(
                video_title=test_example.video_title,
                code_excerpt=test_example.code_excerpt[:2000]  # Limit code excerpt
            )
            
            if prediction.generated_transcript:
                print(f"✅ Transcript generated ({len(prediction.generated_transcript)} chars)")
                print(f"💭 Reasoning: {prediction.reasoning[:200]}...")
            else:
                print("⚠️ No transcript generated")
        
        print(f"\n🎉 Transcript GEPA system ready for optimization!")
        print(f"💡 Next step: Run optimizer.compile() to start GEPA optimization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure dataset splits exist and OPENROUTER_API_KEY is configured")

if __name__ == "__main__":
    main()