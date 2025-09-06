#!/usr/bin/env python3
"""
Complete Video Generation Pipeline
Topic → GEPA Transcript → Synchronized Animation Code → Rendered Video

Usage: python3 video_pipeline.py "Linear Algebra Transformations"
"""

import dspy
import os
import sys
import argparse
import subprocess
from pathlib import Path
import re
import json
from transcript_generator import TranscriptGenerator
from code_generator import CodeGenerator
from kokoro_tts_integration import Kokoro82MGenerator

class SelfRefinementSystem:
    """Integrated self-refinement system for code improvement"""
    
    def __init__(self, model="openrouter/openai/gpt-5-mini"):
        self.model = model
        self.refinement_history = []
        self.setup_model()
    
    def setup_model(self):
        """Configure the LLM for refinement"""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        self.lm = dspy.LM(
            model=self.model,
            api_key=api_key,
            temperature=1.0,
            max_tokens=16000,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def validate_code(self, code):
        """Validate Manim code for common errors"""
        errors = []
        
        # Common Manim errors to fix
        if 'mob.clear()' in code:
            errors.append("VGroup.clear() method doesn't exist - use mob.become(VGroup())")
        
        # Check for Title with match_underline parameter (more precise check)
        import re
        title_pattern = r'Title\([^)]*match_underline[^)]*\)'
        if re.search(title_pattern, code):
            errors.append("Title() doesn't accept match_underline parameter - remove it")
        
        if 'always_redraw' in code and 'clear_updaters' not in code:
            errors.append("Missing clear_updaters() call for always_redraw objects")
        
        # Basic Python syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        
        return len(errors) == 0, errors
    
    def rate_code_quality(self, code, context):
        """Rate code quality from 1-10 using LLM self-assessment"""
        quality_prompt = f"""
        Rate this Manim animation code quality from 1-10 based on:
        1. Code correctness and absence of bugs
        2. Manim API usage best practices  
        3. Animation effectiveness and smoothness
        4. Mathematical accuracy
        5. Code readability and structure
        
        Context: {context}
        
        Code to rate:
        {code}
        
        Provide only a numeric score from 1.0 to 10.0.
        """
        
        with dspy.context(lm=self.lm):
            class QualityRater(dspy.Signature):
                """Rate code quality"""
                code = dspy.InputField(desc="Code to rate")
                context = dspy.InputField(desc="Context for the code")
                quality_score = dspy.OutputField(desc="Quality score 1.0-10.0")
            
            rater = dspy.Predict(QualityRater)
            result = rater(code=code, context=context)
            
            try:
                return float(result.quality_score)
            except:
                return 5.0
    
    def refine_code(self, original_code, context, errors=None, error_info=None, iteration=1):
        """Refine code using LLM with memory of previous attempts and actual error logs"""
        
        refinement_context = f"Refinement iteration {iteration}"
        if self.refinement_history:
            refinement_context += f"\nPrevious attempts: {len(self.refinement_history)} iterations"
        
        # Build error information from validation and/or rendering errors
        error_details = []
        if errors:
            error_details.extend([f"- {error}" for error in errors])
        
        if error_info:
            # Add actual rendering error logs
            if error_info.get('stderr'):
                error_details.append("RENDERING ERROR LOGS:")
                # Extract relevant error information from stderr
                stderr_lines = error_info['stderr'].split('\n')
                for line in stderr_lines:
                    if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed']):
                        error_details.append(f"  {line.strip()}")
            
            if error_info.get('exception'):
                error_details.append(f"EXCEPTION: {error_info['exception']}")
            
            if error_info.get('traceback'):
                error_details.append("TRACEBACK:")
                traceback_lines = error_info['traceback'].split('\n')[:10]  # First 10 lines
                for line in traceback_lines:
                    if line.strip():
                        error_details.append(f"  {line.strip()}")
        
        if not error_details:
            error_details = ["No specific errors identified - general quality improvement needed"]
        
        refinement_prompt = f"""
        You are an expert Manim animation developer debugging and improving your own code.
        
        ORIGINAL TASK: {context}
        
        CURRENT CODE ISSUES:
        {chr(10).join(error_details)}
        
        REFINEMENT CONTEXT: {refinement_context}
        
        ORIGINAL CODE:
        {original_code}
        
        REQUIREMENTS:
        1. Fix all identified errors and bugs based on the actual error logs
        2. Improve code quality and best practices
        3. Optimize animation performance
        4. Maintain 3Blue1Brown style
        5. Ensure smooth, educational animations
        
        CRITICAL FIXES TO CONSIDER:
        - Replace 'mob.clear()' with 'mob.become(VGroup())'
        - Remove 'match_underline' parameter from Title() constructor - THIS IS CRITICAL
        - Add 'mob.clear_updaters()' before scene transitions  
        - Fix AttributeError issues (like VGroup methods)
        - Ensure proper variable scope and cleanup
        - Optimize always_redraw usage
        - Fix any syntax or runtime errors shown in logs
        
        IMMEDIATE ACTION REQUIRED:
        - If you see 'match_underline' in any Title() call, REMOVE IT immediately
        - If you see 'mob.clear()', replace with 'mob.become(VGroup())'
        - These are syntax errors that prevent rendering
        
        MANDATORY TEXT REPLACEMENTS:
        1. Find: Title("([^"]+)", match_underline=True([^)]*)\) 
           Replace: Title("\\1"\\2)
        2. Find: mob\.clear\(\)
           Replace: mob.become(VGroup())
        
        YOU MUST APPLY THESE EXACT REPLACEMENTS NO MATTER WHAT!
        """
        
        with dspy.context(lm=self.lm):
            class CodeRefiner(dspy.Signature):
                """Refine and improve code"""
                original_code = dspy.InputField(desc="Original code to refine")
                context = dspy.InputField(desc="Refinement context and requirements")
                refined_code = dspy.OutputField(desc="Improved and fixed code")
            
            refiner = dspy.Predict(CodeRefiner)
            result = refiner(original_code=original_code, context=refinement_prompt)
            
            return result.refined_code
    
    def self_refine_loop(self, initial_code, context, max_iterations=3, error_info=None):
        """Main self-refinement loop with memory and iterative improvement"""
        current_code = initial_code
        best_code = initial_code
        best_score = 0.0
        
        print(f"🔄 Starting self-refinement loop (max {max_iterations} iterations)")
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n🔍 Refinement iteration {iteration}/{max_iterations}")
            
            # Validate current code
            is_valid, errors = self.validate_code(current_code)
            
            # Rate code quality
            quality_score = self.rate_code_quality(current_code, context)
            
            print(f"📊 Quality score: {quality_score}/10")
            if errors:
                print(f"⚠️  Found {len(errors)} errors: {', '.join(errors[:3])}")
            
            # Track this iteration
            self.refinement_history.append({
                'iteration': iteration,
                'code': current_code,
                'errors': errors,
                'quality_score': quality_score,
                'error_info': error_info
            })
            
            # Update best code
            if quality_score > best_score:
                best_score = quality_score
                best_code = current_code
                print(f"🌟 New best code with score {quality_score}/10")
            
            # Check if we're done - only stop if code is valid (no errors) AND good quality
            if is_valid and quality_score >= 8.0:
                print(f"✅ Code meets quality threshold (≥8.0/10) and has no errors")
                break
            
            # Always continue refining if there are errors, regardless of quality score
            if errors:
                print(f"🔧 {len(errors)} errors detected - must continue refining...")
                # Refine the code with error information immediately
                print(f"🔧 Refining code based on feedback...")
                current_code = self.refine_code(current_code, context, errors, error_info, iteration + 1)
                # Clear error_info for subsequent iterations (only use original rendering error)
                error_info = None
                continue  # Go to next iteration with refined code
            
            if iteration >= max_iterations:
                print(f"⏹️  Reached max iterations ({max_iterations})")
                break
            
            # Refine the code with error information
            print(f"🔧 Refining code based on feedback...")
            current_code = self.refine_code(current_code, context, errors, error_info, iteration + 1)
            
            # Clear error_info for subsequent iterations (only use original rendering error)
            error_info = None
        
        # Validate the final code to get current errors
        final_is_valid, final_errors = self.validate_code(current_code)
        
        return {
            'final_code': current_code,
            'quality_score': best_score,
            'iterations': len(self.refinement_history),
            'is_valid': final_is_valid,
            'errors': final_errors
        }

class VideoGenerationPipeline:
    """Complete pipeline for generating educational videos from topics."""

    def __init__(self, model="openrouter/openai/gpt-5-mini", use_tts=True):
        self.model = model
        self.use_tts = use_tts
        self.tts_generator = None
        self.refinement_system = None
        self.setup_model()
        if use_tts:
            self.setup_tts()
        self.setup_refinement()

    def setup_model(self):
        """Configure the language model for the pipeline."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        # Configure the model
        self.lm = dspy.LM(
            model=self.model,
            api_key=api_key,
            temperature=1.0,
            max_tokens=16000,
            base_url="https://openrouter.ai/api/v1"
        )

        dspy.configure(lm=self.lm)
        print(f"✅ Configured {self.model} for pipeline")

    def setup_tts(self):
        """Initialize Kokoro-TTS for audio generation"""
        try:
            print("🎤 Initializing Kokoro-82M TTS...")
            self.tts_generator = Kokoro82MGenerator()
            print("✅ Kokoro-82M TTS ready")
        except Exception as e:
            print(f"⚠️  TTS initialization failed: {e}")
            print("    Continuing without TTS - videos will be silent")
            self.use_tts = False
            self.tts_generator = None
    
    def setup_refinement(self):
        """Initialize the self-refinement system"""
        try:
            self.refinement_system = SelfRefinementSystem(model=self.model)
            print("✅ Self-refinement system ready")
        except Exception as e:
            print(f"⚠️  Self-refinement initialization failed: {e}")
            self.refinement_system = None

    def generate_transcript(self, video_title):
        """Generate educational transcript using GEPA-optimized prompts."""
        print(f"\n📝 Step 1: Generating transcript for '{video_title}'...")

        # Create a basic code context to guide transcript generation
        code_context = f"""
        from manim import *
        import numpy as np

        class {self._sanitize_class_name(video_title)}(Scene):
            def construct(self):
                self.camera.background_color = "#282828"

                # Educational animation for {video_title}
                # This will include:
                # - Clear introduction of concepts
                # - Step-by-step visual explanations
                # - Interactive demonstrations
                # - Practical applications
                # - Summary and key takeaways

                pass
        """

        # Use GEPA-optimized transcript generator
        transcript_gen = TranscriptGenerator(version="v1")  # 90% score version

        result = transcript_gen(
            video_title=video_title,
            code_excerpt=code_context
        )

        transcript = result.generated_transcript
        print(f"✅ Transcript generated ({len(transcript)} characters)")
        print(f"📄 Preview: {transcript[:300]}...")

        return transcript

    def generate_animation_code(self, video_title, transcript):
        """Generate synchronized Manim animation code from transcript with self-refinement."""
        print(f"\n💻 Step 2: Generating synchronized animation code...")

        # Use enhanced code generator with scene management
        code_gen = CodeGenerator()  # Uses enhanced ULTIMATE_MANIM_PROMPT

        result = code_gen(
            video_title=video_title,
            transcript_excerpt=transcript
        )

        generated_code = result.generated_code
        reasoning = getattr(result, 'reasoning', 'No reasoning provided')

        # Clean markdown formatting
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1]
            if "```" in generated_code:
                generated_code = generated_code.split("```")[0]

        print(f"✅ Animation code generated ({len(generated_code)} characters)")
        if reasoning:
            print(f"🧠 Reasoning preview: {reasoning[:200]}...")

        # Apply self-refinement if available
        if self.refinement_system:
            print(f"\n🔄 Starting self-refinement process...")
            refinement_context = f"Generate Manim animation for: {video_title}"
            
            refinement_result = self.refinement_system.self_refine_loop(
                initial_code=generated_code,
                context=refinement_context,
                max_iterations=3
            )
            
            refined_code = refinement_result['final_code']
            quality_score = refinement_result['quality_score']
            iterations = refinement_result['iterations']
            is_valid = refinement_result['is_valid']
            
            print(f"🎯 Refinement completed:")
            print(f"   Quality score: {quality_score}/10")
            print(f"   Iterations: {iterations}")
            print(f"   Status: {'✅ Valid' if is_valid else '⚠️  Has issues'}")
            
            # Use refined code only if it's valid (no errors)
            if is_valid:
                print(f"🌟 Using refined code (score: {quality_score}/10)")
                generated_code = refined_code
                reasoning += f"\n\n[Self-Refinement Applied: {iterations} iterations, final score: {quality_score}/10]"
            else:
                print(f"❌ Refined code still has {len(refinement_result.get('errors', []))} unresolved errors:")
                for error in refinement_result.get('errors', []):
                    print(f"   • {error}")
                print(f"⚠️  Cannot proceed with flawed code - stopping pipeline")
                print(f"   Please fix the errors manually or increase refinement iterations")
                # Do NOT use flawed code - return None to indicate failure
                return None, f"Self-refinement failed to fix errors: {refinement_result.get('errors', [])}"
        
        return generated_code, reasoning

    def create_narration_script(self, transcript):
        """Convert transcript to narration script using Gemini 2.5 Flash Lite"""
        print(f"\n🎙️  Step 2.3: Converting transcript to narration script...")

        try:
            # Configure Gemini 2.5 Flash Lite
            gemini_api_key = os.getenv('OPENROUTER_API_KEY')
            if not gemini_api_key:
                print("⚠️  No OpenRouter API key - using basic filtering")
                return self._basic_transcript_cleanup(transcript)

            gemini_lm = dspy.LM(
                model="openrouter/google/gemini-2.5-flash-lite",
                api_key=gemini_api_key,
                temperature=0.1,  # Lower temperature for consistent cleaning
                max_tokens=80000,
                base_url="https://openrouter.ai/api/v1"
            )

            # Enhanced narration script conversion prompt
            narration_prompt = """Convert this educational transcript into a clean narration script for text-to-speech.

REQUIREMENTS - Remove ALL of the following:
1. ALL content in [square brackets] - visual stage directions, camera movements, animation descriptions
2. ALL content in (parentheses) - scene notes, timing cues, director instructions, actor notes
3. ALL music and sound cues including:
   - "soft piano music plays/fades in/fades out"
   - "gentle background music"
   - "upbeat music"
   - "dramatic pause"
   - "sound effects"
   - "transition music"
   - "background sounds"
   - "musical interlude"
   - "silence"
   - "beat"
4. Remove ALL quotation marks around spoken text
5. Convert mathematical notation to speakable format:
   - x² → "x squared"
   - x³ → "x cubed" 
   - x⁴ → "x to the fourth power"
   - s² → "s squared"
   - a² + b² → "a squared plus b squared"
   - √x → "square root of x"
   - ∫ → "integral of"
   - dx → "d x"
   - π → "pi"
   - α → "alpha"
   - β → "beta"
   - θ → "theta"
   - Δ → "delta"
6. Remove ALL visual cues, camera directions, animation notes, production instructions
7. Remove ALL scene markers like "Scene 1:", "Cut to:", "Fade in:", "Fade out:"
8. Remove ALL technical directions like "VOICEOVER:", "NARRATOR:", "ON SCREEN:"
9. Keep ONLY the actual narration content that should be spoken aloud
10. Maintain natural speech flow and educational tone
11. Ensure clean, readable text with proper punctuation

EXAMPLES:
Input: [Scene opens with soft piano music] "The formula is a² + b² = c²" (pause for emphasis) [Show triangle with gentle background music]
Output: The formula is a squared plus b squared equals c squared

Input: [Upbeat music fades in] "Welcome to linear algebra!" (dramatic pause) [Show title animation]
Output: Welcome to linear algebra!

Input: (Soft piano tone) A dark canvas with a faint grid appears. [Crisp coordinate axis draws itself] Let's get comfortable with one of the simplest nonlinear shapes.
Output: A dark canvas with a faint grid appears. Let's get comfortable with one of the simplest nonlinear shapes.

Input: [Gentle background music fades out] "The vertex form gives us insight into the parabola's shape" [Show vertex highlighting]
Output: The vertex form gives us insight into the parabola's shape

Convert this transcript:"""

            # Use Gemini to convert transcript
            with dspy.context(lm=gemini_lm):
                class NarrationConverter(dspy.Signature):
                    """Convert educational transcript to clean narration script"""
                    transcript = dspy.InputField(desc="Educational transcript with stage directions")
                    narration_script = dspy.OutputField(desc="Clean narration script for TTS")

                EnhancedNarrationConverter = NarrationConverter.with_instructions(narration_prompt)
                converter = dspy.Predict(EnhancedNarrationConverter)

                result = converter(transcript=transcript)
                narration_script = result.narration_script.strip()

                print(f"✅ Narration script created ({len(narration_script)} characters)")
                print(f"📝 Original → Cleaned: {len(transcript)} → {len(narration_script)} chars")

                return narration_script

        except Exception as e:
            print(f"⚠️  Gemini conversion failed: {e}")
            print("    Falling back to basic filtering...")
            return self._basic_transcript_cleanup(transcript)

    def _basic_transcript_cleanup(self, text):
        """Basic fallback filter for transcript cleaning"""
        import re

        # Remove content in square brackets (stage directions/visual cues)
        cleaned_text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

        # Remove content in parentheses (scene notes)
        cleaned_text = re.sub(r'\(.*?\)', '', cleaned_text, flags=re.DOTALL)

        # Basic math notation conversion
        math_conversions = {
            r'x²': 'x squared',
            r's²': 's squared',
            r'a²': 'a squared',
            r'b²': 'b squared',
            r'c²': 'c squared',
            r'x³': 'x cubed',
            r'²': ' squared',
            r'³': ' cubed',
            r'⁴': ' to the fourth power',
            r'√': 'square root of ',
            r'π': 'pi',
            r'Δ': 'delta '
        }

        for pattern, replacement in math_conversions.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text)

        # Clean up extra whitespace and newlines
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        # Remove quotes around spoken text
        cleaned_text = cleaned_text.replace('"', '')

        return cleaned_text.strip()

    def generate_audio(self, transcript, video_title):
        """Generate audio narration from transcript using Kokoro-TTS"""
        if not self.use_tts or not self.tts_generator:
            print("🔇 TTS not available - skipping audio generation")
            return None

        print(f"\n🎤 Step 2.5: Generating audio narration...")

        try:
            # Ensure media directory exists
            media_dir = Path("media")
            media_dir.mkdir(exist_ok=True)

            # Create safe filename for audio
            safe_title = self._sanitize_filename(video_title)
            audio_file = media_dir / f"{safe_title}_narration.wav"

            # Generate audio with Kokoro-TTS (af_heart voice)
            result = self.tts_generator.generate_audio(
                text=transcript,
                voice='af_heart',
                speed=1.0,
                output_file=str(audio_file)
            )

            if result:
                # Get audio duration for video synchronization
                duration = self.get_audio_duration(result)
                print(f"✅ Audio narration generated: {result}")
                if duration:
                    print(f"🕐 Audio duration: {duration:.2f} seconds")
                return result, duration
            else:
                print("❌ Audio generation failed")
                return None, None

        except Exception as e:
            print(f"❌ Audio generation error: {e}")
            return None, None

    def get_audio_duration(self, audio_file):
        """Get duration of audio file in seconds using ffprobe"""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "csv=p=0", audio_file]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                return None
        except Exception as e:
            print(f"⚠️  Could not determine audio duration: {e}")
            return None

    def save_outputs(self, video_title, transcript, code, reasoning=None, audio_file=None):
        """Save transcript, code, and reasoning to files."""
        print(f"\n💾 Step 3: Saving pipeline outputs...")

        # Create media directory if it doesn't exist
        media_dir = Path("media")
        media_dir.mkdir(exist_ok=True)
        
        # Create safe filename
        safe_title = self._sanitize_filename(video_title)

        # Save transcript
        transcript_file = media_dir / f"{safe_title}_transcript.txt"
        with open(transcript_file, 'w') as f:
            f.write(f"# Transcript for: {video_title}\n\n")
            f.write(transcript)

        # Save animation code
        code_file = media_dir / f"{safe_title}_animation.py"
        with open(code_file, 'w') as f:
            f.write(code)

        # Save reasoning if available
        if reasoning:
            reasoning_file = media_dir / f"{safe_title}_reasoning.txt"
            with open(reasoning_file, 'w') as f:
                f.write(f"# Animation Reasoning for: {video_title}\n\n")
                f.write(reasoning)

        print(f"✅ Files saved:")
        print(f"   📄 Transcript: {transcript_file}")
        print(f"   🎬 Animation: {code_file}")
        if reasoning:
            print(f"   🧠 Reasoning: {reasoning_file}")
        if audio_file:
            print(f"   🎤 Audio: {audio_file}")

        return str(code_file), str(transcript_file)

    def render_video(self, code_file, quality="480p15", debug_mode=False):
        """Render the Manim animation video with error capture for debugging."""
        print(f"\n🎬 Step 4: Rendering video...")
        
        # Check if code file exists
        if not Path(code_file).exists():
            print(f"❌ Code file not found: {code_file}")
            return None
        
        print(f"📁 Code file: {Path(code_file).absolute()}")

        # Extract scene class name from code
        scene_class = self._extract_scene_class(code_file)
        if not scene_class:
            print("❌ Could not find scene class in generated code")
            return None
        
        print(f"🎭 Scene class: {scene_class}")

        # Render command
        quality_map = {
            "480p15": "-pql",
            "720p30": "-pqm",
            "1080p60": "-pqh"
        }

        quality_flag = quality_map.get(quality, "-pql")
        print(f"🎛️  Quality setting: {quality} (flag: {quality_flag})")

        # Check if manim is available
        try:
            print("🔍 Checking Manim installation...")
            subprocess.run(["manim", "--version"], capture_output=True, text=True, check=True)
            print("✅ Manim is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Manim not found - please install Manim")
            return None

        try:
            print(f"🚀 Rendering {scene_class} at {quality}...")
            cmd = [
                "manim", quality_flag, code_file, scene_class
            ]
            
            print(f"📋 Command: {' '.join(cmd)}")

            # Run in virtual environment
            venv_path = Path("venv_new/bin/activate")
            if venv_path.exists():
                print(f"🐍 Using virtual environment: {venv_path.absolute()}")
                cmd = ["bash", "-c", f"source venv_new/bin/activate && {' '.join(cmd)}"]
            else:
                print("⚠️  No virtual environment found - using system Python")

            print("⏳ Starting rendering process...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            print(f"📊 Return code: {result.returncode}")
            
            # Capture detailed error information for debugging
            error_info = {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd),
                'code_file': code_file,
                'scene_class': scene_class
            }
            
            if result.stdout:
                print(f"📄 STDOUT:\n{result.stdout}")
            
            if result.stderr:
                print(f"📄 STDERR:\n{result.stderr}")

            if result.returncode == 0:
                print("✅ Video rendered successfully!")

                # Find the output video file
                print("🔍 Looking for output video file...")
                video_file = self._find_output_video(code_file, scene_class, quality)
                if video_file:
                    print(f"🎥 Video location: {video_file}")
                    print(f"📏 Video size: {Path(video_file).stat().st_size / (1024*1024):.2f} MB")
                    return video_file, error_info
                else:
                    print("⚠️  Video rendered but location not found")
                    print("🔍 Checking media directory structure...")
                    media_dir = Path("media")
                    if media_dir.exists():
                        print(f"📁 Media directory exists at: {media_dir.absolute()}")
                        for item in media_dir.rglob("*.mp4"):
                            print(f"📹 Found video file: {item.relative_to('.')}")
                    else:
                        print("❌ Media directory not found")
                    return "rendered_successfully", error_info
            else:
                print(f"❌ Rendering failed with return code {result.returncode}")
                return None, error_info

        except subprocess.CalledProcessError as e:
            print(f"❌ Manim execution failed: {e}")
            error_info = {
                'return_code': e.returncode,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'exception': str(e)
            }
            print(f"📄 STDOUT: {e.stdout}")
            print(f"📄 STDERR: {e.stderr}")
            return None, error_info
        except Exception as e:
            print(f"❌ Rendering error: {e}")
            import traceback
            error_info = {
                'exception': str(e),
                'traceback': traceback.format_exc(),
                'error_type': type(e).__name__
            }
            print(f"📊 Full traceback:\n{traceback.format_exc()}")
            return None, error_info

    def generate_subtitles(self, narration_script, video_title, estimated_duration=60, target_phrases=None):
        """Generate SRT subtitle file from cleaned narration script."""
        print(f"\n📝 Step 4.5: Generating subtitles from narration script...")

        # Ensure media directory exists
        media_dir = Path("media")
        media_dir.mkdir(exist_ok=True)

        # Use the cleaned narration script instead of original transcript
        # This ensures consistency between narration and subtitles
        script_text = narration_script

        # Break script into manageable chunks for subtitles
        sentences = self._split_transcript_into_sentences(script_text)

        # Estimate timing based on reading speed (150-200 words per minute)
        words_per_minute = 175
        total_words = len(script_text.split())
        actual_duration = max(estimated_duration, (total_words / words_per_minute) * 60)

        # Create subtitle entries
        srt_entries = []
        time_per_sentence = actual_duration / len(sentences)

        for i, sentence in enumerate(sentences):
            start_time = i * time_per_sentence
            end_time = (i + 1) * time_per_sentence

            # Format for SRT
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)

            srt_entry = f"{i + 1}\n{start_srt} --> {end_srt}\n{sentence.strip()}\n"
            srt_entries.append(srt_entry)

        # Save SRT file to media directory
        safe_title = self._sanitize_filename(video_title)
        srt_file = media_dir / f"{safe_title}_subtitles.srt"

        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_entries))

        print(f"✅ Subtitles generated: {srt_file}")
        print(f"📊 {len(sentences)} subtitle entries, ~{actual_duration:.1f}s duration")
        print(f"🎯 Using cleaned narration script (consistent with audio)")

        return str(srt_file)

    def resync_subtitles_to_animation(self, transcript_file, audio_duration, code_file, output_file=None):
        """Regenerate subtitles to match the animation timing structure."""
        print(f"\n🎝 Step 12.5: Regenerating subtitles to match animation timing...")
        
        try:
            # Extract clean narration from transcript
            narration = self._extract_narration_content(transcript_file)
            print(f"📝 Extracted narration: {len(narration)} characters")
            
            # Analyze animation code to determine target phrase count
            target_phrases = self._analyze_animation_timing(code_file)
            print(f"🎯 Animation timing analysis: {target_phrases} cues detected")
            
            # Split narration into phrases matching animation structure
            phrases = self._split_into_phrases(narration, target_phrases=target_phrases)
            print(f"🎯 Split into {len(phrases)} phrases")
            
            # Calculate timing based on animation structure
            cue_duration = audio_duration / len(phrases)
            
            # Generate SRT content
            srt_content = ""
            for i, phrase in enumerate(phrases):
                start_time = i * cue_duration
                end_time = (i + 1) * cue_duration
                
                # Convert to SRT time format
                start_srt = self._seconds_to_srt_time(start_time)
                end_srt = self._seconds_to_srt_time(end_time)
                
                srt_content += f"{i + 1}\n"
                srt_content += f"{start_srt} --> {end_srt}\n"
                srt_content += f"{phrase}\n\n"
            
            # Determine output file name
            if output_file is None:
                transcript_path = Path(transcript_file)
                base_name = transcript_path.stem.replace('_transcript', '')
                # Save to media directory
                media_dir = Path("media")
                media_dir.mkdir(exist_ok=True)
                output_file = media_dir / f"{base_name}_subtitles_synced.srt"
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            print(f"✅ Generated {len(phrases)} synchronized subtitle entries")
            print(f"📁 Saved to: {output_file}")
            print(f"🕐 Each cue: ~{cue_duration:.2f}s (total: {audio_duration:.2f}s)")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Failed to resync subtitles: {e}")
            return None

    def _extract_narration_content(self, transcript_file):
        """Extract clean narration content from transcript"""
        import re
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

    def _analyze_animation_timing(self, code_file):
        """Analyze animation code to determine timing structure"""
        try:
            with open(code_file, 'r') as f:
                code_content = f.read()
            
            # Look for duration arrays or timing patterns
            import re
            
            # Pattern 1: durations array with 50 entries
            durations_match = re.search(r'durations\s*=\s*\[(.*?)\]', code_content, re.DOTALL)
            if durations_match:
                durations_str = durations_match.group(1)
                # Count the number of duration entries
                duration_entries = len([d.strip() for d in durations_str.split(',') if d.strip()])
                print(f"🔍 Found durations array with {duration_entries} entries")
                return duration_entries
            
            # Pattern 2: Look for subtitle cue references
            subtitle_matches = re.findall(r'subtitle.*?cue|cue.*?subtitle', code_content, re.IGNORECASE)
            if subtitle_matches:
                print(f"🔍 Found {len(subtitle_matches)} subtitle cue references")
                # Estimate based on common patterns
                return max(40, min(60, len(subtitle_matches) * 2))
            
            # Default fallback for complex animations
            print(f"🔍 Using default timing structure (50 cues)")
            return 50
            
        except Exception as e:
            print(f"⚠️  Could not analyze animation timing: {e}")
            return 50  # Default fallback

    def _split_into_phrases(self, text, target_phrases=50):
        """Split text into approximately target_phrases phrases"""
        import re
        
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

    def _count_subtitle_entries(self, srt_file):
        """Count the number of subtitle entries for timing context."""
        try:
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Count subtitle entries (each starts with a number)
            return len([line for line in content.split('\n\n') if line.strip() and line.strip()[0].isdigit()])
        except:
            return 0

    def generate_animation_code_with_timing(self, video_title, transcript, narration_script, timing_context, srt_file):
        """Generate synchronized Manim animation code with timing and subtitle context."""
        print(f"\n💻 Step 5: Generating synchronized animation code with timing context...")

        # Read subtitle file for timing reference
        subtitle_content = ""
        if srt_file and Path(srt_file).exists():
            try:
                with open(srt_file, 'r', encoding='utf-8') as f:
                    subtitle_content = f.read()
                print(f"📊 Loaded {self._count_subtitle_entries(srt_file)} subtitle timing cues")
            except Exception as e:
                print(f"⚠️  Could not read subtitle file: {e}")

        # Enhanced prompt with timing context
        enhanced_prompt = (
            f"Generate a complete Manim animation that is perfectly synchronized with the provided narration.\n\n"
            f"VIDEO TOPIC: {video_title}\n\n"
            f"TRANSCRIPT (for content reference):\n{transcript}\n\n"
            f"CLEAN NARRATION SCRIPT (for timing and content):\n{narration_script}\n\n"
            f"{timing_context}\n\n"
            f"SUBTITLE TIMING REFERENCE:\n{subtitle_content if subtitle_content else 'No subtitles available'}\n\n"
            f"CRITICAL SYNCHRONIZATION REQUIREMENTS:\n"
            f"1. TIMING MATCH: Animation timing MUST match the narration flow exactly\n"
            f"2. VISUAL ALIGNMENT: Key visual elements should appear exactly when they are mentioned in the narration\n"
            f"3. PACING: Use subtitle timing cues to pace animations appropriately\n"
            f"4. DURATION: Total animation should match the audio duration ({timing_context.split('Audio duration: ')[1].split(chr(10))[0] if 'Audio duration:' in timing_context else 'unknown'})\n"
            f"5. TRANSITIONS: Smooth transitions between topics as covered in the narration\n"
            f"6. EMPHASIS: Highlight key concepts exactly when they are spoken about\n\n"
            f"ANIMATION REQUIREMENTS:\n"
            f"- Use 3Blue1Brown style: clean, educational, mathematically precise\n"
            f"- Include proper updaters and cleanup for smooth animations\n"
            f"- Add appropriate wait() calls for timing\n"
            f"- Use fade in/out transitions between sections\n"
            f"- Include mathematical notation and visual explanations\n"
            f"- Ensure proper scene management with clear() calls between phases\n\n"
            f"MANIM CODE STRUCTURE:\n"
            f"1. Import all necessary modules\n"
            f"2. Define scene class with proper name\n"
            f"3. Set background color and styling\n"
            f"4. Create clear phases matching narration flow\n"
            f"5. Use ValueTrackers for dynamic elements\n"
            f"6. Include proper updater management\n"
            f"7. Add timing-appropriate wait() calls\n"
            f"8. Clean up updaters between sections\n\n"
            f"MANDATORY MANIM API RULES:\n"
            f"- Title() constructor does NOT accept match_underline parameter\n"
            f"- VGroup.clear() method does NOT exist - use mob.become(VGroup()) or mob.clear_updaters()\n"
            f"- Always use proper updater cleanup with clear_updaters()\n"
            f"- Use safe coordinate plane methods\n\n"
            f"Generate the complete Manim animation code:"
        )

        # Use enhanced code generator with timing context
        code_gen = CodeGenerator()  # Uses enhanced ULTIMATE_MANIM_PROMPT

        with dspy.context(lm=self.lm):
            class TimedAnimationGenerator(dspy.Signature):
                """Generate timed Manim animation from transcript with narration sync"""
                video_title = dspy.InputField(desc="Video topic/title")
                transcript = dspy.InputField(desc="Educational transcript")
                prompt_context = dspy.InputField(desc="Enhanced prompt with timing requirements")
                animation_code = dspy.OutputField(desc="Complete Manim animation code")

            # Add the timing context to the prompt
            from code_generator import ULTIMATE_MANIM_PROMPT
            context_with_timing = f"{ULTIMATE_MANIM_PROMPT}\n\n{enhanced_prompt}"

            TimedAnimationGenerator = TimedAnimationGenerator.with_instructions(
                "Generate complete Manim animation code that is perfectly synchronized with narration timing."
            )
            
            generator = dspy.Predict(TimedAnimationGenerator)
            result = generator(
                video_title=video_title,
                transcript=transcript,
                prompt_context=context_with_timing
            )

            generated_code = result.animation_code
            reasoning = f"Generated with timing context for {video_title}"

            print(f"✅ Animation code generated ({len(generated_code)} characters)")
            print(f"🧠 Reasoning preview: {reasoning[:200]}...")

            # Apply self-refinement if available
            if self.refinement_system:
                print(f"🔄 Starting self-refinement process...")
                refinement_result = self.refinement_system.self_refine_loop(
                    initial_code=generated_code,
                    context=f"Generate timed Manim animation for: {video_title}",
                    max_iterations=3
                )

                refined_code = refinement_result['final_code']
                quality_score = refinement_result['quality_score']
                iterations = refinement_result['iterations']
                is_valid = refinement_result['is_valid']

                print(f"🎯 Refinement completed:")
                print(f"   Quality score: {quality_score}/10")
                print(f"   Iterations: {iterations}")
                print(f"   Status: {'✅ Valid' if is_valid else '⚠️  Has issues'}")

                # Use refined code only if it's valid (no errors)
                if is_valid:
                    print(f"🌟 Using refined code (score: {quality_score}/10)")
                    generated_code = refined_code
                    reasoning += f"\n\n[Self-Refinement Applied: {iterations} iterations, final score: {quality_score}/10]"
                else:
                    print(f"❌ Refined code still has {len(refinement_result.get('errors', []))} unresolved errors:")
                    for error in refinement_result.get('errors', []):
                        print(f"   • {error}")
                    print(f"⚠️  Cannot proceed with flawed code - stopping pipeline")
                    return None, f"Self-refinement failed to fix errors: {refinement_result.get('errors', [])}"

            return generated_code, reasoning

    def add_subtitles_to_video(self, video_file, srt_file):
        """Add subtitles to video using ffmpeg."""
        print(f"\n🎬 Step 10: Adding subtitles to video...")

        if not video_file or video_file == "rendered_successfully":
            print("⚠️  No video file to add subtitles to")
            return None

        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  ffmpeg not found - subtitles created but not embedded")
            return srt_file

        # Create output filename
        video_path = Path(video_file)
        output_file = video_path.parent / f"{video_path.stem}_with_subtitles{video_path.suffix}"

        # ffmpeg command to add subtitles with subtle styling
        subtitle_style = f"subtitles={srt_file}:force_style='FontSize=13,PrimaryColour=&Hffffff&,Outline=1,Bold=0'"

        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-i', video_file,  # Input video
            '-vf', subtitle_style,  # Add subtitles filter with styling
            '-c:a', 'copy',  # Copy audio as-is
            str(output_file)  # Output file
        ]

        try:
            print("🔄 Embedding subtitles...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Subtitled video created: {output_file}")
                return str(output_file)
            else:
                print(f"❌ Failed to add subtitles: {result.stderr}")
                return srt_file

        except Exception as e:
            print(f"❌ Error adding subtitles: {e}")
            return srt_file

    def dry_run_check(self, code_file):
        """Perform dry run compilation check to catch errors early."""
        print(f"\n🔍 Step 5.5: Performing dry run compilation check...")
        
        try:
            # Extract scene class name
            scene_class = self._extract_scene_class(code_file)
            if not scene_class:
                print(f"⚠️  Could not find scene class in {code_file}")
                return True  # Continue anyway
            
            # Run manim in dry mode (compilation only)
            cmd = [
                'manim', 
                '--format', 'png',  # PNG format for dry run
                '--disable_caching',
                code_file,
                scene_class
            ]
            
            print(f"📋 Dry run command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ Dry run compilation successful")
                return True
            else:
                print(f"❌ Dry run compilation failed:")
                print(f"   Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  Dry run timed out - continuing anyway")
            return True
        except Exception as e:
            print(f"⚠️  Dry run error: {e} - continuing anyway")
            return True

    def _auto_fix_common_issues(self, code):
        """Automatically fix common Manim compilation issues."""
        import re
        
        fixed_code = code
        fixes_applied = []
        
        # Fix 1: Title() with match_underline parameter
        title_pattern = r'Title\([^)]*match_underline=True[^)]*\)'
        if re.search(title_pattern, code):
            fixed_code = re.sub(
                title_pattern,
                lambda m: m.group(0).replace('match_underline=True', '').replace('(, ', '(').replace(', )', ')'),
                fixed_code
            )
            fixes_applied.append("Removed match_underline parameter from Title()")
        
        # Fix 2: VGroup.clear() method (doesn't exist)
        if 'mob.clear()' in fixed_code:
            fixed_code = fixed_code.replace('mob.clear()', 'mob.become(VGroup())')
            fixes_applied.append("Replaced mob.clear() with mob.become(VGroup())")
        
        # Fix 3: Unescaped ampersand in LaTeX
        if '&' in fixed_code and '\\&' not in fixed_code:
            # Only fix ampersands in Text/Title/MathTex contexts
            latex_pattern = r'(Text|Title|MathTex)\([^)]*\)&([^)]*\)'
            fixed_code = re.sub(
                latex_pattern,
                lambda m: m.group(0).replace('&', '\\&'),
                fixed_code
            )
            if fixed_code != code:
                fixes_applied.append("Escaped ampersand in LaTeX text")
        
        # Fix 4: Axes.add_coordinate_labels (method doesn't exist)
        if 'add_coordinate_labels' in fixed_code:
            fixed_code = fixed_code.replace('.add_coordinate_labels(', '.add(')
            fixes_applied.append("Fixed add_coordinate_labels method call")
        
        if fixes_applied:
            print(f"🔧 Applied automatic fixes:")
            for fix in fixes_applied:
                print(f"   • {fix}")
        
        return fixed_code

    def merge_audio_video(self, video_file, audio_file, srt_file=None):
        """Merge audio narration with video and optionally add subtitles"""
        print(f"\n🎬 Step 11: Merging audio with video...")

        if not video_file or not Path(video_file).exists():
            print("❌ No video file to merge audio with")
            return None

        if not audio_file or not Path(audio_file).exists():
            print("❌ No audio file to merge with video")
            return None

        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ffmpeg not found - cannot merge audio and video")
            return None

        # Create output filename
        video_path = Path(video_file)
        output_file = video_path.parent / f"{video_path.stem}_with_audio_and_subtitles{video_path.suffix}"

        try:
            # Build ffmpeg command
            cmd = ['ffmpeg', '-y']  # Overwrite output file
            cmd.extend(['-i', video_file])  # Input video
            cmd.extend(['-i', audio_file])  # Input audio

            # Video and audio filters
            if srt_file and Path(srt_file).exists():
                # Add subtitles and merge audio
                subtitle_style = f"subtitles={srt_file}:force_style='FontSize=13,PrimaryColour=&Hffffff&,Outline=1,Bold=0'"
                cmd.extend(['-vf', subtitle_style])
                cmd.extend(['-c:a', 'aac'])  # Encode audio
                cmd.extend(['-shortest'])  # Match shortest stream duration
            else:
                # Just merge audio without subtitles
                cmd.extend(['-c:v', 'copy'])  # Copy video as-is
                cmd.extend(['-c:a', 'aac'])  # Encode audio
                cmd.extend(['-shortest'])  # Match shortest stream duration

            cmd.append(str(output_file))  # Output file

            print("🔄 Merging audio, video, and subtitles...")
            print(f"📋 Command preview: ffmpeg -i video -i audio {'-vf subtitles' if srt_file else ''} -shortest output")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Complete video created: {output_file}")

                # Remove intermediate video file to keep only final version
                if Path(video_file).exists():
                    try:
                        Path(video_file).unlink()
                        print(f"🗑️  Removed intermediate video")
                    except:
                        pass

                return str(output_file)
            else:
                print(f"❌ Failed to merge audio and video: {result.stderr}")
                return None

        except Exception as e:
            print(f"❌ Error merging audio and video: {e}")
            return None

    def run_pipeline(self, video_title, render_quality="480p15", add_subtitles=True):
        """Run the complete pipeline."""
        print("🎭 Complete Video Generation Pipeline")
        print("=" * 60)
        print(f"📋 Topic: {video_title}")
        print(f"🎬 Quality: {render_quality}")
        print(f"🤖 Model: {self.model}")
        print("=" * 60)

        try:
            # Step 1: Generate transcript
            transcript = self.generate_transcript(video_title)

            # Step 2: Create clean narration script using Gemini 2.5 Flash Lite
            narration_script = self.create_narration_script(transcript)

            # Step 3: Generate audio narration (if TTS available) - use cleaned script
            audio_file, audio_duration = None, None
            if self.use_tts:
                audio_file, audio_duration = self.generate_audio(narration_script, video_title)

            # Step 4: Generate subtitles from narration script for timing reference
            estimated_duration = audio_duration if audio_duration else 60
            srt_file = self.generate_subtitles(narration_script, video_title, estimated_duration)

            # Step 5: Generate animation code with timing context
            timing_context = (
                f"TIMING CONTEXT FOR SYNCHRONIZATION:\n"
                f"- Audio duration: {audio_duration or 'unknown'} seconds\n"
                f"- Estimated video duration: {estimated_duration} seconds\n"
                f"- Subtitle entries: {self._count_subtitle_entries(srt_file)} timing cues\n"
                f"- Narration script length: {len(narration_script)} characters\n\n"
                f"SYNCHRONIZATION REQUIREMENTS:\n"
                f"- Animation timing should match narration flow\n"
                f"- Key visual elements should align with spoken explanations\n"
                f"- Use subtitle timing as reference for animation pacing\n"
                f"- Ensure smooth transitions between topics\n"
                f"- Match animation complexity to available time budget\n"
            )
            
            code, reasoning = self.generate_animation_code_with_timing(
                video_title, 
                transcript, 
                narration_script, 
                timing_context,
                srt_file
            )
            
            # Check if refinement failed
            if code is None:
                print(f"❌ Animation code generation failed: {reasoning}")
                return {
                    'success': False,
                    'error': f"Animation code generation failed: {reasoning}",
                    'video_file': None,
                    'audio_file': None,
                    'subtitles_file': None
                }

            # Step 6: Save outputs
            code_file, transcript_file = self.save_outputs(video_title, transcript, code, reasoning, audio_file)
            
            # Step 6.5: Perform dry run compilation check
            dry_run_success = self.dry_run_check(code_file)
            
            # If dry run fails, try to fix common issues automatically
            if not dry_run_success and self.refinement_system:
                print(f"\n🔧 Dry run failed - attempting automatic fix...")
                
                # Read the current code
                with open(code_file, 'r') as f:
                    current_code = f.read()
                
                # Try to fix common issues automatically
                fixed_code = self._auto_fix_common_issues(current_code)
                
                if fixed_code != current_code:
                    print(f"🔧 Applied automatic fixes to code")
                    with open(code_file, 'w') as f:
                        f.write(fixed_code)
                    
                    # Retry dry run with fixed code
                    dry_run_success = self.dry_run_check(code_file)
                    
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
            
            debug_result, error_info = self.render_video(code_file, render_quality, debug_mode=True)
            
            # If rendering failed, use error info for refinement
            if debug_result is None and error_info and self.refinement_system:
                print(f"\n🔧 Rendering failed - starting error-driven refinement...")
                
                # Read the current code
                with open(code_file, 'r') as f:
                    current_code = f.read()
                
                # Refine based on actual rendering errors
                refinement_context = f"Generate Manim animation for: {video_title}"
                refinement_result = self.refinement_system.self_refine_loop(
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
                    debug_result, _ = self.render_video(code_file, render_quality, debug_mode=True)
                    
                    if debug_result:
                        print(f"✅ Rendering successful after refinement!")
                    else:
                        print(f"⚠️  Still failing after refinement")
                            
            video_file = debug_result

            # Step 9: Resync subtitles to match animation timing (if video was rendered)
            synced_srt_file = srt_file
            if video_file and video_file != "rendered_successfully" and srt_file and transcript_file:
                print(f"\n🎝 Step 9: Synchronizing subtitles with animation timing...")
                
                # Get actual audio duration if available, otherwise estimate from video
                sync_audio_duration = audio_duration
                if not sync_audio_duration and audio_file:
                    sync_audio_duration = self.get_audio_duration(audio_file)
                
                if not sync_audio_duration:
                    # Estimate from video file or use default
                    sync_audio_duration = 448.25  # Default fallback
                
                # Resync subtitles to match animation structure
                synced_srt_file = self.resync_subtitles_to_animation(
                    transcript_file=transcript_file,
                    audio_duration=sync_audio_duration,
                    code_file=code_file,
                    output_file=srt_file.replace('.srt', '_synced.srt')
                )
                
                if synced_srt_file:
                    print(f"✅ Subtitles synchronized to animation timing")
                    # Use synced subtitles for final video
                    srt_file = synced_srt_file
                else:
                    print(f"⚠️  Subtitle synchronization failed - using original subtitles")

            # Step 10: Merge audio, video, and subtitles into final output
            final_video = video_file
            if audio_file and video_file:
                # Create complete video with audio and subtitles
                final_video = self.merge_audio_video(video_file, audio_file, srt_file)
            elif add_subtitles and srt_file:
                # Only add subtitles (no audio available)
                subtitled_video = self.add_subtitles_to_video(video_file, srt_file)
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
                if synced_srt_file and synced_srt_file != srt_file:
                    print(f"📝 Original Subtitles: {srt_file}")
                    print(f"🎝 Synchronized Subtitles: {synced_srt_file}")
                else:
                    print(f"📝 Subtitles: {srt_file}")
            if final_video:
                print(f"🎥 Final Video: {final_video}")
            else:
                print(f"⚠️  Video rendering failed - check {code_file} manually")

            return {
                'transcript_file': transcript_file,
                'code_file': code_file,
                'final_video': final_video,
                'srt_file': srt_file,
                'synced_srt_file': synced_srt_file if synced_srt_file and synced_srt_file != srt_file else None,
                'success': final_video is not None
            }

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return {'success': False, 'error': str(e)}

    def _sanitize_class_name(self, title):
        """Convert title to valid Python class name."""
        # Remove special characters and make title case
        clean = ''.join(c for c in title if c.isalnum() or c.isspace())
        return ''.join(word.capitalize() for word in clean.split())

    def _sanitize_filename(self, title):
        """Convert title to safe filename."""
        clean = ''.join(c for c in title if c.isalnum() or c in ' -_')
        return clean.replace(' ', '_').lower()

    def _extract_scene_class(self, code_file):
        """Extract the main Scene class name from generated code."""
        try:
            with open(code_file, 'r') as f:
                content = f.read()

            # Look for class definitions that inherit from Scene
            import re
            matches = re.findall(r'class (\w+)\(Scene\):', content)
            return matches[0] if matches else None
        except:
            return None

    def _find_output_video(self, code_file, scene_class, quality):
        """Find the rendered video file."""
        base_name = Path(code_file).stem
        quality_dir = quality.replace("p", "p")

        # Common Manim output paths
        possible_paths = [
            f"media/videos/{base_name}/{quality_dir}/{scene_class}.mp4",
            f"./media/videos/{base_name}/{quality_dir}/{scene_class}.mp4"
        ]

        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path).absolute())

        return None

    def _split_transcript_into_sentences(self, transcript):
        """Split transcript into readable subtitle chunks."""
        # Clean up the transcript
        text = re.sub(r'\s+', ' ', transcript).strip()

        # Split by sentence endings, but keep reasonable length
        sentences = re.split(r'[.!?]+', text)

        # Clean up and filter
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Ignore very short fragments
                # Break long sentences into smaller chunks for readability
                if len(sentence) > 80:
                    # Try to split on commas or conjunctions
                    parts = re.split(r'[,;]|\band\b|\bbut\b|\bor\b|\bthen\b', sentence)
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 10:
                            clean_sentences.append(part)
                else:
                    clean_sentences.append(sentence)

        return clean_sentences[:50]  # Limit to reasonable number

    def _seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Generate educational videos from topics')
    parser.add_argument('topic', help='Video topic/title')
    parser.add_argument('--model', default='openrouter/openai/gpt-5-mini',
                       help='Language model to use')
    parser.add_argument('--quality', choices=['480p15', '720p30', '1080p60'],
                       default='480p15', help='Render quality')
    parser.add_argument('--no-subtitles', action='store_true',
                       help='Skip subtitle generation')

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = VideoGenerationPipeline(model=args.model)
    result = pipeline.run_pipeline(args.topic, args.quality, add_subtitles=not args.no_subtitles)

    if result['success']:
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
