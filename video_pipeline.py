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

class VideoGenerationPipeline:
    """Complete pipeline for generating educational videos from topics."""

    def __init__(self, model="openrouter/moonshotai/kimi-k2-0905", use_tts=True):
        self.model = model
        self.use_tts = use_tts
        self.tts_generator = None
        self.setup_model()
        if use_tts:
            self.setup_tts()

    def setup_model(self):
        """Configure the language model for the pipeline."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        # Configure the model
        lm = dspy.LM(
            model=self.model,
            api_key=api_key,
            temperature=0.5,
            max_tokens=32000,
            base_url="https://openrouter.ai/api/v1"
        )

        dspy.configure(lm=lm)
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
        """Generate synchronized Manim animation code from transcript."""
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

        return generated_code, reasoning

    def judge_and_refine_code(self, initial_code, video_title, transcript, max_iterations=2):
        """Self-refinement loop: Judge code quality and refine for better visual output"""
        print(f"\n🎨 Step 2.2: Self-refinement loop (max {max_iterations} iterations)...")
        
        current_code = initial_code
        refinement_history = []  # Track all attempts and feedback
        
        for iteration in range(max_iterations):
            print(f"\n🔍 Refinement iteration {iteration + 1}/{max_iterations}")
            
            try:
                # Build refinement history context
                history_context = ""
                if refinement_history:
                    history_context = f"\n**REFINEMENT HISTORY** (CRITICAL - Don't repeat these approaches):\n"
                    for i, hist in enumerate(refinement_history):
                        history_context += f"Attempt {i+1}: Score {hist['score']} - {hist['feedback'][:200]}...\n"
                        history_context += f"Changes made: {hist['changes'][:150]}...\n\n"
                    history_context += "**DO NOT repeat the same suggestions. Try completely different approaches.**\n"

                # Use the main model for judging and refinement
                judge_prompt = f"""You are an expert Manim animation critic and educator. Analyze this generated Manim code and determine if it will create a high-quality educational video.

EVALUATION CRITERIA:
1. **Visual Clarity**: Are elements well-positioned and visible?
2. **Timing**: Are animations paced appropriately for learning?
3. **Educational Flow**: Does it follow the transcript narrative logically?
4. **Scene Management**: Are elements properly introduced, used, and cleared?
5. **Manim Best Practices**: Correct syntax, proper imports, scene inheritance?

TRANSCRIPT CONTEXT:
{transcript[:800]}...

{history_context}

CURRENT CODE:
```python
{current_code}
```

First, provide a SCORE from 1-10 for video quality potential.
Then, if score < 8, provide SPECIFIC improvements that HAVEN'T been tried before:
- Better framing and positioning (camera work, zoom levels)
- Improved timing and pacing (animation duration, wait times)
- Enhanced visual clarity (colors, sizes, contrast)
- Better educational flow (narrative structure, transitions)
- Technical fixes (API issues, syntax problems)

If score >= 8, respond with "APPROVED - No refinement needed"

Your response:"""

                class CodeJudge(dspy.Signature):
                    analysis_prompt = dspy.InputField(desc="Code analysis prompt with criteria")
                    evaluation = dspy.OutputField(desc="Quality score and improvement suggestions")

                judge = dspy.Predict(CodeJudge)
                judge_result = judge(analysis_prompt=judge_prompt)
                evaluation = judge_result.evaluation.strip()
                
                print(f"📊 Code evaluation: {evaluation[:200]}...")
                
                # Check if approved or needs refinement
                if "APPROVED" in evaluation or "score: 10" in evaluation.lower() or "score: 9" in evaluation.lower() or "score: 8" in evaluation.lower():
                    print(f"✅ Code approved after {iteration + 1} iteration(s)")
                    break
                
                # Store current attempt in history
                score_match = re.search(r'SCORE:\s*(\d+)', evaluation) or re.search(r'score:\s*(\d+)', evaluation.lower())
                current_score = int(score_match.group(1)) if score_match else 6
                
                refinement_history.append({
                    'iteration': iteration + 1,
                    'score': current_score,
                    'feedback': evaluation,
                    'code': current_code,
                    'changes': f"Iteration {iteration + 1} changes will be tracked"
                })

                # Refine the code based on evaluation
                print(f"🔄 Refining code based on feedback...")
                
                # Build refinement context with history
                refine_history = ""
                if len(refinement_history) > 1:
                    refine_history = f"\n**PREVIOUS ATTEMPTS** (Don't repeat these approaches):\n"
                    for hist in refinement_history[:-1]:
                        refine_history += f"- Attempt {hist['iteration']}: Score {hist['score']} - {hist['feedback'][:150]}...\n"
                
                refine_prompt = f"""Based on the evaluation feedback, improve this Manim animation code to create a better educational video.

EVALUATION FEEDBACK:
{evaluation}

{refine_history}

CURRENT CODE:
```python
{current_code}
```

**CRITICAL: Manim Community v0.19+ API Knowledge**

**CAMERA OPERATIONS - REMOVED IN v0.19:**
- ❌ `self.camera.frame.*` - DOES NOT EXIST, will cause AttributeError
- ❌ `self.camera.animate.*` - DOES NOT EXIST, will cause AttributeError  
- ✅ Alternative: Use mobject positioning, scene scaling, or remove camera operations

**WORKING v0.19+ ALTERNATIVES:**
- Instead of camera zoom: Use `mobject.animate.scale()` on specific objects
- Instead of camera move: Position mobjects with `.move_to()`, `.shift()`
- For framing: Use `VGroup` to group and position related objects

**OTHER v0.19+ SYNTAX:**
- ❌ `stroke_dasharray` parameter - Not supported
- ❌ `*[list]` unpacking - Invalid syntax  
- ❌ `.interpolate()` - Use `.blend()`
- ✅ Proper syntax: `*list`, `[0]` not `[0)`, `.clear_updaters()`

IMPROVEMENT INSTRUCTIONS:
1. Address all issues mentioned in the feedback with NOVEL approaches
2. If previous attempts focused on camera work, use mobject scaling/positioning instead
3. If previous attempts changed timing, try different narrative structure  
4. **NEVER use self.camera.frame or self.camera.animate - they don't exist in v0.19**
5. Make SIGNIFICANT improvements, not minor tweaks
6. Keep the same class name and overall structure

Provide the substantially improved code:"""

                class CodeRefiner(dspy.Signature):
                    refinement_prompt = dspy.InputField(desc="Code refinement prompt with feedback")
                    improved_code = dspy.OutputField(desc="Enhanced Manim animation code")

                refiner = dspy.Predict(CodeRefiner)
                refine_result = refiner(refinement_prompt=refine_prompt)
                improved_code = refine_result.improved_code.strip()
                
                # Clean the improved code
                if "```python" in improved_code:
                    improved_code = improved_code.split("```python")[1]
                    if "```" in improved_code:
                        improved_code = improved_code.split("```")[0]
                
                current_code = improved_code
                print(f"✅ Code refined ({len(current_code)} characters)")
                
            except Exception as e:
                print(f"⚠️  Refinement iteration {iteration + 1} failed: {e}")
                break
        
        print(f"🎨 Self-refinement complete")
        return current_code

    def debug_and_fix_code(self, code, error_output, video_title, max_attempts=2):
        """Self-debugging loop: Analyze errors and fix code automatically"""
        print(f"\n🔧 Step 4.1: Self-debugging loop (max {max_attempts} attempts)...")
        
        current_code = code
        debug_history = []  # Track all debug attempts and errors
        
        for attempt in range(max_attempts):
            print(f"\n🐛 Debug attempt {attempt + 1}/{max_attempts}")
            
            try:
                # Build debug history context
                history_context = ""
                if debug_history:
                    history_context = f"\n**DEBUG HISTORY** (Learn from previous failures):\n"
                    for i, hist in enumerate(debug_history):
                        history_context += f"Attempt {i+1}: {hist['error'][:100]}... → {hist['fix_attempted'][:100]}...\n"
                    history_context += "**Try a DIFFERENT approach than previous attempts.**\n"

                debug_prompt = f"""You are an expert Manim Community v0.19+ debugger. Analyze this EXACT error and fix the code.

ERROR OUTPUT:
{error_output}

{history_context}

CURRENT CODE:
```python
{current_code}
```

CRITICAL: The error shows the EXACT line and issue. Fix ONLY what's broken:

**CRITICAL: Manim Community v0.19+ API Knowledge**

**CAMERA OPERATIONS - REMOVED IN v0.19:**
- ❌ `self.camera.frame.animate.*` - DOES NOT EXIST
- ❌ `self.camera.animate.*` - DOES NOT EXIST  
- ❌ `self.camera.frame.*` - DOES NOT EXIST
- ✅ Alternative: Remove camera operations or use scene-level scaling

**COMMON SYNTAX ISSUES:**
- ❌ `[0)` - Wrong bracket type
- ✅ `[0]` - Correct syntax
- ❌ `*[list]` - Invalid unpacking
- ✅ `*list` or `*(list)` - Correct unpacking

**PARAMETER ISSUES:**
- ❌ `stroke_dasharray` - Not supported in v0.19
- ❌ `.interpolate()` - Use `.blend()`
- ❌ `.remove_updater()` - Use `.clear_updaters()`

**DEBUGGING STRATEGY:**
1. Find the EXACT line mentioned in the traceback
2. Check if it uses REMOVED v0.19+ API (especially camera operations)
3. Replace with working equivalent or remove/comment out
4. Fix syntax errors (brackets, unpacking, parameters)
5. Keep all educational content unchanged

**SPECIFIC ERROR ANALYSIS:**
Look at the error traceback - it shows the exact file, line number, and error type. Most common: AttributeError for removed camera API.

Provide ONLY the corrected code with minimal changes:"""

                class CodeDebugger(dspy.Signature):
                    debug_prompt = dspy.InputField(desc="Code debugging prompt with error details")
                    fixed_code = dspy.OutputField(desc="Debugged and corrected Manim code")

                debugger = dspy.Predict(CodeDebugger)
                debug_result = debugger(debug_prompt=debug_prompt)
                fixed_code = debug_result.fixed_code.strip()
                
                # Clean the fixed code
                if "```python" in fixed_code:
                    fixed_code = fixed_code.split("```python")[1]
                    if "```" in fixed_code:
                        fixed_code = fixed_code.split("```")[0]
                
                current_code = fixed_code
                print(f"✅ Code debugged ({len(current_code)} characters)")
                
                # Store debug attempt in history
                debug_history.append({
                    'attempt': attempt + 1,
                    'error': error_output[:200],
                    'fix_attempted': fixed_code[:200],
                    'success': True
                })
                
                # Apply only minimal critical fixes to preserve audio/visual coherence
                current_code = self._apply_minimal_critical_fixes(current_code)
                
                # Save the debugged code for testing
                safe_title = self._sanitize_filename(video_title)
                debug_file = f"{safe_title}_debug_attempt_{attempt + 1}.py"
                with open(debug_file, 'w') as f:
                    f.write(current_code)
                
                return current_code
                
            except Exception as e:
                print(f"⚠️  Debug attempt {attempt + 1} failed: {e}")
                # Store failed attempt in history
                debug_history.append({
                    'attempt': attempt + 1,
                    'error': error_output[:200],
                    'fix_attempted': f"Failed: {str(e)}",
                    'success': False
                })
        
        print(f"❌ All debug attempts exhausted - using last version")
        return current_code

    def _apply_minimal_critical_fixes(self, code):
        """Apply only critical syntax fixes that prevent compilation - preserve visual coherence"""
        import re
        
        # MINIMAL FIXES - Only fix what absolutely prevents code from running
        critical_fixes = [
            # Critical syntax errors that prevent compilation
            (r'\[0\)', '[0]'),  # Fix bracket mismatch: [0) -> [0]
            (r'\[1\)', '[1]'),  # Fix bracket mismatch: [1) -> [1] 
            (r'\[([^]]+)\)', r'[\1]'),  # Fix any [content) -> [content]
            (r'\*\[([^\]]+)\]', r'*(\1)'),  # Fix unpacking: *[...] -> *(...)
        ]
        
        # Apply only critical syntax fixes
        for pattern, replacement in critical_fixes:
            code = re.sub(pattern, replacement, code)
        
        # Check if code has camera API issues and handle more intelligently
        if 'self.camera.frame' in code or 'self.camera.animate' in code:
            print("⚠️  Code contains camera API that may cause errors - letting AI debugger handle it intelligently")
        
        # Ensure proper imports
        if 'from manim import *' not in code and 'import manim' not in code:
            code = 'from manim import *\nimport numpy as np\n\n' + code
        
        return code

    def render_video_with_debugging(self, code_file, video_title, quality="480p15"):
        """Enhanced render with self-debugging on failure"""
        print(f"\n🎬 Step 4: Rendering video with self-debugging...")
        
        # Try initial render
        video_file = self.render_video(code_file, quality)
        
        if video_file:
            return video_file
        
        # Read the current code for debugging
        try:
            with open(code_file, 'r') as f:
                current_code = f.read()
        except:
            print("❌ Could not read code file for debugging")
            return None
        
        # Get the last error output (we'll need to capture this in render_video)
        last_error = getattr(self, '_last_render_error', 'Unknown rendering error')
        
        # Apply iterative debugging with multiple attempts
        current_debug_code = current_code
        debug_attempts = 0
        max_debug_cycles = 3  # Allow up to 3 full debug cycles
        global_debug_history = []  # Track across all cycles
        
        for cycle in range(max_debug_cycles):
            print(f"\n🔄 Debug cycle {cycle + 1}/{max_debug_cycles}")
            
            # Apply debugging to current code with accumulated history
            debugged_code = self.debug_and_fix_code(current_debug_code, last_error, video_title, max_attempts=2)
            
            # Store cycle results in global history
            global_debug_history.append({
                'cycle': cycle + 1,
                'error': last_error[:150],
                'code_length': len(debugged_code),
                'changes': f"Cycle {cycle + 1} debugging applied"
            })
            
            # Save debugged code and try rendering
            debug_code_file = code_file.replace('.py', f'_debugged_cycle_{cycle + 1}.py')
            with open(debug_code_file, 'w') as f:
                f.write(debugged_code)
            
            print(f"🔄 Attempting render with debugged code (cycle {cycle + 1})...")
            final_video = self.render_video(debug_code_file, quality)
            
            if final_video:
                print(f"✅ Debugging successful after {cycle + 1} cycles - video rendered")
                return final_video
            
            # If rendering failed, get the new error for next cycle
            if hasattr(self, '_last_render_error'):
                last_error = self._last_render_error
                current_debug_code = debugged_code
                print(f"⚠️  Cycle {cycle + 1} failed, trying next cycle with new error info")
            else:
                break
        
        print(f"❌ All {max_debug_cycles} debug cycles exhausted - proceeding without video")
        return None

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
                temperature=0.2,
                max_tokens=8000,
                base_url="https://openrouter.ai/api/v1"
            )

            # Create narration script conversion prompt
            narration_prompt = """Convert this educational transcript into a clean narration script for text-to-speech.

REQUIREMENTS:
1. Remove ALL content in [square brackets] - these are visual stage directions
2. Remove ALL content in (parentheses) - these are scene notes
3. Remove quotation marks around spoken text
4. Convert mathematical notation to speakable format:
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
5. Keep only the actual narration content that should be spoken aloud
6. Maintain natural speech flow and educational tone
7. Remove any visual cues, camera directions, or animation notes

EXAMPLE:
Input: [Scene opens] "The formula is a² + b² = c²" (pause for emphasis) [Show triangle]
Output: The formula is a squared plus b squared equals c squared

Convert this transcript:"""

            # Use Gemini to convert transcript
            with dspy.context(lm=gemini_lm):
                class NarrationConverter(dspy.Signature):
                    transcript = dspy.InputField(desc="Educational transcript with stage directions")
                    narration_script = dspy.OutputField(desc="Clean narration script for TTS")

                # Set instructions as signature docstring
                NarrationConverter.__doc__ = narration_prompt
                
                converter = dspy.Predict(NarrationConverter)
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

        print(f"\n🎤 Step 2.4: Generating audio narration...")

        try:
            # Convert transcript to clean narration script
            narration_script = self.create_narration_script(transcript)
            
            # Create safe filename for audio
            safe_title = self._sanitize_filename(video_title)
            audio_file = f"{safe_title}_narration.wav"

            # Generate audio with Kokoro-TTS using cleaned narration script
            result = self.tts_generator.generate_audio(
                text=narration_script,  # Use cleaned script instead of raw transcript
                voice='af_heart',
                speed=1.0,
                output_file=audio_file
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

        # Create safe filename
        safe_title = self._sanitize_filename(video_title)

        # Save transcript
        transcript_file = f"{safe_title}_transcript.txt"
        with open(transcript_file, 'w') as f:
            f.write(f"# Transcript for: {video_title}\n\n")
            f.write(transcript)

        # Save animation code
        code_file = f"{safe_title}_animation.py"
        with open(code_file, 'w') as f:
            f.write(code)

        # Save reasoning if available
        if reasoning:
            reasoning_file = f"{safe_title}_reasoning.txt"
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

        return code_file, transcript_file

    def render_video(self, code_file, quality="480p15"):
        """Render the Manim animation video."""
        print(f"\n🎬 Step 4: Rendering video...")

        # Extract scene class name from code
        scene_class = self._extract_scene_class(code_file)
        if not scene_class:
            print("❌ Could not find scene class in generated code")
            return None

        # Render command
        quality_map = {
            "480p15": "-pql",
            "720p30": "-pqm",
            "1080p60": "-pqh"
        }

        quality_flag = quality_map.get(quality, "-pql")

        try:
            print(f"🚀 Rendering {scene_class} at {quality}...")
            cmd = [
                "manim", quality_flag, code_file, scene_class
            ]

            # Run in virtual environment
            if Path("venv_new/bin/activate").exists():
                cmd = ["bash", "-c", f"source venv_new/bin/activate && {' '.join(cmd)}"]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

            if result.returncode == 0:
                print("✅ Video rendered successfully!")

                # Find the output video file
                video_file = self._find_output_video(code_file, scene_class, quality)
                if video_file:
                    print(f"🎥 Video location: {video_file}")
                    return video_file
                else:
                    print("⚠️  Video rendered but location not found")
                    return "rendered_successfully"
            else:
                print(f"❌ Rendering failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                
                # Store error for debugging system
                self._last_render_error = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                return None

        except Exception as e:
            print(f"❌ Rendering error: {e}")
            return None

    def generate_subtitles(self, transcript, video_title, estimated_duration=60):
        """Generate SRT subtitle file from cleaned narration script (same as TTS audio)."""
        print(f"\n📝 Step 4.5: Generating subtitles...")

        # Convert transcript to clean narration script (same as used for TTS)
        narration_script = self.create_narration_script(transcript)
        
        # Break cleaned narration script into manageable chunks for subtitles
        sentences = self._split_transcript_into_sentences(narration_script)

        # Estimate timing based on reading speed (150-200 words per minute)
        words_per_minute = 175
        total_words = len(narration_script.split())
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

        # Save SRT file
        safe_title = self._sanitize_filename(video_title)
        srt_file = f"{safe_title}_subtitles.srt"

        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_entries))

        print(f"✅ Subtitles generated: {srt_file}")
        print(f"📊 {len(sentences)} subtitle entries, ~{actual_duration:.1f}s duration")

        return srt_file

    def add_subtitles_to_video(self, video_file, srt_file):
        """Add subtitles to video using ffmpeg."""
        print(f"\n🎬 Step 5: Adding subtitles to video...")

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

    def merge_audio_video(self, video_file, audio_file, srt_file=None):
        """Merge audio narration with video and optionally add subtitles"""
        print(f"\n🎬 Step 6: Merging audio with video...")

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

            # Step 2: Generate initial animation code
            initial_code, reasoning = self.generate_animation_code(video_title, transcript)

            # Step 2.2: Self-refinement loop (judge and improve code quality)
            refined_code = self.judge_and_refine_code(initial_code, video_title, transcript)

            # Step 2.4: Generate audio narration (if TTS available)
            audio_file, audio_duration = None, None
            if self.use_tts:
                audio_file, audio_duration = self.generate_audio(transcript, video_title)

            # Step 3: Save outputs (use refined code)
            code_file, transcript_file = self.save_outputs(video_title, transcript, refined_code, reasoning, audio_file)

            # Step 4: Render video with self-debugging
            video_file = self.render_video_with_debugging(code_file, video_title, render_quality)

            # Step 5: Generate subtitles (if requested)
            srt_file = None
            if add_subtitles:
                # Use audio duration for better subtitle timing if available
                estimated_duration = audio_duration if audio_duration else 60
                srt_file = self.generate_subtitles(transcript, video_title, estimated_duration)

            # Step 6: Merge audio, video, and subtitles into final output
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
    parser.add_argument('--model', default='openrouter/moonshotai/kimi-k2-0905',
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
