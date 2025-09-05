# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv_new
source venv_new/bin/activate  # Linux/Mac
# OR: venv_new\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set required API key
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### Quick Testing
```bash
# Test code generation
python3 -c "
from code_generator import CodeGenerator
gen = CodeGenerator()
result = gen(video_title='Linear Transformations', transcript_excerpt='Let us visualize matrices...')
print('Success:', len(result.generated_code) > 0)
"

# Test transcript generation (v1 = 90% score, v2 = 82% score)
python3 -c "
from transcript_generator import TranscriptGenerator
gen = TranscriptGenerator(version='v1')
result = gen(video_title='Complex Numbers', code_excerpt='class Scene(Scene): def construct(self):...')
print('Success:', len(result.generated_transcript) > 0)
"

# Run full training pipeline (requires dataset)
python3 training_pipeline.py
```

### Code Quality
```bash
# Format code
black *.py

# Lint
flake8 *.py

# Type checking
mypy *.py
```

## Architecture Overview

### Core GEPA-DSPy Integration Pattern
This codebase implements a critical pattern for DSPy prompt optimization:

**DO NOT put comprehensive prompts in signature docstrings** - they won't be used by DSPy. Instead:

1. **Store comprehensive prompts as constants**: `ULTIMATE_MANIM_PROMPT`, `ULTIMATE_TRANSCRIPT_PROMPT_V1/V2`
2. **Use concise signature docstrings**: 1-2 sentences describing the task
3. **Apply prompts via `with_instructions()`**: `enhanced_signature = Signature.with_instructions(PROMPT)`

Example from `code_generator.py`:
```python
ULTIMATE_MANIM_PROMPT = """11,247-character comprehensive prompt..."""

class GenerateManimCode(dspy.Signature):
    """Generate high-quality Manim animation code in 3Blue1Brown style."""  # Concise!
    
class CodeGenerator(dspy.Module):
    def __init__(self):
        enhanced_signature = GenerateManimCode.with_instructions(ULTIMATE_MANIM_PROMPT)
        self.generate_code = dspy.ChainOfThought(enhanced_signature)
```

### Dual-System Architecture

#### Code Generation Pipeline
- **`code_generator.py`**: GEPA-optimized Manim code generation with 100% test accuracy
- **`code_judge.py`**: Multi-metric evaluation (syntax, API usage, visual effectiveness, style)
- **Core prompt**: 11,247-character `ULTIMATE_MANIM_PROMPT` with 87+ Manim objects/animations

#### Transcript Generation Pipeline  
- **`transcript_generator.py`**: Dual-prompt architecture (90% and 82% GEPA scores)
- **`transcript_judge.py`**: Educational quality evaluation (clarity, accuracy, flow, engagement)
- **Two optimized prompts**: V1 (JSON format, 90%) and V2 (expert style, 82%)

#### Training & Optimization
- **`training_pipeline.py`**: Integrated GEPA training with `ManimGEPA` class
- **GEPA optimization**: Iterative prompt improvement via reflection-based learning
- **Evaluation metrics**: Uses judge systems, NOT simple "did it generate?" checks

### Data Architecture

#### Dataset Structure
```
dataset/
├── examples/           # Raw video examples with code/transcript pairs
├── splits/            # Train/validation/test JSON splits
│   ├── train_code.json, val_code.json, test_code.json
│   └── train_transcript.json, val_transcript.json, test_transcript.json
```

#### Data Format
Each JSON contains arrays of objects:
```json
{
  "video_title": "string",
  "transcript_excerpt": "string",  // for code generation
  "code_excerpt": "string",       // for transcript generation  
  "target_code": "string",         // for code generation
  "target_transcript": "string",   // for transcript generation
  "quality_score": 0.0-1.0
}
```

### Critical DSPy Integration Notes

1. **Language Model Setup**: Uses OpenRouter with Gemini 2.5 Flash via `dspy.LM()`
2. **GEPA Configuration**: 50 max metric calls, 2-4 threads, reflection-based optimization
3. **Evaluation Trap**: The `evaluate_*_program()` functions use dummy metrics (return 1.0 if anything generated). Real evaluation happens in `*_quality_metric()` functions during GEPA training.
4. **Prompt Versioning**: TranscriptGenerator supports `version="v1"` or `version="v2"` parameter

### 3Blue1Brown Style Implementation

#### Visual Guidelines (in prompts)
- Dark backgrounds (`#282828`) with harmonious color palettes  
- Smooth transitions: `Transform`, `FadeIn/Out`, `Create`, `Write`
- Mathematical notation via `MathTex`, general text via `Text`
- Progressive concept disclosure and in-place transformations
- 87+ specific Manim mobjects and animations documented

#### Content Patterns
- Problem-driven narratives with visual metaphors
- Gradual complexity building from familiar concepts
- Explicit mathematical reasoning and step-by-step transformations
- Color coding for different states/concepts (BLUE=correct, RED=error, etc.)

## Usage Patterns

### Standalone Generation
```python
# Code generation with ultimate prompt
from code_generator import CodeGenerator
generator = CodeGenerator()
result = generator(video_title="...", transcript_excerpt="...")

# Transcript generation with version selection  
from transcript_generator import TranscriptGenerator
gen_v1 = TranscriptGenerator(version="v1")  # 90% score
gen_v2 = TranscriptGenerator(version="v2")  # 82% score
```

### GEPA Training
```python
from training_pipeline import ManimGEPA
system = ManimGEPA(api_key=os.getenv('OPENROUTER_API_KEY'))
results = system.run_complete_training(code_iterations=10, transcript_iterations=10)
```

### Dataset Management
```python
# Create train/val/test splits
python3 create_data_splits.py

# Filter dataset by quality
python3 filter_dataset.py --min_score 0.8
```

## Important Implementation Details

- **API Dependencies**: Requires `OPENROUTER_API_KEY` environment variable
- **Virtual Environment**: Always use `venv_new/` to avoid package conflicts
- **Performance**: Code generator achieves 100% on test metrics; transcript generators reach 90%/82%
- **GEPA Training**: Computationally intensive, uses reflection LM for prompt evolution
- **3Blue1Brown Fidelity**: Prompts contain extensive style guidelines and visual patterns derived from actual 3B1B content analysis