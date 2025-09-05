# Setup Guide 🚀

## Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd manim-gepa-system

# Create and activate virtual environment
python3 -m venv venv_new
source venv_new/bin/activate  # Linux/Mac
# OR
venv_new\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration
```bash
# Set OpenRouter API key (required for GEPA optimization)
export OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Or create .env file
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

### 3. Quick Test
```python
from code_generator import CodeGenerator

# Test code generation
generator = CodeGenerator()
result = generator(
    video_title="Complex Numbers",
    transcript_excerpt="Let's visualize complex multiplication..."
)
print(result.generated_code[:500] + "...")
```

## Dataset Setup (Optional)

For training and optimization, you'll need dataset splits:

```bash
mkdir -p dataset/splits

# Training data should be structured as:
# dataset/splits/train_code.json
# dataset/splits/val_code.json  
# dataset/splits/train_transcript.json
# dataset/splits/val_transcript.json
```

Each JSON file should contain arrays of objects with these fields:
- `video_title`: String
- `transcript_excerpt` (for code) or `code_excerpt` (for transcripts): String
- `target_code` or `target_transcript`: String
- `quality_score`: Float (0-1)

## Usage Examples

### Generate Manim Code
```python
from code_generator import CodeGenerator

generator = CodeGenerator()
result = generator(
    video_title="Linear Transformations",
    transcript_excerpt="Watch as we transform this unit square..."
)

print("Generated Code:")
print(result.generated_code)
print("\nReasoning:")
print(result.reasoning)
```

### Generate Educational Transcript
```python
from transcript_generator import TranscriptGenerator

# Choose between two optimized prompt versions
generator_v1 = TranscriptGenerator(version="v1")  # 90% score
generator_v2 = TranscriptGenerator(version="v2")  # 82% score

result = generator_v1(
    video_title="Matrix Multiplication Visualization", 
    code_excerpt="class MatrixScene(Scene): def construct(self)..."
)

print(result.generated_transcript)
```

### Run GEPA Training
```python
from training_pipeline import ManimGEPA
import os

# Initialize system
system = ManimGEPA(api_key=os.getenv('OPENROUTER_API_KEY'))

# Run complete training (requires dataset)
results = system.run_complete_training(
    code_iterations=10,
    transcript_iterations=10
)

print(f"Training completed in {results['training_time']:.1f} seconds")
```

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: OPENROUTER_API_KEY environment variable not set
   ```
   Solution: Set the environment variable or create a `.env` file

2. **Missing Dataset**
   ```
   FileNotFoundError: Data file not found: dataset/splits/train_code.json
   ```
   Solution: Create dataset files or use the systems without training

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'dspy'
   ```
   Solution: Ensure virtual environment is activated and dependencies are installed

### Performance Tips

- Use `version="v1"` for TranscriptGenerator for best quality (90% score)
- Code generator achieves 100% on test metrics
- GEPA training requires significant computational resources
- Consider using smaller datasets for initial testing

## Project Structure

```
manim-gepa-system/
├── code_generator.py          # GEPA-optimized code generation
├── transcript_generator.py    # GEPA-optimized transcript generation
├── training_pipeline.py       # Integrated training system
├── code_judge.py             # Code evaluation
├── transcript_judge.py       # Transcript evaluation
├── requirements.txt          # Dependencies
├── README.md                 # Project overview
├── SETUP.md                 # This file
├── LICENSE                  # MIT license
└── dataset/                 # Training data (optional)
    └── splits/             # Train/validation splits
```

## Next Steps

1. **Basic Usage**: Start with the code and transcript generators
2. **Dataset Creation**: Prepare your own training data
3. **GEPA Training**: Run optimization to improve prompts
4. **Custom Evaluation**: Modify judge systems for your needs
5. **Integration**: Embed into your educational content pipeline

---

For detailed documentation, see [README.md](README.md)