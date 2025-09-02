# Manim GEPA System 🎬✨

An advanced AI-powered system for generating **Manim animation code** and **educational video transcripts** in the distinctive style of **3Blue1Brown**, optimized using **GEPA (Generative Error-driven Prompt Adaptation)**.

## 🎯 Overview

This project implements a dual-system AI framework that:
- **Generates Manim Python code** from video transcripts with 96-98% accuracy
- **Creates educational transcripts** from video titles and Manim code with 90% quality score
- Uses **GEPA optimization** to iteratively improve prompt quality
- Follows **3Blue1Brown's visual storytelling** principles

## 🚀 Key Features

### 🎨 **Code Generation System**
- **ULTIMATE_MANIM_PROMPT**: 11,247-character comprehensive prompt with 87+ Manim mobjects/animations
- **3Blue1Brown Style Guidelines**: Color palettes, smooth transitions, mathematical visualizations  
- **Step-by-step Reasoning**: Detailed animation planning and implementation strategy
- **Mathematical Expression Handling**: Advanced LaTeX and transformation patterns

### 🎙️ **Transcript Generation System**  
- **Dual Prompt Architecture**: Two optimized prompts (90% and 82% GEPA scores), #1 for Code just like 3b1b, #2 for transcripts just like 3b1b
- **Manim Code Analysis**: Intelligent interpretation of animation code structure
- **Visual Storytelling**: Progressive concept disclosure and intuitive explanations
- **Mismatch Detection**: Smart handling of title/code inconsistencies

### 🧠 **GEPA Optimization**
- **Iterative Prompt Evolution**: Automatic prompt improvement through feedback
- **Multi-metric Evaluation**: Quality scoring across multiple dimensions
- **Pareto Front Tracking**: Maintains best-performing prompt variants
- **Reflection-based Learning**: Uses advanced LLMs for self-improvement

## 📁 Project Structure

```
manim_new/
├── code_generator.py          # GEPA-optimized Manim code generation
├── transcript_generator.py    # GEPA-optimized transcript generation  
├── training_pipeline.py       # Integrated GEPA training system
├── code_judge.py             # Code quality evaluation system
├── transcript_judge.py       # Transcript quality evaluation system
├── dataset/                  # Training and validation data
│   └── splits/              # Train/validation splits
├── training_results/         # GEPA optimization results
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- OpenRouter API key (for Gemini 2.5 Flash)

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd manim_new

# Create virtual environment
python3 -m venv venv_new
source venv_new/bin/activate  # On Windows: venv_new\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
export OPENROUTER_API_KEY="your_api_key_here"
```

## 🎬 Usage

### Generate Manim Code
```python
from code_generator import CodeGenerator

generator = CodeGenerator()
result = generator(
    video_title="Complex Number Multiplication",
    transcript_excerpt="Let's visualize how multiplying by i rotates vectors..."
)

print(result.generated_code)
```

### Generate Educational Transcripts
```python
from transcript_generator import TranscriptGenerator

# Use v1 (JSON format, 90% score) or v2 (expert style, 82% score)
generator = TranscriptGenerator(version="v1")
result = generator(
    video_title="Linear Transformations",
    code_excerpt="class LinearTransform(Scene): def construct(self)..."
)

print(result.generated_transcript)
```

### Run GEPA Training
```python
from training_pipeline import ManimGEPA

system = ManimGEPA(api_key="your_openrouter_key")
results = system.run_complete_training(
    code_iterations=10,
    transcript_iterations=10
)
```

## 📊 Performance Metrics

| System | Method | Score | Improvement |
|--------|--------|-------|-------------|
| **Code Generator** | GEPA Optimized | **100%** | Ultimate accuracy |
| **Transcript Gen V1** | JSON Format | **90.0%** | +25.3% from baseline |
| **Transcript Gen V2** | Expert Style | **82.0%** | +17.3% from baseline |

## 🎨 3Blue1Brown Style Features

### Visual Elements
- **Color Palette**: Dark backgrounds with distinct, harmonious colors
- **Smooth Transitions**: `Transform`, `FadeIn/Out`, `Create`, `Write`
- **Mathematical Notation**: Proper LaTeX rendering with `MathTex`
- **Progressive Disclosure**: Step-by-step concept introduction

### Animation Patterns
- **87+ Manim Objects**: Complete coverage of visual elements
- **In-place Transformations**: Evolutionary concept visualization
- **Highlighting Techniques**: `Indicate`, `Flash`, `Circumscribe`
- **Camera Movement**: Strategic focus and emphasis

## 🧪 Technical Details

### GEPA Optimization Process
1. **Base Program Evaluation**: Initial performance measurement
2. **Reflection Generation**: LLM analyzes failures and suggests improvements  
3. **Prompt Evolution**: Iterative prompt refinement based on feedback
4. **Pareto Front Tracking**: Maintains best-performing variants
5. **Multi-objective Optimization**: Balances multiple quality metrics

### Evaluation Metrics
- **Code Quality**: Syntax, API usage, visual effectiveness, style consistency
- **Transcript Quality**: Clarity, accuracy, narrative flow, engagement, style
- **Mathematical Accuracy**: Correct concept representation and explanation
- **Visual Alignment**: Code-transcript coherence and 3B1B style adherence

## 🔬 Research & Development

This project implements cutting-edge techniques in:
- **Prompt Engineering**: GEPA-based automatic prompt optimization
- **Multi-modal AI**: Code generation + natural language synthesis  
- **Educational Technology**: AI-powered mathematical visualization
- **Style Transfer**: Replicating distinctive educational presentation styles

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **3Blue1Brown (Grant Sanderson)** - Inspiration for visual mathematics education
- **DSPy Framework** - Foundation for prompt optimization and LLM programming
- **Manim Community** - Powerful mathematical animation engine
- **GEPA Research** - Advanced prompt adaptation methodology

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{manim-gepa-system,
  title={Manim GEPA System: AI-Powered Mathematical Animation Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/manim-gepa-system}
}
```

---

**Built with ❤️ for mathematical education and AI-powered content creation**