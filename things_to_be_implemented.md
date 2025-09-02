Absolutely! I’ll give you a **full Python+DSPy pipeline** for your Manim repo, using **OpenRouter** for both models as you requested:
- **Task LLM:** `"moonshotai/kimi-k2"` (generates Manim code from high-level prompts)
- **Judge LLM:** Any OpenRouter model (e.g., `"google/gemini-2.5-flash"` for prompting/code eval)
- **GEPA**: For reflective prompt optimization (via DSPy!)

The template below is modeled on best practices from your attached **GEPA** tutorials, but adapted **for prompt-optimized Manim code generation**, not the generic demos:

***

## **1. Install Dependencies**

```bash
pip install openai dspy-ai python-dotenv
# Optionally: pip install manim google-generativeai (if doing actual video evals)
```

***

## **2. Setup OpenRouter for DSPy**

```python
import os
from dotenv import load_dotenv
import dspy
import openai

# Load OR key from .env, or set here directly
load_dotenv()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]  # or manually: OPENROUTER_API_KEY = "<your-key>"

def openrouter_llm(model_name):
    return dspy.LM(
        client=openai.OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://your-app.com", "X-Title": "ManimGEPA"}
        ),
        model=model_name
    )

# ------- LLMs -------
KIMI_MODEL = "moonshotai/kimi-k2"
JUDGE_MODEL = "google/gemini-2.5-flash"  # Change to any OR-supported judge

kimi_lm = openrouter_llm(KIMI_MODEL)
judge_lm = openrouter_llm(JUDGE_MODEL)
dspy.configure(lm=kimi_lm)  # Set kimi-k2 as default in DSPy

# (You can switch defaults per module with set_lm if needed)
```

***

## **3. Prepare Dataset**

You need a list of examples: each with a **prompt** and its corresponding **Manim code** (and reference video path, if available).

```python
import dspy

# Skeleton: replace with your dataset loading logic
def load_manim_examples():
    # Fake example (replace with reading your repo files)
    return [
        dspy.Example({
            "description": "Animate a circle transforming into a square",
            "manim_code": """from manim import *
class CircleToSquare(Scene):
    def construct(self):
        c = Circle()
        self.play(Create(c))
        self.play(Transform(c, Square()))""",
            "reference_video": "videos/circle_to_square.mp4"
        }).with_inputs("description")
        # Add more...
    ]

examples = load_manim_examples()
trainset, valset = examples[:int(0.8*len(examples))], examples[int(0.8*len(examples)):]
```

***

## **4. Define DSPy Program**

Signature: Prompt → Manim code

```python
class ManimGenSignature(dspy.Signature):
    """Given an animation description, generate Manim code."""
    description = dspy.InputField(desc="High-level animation description")
    manim_code = dspy.OutputField(desc="Python Manim code implementing the animation")

class ManimGenModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.chain = dspy.ChainOfThought(ManimGenSignature)
    def forward(self, description):
        return self.chain(description=description)
```

***

## **5. Judge/Metric Function**

**Judge**: evaluates how close the generated code is to the reference.  
Here, we’ll use the *judge LLM* to score on a 0–1 scale. Here’s a simple implementation (adapt/extend for your real needs):

```python
def judge_metric(example, pred, trace=None):
    # Use any LLM to grade similarity of generated vs reference code (text, optionally video)
    prompt = (
        "On a scale from 0 (no match) to 1 (exact), how similar is this Manim code to the reference?\n\n"
        f"Description: {example['description']}\n"
        f"Reference code:\n{example['manim_code']}\n"
        f"GENERATED code:\n{getattr(pred, 'manim_code', pred)}\n"
        "Score only for faithfulness to the description and code structure. Return a single numeric score."
    )
    # You can improve this with a more detailed rubric or even attach video info!
    judge_lm_response = judge_lm(prompt)[0]
    # Try to extract score (fallback to 0)
    try:
        score = float(next(s for s in judge_lm_response.split() if s.replace('.','',1).isdigit()))
    except Exception:
        score = 0.0
    return score
```

For *true* GEPA, you may want to use more structured judge prompts, but this template works.

***

## **6. GEPA Optimization Loop**

```python
from dspy.teleprompt import GEPA

gepa = GEPA(
    metric=judge_metric,
    prompt_model=judge_lm,    # openrouter judge (reflection proposals)
    task_model=kimi_lm,       # kimi-k2 (Manim gen)
    max_metric_calls=100      # increase for large data
)

manim_prog = ManimGenModule()

manim_prog_opt = gepa.compile(
    manim_prog, trainset=trainset, valset=valset
)

# To get the optimized prompt for a new description:
res = manim_prog_opt(description="Animate a triangle morphing into a star")
print(res.manim_code)
```

***

## **Optional: Add video-based judge using Gemini or Manim rendering**

(This is advanced—see above for textual version.)

```python
# If you want to use video, render with Manim, then
# - call Gemini 2.5, or
# - send a video comparison prompt to your judge LLM (if it supports video)
# Example not shown fully: see previous responses for Gemini integration!
```

***

## **Summary**

- Edit data ingestion to match your repo.
- `kimi-k2` will write code, the judge LLM will score similarity and drive GEPA.
- Adapt scoring or program complexity as needed.

If you want this as a **colab-ready notebook or end-to-end repo template**, just say so!


