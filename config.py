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
        model=model_name,
        # GPT-5 Mini requires temperature=1.0 (fixed default)
        temperature=1.0 if "gpt-5-mini" in model_name else 0.7,
        # Enable high thinking for GPT-5 Mini
        reasoning={"effort": "high", "exclude": False} if "gpt-5-mini" in model_name else None
    )

# ------- LLMs -------
GPT5_MINI_MODEL = "openai/gpt-5-mini"
JUDGE_MODEL = "google/gemini-2.5-flash"  # Change to any OR-supported judge

gpt5_mini_lm = openrouter_llm(GPT5_MINI_MODEL)
judge_lm = openrouter_llm(JUDGE_MODEL)
dspy.configure(lm=gpt5_mini_lm)  # Set GPT-5 Mini as default in DSPy

# (You can switch defaults per module with set_lm if needed)
