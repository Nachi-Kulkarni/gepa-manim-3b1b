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
