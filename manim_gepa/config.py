import os
from dotenv import load_dotenv
import dspy
import openai

# Load environment variables
load_dotenv()

def openrouter_llm(model_name):
    """Create an OpenRouter LLM instance."""
    return dspy.LM(
        client=openai.OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://your-app.com",
                "X-Title": "ManimGEPA"
            }
        ),
        model=model_name,
        # GPT-5 Mini requires temperature=1.0 (fixed default)
        temperature=1.0 if "gpt-5-mini" in model_name else 0.7,
        # Enable high thinking for GPT-5 Mini
        reasoning={"effort": "high", "exclude": False} if "gpt-5-mini" in model_name else None
    )

def setup_models():
    """Set up and return the configured LLM models."""
    # Get model names from environment or use defaults
    gpt5_mini_model = os.environ.get("GPT5_MINI_MODEL", "openai/gpt-5-mini")
    judge_model = os.environ.get("JUDGE_MODEL", "google/gemini-2.5-flash")
    
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    print(f"Using task model: {gpt5_mini_model}")
    print(f"Using judge model: {judge_model}")
    
    # Create model instances
    gpt5_mini_lm = openrouter_llm(gpt5_mini_model)
    judge_lm = openrouter_llm(judge_model)
    
    # Configure GPT-5 Mini as default in DSPy
    dspy.configure(lm=gpt5_mini_lm)
    
    return gpt5_mini_lm, judge_lm

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = ["OPENROUTER_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return True