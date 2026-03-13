import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from config.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL


def get_gemini_model():
    """
    Returns the OpenRouter LLM client.
    Named get_gemini_model() for backward compatibility with any existing imports.
    """
    try:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or Streamlit Cloud secrets."
            )
        model = ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.3,
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenRouter model: {str(e)}")
