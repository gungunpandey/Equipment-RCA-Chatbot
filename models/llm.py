import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from config.config import OPENROUTER_BASE_URL


def _resolve_secret(key: str) -> str:
    """Read from env first, then st.secrets at call time (works on Streamlit Cloud)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""


def get_gemini_model():
    """
    Returns the OpenRouter LLM client.
    Named get_gemini_model() for backward compatibility with any existing imports.
    """
    try:
        api_key = _resolve_secret("OPENROUTER_API_KEY")
        model_name = _resolve_secret("OPENROUTER_MODEL") or "openai/gpt-5.2"

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or Streamlit Cloud secrets."
            )
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.3,
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenRouter model: {str(e)}")
