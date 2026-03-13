import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """Read from env first, then Streamlit secrets (for Cloud deployment)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ── OpenRouter ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = _get("OPENROUTER_API_KEY")
OPENROUTER_MODEL    = _get("OPENROUTER_MODEL") or "nvidia/nemotron-3-super-120b-a12b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Vision model used for image damage analysis (must support vision input)
VISION_MODEL = _get("VISION_MODEL") or "qwen/qwen2.5-vl-72b-instruct"

# ── Weaviate (pre-existing vector DB with equipment manuals) ────────────────
WEAVIATE_URL        = _get("WEAVIATE_URL")
WEAVIATE_API_KEY    = _get("WEAVIATE_API_KEY")
WEAVIATE_COLLECTION = _get("WEAVIATE_COLLECTION") or "Rca"

# ── RCA intake fields (collected before analysis triggers) ──────────────────
REQUIRED_FAULT_FIELDS = [
    "fault_description",    # seeded from first user message
    "equipment_name",       # auto-extracted; asked if not clear
    "downtime_minutes",     # asked — any format accepted, converted to minutes
    "production_loss",      # asked — was there feed/production loss?
    "observations",         # asked — specific operator observations
    "attachments_note",     # asked — prompt user to upload images/PDF; 'skip' accepted
]

FIELD_QUESTIONS = {
    "fault_description": (
        "Please describe the fault or failure you observed."
    ),
    "equipment_name": (
        "What is the name of the equipment that failed? "
        "(e.g., ID Fan, Motor, Compressor, Rotary Kiln, Cooling Tower)"
    ),
    "downtime_minutes": (
        "How long was the equipment down? "
        "(e.g., '2 hours', '45 minutes', '1.5 days' — I'll convert it to minutes)"
    ),
    "production_loss": (
        "Was there any feed or production loss due to this failure? "
        "If yes, please describe the impact."
    ),
    "observations": (
        "Any specific observations before or during the failure? "
        "(e.g., unusual readings, alarms triggered, recent maintenance) "
        "— type 'none' if not applicable."
    ),
    "attachments_note": (
        "Do you have any images of the damaged component or a PDF failure report? "
        "Upload them using the panel in the sidebar — then type 'done'. "
        "If you don't have any, type 'skip' and I'll proceed with the analysis."
    ),
}

# ── Response mode instructions injected into prompts ───────────────────────
CONCISE_INSTRUCTION = (
    "Be brief and direct. Root cause in 1 sentence. Each Why in 1 line only. "
    "No extra commentary after the 5 Whys."
)
DETAILED_INSTRUCTION = (
    "Provide thorough analysis. Include evidence for each Why step. "
    "After the 5 Whys, add a short paragraph with additional context or notes."
)
