"""
Image damage analysis utility.
"""

import os
import re
import sys
import json
import base64
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import OPENROUTER_API_KEY, VISION_MODEL

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

ANALYSIS_PROMPT = """\
You are an expert industrial equipment failure analyst.

Look at the image. Identify the component and any damage, tears, wear, corrosion, or defects.

You MUST respond with ONLY a raw JSON object. No markdown. No backticks. No bullet points.
No explanation text before or after. Start your response with { and end with }.

Required format:
{"component": "name of the part", "damage_type": "type of damage", "severity": "None|Minor|Moderate|Severe|Critical", "visual_symptoms": ["symptom1", "symptom2", "symptom3"], "possible_causes": ["cause1", "cause2", "cause3"], "description": "2-3 sentence plain English summary of the damage and why it matters"}

Rules:
- component: real part name e.g. ball bearing, conveyor belt, gear shaft, motor winding
- damage_type: short label e.g. cage fracture, surface pitting, belt tear, thermal cracking
- severity: pick one of None / Minor / Moderate / Severe / Critical
- visual_symptoms: 3-5 specific visible signs in the image
- possible_causes: 3-5 engineering root causes
- description: plain English, 2-3 sentences, no jargon
- If no damage visible: damage_type = "No damage detected", severity = "None"
- YOUR ENTIRE RESPONSE MUST BE VALID JSON ONLY.
"""


def analyze_image_bytes(image_bytes: bytes, mime_type: str) -> dict:
    """
    Send image bytes to vision model via OpenRouter and return structured analysis.
    mime_type: e.g. 'image/jpeg', 'image/png'
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set.")

    img_b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": ANALYSIS_PROMPT,
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    resp = requests.post(
        OPENROUTER_ENDPOINT,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RCA Image Analyzer",
        },
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    resp_json = resp.json()

    message  = resp_json.get("choices", [{}])[0].get("message", {})
    raw_text = (message.get("content") or message.get("reasoning") or "").strip()

    if not raw_text:
        raise ValueError("Vision model returned empty response.")

    return _extract_json(raw_text)


def _extract_json(text: str) -> dict:
    """Robustly extract a JSON object from model output (copied from analyze_image.py)."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(clean)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    for search_text in (text[-1500:], text):
        matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', search_text, re.DOTALL))
        if not matches:
            matches = list(re.finditer(r'\{[\s\S]*\}', search_text))
        if matches:
            for m in reversed(matches):
                try:
                    result = json.loads(m.group())
                    if isinstance(result, dict) and len(result) > 1:
                        return result
                except json.JSONDecodeError:
                    continue

    raise ValueError(f"Could not parse JSON from vision model response:\n{text[:500]}")


def format_image_analysis(analysis: dict) -> str:
    """Format structured image analysis dict into a string for LLM prompt injection."""
    lines = [
        f"Component     : {analysis.get('component', 'N/A')}",
        f"Damage Type   : {analysis.get('damage_type', 'N/A')}",
        f"Severity      : {analysis.get('severity', 'N/A')}",
        f"Visual Symptoms: {', '.join(analysis.get('visual_symptoms', []))}",
        f"Possible Causes: {', '.join(analysis.get('possible_causes', []))}",
        f"Description   : {analysis.get('description', 'N/A')}",
    ]
    return "\n".join(lines)


def get_mime_type(filename: str) -> str:
    """Derive MIME type from filename extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "bmp": "image/bmp",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
