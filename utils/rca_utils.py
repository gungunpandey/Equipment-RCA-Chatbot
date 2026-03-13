"""
RCA conversation utilities: stage management, prompt building, field extraction.
"""

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    REQUIRED_FAULT_FIELDS,
    FIELD_QUESTIONS,
    CONCISE_INSTRUCTION,
    DETAILED_INSTRUCTION,
)


# ── Intake helpers ─────────────────────────────────────────────────────────

def get_missing_fields(fault_info: dict) -> list:
    """Return list of required fields that are still empty."""
    return [f for f in REQUIRED_FAULT_FIELDS if not fault_info.get(f, "").strip()]


def get_next_question(missing_fields: list) -> str:
    """Return the question for the first missing field."""
    if not missing_fields:
        return ""
    return FIELD_QUESTIONS.get(missing_fields[0], f"Please provide: {missing_fields[0]}")


def build_intake_prompt(fault_info: dict, missing_fields: list) -> str:
    """System prompt for the intake stage — ask for ONE missing field at a time."""
    collected = {k: v for k, v in fault_info.items() if v.strip()}
    collected_str = (
        "\n".join(f"  - {k.replace('_', ' ').title()}: {v}" for k, v in collected.items())
        if collected else "  (none yet)"
    )
    next_q = get_next_question(missing_fields)

    return f"""You are an expert industrial equipment RCA (Root Cause Analysis) assistant.

Your goal is to collect the following information before performing RCA:
{chr(10).join(f'  - {f.replace("_", " ").title()}' for f in REQUIRED_FAULT_FIELDS)}

Information already collected:
{collected_str}

Still needed: {', '.join(f.replace('_', ' ') for f in missing_fields)}

Ask the user for exactly ONE piece of missing information:
"{next_q}"

Rules:
- Be professional and conversational. One question only.
- Do not perform any analysis yet.
- If the user's latest message already answers the question, acknowledge briefly
  and ask for the next missing item instead.
- For downtime: accept any format — you will confirm what you understood in minutes."""


def extract_equipment_name(fault_description: str, model) -> str:
    """
    Use LLM to extract equipment name from fault description.
    Falls back to 'Unknown Equipment'.
    """
    try:
        from langchain_core.messages import HumanMessage
        prompt = (
            "Extract only the equipment name from this fault report "
            "(e.g., 'Rotary Kiln', 'ID Fan', 'Compressor', 'Fan Motor', 'Cooling Tower').\n"
            "Return ONLY the equipment name — nothing else. "
            "If unclear, return 'Unknown Equipment'.\n\n"
            f"Fault report: {fault_description}"
        )
        result = model.invoke([HumanMessage(content=prompt)])
        name = result.content.strip()
        if name and len(name) < 60 and name.lower() != "unknown equipment":
            return name
        return "Unknown Equipment"
    except Exception:
        return "Unknown Equipment"


def _regex_extract_downtime(text: str) -> str:
    """
    Extract downtime from free text using regex, no LLM needed.
    Returns a string like '90 minutes', or '' if nothing found.
    """
    t = text.lower()
    # Order matters: days → hours → minutes (most specific first)
    day_m  = re.search(r'(\d+(?:\.\d+)?)\s*days?', t)
    hour_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)', t)
    min_m  = re.search(r'(\d+)\s*(?:minutes?|mins?)', t)

    if day_m:
        mins = int(float(day_m.group(1)) * 24 * 60)
        return f"{mins} minutes"
    if hour_m:
        mins = int(float(hour_m.group(1)) * 60)
        return f"{mins} minutes"
    if min_m:
        return f"{min_m.group(1)} minutes"
    return ""


def extract_fields_from_description(fault_description: str, model) -> dict:
    """
    Extract all available intake fields from the initial fault description.
    Uses regex for downtime (reliable) and LLM for equipment name / other fields.
    Returns only the fields clearly present — caller should not overwrite existing values.
    """
    result: dict = {}

    # ── Downtime: regex first (no LLM needed, very reliable) ──────────────
    downtime = _regex_extract_downtime(fault_description)
    if downtime:
        result["downtime_minutes"] = downtime

    # ── Equipment name + other fields: LLM call ────────────────────────────
    import json as _json
    try:
        from langchain_core.messages import HumanMessage
        prompt = (
            "Extract the following fields from this fault description if they are clearly stated. "
            "Return ONLY a raw JSON object — no markdown, no explanation.\n\n"
            "Fields to extract:\n"
            "- equipment_name: name of the equipment (e.g. 'Hydraulic Pump', 'ID Fan') — short name only\n"
            "- production_loss: any mention of production/feed loss impact (string or omit if not mentioned)\n"
            "- observations: specific technical observations stated (string or omit if not mentioned)\n\n"
            "Rules:\n"
            "- Only include a field if the information is explicitly stated — do not infer.\n"
            "- If a field is not mentioned, omit it from the JSON entirely.\n"
            "- equipment_name must be under 60 characters.\n\n"
            f"Fault description: {fault_description}\n\n"
            "Respond with ONLY the JSON object."
        )
        llm_result = model.invoke([HumanMessage(content=prompt)])
        raw = llm_result.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        parsed = _json.loads(raw)
        if isinstance(parsed, dict):
            if "equipment_name" in parsed:
                name = str(parsed["equipment_name"]).strip()
                if name and len(name) <= 60 and name.lower() not in ("unknown", "unknown equipment", ""):
                    result["equipment_name"] = name
            for field in ("production_loss", "observations"):
                if field in parsed:
                    val = str(parsed[field]).strip()
                    if val and val.lower() not in ("null", "none", ""):
                        result[field] = val
    except Exception:
        # LLM extraction failed — still return whatever regex found
        pass

    return result


def convert_downtime_to_minutes(downtime_str: str, model) -> str:
    """
    Convert any downtime string to minutes using Gemini.
    e.g. '2 hours' → '120', '1.5 days' → '2160', '45 minutes' → '45'
    Returns the converted string like '120 minutes'.
    """
    try:
        from langchain_core.messages import HumanMessage
        prompt = (
            "Convert this downtime duration to minutes. "
            "Return ONLY a number followed by 'minutes', e.g. '120 minutes'. "
            "Nothing else.\n\n"
            f"Duration: {downtime_str}"
        )
        result = model.invoke([HumanMessage(content=prompt)])
        converted = result.content.strip()
        # Sanity check: must contain a number
        if re.search(r'\d', converted):
            return converted
        return downtime_str
    except Exception:
        return downtime_str


# ── Domain analysis prompt ─────────────────────────────────────────────────

def build_domain_analysis_prompt(
    fault_info: dict,
    equipment_name: str,
    rag_context: str,
    web_context: str,
) -> str:
    """
    Build prompt for domain expert analysis step.
    Analyzes failure from Mechanical, Electrical, and Process perspectives
    in a single call — adapted from rca_ref domain agent pattern.
    """
    context_section = ""
    if rag_context:
        context_section += f"\nEQUIPMENT DOCUMENTATION (OEM Manuals):\n{rag_context}\n"
    if web_context:
        context_section += f"\nWEB SEARCH RESULTS:\n{web_context}\n"
    if not rag_context and not web_context:
        context_section = "\nNo external documentation available. Use engineering knowledge.\n"

    return f"""You are a panel of 3 domain expert engineers analyzing an industrial equipment failure.

FAILURE DETAILS:
- Equipment     : {equipment_name}
- Fault         : {fault_info.get('fault_description', 'Not specified')}
- Downtime      : {fault_info.get('downtime_minutes', 'Not specified')}
- Production Loss: {fault_info.get('production_loss', 'Not specified')}
- Observations  : {fault_info.get('observations', 'Not specified')}
{context_section}
Analyze this failure from THREE domain perspectives. For each domain:
- State the most likely contributing factor from that domain's viewpoint
- Reference any evidence from documentation if available
- State confidence: High / Medium / Low

Respond in EXACTLY this format:

MECHANICAL: [1-2 lines — mechanical cause hypothesis and confidence]
ELECTRICAL: [1-2 lines — electrical cause hypothesis and confidence]
PROCESS: [1-2 lines — process/operational cause hypothesis and confidence]
COMBINED HYPOTHESIS: [1-2 lines — most probable combined root cause direction]"""


# ── 5 Whys prompt ──────────────────────────────────────────────────────────

def build_rca_prompt(
    fault_info: dict,
    equipment_name: str,
    domain_analysis: str,
    rag_context: str,
    web_context: str,
    response_mode: str,
) -> str:
    """
    Build the 5 Whys prompt using domain analysis as foundation.
    Produces ROOT CAUSE (1 line) + 5 Whys (1 line each).
    Adapted from rca_ref/llm/tools/five_whys_tool.py prompts.
    """
    mode_instruction = (
        CONCISE_INSTRUCTION if response_mode == "Concise" else DETAILED_INSTRUCTION
    )

    context_section = ""
    if rag_context:
        context_section += f"\nEQUIPMENT DOCUMENTATION (OEM Manuals):\n{rag_context}\n"
    if web_context:
        context_section += f"\nWEB SEARCH RESULTS:\n{web_context}\n"
    if not rag_context and not web_context:
        context_section = "\nNo external documentation available. Use engineering knowledge.\n"

    return f"""You are an expert industrial equipment failure analyst performing a 5 Whys Root Cause Analysis.

EQUIPMENT FAILURE DETAILS:
- Equipment      : {equipment_name}
- Fault          : {fault_info.get('fault_description', 'Not specified')}
- Downtime       : {fault_info.get('downtime_minutes', 'Not specified')}
- Production Loss: {fault_info.get('production_loss', 'Not specified')}
- Observations   : {fault_info.get('observations', 'Not specified')}
{context_section}
DOMAIN EXPERT PRE-ANALYSIS:
{domain_analysis}

ANALYSIS RULES (from industrial RCA best practices):
1. Never use HTTP/API errors as plant failure modes.
2. Plant signal failures are: "Bad Quality", "Comm Fail", "Signal Unhealthy", "Input Forced".
3. Back every claim with evidence where possible. If inferring, say "Based on inference".
4. Root cause = first equipment-level failure that explains ALL observed symptoms.
5. Do NOT escalate to governance/design failures without direct evidence.
6. Stop at 3-5 Whys — stop as soon as root cause is clearly identified.
7. Do NOT use any markdown formatting — no **, no *, no #, no headings.
8. Each Why must be exactly 1 line.
9. {mode_instruction}

CRITICAL OUTPUT RULES:
- Your ENTIRE response MUST start with the literal text "ROOT CAUSE:" — nothing before it.
- Do NOT write any introduction, preamble, summary, problem statement, or section header.
- Do NOT generate any text outside the exact format shown below.
- Do NOT use markdown headings (#, ##, ###) anywhere in your response.

Your response MUST follow this EXACT format — nothing more, nothing less:

ROOT CAUSE: [1 concise sentence — the fundamental cause]

5 WHYS:
Why 1: [1 line — why did the equipment fail?]
Why 2: [1 line — why did that happen?]
Why 3: [1 line — why did that happen?]
Why 4: [1 line — only if chain is not yet complete]
Why 5: [1 line — only if needed]

Example of correct output:
ROOT CAUSE: Bearing failure caused by prolonged operation without lubrication.

5 WHYS:
Why 1: Fan motor tripped on thermal overload due to excessive current draw.
Why 2: Current draw increased because the motor was starting against full load.
Why 3: Fan was starting against full load because the damper linkage was broken.
Why 4: Damper linkage broke due to fatigue from continuous vibration.
Why 5: Vibration went undetected because routine inspection intervals were missed.

IMPORTANT: Each Why MUST be on its own separate line. Do NOT merge them into a paragraph. START your response with ROOT CAUSE:"""


# ── Corrective measures prompt ─────────────────────────────────────────────

def build_corrective_prompt(
    fault_info: dict,
    equipment_name: str,
    rca_result: str,
    rag_context: str,
    response_mode: str,
) -> str:
    """System prompt for corrective measures follow-up stage."""
    mode_instruction = (
        CONCISE_INSTRUCTION if response_mode == "Concise" else DETAILED_INSTRUCTION
    )
    context_section = (
        f"\nEQUIPMENT DOCUMENTATION:\n{rag_context}\n" if rag_context else ""
    )
    return f"""You are an expert industrial equipment maintenance engineer.

FAULT CONTEXT:
- Equipment      : {equipment_name}
- Fault          : {fault_info.get('fault_description', 'Unknown')}
- Downtime       : {fault_info.get('downtime_minutes', 'Unknown')}
- Production Loss: {fault_info.get('production_loss', 'Unknown')}
- Observations   : {fault_info.get('observations', 'Unknown')}
{context_section}
ROOT CAUSE IDENTIFIED:
{rca_result}

Answer the user's follow-up question about corrective or preventive measures.
Be specific and actionable. Reference documentation procedures where available.
{mode_instruction}"""


# ── Response parser ────────────────────────────────────────────────────────

def parse_rca_response(response: str) -> dict:
    """Parse structured RCA response. Returns root_cause, why_steps, source."""
    try:
        # Primary: look for explicit ROOT CAUSE: marker
        rc_match = re.search(
            r'ROOT CAUSE:\s*(.+?)(?=\n5 WHYS:|\n\nWhy|\nWhy 1:|\n\n|\Z)',
            response, re.IGNORECASE | re.DOTALL
        )
        root_cause = rc_match.group(1).strip() if rc_match else ""

        # Primary: look for Why N: lines
        why_steps = re.findall(r'Why\s*\d+\s*:\s*(.+)', response, re.IGNORECASE)

        # Fallback root_cause: look for "root cause" keyword anywhere in text
        if not root_cause:
            rc_fallback = re.search(
                r'(?:root\s+cause[:\s]+|fundamental\s+cause[:\s]+)(.+?)(?:\n|$)',
                response, re.IGNORECASE
            )
            if rc_fallback:
                root_cause = rc_fallback.group(1).strip()

        # Fallback why_steps: look for numbered list items like "1.", "1)"
        if not why_steps:
            why_steps = re.findall(r'^\s*\d+[.)]\s*(.+)', response, re.MULTILINE)

        # Last-resort root_cause: use first non-empty sentence
        if not root_cause and response.strip():
            first_line = next(
                (l.strip() for l in response.splitlines() if l.strip() and not l.strip().startswith('#')),
                ""
            )
            root_cause = first_line[:200] if first_line else "See analysis below."

        src_match = re.search(r'SOURCE:\s*(.+)', response, re.IGNORECASE)
        source    = src_match.group(1).strip() if src_match else "Engineering Knowledge"
        return {"root_cause": root_cause, "why_steps": why_steps, "source": source}
    except Exception:
        return {"root_cause": response[:300], "why_steps": [], "source": "Engineering Knowledge"}


# ── Display formatter ──────────────────────────────────────────────────────

def format_rca_for_display(rca_text: str, rag_context: str, web_context: str) -> str:
    """
    Parse the raw LLM output and reformat it into clean markdown:
    - Root cause shown as bold heading
    - Each Why on its own line
    - Single source badge at the bottom (not duplicated)
    """
    parsed = parse_rca_response(rca_text)
    root_cause = parsed["root_cause"]
    why_steps  = parsed["why_steps"]

    lines = []

    # Root cause — bold and prominent
    lines.append("### Root Cause")
    lines.append(f"**{root_cause}**")
    lines.append("")
    lines.append("---")

    # 5 Whys — one per line
    lines.append("### 5 Whys")
    if why_steps:
        for i, why in enumerate(why_steps, 1):
            lines.append(f"**Why {i}:** {why}  ")
    else:
        # Fallback: LLM ignored format — strip markdown headers and show cleaned text
        cleaned = re.sub(r'^#+\s*', '', rca_text, flags=re.MULTILINE)
        lines.append(cleaned)

    lines.append("")

    # Single source badge
    if rag_context:
        lines.append("> **Source:** OEM Equipment Manuals (RAG)")
    elif web_context:
        lines.append("> **Source:** Web Search + Engineering Knowledge")
    else:
        lines.append("> **Source:** Engineering Knowledge")

    return "\n".join(lines)
