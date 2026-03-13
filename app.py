import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from models.llm import get_gemini_model
from utils.rag_utils import retrieve_equipment_context, format_context_for_llm
from utils.web_search import search_web, format_search_results
from utils.image_utils import analyze_image_bytes, format_image_analysis, get_mime_type
from utils.pdf_utils import extract_pdf_text
from utils.rca_utils import (
    get_missing_fields,
    build_intake_prompt,
    build_domain_analysis_prompt,
    build_rca_prompt,
    build_corrective_prompt,
    extract_equipment_name,
    extract_fields_from_description,
    convert_downtime_to_minutes,
    format_rca_for_display,
)
from config.config import REQUIRED_FAULT_FIELDS


# ── Session state ──────────────────────────────────────────────────────────

def _init_session_state():
    defaults = {
        "messages":          [],
        "fault_info":        {f: "" for f in REQUIRED_FAULT_FIELDS},
        "equipment_name":    "",
        "rca_stage":         "intake",   # intake | ask_corrective | corrective
        "rca_result":        "",
        "last_asked_field":  None,
        "response_mode":     "Detailed",
        "rag_context_cache": "",
        "image_analysis":    None,   # dict from vision model
        "pdf_context":       "",     # extracted PDF text
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_case():
    st.session_state.messages               = []
    st.session_state.fault_info             = {f: "" for f in REQUIRED_FAULT_FIELDS}
    st.session_state.equipment_name         = ""
    st.session_state.rca_stage              = "intake"
    st.session_state.rca_result             = ""
    st.session_state.last_asked_field       = None
    st.session_state.rag_context_cache      = ""
    st.session_state.image_analysis         = None
    st.session_state.pdf_context            = ""
    st.session_state["_uploaded_image_data"] = None
    st.session_state["_uploaded_pdf_data"]   = None


# ── RCA pipeline ───────────────────────────────────────────────────────────

def _run_analysis(model) -> str:
    """
    Full RCA pipeline wrapped in a collapsible st.status thinking box.
    Steps: extract name → RAG → web fallback → domain analysis → 5 Whys.
    Logs disappear into a collapsed box once the result is ready.
    """
    fault_info    = st.session_state.fault_info
    response_mode = st.session_state.response_mode
    rag_context   = ""
    web_context   = ""
    domain_analysis = ""
    rca_result    = ""

    with st.status("Analyzing failure...", expanded=True) as status:

        # ── Step 1: Equipment name ─────────────────────────────────────────
        st.write("**[1/5] Extracting equipment details...**")
        eq_name = st.session_state.equipment_name or extract_equipment_name(
            fault_info.get("fault_description", ""), model
        )
        st.session_state.equipment_name = eq_name
        st.write(f"  → Equipment: {eq_name}")

        # ── Step 2: RAG retrieval ──────────────────────────────────────────
        st.write("**[2/5] Searching equipment manuals (RAG)...**")
        try:
            symptom_text = (
                fault_info.get("fault_description", "") + " " +
                fault_info.get("observations", "")
            )
            symptoms = [s.strip() for s in symptom_text.split(",") if s.strip()][:5]
            rag_docs = retrieve_equipment_context(eq_name, symptoms, top_k=5)
            rag_context = format_context_for_llm(rag_docs)
            if rag_context:
                st.write(f"  → Found {len(rag_docs)} relevant manual sections.")
            else:
                st.write("  → No manual found for this equipment.")
        except Exception as e:
            st.write(f"  → RAG unavailable: {e}")

        # ── Step 3: Web search fallback ────────────────────────────────────
        if not rag_context:
            st.write("**[3/5] Searching web for failure patterns...**")
            try:
                query = (
                    f"{eq_name} failure "
                    f"{fault_info.get('fault_description', '')} "
                    "root cause troubleshooting"
                )
                web_results = search_web(query, max_results=3)
                web_context = format_search_results(web_results)
                if web_context:
                    st.write(f"  → Found {len(web_results)} web sources.")
                else:
                    st.write("  → No web results — using engineering knowledge.")
            except Exception as e:
                st.write(f"  → Web search failed: {e}")
        else:
            st.write("**[3/5] RAG context available — skipping web search.**")

        st.session_state.rag_context_cache = rag_context

        # ── Step 3b: Process uploaded attachments ─────────────────────────
        uploaded_image = st.session_state.get("_uploaded_image_data")
        uploaded_pdf   = st.session_state.get("_uploaded_pdf_data")

        if uploaded_image:
            st.write("**[3b] Analyzing uploaded image...**")
            try:
                img_bytes, img_name = uploaded_image
                mime = get_mime_type(img_name)
                analysis = analyze_image_bytes(img_bytes, mime)
                st.session_state.image_analysis = analysis
                st.write(
                    f"  → Image: {analysis.get('component', 'N/A')} — "
                    f"{analysis.get('damage_type', 'N/A')} "
                    f"(Severity: {analysis.get('severity', 'N/A')})"
                )
            except Exception as e:
                st.write(f"  → Image analysis failed: {e}")

        if uploaded_pdf:
            st.write("**[3b] Extracting uploaded PDF context...**")
            try:
                pdf_bytes, pdf_name = uploaded_pdf
                pdf_text = extract_pdf_text(pdf_bytes)
                st.session_state.pdf_context = pdf_text
                st.write(f"  → Extracted {len(pdf_text)} chars from {pdf_name}")
            except Exception as e:
                st.write(f"  → PDF extraction failed: {e}")

        # Build combined context string for prompts
        image_analysis = st.session_state.get("image_analysis")
        pdf_context    = st.session_state.get("pdf_context", "")

        if image_analysis:
            rag_context += (
                "\n\nIMAGE ANALYSIS (uploaded component photo):\n"
                + format_image_analysis(image_analysis)
            )
        if pdf_context:
            rag_context += (
                "\n\nUPLOADED DOCUMENT CONTEXT:\n"
                + pdf_context
            )

        # ── Step 4: Domain expert analysis ────────────────────────────────
        st.write("**[4/5] Running domain expert analysis (Mechanical / Electrical / Process)...**")
        domain_prompt = build_domain_analysis_prompt(
            fault_info, eq_name, rag_context, web_context
        )
        domain_response = model.invoke([
            SystemMessage(content=domain_prompt),
            HumanMessage(content=f"Analyze the {eq_name} failure from all domain perspectives."),
        ])
        domain_analysis = domain_response.content
        st.write("  → Domain analysis complete.")
        with st.expander("Domain Expert Findings", expanded=False):
            st.text(domain_analysis)

        # ── Step 5: 5 Whys ────────────────────────────────────────────────
        st.write("**[5/5] Performing 5 Whys Root Cause Analysis...**")
        rca_system_prompt = build_rca_prompt(
            fault_info, eq_name, domain_analysis, rag_context, web_context, response_mode
        )
        rca_response = model.invoke([
            SystemMessage(content=rca_system_prompt),
            HumanMessage(content=f"Perform the 5 Whys RCA for the {eq_name} failure."),
        ])
        rca_result = rca_response.content
        st.write("  → RCA complete.")

        # Collapse the thinking box
        status.update(label="Analysis complete", state="complete", expanded=False)

    # Store result and move to follow-up stage (user can ask for corrective measures freely)
    st.session_state.rca_result = rca_result
    st.session_state.rca_stage  = "corrective"

    # Format result with proper markdown (bold root cause, line-broken Whys, single source)
    formatted = format_rca_for_display(rca_result, rag_context, web_context)

    return (
        formatted +
        "\n\n---\n"
        "*You can ask for corrective measures, prevention steps, or any follow-up questions.*"
    )


# ── Message handler ────────────────────────────────────────────────────────

def _handle_message(user_message: str, model) -> str:
    """Route user message to the correct stage handler."""
    stage      = st.session_state.rca_stage
    fault_info = st.session_state.fault_info

    # ── INTAKE STAGE ───────────────────────────────────────────────────────
    if stage == "intake":
        # First message — seed fault_description and auto-extract any stated fields
        if not any(fault_info.values()):
            fault_info["fault_description"] = user_message
            extracted = extract_fields_from_description(user_message, model)
            for field, value in extracted.items():
                if field in fault_info and not fault_info[field]:
                    fault_info[field] = value
            # Sync equipment_name to session state
            if fault_info.get("equipment_name"):
                st.session_state.equipment_name = fault_info["equipment_name"]
        else:
            # Map reply to the last asked field
            field = st.session_state.last_asked_field
            if field:
                value = user_message
                # Convert downtime to minutes if needed
                if field == "downtime_minutes":
                    value = convert_downtime_to_minutes(user_message, model)
                fault_info[field] = value

        # Auto-fill attachments_note if files are already uploaded in sidebar
        if not fault_info.get("attachments_note", "").strip():
            has_image = st.session_state.get("_uploaded_image_data") is not None
            has_pdf   = st.session_state.get("_uploaded_pdf_data") is not None
            if has_image or has_pdf:
                parts = []
                if has_image:
                    parts.append("image")
                if has_pdf:
                    parts.append("PDF")
                fault_info["attachments_note"] = f"uploaded via sidebar: {', '.join(parts)}"

        missing = get_missing_fields(fault_info)

        # ── Special case: only attachments_note remains ────────────────────
        # Bypass LLM entirely to prevent it from hallucinating an RCA response.
        if missing == ["attachments_note"]:
            user_lower = user_message.lower().strip()
            skip_words = {"skip", "no", "none", "n/a", "nope", "nothing", "na"}
            done_words = {"done", "uploaded", "yes", "ok", "okay", "added", "attached"}
            if any(w in user_lower for w in skip_words):
                fault_info["attachments_note"] = "skip"
            elif any(w in user_lower for w in done_words):
                fault_info["attachments_note"] = "uploaded"
            else:
                # Not answered yet — ask directly without LLM
                st.session_state.last_asked_field = "attachments_note"
                return (
                    "Do you have any images of the damaged component or a PDF failure report? "
                    "Upload them using the **Attachments** panel in the sidebar, then type **done**. "
                    "If you don't have any, type **skip**."
                )
            # Attachments handled — run analysis
            return _run_analysis(model)

        if missing:
            st.session_state.last_asked_field = missing[0]
            system_prompt = build_intake_prompt(fault_info, missing)
            response = model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ])
            return response.content
        else:
            # All fields collected — trigger full analysis pipeline
            return _run_analysis(model)

    # ── CORRECTIVE STAGE ───────────────────────────────────────────────────
    elif stage == "corrective":
        # Detect explicit web search intent in the user's message
        web_keywords = [
            "search", "internet", "online", "web", "look up", "lookup",
            "find online", "google", "browse", "search for",
        ]
        user_lower = user_message.lower()
        wants_web = any(kw in user_lower for kw in web_keywords)

        live_web_context = ""
        if wants_web:
            with st.status("Searching the web...", expanded=True) as ws:
                try:
                    eq_name = st.session_state.get("equipment_name", "")
                    fault   = st.session_state.fault_info.get("fault_description", "")

                    # Strip conversational filler to extract just the topic keywords
                    filler = [
                        "can you", "could you", "please", "search the", "search for",
                        "search internet", "internet and", "provide me", "give me",
                        "list of", "a list", "look up", "find me", "tell me",
                        "browse the web", "go online", "on the web", "on the internet",
                        "online", "internet", "google", "web search", "search",
                    ]
                    clean = user_message.lower()
                    for f in filler:
                        clean = clean.replace(f, " ")
                    topic = " ".join(clean.split()).strip()

                    # Build a short, clean query (≤60 chars total)
                    if topic and topic not in ("", "the"):
                        query = f"{eq_name} {topic}"
                    else:
                        query = f"{eq_name} failure causes troubleshooting"

                    st.write(f"  → Query: `{query[:100]}`")
                    results = search_web(query, max_results=5)
                    live_web_context = format_search_results(results)
                    st.write(f"  → Found {len(results)} web results.")
                    ws.update(label="Web search complete", state="complete", expanded=False)
                except Exception as e:
                    st.write(f"  → Web search failed: {e}")
                    ws.update(label="Web search failed", state="error", expanded=False)

        # Build context: combine cached RAG + any live web results
        combined_context = st.session_state.rag_context_cache
        if live_web_context:
            combined_context += f"\n\nLIVE WEB SEARCH RESULTS:\n{live_web_context}"

        system_prompt = build_corrective_prompt(
            fault_info=fault_info,
            equipment_name=st.session_state.equipment_name,
            rca_result=st.session_state.rca_result,
            rag_context=combined_context,
            response_mode=st.session_state.response_mode,
        )
        messages = [SystemMessage(content=system_prompt)]
        for msg in st.session_state.messages[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_message))
        response = model.invoke(messages)

        suffix = (
            "\n\n> **Source:** Web Search + Engineering Knowledge"
            if live_web_context else ""
        )
        return response.content + suffix

    return "Unexpected state. Please click New Case to restart."


# ── Pages ──────────────────────────────────────────────────────────────────

def chat_page():
    st.title("Equipment RCA Assistant")
    st.caption("Guided Root Cause Analysis for Industrial Equipment Failures")

    try:
        model = get_gemini_model()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    _init_session_state()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Settings")
        mode = st.radio("Response Mode", ["Concise", "Detailed"], index=1)
        st.session_state.response_mode = mode

        st.divider()
        st.subheader("Current Case")

        stage_labels = {
            "intake":     "Gathering Info",
            "corrective": "RCA Complete — Follow-up Q&A",
        }
        stage_colors = {
            "intake":     "orange",
            "corrective": "green",
        }
        current_stage = st.session_state.get("rca_stage", "intake")
        label = stage_labels.get(current_stage, current_stage)
        color = stage_colors.get(current_stage, "gray")
        st.markdown(f"**Stage:** :{color}[{label}]")

        fault_info = st.session_state.fault_info
        if any(fault_info.values()):
            for field, value in fault_info.items():
                if value:
                    label_text = field.replace("_", " ").title()
                    short = value[:45] + ("..." if len(value) > 45 else "")
                    st.caption(f"**{label_text}:** {short}")
        else:
            st.caption("No case started yet.")

        st.divider()
        st.subheader("Attachments")
        st.caption("Upload before the analysis runs")

        img_file = st.file_uploader(
            "Component image (jpg/png)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_uploader",
        )
        pdf_file = st.file_uploader(
            "Failure report / PDF",
            type=["pdf"],
            key="pdf_uploader",
        )
        # Store bytes in session state so _run_analysis can pick them up.
        # Only update when a file is present — clearing is handled by _reset_case()
        # to prevent Streamlit reruns from wiping already-uploaded file bytes.
        if img_file is not None:
            st.session_state["_uploaded_image_data"] = (img_file.read(), img_file.name)
            img_file.seek(0)

        if pdf_file is not None:
            st.session_state["_uploaded_pdf_data"] = (pdf_file.read(), pdf_file.name)
            pdf_file.seek(0)

        # Show status of what's ready
        img_data = st.session_state.get("_uploaded_image_data")
        pdf_data = st.session_state.get("_uploaded_pdf_data")
        if img_data:
            st.caption(f"Image ready: {img_data[1]}")
        if pdf_data:
            st.caption(f"PDF ready: {pdf_data[1]}")

        st.divider()
        if st.button("New Case", use_container_width=True):
            _reset_case()
            st.rerun()

    # ── Welcome message ────────────────────────────────────────────────────
    if not st.session_state.messages:
        welcome = (
            "Hello! I'm your **RCA Assistant**. I'll guide you through a structured "
            "Root Cause Analysis for any industrial equipment failure.\n\n"
            "**To get started:** Describe the fault or failure you're experiencing.\n\n"
            "*Example: 'The ID Fan tripped due to a high vibration alarm'*"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})

    # ── Display chat history ───────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat input ─────────────────────────────────────────────────────────
    stage_now = st.session_state.rca_stage
    if stage_now == "intake":
        placeholder = "Describe the fault or answer the question above..."
    else:
        placeholder = "Ask about corrective measures, prevention, or any follow-up..."

    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = _handle_message(prompt, model)
            except Exception as e:
                response = (
                    f"An error occurred: {str(e)}\n\n"
                    "Please try again or click **New Case** to restart."
                )
            # st.write() log lines from _run_analysis already rendered above.
            # st.markdown() renders the final response text below the logs.
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Equipment RCA Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    chat_page()


if __name__ == "__main__":
    main()
