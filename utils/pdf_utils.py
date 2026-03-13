"""
PDF text extraction utility for uploaded failure reports / manuals.
Uses pypdf to extract text from uploaded PDF bytes.
"""

import io
import logging

logger = logging.getLogger(__name__)

MAX_CHARS = 4000  # cap to avoid bloating the LLM prompt


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes.
    Returns up to MAX_CHARS of text, or empty string on failure.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        total = 0

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            total += len(page_text)
            if total >= MAX_CHARS:
                break

        full_text = "\n".join(text_parts)[:MAX_CHARS]
        logger.info(f"Extracted {len(full_text)} chars from PDF ({len(reader.pages)} pages)")
        return full_text.strip()

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""
