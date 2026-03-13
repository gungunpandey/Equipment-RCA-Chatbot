"""
Web search utility using DuckDuckGo (no API key required).

Used as a fallback when RAG retrieval returns no results for the
reported equipment. Adapted from xelec_ref internet search pattern
but simplified to use duckduckgo-search instead of Google Custom Search.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search the web using DuckDuckGo.

    Returns a list of {title, body, href} dicts.
    Returns empty list on any failure — caller handles gracefully.
    """
    try:
        from ddgs import DDGS

        results = []
        ddgs = DDGS()
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "body":  r.get("body", ""),
                "href":  r.get("href", ""),
            })

        logger.info(f"Web search returned {len(results)} results for: {query}")
        return results

    except Exception as e:
        logger.warning(f"Web search failed (DuckDuckGo): {e}")
        return []


def format_search_results(results: List[Dict]) -> str:
    """
    Format web search results into a string for LLM prompt injection.
    Returns empty string if no results.
    """
    if not results:
        return ""

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Web Result {i}] {r['title']}\n"
            f"{r['body']}\n"
            f"Source: {r['href']}"
        )
    return "\n---\n".join(parts)
