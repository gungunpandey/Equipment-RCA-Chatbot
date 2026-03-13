"""
RAG utilities equipment context retrieval
"""

import os
import sys
import logging
from typing import List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_COLLECTION

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: str
    source: str
    score: float


def _get_weaviate_client():
    """Connect to Weaviate Cloud. Returns client or None if unavailable."""
    try:
        import weaviate
        from weaviate.classes.init import Auth, AdditionalConfig, Timeout

        if not WEAVIATE_URL or not WEAVIATE_API_KEY:
            logger.warning("WEAVIATE_URL or WEAVIATE_API_KEY not set — RAG disabled.")
            return None

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            skip_init_checks=True,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=10, query=15, insert=120)
            ),
        )
        return client
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}")
        return None


def retrieve_equipment_context(
    equipment_name: str,
    symptoms: List[str],
    top_k: int = 5,
    collection_name: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve relevant equipment documentation from Weaviate using BM25.

    Builds a combined query from equipment name + symptoms and performs
    keyword search. Returns empty list if Weaviate is unavailable,
    caller should fall back to web search.
    """
    if collection_name is None:
        collection_name = WEAVIATE_COLLECTION

    client = _get_weaviate_client()
    if client is None:
        return []

    try:
        symptom_str = " ".join(s for s in symptoms if s.strip())
        query_text = f"{equipment_name} {symptom_str}".strip()

        collection = client.collections.get(collection_name)
        result = collection.query.bm25(query=query_text, limit=top_k)

        documents = []
        for obj in result.objects:
            score = 0.5
            if hasattr(obj.metadata, "score") and obj.metadata.score:
                score = obj.metadata.score
            doc = Document(
                content=obj.properties.get("content", ""),
                source=obj.properties.get("sourcePdf", "Unknown"),
                score=score,
            )
            documents.append(doc)

        logger.info(f"Retrieved {len(documents)} documents from Weaviate for '{query_text}'")
        return documents

    except Exception as e:
        logger.error(f"Weaviate BM25 retrieval failed: {e}")
        return []

    finally:
        try:
            client.close()
        except Exception:
            pass


def format_context_for_llm(documents: List[Document]) -> str:
    """
    Format retrieved documents into a string ready for LLM prompt injection.
    Returns empty string if no documents.
    """
    if not documents:
        return ""

    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(
            f"[Document {i}] Source: {doc.source} (score: {doc.score:.2f})\n"
            f"{doc.content}"
        )
    return "\n---\n".join(parts)
