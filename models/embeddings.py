"""
Embedding model module.

This project uses Weaviate Cloud as the vector store, which handles
embeddings server-side via its configured vectorizer. RAG retrieval
uses BM25 keyword search against the pre-ingested equipment manuals
collection, so no local embedding model is required for retrieval.

This module exposes get_embedding_info() for transparency and can be
extended with a local HuggingFace model if needed in the future.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import WEAVIATE_COLLECTION


def get_embedding_info() -> dict:
    """Return metadata about the embedding setup used in this project."""
    return {
        "type": "weaviate_managed",
        "description": (
            "Embeddings are managed by Weaviate Cloud. "
            "RAG retrieval uses BM25 keyword search on the existing "
            f"'{WEAVIATE_COLLECTION}' collection of equipment manuals."
        ),
        "collection": WEAVIATE_COLLECTION,
    }
