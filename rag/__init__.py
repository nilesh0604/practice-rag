"""RAG orchestration package.

Holds the online RAG components (retriever, generator, post-processor) and the
shared Qdrant collection helper. Built as plain Python so each piece is
unit-testable without FastAPI.
"""

from rag.qdrant_collection import (
    COLLECTION_NAME,
    CollectionConfig,
    SPARSE_VECTOR_NAME,
    VECTOR_NAME,
    VECTOR_SIZE,
    create_collection,
    ensure_collection,
    get_qdrant_client,
)

__all__ = [
    "COLLECTION_NAME",
    "CollectionConfig",
    "SPARSE_VECTOR_NAME",
    "VECTOR_NAME",
    "VECTOR_SIZE",
    "create_collection",
    "ensure_collection",
    "get_qdrant_client",
]
