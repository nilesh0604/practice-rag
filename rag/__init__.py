"""RAG orchestration package.

Holds the online RAG components (retriever, generator, post-processor) and the
shared Qdrant collection helper. Built as plain Python so each piece is
unit-testable without FastAPI.
"""

from rag.context_assembler import ContextAssembler
from rag.generator import Generator, build_system_prompt
from rag.orchestrator import RAGOrchestrator
from rag.post_processor import PostProcessor, PostProcessResult
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
from rag.query_rewriter import (
    LLMQueryRewriter,
    PassthroughQueryRewriter,
    QueryRewriter,
)
from rag.retriever import HybridRetriever

__all__ = [
    "COLLECTION_NAME",
    "CollectionConfig",
    "ContextAssembler",
    "Generator",
    "HybridRetriever",
    "LLMQueryRewriter",
    "PassthroughQueryRewriter",
    "PostProcessResult",
    "PostProcessor",
    "QueryRewriter",
    "RAGOrchestrator",
    "SPARSE_VECTOR_NAME",
    "VECTOR_NAME",
    "VECTOR_SIZE",
    "build_system_prompt",
    "create_collection",
    "ensure_collection",
    "get_qdrant_client",
]
