"""FastAPI dependency providers — wire the Step 3 RAG components into the HTTP layer.

Each provider is a small factory exposed via ``Depends(...)``. The
heavyweight singletons (Qdrant client, embedder, conversation store, cache)
are cached with ``functools.lru_cache`` so one instance serves the whole
process. Tests override these via ``app.dependency_overrides`` to inject
mocks without touching the network or disk.

The orchestrator is constructed from its real collaborators by default
(host Ollama + Qdrant), but because every collaborator is injected, tests
replace ``get_orchestrator`` with a mock and never construct a real one.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from ingestion.embedder import Embedder
from qdrant_client import QdrantClient

from api.cache import DEFAULT_MAX_SIZE, LRUCache
from api.conversation import DEFAULT_DB_PATH, ConversationStore
from api.guardrails import GuardrailSuite, InputGuardrail, OutputGuardrail, QueryClassifier
from rag.context_assembler import ContextAssembler
from rag.generator import Generator
from rag.orchestrator import RAGOrchestrator
from rag.post_processor import PostProcessor
from rag.qdrant_collection import get_qdrant_client
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR: str = os.getenv("CORPUS_DIR", "data/corpus")
"""Default corpus directory for the ``/api/v1/ingest`` endpoint."""


@lru_cache
def get_qdrant_client_dep() -> QdrantClient:
    """Process-wide Qdrant client (built from ``QDRANT_URL`` env)."""
    return get_qdrant_client()


@lru_cache
def get_embedder() -> Embedder:
    """Process-wide Ollama embedder (nomic-embed-text)."""
    return Embedder()


def get_retriever() -> HybridRetriever:
    """Construct the hybrid retriever from the cached client + embedder.

    Not cached itself — it holds no expensive state beyond the injected
    client/embedder (which are themselves cached).
    """
    return HybridRetriever(get_qdrant_client_dep(), get_embedder())


def get_orchestrator() -> RAGOrchestrator:
    """Construct the RAG orchestrator from its real collaborators.

    Tests override this provider entirely (``app.dependency_overrides``) so
    the real Ollama/Qdrant stack is never reached.
    """
    return RAGOrchestrator(
        get_retriever(),
        ContextAssembler(),
        Generator(),
        PostProcessor(get_embedder()),
        guardrail_suite=get_guardrail_suite(),
    )


@lru_cache
def get_guardrail_suite() -> GuardrailSuite:
    """Process-wide guardrail suite (Step 6).

    LLM-backed checks are enabled by default; they degrade to regex/keyword
    fallbacks if Ollama is unreachable (see ``api/guardrails.py``).
    """
    return GuardrailSuite(
        input_guardrail=InputGuardrail(),
        output_guardrail=OutputGuardrail(),
        classifier=QueryClassifier(),
    )


@lru_cache
def get_conversation_store() -> ConversationStore:
    """Process-wide SQLite conversation store (``CHAT_DB_PATH`` env)."""
    return ConversationStore(DEFAULT_DB_PATH)


@lru_cache
def get_cache() -> LRUCache:
    """Process-wide LRU response cache."""
    return LRUCache(max_size=DEFAULT_MAX_SIZE)


def get_corpus_dir() -> str:
    """Resolve the corpus directory for the ingest endpoint."""
    return DEFAULT_CORPUS_DIR
