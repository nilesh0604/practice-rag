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
from api.guardrails import GuardrailSuite
from api.nim_guardrails import build_guardrail_suite
from api.observability import CircuitBreaker, LangfuseTracer, MetricsCollector
from rag.context_assembler import ContextAssembler
from rag.nim_generator import build_generator
from rag.nim_reranker import build_reranker, get_rerank_candidate_k
from rag.orchestrator import RAGOrchestrator
from rag.post_processor import PostProcessor
from rag.qdrant_collection import get_qdrant_client
from rag.query_rewriter import LLMQueryRewriter
from rag.retriever import DEFAULT_TOP_K, HybridRetriever

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
    client/embedder (which are themselves cached). When the NIM reranker is
    active (Phase 4), the retriever fetches a larger candidate pool
    (``NIM_RERANK_CANDIDATE_K``, default 20) so the reranker has enough
    signal to reorder meaningfully; otherwise the default ``top_k=5``.
    """
    reranker = get_reranker()
    top_k = get_rerank_candidate_k() if reranker is not None else DEFAULT_TOP_K
    return HybridRetriever(get_qdrant_client_dep(), get_embedder(), top_k=top_k)


@lru_cache
def get_circuit_breaker() -> CircuitBreaker:
    """Process-wide Ollama circuit breaker (Step 7, build-order item 41).

    Opens after 3 consecutive Ollama failures, stays open for 30 s, then
    half-opens for one probe call. Shared by the generator so all Ollama
    generation calls go through one breaker.
    """
    return CircuitBreaker(threshold=3, timeout=30.0)


@lru_cache
def get_tracer() -> LangfuseTracer:
    """Process-wide Langfuse tracer (Step 7, build-order item 39).

    Auto-disables when the ``langfuse`` package is absent or
    ``LANGFUSE_BASE_URL`` / public / secret keys are unset — in that mode every
    method is a no-op that logs via structlog (the doc's "Langfuse-down
    fallback to structlog" mitigation).
    """
    return LangfuseTracer()


@lru_cache
def get_metrics() -> MetricsCollector:
    """Process-wide online metrics collector (Step 7, build-order item 40)."""
    return MetricsCollector()


def get_orchestrator() -> RAGOrchestrator:
    """Construct the RAG orchestrator from its real collaborators.

    Tests override this provider entirely (``app.dependency_overrides``) so
    the real Ollama/Qdrant stack is never reached.
    """
    return RAGOrchestrator(
        get_retriever(),
        ContextAssembler(),
        build_generator(circuit_breaker=get_circuit_breaker()),
        PostProcessor(get_embedder()),
        query_rewriter=LLMQueryRewriter(),
        guardrail_suite=get_guardrail_suite(),
        tracer=get_tracer(),
        reranker=get_reranker(),
    )


@lru_cache
def get_reranker():
    """Process-wide NIM reranker (Phase 4).

    Returns ``None`` when NIM reranking is disabled (``NIM_ENABLED`` unset or
    ``NIM_RERANK_ENABLED=false``) — the orchestrator skips the rerank step
    entirely. When enabled, the retriever fetches a larger candidate pool
    (see ``get_retriever``) and the reranker reorders down to ``top_n``.
    The reranker owns a dedicated ``CircuitBreaker`` and a lazy httpx client.
    """
    return build_reranker()


@lru_cache
def get_guardrail_suite() -> GuardrailSuite:
    """Process-wide guardrail suite (Step 6 + NIM Phase 2).

    When ``NIM_ENABLED=true`` the suite uses the NIM-augmented guardrails with
    3-tier fallback (NIM → Ollama → regex/keyword). Otherwise (default) the
    plain Ollama guardrails are used. LLM-backed checks degrade to
    regex/keyword fallbacks if Ollama is unreachable (see
    ``api/guardrails.py`` + ``api/nim_guardrails.py``).
    """
    return build_guardrail_suite()


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
