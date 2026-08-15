"""NVIDIA NIM embedder + NIM ingestion path (Phase 3 of the NIM integration plan).

Provides a NIM-backed embedder that implements the same ``embed_texts`` /
``embed_text`` interface as the Ollama ``Embedder`` so it is a drop-in for the
existing ``IndexWriter`` and ``HybridRetriever``. Unlike the generator and
guardrails (Phases 1–2), the NIM embedder is **not** a runtime fallback and is
**not** wired into the live app's default path — embeddings are not
interoperable across models, so the NIM embedder populates a **separate**
Qdrant collection (``ctc_rag_nim``) for retrieval-quality (recall@5)
comparison only. The live app continues to query the Ollama ``docs-knowledge``
collection.

Design notes (from ``docs/NVIDIA_NIM_INTEGRATION_PLAN.md`` §2.5 + §3.3):

- NIM is **opt-in** (``NIM_ENABLED=true``). The default embedder (Ollama
  ``nomic-embed-text``) is unchanged — ``api/deps.py`` is not modified here.
- The NIM embedder calls the OpenAI-compatible ``/embeddings`` endpoint
  (``POST https://integrate.api.nvidia.com/v1/embeddings``) with ``model``,
  ``input``, and ``input_type`` (``passage`` at index time, ``query`` at query
  time — the nemotron model is tuned for asymmetric bi-encoder retrieval).
- The nemotron model supports Matryoshka reduced dimensions (384/512/768/
  1024/2048) via the optional ``dimensions`` API param. The native 2048-d is
  used by default; override with ``NIM_EMBEDDING_DIM`` to trade storage for
  speed. The collection is created at the configured dimensionality.
- The NIM embedder gets its **own** ``CircuitBreaker`` (separate from the
  generator's and guardrails' NIM breakers and the Ollama breaker) so an
  embedding outage does not trip other breakers.
- NIM-specific HTTP errors (429 rate limit, 404 model retired) are mapped to
  ``NIMEmbeddingError`` — **not** retried (retrying a 429 against a shared
  global limit would waste the budget). There is **no fallback embedder**:
  vectors are not interoperable, so a NIM embedding failure aborts the
  ingestion batch (the operator re-runs ingestion; the Ollama collection is
  unaffected).
- The sparse vector tier (local hashing-trick BM25) is embedder-agnostic and
  unchanged — the NIM collection keeps the same hybrid dense+sparse design.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from ingestion.embedder import Embedder
from rag.qdrant_collection import (
    NIM_COLLECTION_NAME,
    NIM_EMBEDDING_MODEL_DEFAULT,
    NIM_VECTOR_SIZE,
    build_nim_collection_config,
    ensure_nim_collection,
)

if TYPE_CHECKING:
    from api.observability import CircuitBreaker
    from rag.qdrant_collection import CollectionConfig

logger = logging.getLogger(__name__)

# ── NIM constants ──────────────────────────────────────────────────────

NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
"""NVIDIA NIM OpenAI-compatible API base URL (free tier).

Duplicated from ``rag.nim_generator`` / ``api.nim_guardrails`` to keep the
embedder module decoupled from the generator/guardrail modules (no
cross-package import for a constant)."""

NIM_EMBED_TIMEOUT: float = 60.0
"""Per-request timeout for NIM embedding calls (seconds).

Longer than the guardrail timeout (30s) because embedding a batch of chunks
is heavier than a single guardrail verdict, and ingestion is a batch job
(not in the per-request hot path)."""

INPUT_TYPE_PASSAGE: str = "passage"
"""NIM ``input_type`` for index-time embeddings (document passages)."""

INPUT_TYPE_QUERY: str = "query"
"""NIM ``input_type`` for query-time embeddings (search queries)."""


# ── exceptions ─────────────────────────────────────────────────────────


class NIMEmbeddingError(Exception):
    """Raised when a NIM embedding API call fails.

    Unlike the generator/guardrail paths there is **no fallback embedder**
    (vectors are not interoperable), so this is raised to the caller — the
    ingestion orchestrator aborts the batch. The Ollama collection is never
    touched by the NIM ingestion path, so a NIM failure cannot corrupt the
    live app's default collection.
    """


# ── NIM embedder ───────────────────────────────────────────────────────


class NIMEmbedder:
    """OpenAI-compatible embedding client for NVIDIA NIM.

    Calls ``POST /embeddings`` with ``input_type`` (``passage`` at index time,
    ``query`` at query time) and returns dense float vectors. The httpx client
    is lazily constructed (like the Ollama ``Embedder``) so unit tests inject a
    mock without hitting the network.

    Owns a dedicated ``CircuitBreaker`` (separate from all other breakers).
    When the circuit is open, ``embed_texts`` raises ``CircuitOpenError``
    immediately (no network round-trip).

    There is intentionally **no fallback** to the Ollama embedder: vectors
    from different models live in different spaces and cannot be mixed in one
    Qdrant collection. A NIM embedding failure aborts the ingestion batch.
    """

    def __init__(
        self,
        model: str = NIM_EMBEDDING_MODEL_DEFAULT,
        api_key: str | None = None,
        base_url: str = NIM_BASE_URL,
        dimensions: int | None = None,
        timeout: float = NIM_EMBED_TIMEOUT,
        circuit_breaker: "CircuitBreaker | None" = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        # Resolve dimensionality: explicit arg > NIM_EMBEDDING_DIM env > native.
        self.dimensions = dimensions or int(
            os.getenv("NIM_EMBEDDING_DIM", str(NIM_VECTOR_SIZE))
        )
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker
        self._client = None

    @property
    def client(self):
        """Lazily construct the httpx client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _build_payload(
        self,
        texts: list[str],
        input_type: str,
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
        }
        # Only send `dimensions` when it differs from the model's native size
        # (sending the native size explicitly is harmless but unnecessary).
        if self.dimensions != NIM_VECTOR_SIZE:
            payload["dimensions"] = self.dimensions
        return payload

    def _do_request(self, texts: list[str], input_type: str) -> list[list[float]]:
        """POST to NIM ``/embeddings`` and return a list of float vectors.

        Raises ``NIMEmbeddingError`` for any failure (429, 404, connection
        error, timeout, missing/empty data).
        """
        import httpx

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = self._build_payload(texts, input_type)
        try:
            resp = self.client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                raise NIMEmbeddingError(
                    f"NIM embedding rate limited (429) for model {self.model}",
                )
            if resp.status_code == 404:
                raise NIMEmbeddingError(
                    f"NIM embedding model not found (404): {self.model}",
                )
            if resp.status_code >= 400:
                body = resp.text[:200]
                raise NIMEmbeddingError(
                    f"NIM embedding HTTP {resp.status_code}: {body}",
                )
            data = resp.json()
            items = data.get("data", [])
            if not items or len(items) != len(texts):
                raise NIMEmbeddingError(
                    f"NIM embedding returned {len(items)} vectors for "
                    f"{len(texts)} inputs",
                )
            vectors = [list(item["embedding"]) for item in items]
            # Sanity-check dimensionality on the first vector.
            if vectors and len(vectors[0]) != self.dimensions:
                raise NIMEmbeddingError(
                    f"NIM embedding dimension mismatch: expected "
                    f"{self.dimensions}, got {len(vectors[0])} from model "
                    f"{self.model}",
                )
            return vectors
        except httpx.TimeoutException as exc:
            raise NIMEmbeddingError(f"NIM embedding timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise NIMEmbeddingError(
                f"NIM embedding connection error: {exc}",
            ) from exc
        except NIMEmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001 — map unknown errors
            raise NIMEmbeddingError(f"NIM embedding unexpected error: {exc}") from exc

    def embed_texts(
        self,
        texts: list[str],
        input_type: str = INPUT_TYPE_PASSAGE,
    ) -> list[list[float]]:
        """Embed a batch of texts → list of float vectors (NIM dimensionality).

        Args:
            texts: strings to embed (preserved order in the response).
            input_type: ``passage`` (index time) or ``query`` (query time).
                The nemotron model is tuned for asymmetric bi-encoder
                retrieval, so the right value matters for recall quality.
        """
        if not texts:
            return []
        if self.circuit_breaker is not None:
            return self.circuit_breaker.call(
                self._do_request, texts, input_type,
            )
        return self._do_request(texts, input_type)

    def embed_text(
        self,
        text: str,
        input_type: str = INPUT_TYPE_QUERY,
    ) -> list[float]:
        """Embed a single text → one float vector.

        Defaults to ``input_type=query`` (the retriever is the single-text
        caller); ingestion uses ``embed_texts`` with ``passage``.
        """
        return self.embed_texts([text], input_type=input_type)[0]

    def close(self) -> None:
        """Release the httpx client if one was constructed."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None


# ── factory used by the NIM ingestion path ─────────────────────────────


def build_nim_embedder(circuit_breaker: "CircuitBreaker | None" = None) -> NIMEmbedder:
    """Build a ``NIMEmbedder`` from env vars.

    Reads ``NVIDIA_API_KEY``, ``NIM_EMBEDDING_MODEL`` (override the default
    nemotron model), and ``NIM_EMBEDDING_DIM`` (override the dimensionality).
    The caller supplies a dedicated circuit breaker (the ingestion path owns
    it so it is not shared with the live app's breakers).
    """
    model = os.getenv("NIM_EMBEDDING_MODEL", NIM_EMBEDDING_MODEL_DEFAULT)
    return NIMEmbedder(model=model, circuit_breaker=circuit_breaker)


def build_nim_retriever(
    client=None,
    embedder: NIMEmbedder | None = None,
    collection_config: "CollectionConfig | None" = None,
):
    """Build a ``HybridRetriever`` pointed at the NIM comparison collection.

    Used for recall@5 comparison against the default Ollama collection. The
    retriever is the existing ``HybridRetriever`` — no subclass needed because
    it already accepts ``collection_name`` as a constructor param. The dense
    query embedding comes from the NIM embedder (``input_type=query``); the
    sparse query vector is the local hashing-trick BM25 (unchanged).

    Not wired into ``api/deps.py`` — the live app keeps using the Ollama
    retriever. This factory is for the comparison/eval script only.
    """
    from rag.qdrant_collection import get_qdrant_client
    from rag.retriever import HybridRetriever

    cfg = collection_config or build_nim_collection_config()
    qdrant = client or get_qdrant_client()
    nim_emb = embedder or build_nim_embedder()
    return HybridRetriever(
        client=qdrant,
        embedder=nim_emb,
        collection_name=cfg.collection_name,
        vector_name=cfg.vector_name,
        sparse_vector_name=cfg.sparse_vector_name,
        hnsw_ef=cfg.hnsw_ef,
    )


# ── NIM ingestion orchestrator ─────────────────────────────────────────


def run_nim_ingestion(
    corpus_dir,
    *,
    full_reindex: bool = False,
    manifest_path=None,
    embedder: NIMEmbedder | None = None,
    circuit_breaker: "CircuitBreaker | None" = None,
):
    """Run ingestion populating the separate NIM comparison collection.

    Thin wrapper over ``ingestion.run.run_ingestion`` that injects a
    ``NIMEmbedder`` (dense) + an ``IndexWriter`` pointed at ``ctc_rag_nim``,
    and ensures the NIM collection exists at the configured dimensionality.
    The default ``docs-knowledge`` collection is never touched.

    Requires ``NIM_ENABLED=true`` — refuses to run otherwise (the NIM
    ingestion path is opt-in, like the generator/guardrail paths).

    Returns the same summary dict as ``run_ingestion``.
    """
    from ingestion.index_writer import IndexWriter
    from ingestion.run import run_ingestion

    nim_enabled = os.getenv("NIM_ENABLED", "").strip().lower() in ("true", "1", "yes")
    if not nim_enabled:
        raise RuntimeError(
            "run_nim_ingestion requires NIM_ENABLED=true — the NIM ingestion "
            "path is opt-in and never runs by default."
        )

    client, cfg = ensure_nim_collection()
    emb = embedder or build_nim_embedder(circuit_breaker=circuit_breaker)
    writer = IndexWriter(client, collection_name=cfg.collection_name)
    logger.info(
        "NIM ingestion into collection %r (model=%s, dim=%d)",
        cfg.collection_name,
        emb.model,
        emb.dimensions,
    )
    return run_ingestion(
        corpus_dir,
        full_reindex=full_reindex,
        manifest_path=manifest_path,
        embedder=emb,
        index_writer=writer,
        collection_config=cfg,
    )
