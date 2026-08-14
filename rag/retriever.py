"""Hybrid retriever — Qdrant dense + sparse search fused with Reciprocal Rank Fusion.

Implements the three-stage retrieval from the architecture doc's Component
Deep-Dive > Qdrant section:

    Stage 1 — Dense vector search   (semantic meaning, nomic-embed-text)
    Stage 2 — Sparse vector search  (exact keywords / API names, BM25+IDF)
    Stage 3 — RRF fusion            (merge both rankings into one score)

Two separate ``query_points`` calls are issued (one dense, one sparse) and the
results are fused client-side with Reciprocal Rank Fusion. Doing the fusion in
Python (rather than Qdrant's native ``RrfQuery``) keeps the RRF logic
unit-testable without a live Qdrant instance and makes the fusion constant
explicit.

RRF formula (per the standard Cormack et al. 2009 definition)::

    score(d) = Σ  1 / (k + rank_L(d))
               L

where ``rank`` is 1-based and ``k`` is a smoothing constant (default 60). A
document absent from a list contributes 0 from that list. The raw RRF score is
normalized to ``[0, 1]`` by dividing by the theoretical maximum
``2 / (k + 1)`` (rank 1 in both lists) so it fits ``RetrievedDoc.score``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

from ingestion.embedder import Embedder, sparse_embed_text
from rag.qdrant_collection import (
    COLLECTION_NAME,
    HNSW_EF,
    SPARSE_VECTOR_NAME,
    VECTOR_NAME,
)
from schemas.documents import RetrievedDoc

if TYPE_CHECKING:
    from qdrant_client.http.models.models import ScoredPoint

logger = logging.getLogger(__name__)

# ── retrieval constants ────────────────────────────────────────────────

DEFAULT_TOP_K: int = 5
"""Number of fused results to return. Matches the architecture doc's top-K=5."""

DEFAULT_RRF_K: int = 60
"""RRF smoothing constant. 60 is the value from the original RRF paper."""

PREFETCH_LIMIT: int = 20
"""How many candidates to fetch from each branch (dense / sparse) before fusion.
Larger than ``top_k`` so RRF has enough overlap signal to reorder meaningfully."""

MAX_RRF_SCORE: float = 2.0 / (DEFAULT_RRF_K + 1)
"""Theoretical max RRF score (rank 1 in both lists). Used to normalize to [0,1]."""


class HybridRetriever:
    """Qdrant hybrid (dense + sparse) retriever with RRF fusion.

    The dense query embedding is produced by the injected ``Embedder`` (reusing
    the Step 2 wrapper). The sparse query vector is generated client-side via
    the shared ``sparse_embed_text`` function (same tokenizer as ingestion).
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection_name: str = COLLECTION_NAME,
        vector_name: str = VECTOR_NAME,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        prefetch_limit: int = PREFETCH_LIMIT,
        hnsw_ef: int = HNSW_EF,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.sparse_vector_name = sparse_vector_name
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.prefetch_limit = prefetch_limit
        self.hnsw_ef = hnsw_ef

    # ── public API ─────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """Run hybrid retrieval for a query string → fused top-K RetrievedDocs.

        Embeds the query (dense + sparse), queries both Qdrant indexes, fuses
        with RRF, and returns the top-K results sorted by fused score.
        """
        dense_vec = self.embedder.embed_text(query)
        sparse_vec = sparse_embed_text(query)

        dense_hits = self._search_dense(dense_vec)
        sparse_hits = self._search_sparse(sparse_vec)

        fused = self._rrf_fuse(dense_hits, sparse_hits)
        logger.info(
            "Hybrid retrieve %r → %d dense, %d sparse, %d fused (top %d returned)",
            query,
            len(dense_hits),
            len(sparse_hits),
            len(fused),
            min(self.top_k, len(fused)),
        )
        return fused[: self.top_k]

    # ── Qdrant queries ─────────────────────────────────────────────────

    def _search_dense(self, dense_vec: list[float]) -> list["ScoredPoint"]:
        """Dense vector search via Qdrant ``query_points``."""
        from qdrant_client.http.models import SearchParams

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using=self.vector_name,
            limit=self.prefetch_limit,
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(hnsw_ef=self.hnsw_ef),
        )
        return list(response.points)

    def _search_sparse(self, sparse_vec) -> list["ScoredPoint"]:
        """Sparse vector search via Qdrant ``query_points``."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vec,
            using=self.sparse_vector_name,
            limit=self.prefetch_limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)

    # ── RRF fusion ─────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense_hits: list["ScoredPoint"],
        sparse_hits: list["ScoredPoint"],
    ) -> list[RetrievedDoc]:
        """Fuse two ranked lists with Reciprocal Rank Fusion.

        Returns ``RetrievedDoc`` objects sorted by fused score (descending),
        normalized to ``[0, 1]``.
        """
        rrf_scores: dict[str, float] = {}
        points_by_id: dict[str, "ScoredPoint"] = {}

        for hits in (dense_hits, sparse_hits):
            for rank, point in enumerate(hits, start=1):
                pid = str(point.id)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )
                # Keep the first-seen point (dense branch preferred for payload).
                if pid not in points_by_id:
                    points_by_id[pid] = point

        max_score = 2.0 / (self.rrf_k + 1)
        results: list[RetrievedDoc] = []
        for pid, raw_score in rrf_scores.items():
            point = points_by_id[pid]
            normalized = raw_score / max_score if max_score > 0 else 0.0
            normalized = min(max(normalized, 0.0), 1.0)
            results.append(self._to_retrieved_doc(point, normalized))

        results.sort(key=lambda d: d.score, reverse=True)
        return results

    # ── mapping ────────────────────────────────────────────────────────

    @staticmethod
    def _to_retrieved_doc(point: "ScoredPoint", score: float) -> RetrievedDoc:
        """Map a Qdrant ScoredPoint + fused score to a RetrievedDoc."""
        payload = point.payload or {}
        return RetrievedDoc(
            id=str(point.id),
            content=payload.get("content", ""),
            title=payload.get("title", ""),
            source_url=payload.get("source_url", ""),
            section=payload.get("section"),
            score=score,
            chunk_index=payload.get("chunk_index"),
            parent_doc_id=payload.get("parent_doc_id"),
        )
