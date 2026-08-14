"""Qdrant collection schema + creation helper for the `docs-knowledge` collection.

Implements the collection config documented in the architecture doc's
Component Deep-Dive > Qdrant section:

    collection_name: docs-knowledge
    vectors:          size 768, distance Cosine   (nomic-embed-text)
    sparse_vectors:   "text" (on_disk: false)      (SPLADE / BM25 fallback)
    hnsw_config:      m=16, ef_construct=128, ef=64

The helper is idempotent: by default it creates the collection only if it does
not already exist, so it is safe to call on every ingestion run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

logger = logging.getLogger(__name__)

# ── Collection constants (the single source of truth for the schema) ────────

COLLECTION_NAME: str = "docs-knowledge"
"""Name of the Qdrant collection holding the documentation corpus."""

VECTOR_SIZE: int = 768
"""Dense embedding dimensionality (nomic-embed-text)."""

VECTOR_NAME: str = "dense"
"""Named dense vector field. Qdrant allows multiple named vectors; we use one."""

SPARSE_VECTOR_NAME: str = "text"
"""Named sparse vector field for hybrid (SPLADE / BM25) search."""

SPARSE_MODIFIER: Modifier = Modifier.IDF
"""Sparse vector modifier — Qdrant applies IDF weighting at query time,
producing BM25-style scoring from the raw term-frequency sparse vectors
generated client-side by the ingestion embedder."""

HNSW_M: int = 16
HNSW_EF_CONSTRUCT: int = 128
HNSW_EF: int = 64
"""HNSW graph parameters. Defaults are Qdrant-recommended for small corpora."""

DEFAULT_QDRANT_URL: str = "http://localhost:6333"
"""Default Qdrant REST URL (matches docker-compose port mapping)."""


@dataclass(frozen=True)
class CollectionConfig:
    """Resolved collection configuration. Exposed for tests and introspection."""

    collection_name: str = COLLECTION_NAME
    vector_name: str = VECTOR_NAME
    vector_size: int = VECTOR_SIZE
    distance: Distance = Distance.COSINE
    sparse_vector_name: str = SPARSE_VECTOR_NAME
    sparse_modifier: Modifier = SPARSE_MODIFIER
    hnsw_m: int = HNSW_M
    hnsw_ef_construct: int = HNSW_EF_CONSTRUCT
    hnsw_ef: int = HNSW_EF


def get_qdrant_client(url: str | None = None) -> QdrantClient:
    """Build a QdrantClient from the `QDRANT_URL` env var (or the given url).

    Defaults to `http://localhost:6333`, matching the docker-compose port
    mapping. Kept as a tiny factory so callers (ingestion, retriever, tests)
    share one resolution path and never hardcode the URL.
    """
    resolved = url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    logger.debug("Connecting to Qdrant at %s", resolved)
    return QdrantClient(url=resolved)


def _build_vector_params(config: CollectionConfig) -> dict[str, VectorParams]:
    return {
        config.vector_name: VectorParams(
            size=config.vector_size,
            distance=config.distance,
        ),
    }


def _build_sparse_vectors_config(
    config: CollectionConfig,
) -> dict[str, SparseVectorParams]:
    return {
        config.sparse_vector_name: SparseVectorParams(
            index=SparseIndexParams(on_disk=False),
            modifier=config.sparse_modifier,
        ),
    }


def _build_hnsw_config(config: CollectionConfig) -> HnswConfigDiff:
    # NOTE: `ef` is a *search-time* parameter (set per query in the retriever),
    # not a build-time HnswConfigDiff field, so it is not passed here. The
    # `hnsw_ef` constant on CollectionConfig is consumed by the retriever.
    return HnswConfigDiff(
        m=config.hnsw_m,
        ef_construct=config.hnsw_ef_construct,
    )


def create_collection(
    client: QdrantClient,
    config: CollectionConfig | None = None,
    *,
    recreate: bool = False,
) -> bool:
    """Create the `docs-knowledge` collection if needed.

    Args:
        client: an active QdrantClient.
        config: optional override of the default collection config.
        recreate: if True, drop and recreate the collection even if it exists.
            Destructive — only use for full re-indexes.

    Returns:
        True if the collection was created (or recreated), False if it already
        existed and `recreate` was False.
    """
    cfg = config or CollectionConfig()

    if recreate:
        try:
            client.delete_collection(collection_name=cfg.collection_name)
            logger.info("Deleted existing collection %r (recreate=True)", cfg.collection_name)
        except UnexpectedResponse as exc:
            # 404 means nothing to delete — safe to ignore.
            if exc.status_code != 404:
                raise

    if not recreate and _collection_exists(client, cfg.collection_name):
        logger.info("Collection %r already exists; skipping creation", cfg.collection_name)
        return False

    client.create_collection(
        collection_name=cfg.collection_name,
        vectors_config=_build_vector_params(cfg),
        sparse_vectors_config=_build_sparse_vectors_config(cfg),
        hnsw_config=_build_hnsw_config(cfg),
    )
    logger.info(
        "Created collection %r (dense=%s/%s, sparse=%s, hnsw m=%s ef_construct=%s ef=%s)",
        cfg.collection_name,
        cfg.vector_name,
        cfg.vector_size,
        cfg.sparse_vector_name,
        cfg.hnsw_m,
        cfg.hnsw_ef_construct,
        cfg.hnsw_ef,
    )
    return True


def ensure_collection(
    client: QdrantClient | None = None,
    config: CollectionConfig | None = None,
) -> QdrantClient:
    """Convenience wrapper: get a client (if not given) and create the collection if missing.

    Typical ingestion entrypoint usage::

        from rag.qdrant_collection import ensure_collection
        client = ensure_collection()
    """
    qdrant = client or get_qdrant_client()
    create_collection(qdrant, config, recreate=False)
    return qdrant


def _collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """True if the collection exists. Uses the collection listing to stay compatible across client versions."""
    try:
        names = {c.name for c in client.get_collections().collections}
    except UnexpectedResponse:
        return False
    return collection_name in names
