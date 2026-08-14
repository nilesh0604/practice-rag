"""Tests for the Qdrant collection helper in `rag/qdrant_collection.py`.

Uses a fully mocked QdrantClient — no live Qdrant required. Verifies the
collection is created with the documented schema (768-d cosine, sparse
"text" vector, HNSW m=16/ef_construct=128/ef=64) and that creation is
idempotent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from rag.qdrant_collection import (
    COLLECTION_NAME,
    SPARSE_VECTOR_NAME,
    VECTOR_NAME,
    VECTOR_SIZE,
    CollectionConfig,
    create_collection,
    get_qdrant_client,
)


def _make_client(*, existing_collections: list[str] | None = None) -> MagicMock:
    """Build a MagicMock that quacks like a QdrantClient for collection ops."""
    client = MagicMock(name="QdrantClient")
    existing = existing_collections or []

    collection_objs = []
    for c in existing:
        m = MagicMock()
        m.name = c
        collection_objs.append(m)
    collections_response = MagicMock()
    collections_response.collections = collection_objs
    client.get_collections.return_value = collections_response
    client.create_collection.return_value = MagicMock()
    client.delete_collection.return_value = MagicMock()
    return client


class TestCollectionConfig:
    def test_defaults_match_architecture_doc(self):
        cfg = CollectionConfig()
        assert cfg.collection_name == "docs-knowledge"
        assert cfg.vector_name == "dense"
        assert cfg.vector_size == 768
        assert cfg.distance == Distance.COSINE
        assert cfg.sparse_vector_name == "text"
        assert cfg.hnsw_m == 16
        assert cfg.hnsw_ef_construct == 128
        assert cfg.hnsw_ef == 64


class TestCreateCollection:
    def test_creates_when_missing(self):
        client = _make_client(existing_collections=[])
        created = create_collection(client)
        assert created is True
        client.create_collection.assert_called_once()
        client.delete_collection.assert_not_called()

        call = client.create_collection.call_args
        assert call.kwargs["collection_name"] == COLLECTION_NAME

        vectors = call.kwargs["vectors_config"]
        assert VECTOR_NAME in vectors
        vp: VectorParams = vectors[VECTOR_NAME]
        assert vp.size == VECTOR_SIZE
        assert vp.distance == Distance.COSINE

        sparse = call.kwargs["sparse_vectors_config"]
        assert SPARSE_VECTOR_NAME in sparse
        assert isinstance(sparse[SPARSE_VECTOR_NAME], SparseVectorParams)
        assert isinstance(sparse[SPARSE_VECTOR_NAME].index, SparseIndexParams)
        assert sparse[SPARSE_VECTOR_NAME].index.on_disk is False

        hnsw: HnswConfigDiff = call.kwargs["hnsw_config"]
        assert hnsw.m == 16
        assert hnsw.ef_construct == 128
        # `ef` is a search-time param, not part of HnswConfigDiff (build-time).
        assert not hasattr(hnsw, "ef") or hnsw.ef is None

    def test_skips_when_exists(self):
        client = _make_client(existing_collections=[COLLECTION_NAME])
        created = create_collection(client)
        assert created is False
        client.create_collection.assert_not_called()
        client.delete_collection.assert_not_called()

    def test_recreate_drops_then_creates(self):
        client = _make_client(existing_collections=[COLLECTION_NAME])
        created = create_collection(client, recreate=True)
        assert created is True
        client.delete_collection.assert_called_once_with(collection_name=COLLECTION_NAME)
        client.create_collection.assert_called_once()

    def test_recreate_when_missing_does_not_error(self):
        client = _make_client(existing_collections=[])
        # delete_collection raises nothing here; should just proceed to create.
        created = create_collection(client, recreate=True)
        assert created is True
        client.delete_collection.assert_called_once()
        client.create_collection.assert_called_once()


class TestGetQdrantClient:
    def test_env_url_used(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:6333")
        client = get_qdrant_client()
        try:
            # qdrant-client exposes the resolved REST URI on the inner client.
            assert str(client._client.rest_uri).rstrip("/") == "http://qdrant.example:6333"
        finally:
            client.close()

    def test_default_url_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("QDRANT_URL", raising=False)
        client = get_qdrant_client()
        try:
            assert str(client._client.rest_uri).rstrip("/") == "http://localhost:6333"
        finally:
            client.close()
