"""Tests for the index writer (ingestion/index_writer.py).

Uses a fully mocked QdrantClient — no live Qdrant required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import PointStruct, SparseVector

from ingestion.embedder import sparse_embed_text
from ingestion.index_writer import IndexWriter
from rag.qdrant_collection import COLLECTION_NAME, SPARSE_VECTOR_NAME, VECTOR_NAME
from schemas.documents import DocumentChunk

_NO_EMBEDDING = object()  # sentinel to distinguish "not provided" from "explicitly None"


def _make_chunk(
    chunk_id: str = "abc123",
    content: str = "FastAPI path parameters",
    parent_doc_id: str = "fastapi-path-params",
    embedding: object = _NO_EMBEDDING,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        content=content,
        title="Path Parameters",
        source_url="https://fastapi.tiangolo.com/tutorial/path-params/",
        section="tutorial",
        last_modified=datetime(2026, 1, 1, tzinfo=timezone.utc),
        chunk_index=0,
        parent_doc_id=parent_doc_id,
        embedding=[0.1] * 768 if embedding is _NO_EMBEDDING else embedding,  # type: ignore[arg-type]
    )


class TestUpsertChunks:
    def test_upsert_single_chunk(self):
        client = MagicMock()
        writer = IndexWriter(client)
        chunk = _make_chunk()
        count = writer.upsert_chunks([chunk])
        assert count == 1
        client.upsert.assert_called_once()
        call = client.upsert.call_args
        assert call.kwargs["collection_name"] == COLLECTION_NAME
        points = call.kwargs["points"]
        assert len(points) == 1
        point: PointStruct = points[0]
        assert point.id == "abc123"
        assert VECTOR_NAME in point.vector
        assert SPARSE_VECTOR_NAME in point.vector
        assert len(point.vector[VECTOR_NAME]) == 768
        assert isinstance(point.vector[SPARSE_VECTOR_NAME], SparseVector)

    def test_upsert_empty_list(self):
        client = MagicMock()
        writer = IndexWriter(client)
        assert writer.upsert_chunks([]) == 0
        client.upsert.assert_not_called()

    def test_upsert_batching(self):
        client = MagicMock()
        writer = IndexWriter(client, batch_size=2)
        chunks = [_make_chunk(chunk_id=f"id-{i}", content=f"content {i}") for i in range(5)]
        count = writer.upsert_chunks(chunks)
        assert count == 5
        # 5 chunks / batch_size 2 → 3 upsert calls (2, 2, 1)
        assert client.upsert.call_count == 3

    def test_payload_excludes_id_and_embedding(self):
        client = MagicMock()
        writer = IndexWriter(client)
        chunk = _make_chunk()
        writer.upsert_chunks([chunk])
        point = client.upsert.call_args.kwargs["points"][0]
        assert "id" not in point.payload
        assert "embedding" not in point.payload
        assert "content" in point.payload
        assert "title" in point.payload
        assert "parent_doc_id" in point.payload

    def test_raises_on_missing_embedding(self):
        client = MagicMock()
        writer = IndexWriter(client)
        chunk = _make_chunk(embedding=None)
        with pytest.raises(ValueError, match="no dense embedding"):
            writer.upsert_chunks([chunk])

    def test_sparse_vector_generated_from_content(self):
        client = MagicMock()
        writer = IndexWriter(client)
        chunk = _make_chunk(content="FastAPI path parameters declaration")
        writer.upsert_chunks([chunk])
        point = client.upsert.call_args.kwargs["points"][0]
        sparse: SparseVector = point.vector[SPARSE_VECTOR_NAME]
        expected = sparse_embed_text("FastAPI path parameters declaration")
        assert sparse.indices == expected.indices

    def test_custom_collection_name(self):
        client = MagicMock()
        writer = IndexWriter(client, collection_name="custom-collection")
        chunk = _make_chunk()
        writer.upsert_chunks([chunk])
        assert client.upsert.call_args.kwargs["collection_name"] == "custom-collection"


class TestDeleteByParent:
    def test_delete_calls_client_delete(self):
        client = MagicMock()
        writer = IndexWriter(client)
        writer.delete_by_parent("fastapi-path-params")
        client.delete.assert_called_once()
        call = client.delete.call_args
        assert call.kwargs["collection_name"] == COLLECTION_NAME
        flt = call.kwargs["points_selector"]
        # The filter should match parent_doc_id
        assert flt.must[0].key == "parent_doc_id"
        assert flt.must[0].match.value == "fastapi-path-params"

    def test_delete_handles_404(self):
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        exc = UnexpectedResponse(status_code=404, reason_phrase="Not Found", content=b"{}", headers=None)
        client.delete.side_effect = exc
        writer = IndexWriter(client)
        result = writer.delete_by_parent("missing-doc")
        assert result == 0  # no error

    def test_delete_reraises_non_404(self):
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        exc = UnexpectedResponse(status_code=500, reason_phrase="Server Error", content=b"{}", headers=None)
        client.delete.side_effect = exc
        writer = IndexWriter(client)
        with pytest.raises(UnexpectedResponse):
            writer.delete_by_parent("some-doc")
