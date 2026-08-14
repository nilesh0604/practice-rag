"""Tests for the hybrid retriever (rag/retriever.py).

Qdrant and Ollama are mocked. The RRF fusion logic is exercised directly
against fake ScoredPoint objects so the math is verified without a live
Qdrant instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion.embedder import EMBEDDING_DIM
from rag.retriever import (
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    HybridRetriever,
    MAX_RRF_SCORE,
    PREFETCH_LIMIT,
)


def _make_point(pid: str, score: float, title: str, content: str, section=None):
    """Build a fake ScoredPoint-like object with the fields the retriever reads."""
    point = MagicMock()
    point.id = pid
    point.score = score
    point.payload = {
        "content": content,
        "title": title,
        "source_url": f"https://example.com/{pid}",
        "section": section,
        "chunk_index": 0,
        "parent_doc_id": f"doc-{pid}",
    }
    return point


def _make_query_response(points):
    resp = MagicMock()
    resp.points = points
    return resp


def _make_embedder(dim=EMBEDDING_DIM):
    """Embedder mock that returns deterministic vectors."""
    emb = MagicMock()
    emb.embed_text.return_value = [0.01] * dim
    return emb


class TestRetrieverRetrieve:
    def test_returns_top_k_results(self):
        client = MagicMock()
        dense_pts = [_make_point(f"d{i}", 0.9 - i * 0.1, f"Dense {i}", f"content {i}") for i in range(8)]
        sparse_pts = [_make_point(f"s{i}", 10 - i, f"Sparse {i}", f"content s{i}") for i in range(8)]
        client.query_points.side_effect = [
            _make_query_response(dense_pts),
            _make_query_response(sparse_pts),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("how to use FastAPI")
        assert len(results) == 5
        assert all(r.score <= 1.0 for r in results)
        assert all(r.score >= 0.0 for r in results)

    def test_results_sorted_by_score_desc(self):
        client = MagicMock()
        dense_pts = [_make_point("a", 0.9, "A", "aaa"), _make_point("b", 0.5, "B", "bbb")]
        sparse_pts = [_make_point("c", 8.0, "C", "ccc"), _make_point("a", 5.0, "A", "aaa")]
        client.query_points.side_effect = [
            _make_query_response(dense_pts),
            _make_query_response(sparse_pts),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=10)
        results = retriever.retrieve("query")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_doc_in_both_lists_ranks_higher(self):
        """A doc that appears in both dense and sparse should outscore one in only one."""
        client = MagicMock()
        # "shared" is rank 1 in dense, rank 2 in sparse → RRF = 1/(k+1) + 1/(k+2)
        # "dense_only" is rank 2 in dense only → RRF = 1/(k+2)
        dense_pts = [_make_point("shared", 0.9, "Shared", "x"), _make_point("dense_only", 0.8, "DOnly", "y")]
        sparse_pts = [_make_point("other", 10, "Other", "z"), _make_point("shared", 5, "Shared", "x")]
        client.query_points.side_effect = [
            _make_query_response(dense_pts),
            _make_query_response(sparse_pts),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=10)
        results = retriever.retrieve("query")
        by_id = {r.id: r for r in results}
        assert by_id["shared"].score > by_id["dense_only"].score

    def test_retrieved_doc_fields_mapped(self):
        client = MagicMock()
        pt = _make_point("abc", 0.88, "Path Parameters - FastAPI", "path params content", section="tutorial")
        client.query_points.side_effect = [
            _make_query_response([pt]),
            _make_query_response([]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("path params")
        assert len(results) == 1
        doc = results[0]
        assert doc.id == "abc"
        assert doc.title == "Path Parameters - FastAPI"
        assert doc.content == "path params content"
        assert doc.section == "tutorial"
        assert str(doc.source_url) == "https://example.com/abc"
        assert doc.chunk_index == 0
        assert doc.parent_doc_id == "doc-abc"

    def test_empty_results_from_both(self):
        client = MagicMock()
        client.query_points.side_effect = [
            _make_query_response([]),
            _make_query_response([]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("nothing")
        assert results == []

    def test_dense_only_results(self):
        client = MagicMock()
        dense_pts = [_make_point("d1", 0.9, "D1", "c1"), _make_point("d2", 0.7, "D2", "c2")]
        client.query_points.side_effect = [
            _make_query_response(dense_pts),
            _make_query_response([]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("query")
        assert len(results) == 2
        assert results[0].score > results[1].score  # rank 1 > rank 2

    def test_sparse_only_results(self):
        client = MagicMock()
        sparse_pts = [_make_point("s1", 10, "S1", "c1"), _make_point("s2", 5, "S2", "c2")]
        client.query_points.side_effect = [
            _make_query_response([]),
            _make_query_response(sparse_pts),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("query")
        assert len(results) == 2

    def test_embedder_called_with_query(self):
        client = MagicMock()
        client.query_points.side_effect = [
            _make_query_response([]),
            _make_query_response([]),
        ]
        emb = _make_embedder()
        retriever = HybridRetriever(client, emb, top_k=5)
        retriever.retrieve("my search query")
        emb.embed_text.assert_called_once_with("my search query")

    def test_query_points_called_twice(self):
        client = MagicMock()
        client.query_points.side_effect = [
            _make_query_response([]),
            _make_query_response([]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        retriever.retrieve("query")
        assert client.query_points.call_count == 2

    def test_prefetch_limit_passed_to_qdrant(self):
        client = MagicMock()
        client.query_points.side_effect = [
            _make_query_response([]),
            _make_query_response([]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5, prefetch_limit=15)
        retriever.retrieve("query")
        for call in client.query_points.call_args_list:
            assert call.kwargs["limit"] == 15

    def test_section_none_handled(self):
        client = MagicMock()
        pt = _make_point("x", 0.9, "Title", "content", section=None)
        client.query_points.side_effect = [
            _make_query_response([pt]),
            _make_query_response([]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("query")
        assert results[0].section is None


class TestRRFFusion:
    def test_rank1_in_both_normalizes_to_one(self):
        """A doc at rank 1 in both lists should normalize to 1.0."""
        client = MagicMock()
        pt = _make_point("top", 0.99, "Top", "best")
        client.query_points.side_effect = [
            _make_query_response([pt]),
            _make_query_response([pt]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("query")
        assert results[0].score == pytest.approx(1.0)

    def test_rank1_in_one_only(self):
        """A doc at rank 1 in only one list → score = (1/(k+1)) / max."""
        client = MagicMock()
        pt = _make_point("solo", 0.9, "Solo", "content")
        client.query_points.side_effect = [
            _make_query_response([pt]),
            _make_query_response([]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("query")
        expected = (1.0 / (DEFAULT_RRF_K + 1)) / MAX_RRF_SCORE
        assert results[0].score == pytest.approx(expected)

    def test_custom_rrf_k(self):
        client = MagicMock()
        pt = _make_point("x", 0.9, "X", "c")
        client.query_points.side_effect = [
            _make_query_response([pt]),
            _make_query_response([pt]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5, rrf_k=30)
        results = retriever.retrieve("query")
        # rank 1 in both with k=30 → raw = 2/31, max = 2/31 → normalized = 1.0
        assert results[0].score == pytest.approx(1.0)

    def test_dedup_by_id(self):
        """Same doc id in both lists should produce one result, not two."""
        client = MagicMock()
        pt = _make_point("dup", 0.9, "Dup", "content")
        client.query_points.side_effect = [
            _make_query_response([pt]),
            _make_query_response([pt]),
        ]
        retriever = HybridRetriever(client, _make_embedder(), top_k=5)
        results = retriever.retrieve("query")
        assert len(results) == 1


class TestRetrieverDefaults:
    def test_default_top_k(self):
        assert DEFAULT_TOP_K == 5

    def test_default_rrf_k(self):
        assert DEFAULT_RRF_K == 60

    def test_prefetch_limit_greater_than_top_k(self):
        assert PREFETCH_LIMIT > DEFAULT_TOP_K

    def test_max_rrf_score_positive(self):
        assert MAX_RRF_SCORE > 0
