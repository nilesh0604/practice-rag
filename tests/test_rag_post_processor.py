"""Tests for the post-processor (rag/post_processor.py).

Citation extraction is tested as pure functions (no embedder). Groundedness
scoring uses a mocked embedder returning deterministic vectors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.post_processor import (
    CONFIDENCE_THRESHOLD,
    PostProcessor,
    PostProcessResult,
    _cosine_similarity,
    compute_groundedness,
    extract_citations,
)
from schemas.documents import RetrievedDoc


def _make_doc(title="Doc", content="content", url="https://example.com/d"):
    return RetrievedDoc(
        id="x",
        content=content,
        title=title,
        source_url=url,
        section=None,
        score=0.9,
    )


def _mock_embedder(answer_vec, doc_vecs):
    """Embedder mock: embed_text → answer_vec, embed_texts → doc_vecs."""
    emb = MagicMock()
    emb.embed_text.return_value = answer_vec
    emb.embed_texts.return_value = doc_vecs
    return emb


class TestExtractCitations:
    def test_single_citation_matched(self):
        answer = "Use Depends for DI. [Source: Dependency Injection - FastAPI]"
        docs = [_make_doc(title="Dependency Injection - FastAPI", content="di content")]
        cites = extract_citations(answer, docs)
        assert len(cites) == 1
        assert cites[0].title == "Dependency Injection - FastAPI"

    def test_multiple_citations(self):
        answer = "See [Source: Query Parameters - FastAPI] and [Source: Pydantic Models]."
        docs = [
            _make_doc(title="Query Parameters - FastAPI", content="q", url="https://a.com"),
            _make_doc(title="Pydantic Models", content="p", url="https://b.com"),
        ]
        cites = extract_citations(answer, docs)
        assert len(cites) == 2

    def test_dedup_by_url(self):
        answer = "[Source: FastAPI Docs] and again [Source: FastAPI Docs]"
        docs = [_make_doc(title="FastAPI Docs", content="x")]
        cites = extract_citations(answer, docs)
        assert len(cites) == 1

    def test_unmatched_title_dropped(self):
        answer = "See [Source: Nonexistent Page]."
        docs = [_make_doc(title="Real Page", content="x")]
        cites = extract_citations(answer, docs)
        assert cites == []

    def test_case_insensitive_match(self):
        answer = "[Source: fastapi docs]"
        docs = [_make_doc(title="FastAPI Docs", content="x")]
        cites = extract_citations(answer, docs)
        assert len(cites) == 1

    def test_empty_answer(self):
        assert extract_citations("", [_make_doc()]) == []

    def test_no_docs(self):
        assert extract_citations("[Source: X]", []) == []

    def test_no_citations_in_answer(self):
        answer = "Just a plain answer with no source markers."
        docs = [_make_doc(title="Doc", content="x")]
        assert extract_citations(answer, docs) == []

    def test_preserves_first_seen_order(self):
        answer = "[Source: B] then [Source: A]"
        docs = [
            _make_doc(title="A", content="a", url="https://a"),
            _make_doc(title="B", content="b", url="https://b"),
        ]
        cites = extract_citations(answer, docs)
        assert len(cites) == 2
        assert cites[0].title == "B"
        assert cites[1].title == "A"

    def test_citation_has_source_url(self):
        answer = "[Source: FastAPI Docs]"
        docs = [_make_doc(title="FastAPI Docs", content="x", url="https://fastapi.tiangolo.com")]
        cites = extract_citations(answer, docs)
        assert str(cites[0].source_url) == "https://fastapi.tiangolo.com/"


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestComputeGroundedness:
    def test_perfect_match(self):
        emb = _mock_embedder([1.0, 0.0], [[1.0, 0.0]])
        score = compute_groundedness("answer", [_make_doc()], emb)
        assert score == pytest.approx(1.0)

    def test_no_match(self):
        emb = _mock_embedder([1.0, 0.0], [[0.0, 1.0]])
        score = compute_groundedness("answer", [_make_doc()], emb)
        assert score == pytest.approx(0.0)

    def test_takes_max_over_docs(self):
        emb = _mock_embedder([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]])
        score = compute_groundedness("answer", [_make_doc(), _make_doc()], emb)
        assert score == pytest.approx(1.0)

    def test_empty_answer(self):
        emb = _mock_embedder([1.0], [[1.0]])
        assert compute_groundedness("", [_make_doc()], emb) == 0.0

    def test_no_docs(self):
        emb = _mock_embedder([1.0], [])
        assert compute_groundedness("answer", [], emb) == 0.0

    def test_clamped_to_one(self):
        emb = _mock_embedder([1.0, 0.0], [[1.0, 0.0]])
        score = compute_groundedness("answer", [_make_doc()], emb)
        assert score <= 1.0

    def test_embedder_called_with_answer_and_doc_contents(self):
        emb = _mock_embedder([1.0], [[1.0]])
        docs = [_make_doc(content="doc content 1"), _make_doc(content="doc content 2")]
        compute_groundedness("the answer", docs, emb)
        emb.embed_text.assert_called_once_with("the answer")
        emb.embed_texts.assert_called_once_with(["doc content 1", "doc content 2"])


class TestPostProcessor:
    def test_post_process_returns_result(self):
        emb = _mock_embedder([1.0, 0.0], [[1.0, 0.0]])
        pp = PostProcessor(emb)
        answer = "Use Depends. [Source: DI - FastAPI]"
        docs = [_make_doc(title="DI - FastAPI", content="di")]
        result = pp.post_process(answer, docs)
        assert isinstance(result, PostProcessResult)
        assert result.answer == answer
        assert len(result.citations) == 1
        assert result.confidence == pytest.approx(1.0)

    def test_low_confidence_flag(self):
        emb = _mock_embedder([1.0, 0.0], [[0.0, 1.0]])  # score 0.0
        pp = PostProcessor(emb)
        result = pp.post_process("answer", [_make_doc()])
        assert result.low_confidence is True
        assert result.confidence < CONFIDENCE_THRESHOLD

    def test_high_confidence_no_flag(self):
        emb = _mock_embedder([1.0, 0.0], [[1.0, 0.0]])  # score 1.0
        pp = PostProcessor(emb)
        result = pp.post_process("answer", [_make_doc()])
        assert result.low_confidence is False

    def test_confidence_threshold_is_065(self):
        assert CONFIDENCE_THRESHOLD == 0.65

    def test_empty_answer_zero_confidence(self):
        emb = _mock_embedder([1.0], [[1.0]])
        pp = PostProcessor(emb)
        result = pp.post_process("", [_make_doc()])
        assert result.confidence == 0.0
        assert result.low_confidence is True
