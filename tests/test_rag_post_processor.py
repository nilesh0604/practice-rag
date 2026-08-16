"""Tests for the post-processor (rag/post_processor.py).

Citation extraction is tested as pure functions (no embedder). Groundedness
scoring uses a mocked embedder returning deterministic vectors.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from rag.post_processor import (
    CONFIDENCE_THRESHOLD,
    SNIPPET_MAX_CHARS,
    PostProcessor,
    PostProcessResult,
    _cosine_similarity,
    _make_snippet,
    compute_groundedness,
    extract_citations,
)
from schemas.documents import RetrievedDoc


def _make_doc(
    title="Doc",
    content="content",
    url="https://example.com/d",
    score=0.9,
    last_modified=None,
):
    return RetrievedDoc(
        id="x",
        content=content,
        title=title,
        source_url=url,
        section=None,
        score=score,
        last_modified=last_modified,
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


class TestCitationEnrichedFields:
    """Verify snippet, relevanceScore, and lastModified are populated."""

    def test_snippet_from_content(self):
        answer = "[Source: FastAPI Docs]"
        docs = [_make_doc(title="FastAPI Docs", content="Use Depends for DI.")]
        cites = extract_citations(answer, docs)
        assert cites[0].snippet == "Use Depends for DI."

    def test_snippet_truncated_to_max(self):
        long = "word " * (SNIPPET_MAX_CHARS // 2)
        answer = "[Source: Big Doc]"
        docs = [_make_doc(title="Big Doc", content=long)]
        cites = extract_citations(answer, docs)
        assert cites[0].snippet is not None
        assert len(cites[0].snippet) <= SNIPPET_MAX_CHARS + 1  # +1 for ellipsis
        assert cites[0].snippet.endswith("…")

    def test_snippet_collapses_whitespace(self):
        answer = "[Source: Spaced Doc]"
        docs = [_make_doc(title="Spaced Doc", content="  hello\n\n  world  ")]
        cites = extract_citations(answer, docs)
        assert cites[0].snippet == "hello world"

    def test_relevance_score_populated(self):
        answer = "[Source: FastAPI Docs]"
        docs = [_make_doc(title="FastAPI Docs", content="x", score=0.75)]
        cites = extract_citations(answer, docs)
        assert cites[0].relevanceScore == pytest.approx(0.75)

    def test_last_modified_populated(self):
        ts = datetime(2026, 1, 15, tzinfo=timezone.utc)
        answer = "[Source: FastAPI Docs]"
        docs = [_make_doc(title="FastAPI Docs", content="x", last_modified=ts)]
        cites = extract_citations(answer, docs)
        assert cites[0].lastModified == ts

    def test_last_modified_none_when_absent(self):
        answer = "[Source: FastAPI Docs]"
        docs = [_make_doc(title="FastAPI Docs", content="x", last_modified=None)]
        cites = extract_citations(answer, docs)
        assert cites[0].lastModified is None

    def test_all_three_fields_together(self):
        ts = datetime(2026, 3, 1, tzinfo=timezone.utc)
        answer = "[Source: Rich Doc]"
        docs = [_make_doc(
            title="Rich Doc",
            content="Some supporting text here.",
            score=0.88,
            last_modified=ts,
        )]
        cites = extract_citations(answer, docs)
        c = cites[0]
        assert c.snippet == "Some supporting text here."
        assert c.relevanceScore == pytest.approx(0.88)
        assert c.lastModified == ts


class TestMakeSnippet:
    def test_short_content_unchanged(self):
        assert _make_snippet("hello world") == "hello world"

    def test_long_content_truncated_with_ellipsis(self):
        text = " ".join(["word"] * 300)
        result = _make_snippet(text)
        assert len(result) <= SNIPPET_MAX_CHARS + 1
        assert result.endswith("…")

    def test_collapses_whitespace(self):
        assert _make_snippet("  a\n b\t c  ") == "a b c"

    def test_empty_content(self):
        assert _make_snippet("") == ""


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
