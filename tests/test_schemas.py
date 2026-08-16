"""Tests for the shared Pydantic contracts in `schemas/`.

These are pure validation/serialization tests — no I/O, no network. They lock
the seam between the ingestion pipeline and the serving layer.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentChunk,
    FeedbackRequest,
    RetrievedDoc,
)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def valid_chunk_kwargs():
    return {
        "id": "fastapi-query-params-chunk-3",
        "content": "Optional query parameters are declared with a default value.",
        "title": "Query Parameters - FastAPI",
        "source_url": "https://fastapi.tiangolo.com/tutorial/query-params/",
        "section": "tutorial",
        "last_modified": "2026-01-15T00:00:00Z",
        "chunk_index": 3,
        "parent_doc_id": "fastapi-query-params",
    }


# ── DocumentChunk ───────────────────────────────────────────────────────────


class TestDocumentChunk:
    def test_valid_chunk_without_embedding(self, valid_chunk_kwargs):
        chunk = DocumentChunk(**valid_chunk_kwargs)
        assert chunk.embedding is None
        assert chunk.chunk_index == 3

    def test_valid_chunk_with_embedding(self, valid_chunk_kwargs):
        chunk = DocumentChunk(**valid_chunk_kwargs, embedding=[0.1] * 768)
        assert len(chunk.embedding) == 768

    def test_naive_last_modified_normalized_to_utc(self):
        chunk = DocumentChunk(
            id="x",
            content="c",
            title="t",
            source_url="https://example.com/x",
            last_modified=datetime(2026, 1, 15, 0, 0, 0),  # naive
            chunk_index=0,
            parent_doc_id="x",
        )
        assert chunk.last_modified.tzinfo is timezone.utc

    def test_aware_last_modified_preserved(self):
        chunk = DocumentChunk(
            id="x",
            content="c",
            title="t",
            source_url="https://example.com/x",
            last_modified=datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            chunk_index=0,
            parent_doc_id="x",
        )
        assert chunk.last_modified.utcoffset() == timezone.utc.utcoffset(None)

    def test_negative_chunk_index_rejected(self, valid_chunk_kwargs):
        valid_chunk_kwargs["chunk_index"] = -1
        with pytest.raises(ValidationError):
            DocumentChunk(**valid_chunk_kwargs)

    def test_empty_content_rejected(self, valid_chunk_kwargs):
        valid_chunk_kwargs["content"] = ""
        with pytest.raises(ValidationError):
            DocumentChunk(**valid_chunk_kwargs)

    def test_extra_field_rejected(self, valid_chunk_kwargs):
        valid_chunk_kwargs["unexpected"] = "nope"
        with pytest.raises(ValidationError):
            DocumentChunk(**valid_chunk_kwargs)

    def test_to_payload_excludes_id_and_embedding(self, valid_chunk_kwargs):
        chunk = DocumentChunk(**valid_chunk_kwargs, embedding=[0.1] * 768)
        payload = chunk.to_payload()
        assert "id" not in payload
        assert "embedding" not in payload
        assert payload["parent_doc_id"] == "fastapi-query-params"
        assert payload["chunk_index"] == 3
        # source_url serializes to a string
        assert payload["source_url"].startswith("https://")


# ── RetrievedDoc ────────────────────────────────────────────────────────────


class TestRetrievedDoc:
    def test_valid_retrieved_doc(self):
        doc = RetrievedDoc(
            id="x",
            content="c",
            title="t",
            source_url="https://example.com/x",
            score=0.87,
        )
        assert doc.section is None
        assert doc.chunk_index is None

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            RetrievedDoc(
                id="x",
                content="c",
                title="t",
                source_url="https://example.com/x",
                score=1.5,
            )


# ── ChatRequest ─────────────────────────────────────────────────────────────


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(message="How do I declare path parameters?", session_id="sess_1")
        assert req.session_id == "sess_1"

    def test_session_id_optional(self):
        req = ChatRequest(message="hi")
        assert req.session_id is None

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_oversized_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 4001)

    def test_department_and_role_optional(self):
        req = ChatRequest(
            message="hi",
            department="engineering",
            role="manager",
        )
        assert req.department == "engineering"
        assert req.role == "manager"

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", unknown="nope")


# ── ChatResponse / Citation ─────────────────────────────────────────────────


class TestChatResponse:
    def test_dedupes_citations_by_url(self):
        resp = ChatResponse(
            session_id="sess_1",
            answer="...",
            citations=[
                Citation(title="A", source_url="https://example.com/a"),
                Citation(title="A again", source_url="https://example.com/a"),
                Citation(title="B", source_url="https://example.com/b"),
            ],
        )
        assert len(resp.citations) == 2
        assert resp.citations[0].title == "A"
        assert resp.citations[1].title == "B"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ChatResponse(session_id="s", answer="a", confidence=1.2)


class TestCitation:
    def test_minimal_title_and_url(self):
        c = Citation(title="A", source_url="https://example.com/a")
        assert c.title == "A"
        assert c.snippet is None
        assert c.relevanceScore is None
        assert c.lastModified is None

    def test_all_fields(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        c = Citation(
            title="A",
            source_url="https://example.com/a",
            snippet="excerpt",
            relevanceScore=0.8,
            lastModified=ts,
        )
        assert c.snippet == "excerpt"
        assert c.relevanceScore == pytest.approx(0.8)
        assert c.lastModified == ts

    def test_relevance_score_bounds(self):
        with pytest.raises(ValidationError):
            Citation(title="A", source_url="https://x.com", relevanceScore=1.5)

    def test_serializes_new_fields_to_json(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        c = Citation(
            title="A",
            source_url="https://example.com/a",
            snippet="excerpt",
            relevanceScore=0.5,
            lastModified=ts,
        )
        dumped = c.model_dump(mode="json")
        assert dumped["snippet"] == "excerpt"
        assert dumped["relevanceScore"] == 0.5
        assert dumped["lastModified"].startswith("2026-01-01")


# ── FeedbackRequest ─────────────────────────────────────────────────────────


class TestFeedbackRequest:
    def test_valid_up(self):
        fb = FeedbackRequest(session_id="s", message_index=2, rating="up")
        assert fb.rating == "up"
        assert fb.comment is None

    def test_valid_down_with_comment(self):
        fb = FeedbackRequest(
            session_id="s", message_index=0, rating="down", comment="wrong"
        )
        assert fb.comment == "wrong"

    def test_invalid_rating_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(session_id="s", message_index=0, rating="meh")

    def test_negative_message_index_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(session_id="s", message_index=-1, rating="up")
