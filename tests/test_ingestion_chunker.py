"""Tests for the chunker (ingestion/chunker.py)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ingestion.chunker import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, chunk_document
from ingestion.parser import ParsedDocument


def _make_doc(content: str, parent_doc_id: str = "test-doc") -> ParsedDocument:
    return ParsedDocument(
        file_path="/tmp/test.md",
        content=content,
        title="Test Doc",
        source_url="https://example.com/test",
        last_modified=datetime(2026, 1, 1, tzinfo=timezone.utc),
        parent_doc_id=parent_doc_id,
    )


class TestChunkDocument:
    def test_short_doc_produces_one_chunk(self):
        doc = _make_doc("# Title\n\nShort content here.\n")
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].parent_doc_id == "test-doc"
        assert chunks[0].title == "Test Doc"

    def test_empty_content_returns_empty(self):
        doc = _make_doc("")
        assert chunk_document(doc) == []

    def test_whitespace_only_returns_empty(self):
        doc = _make_doc("   \n\n   \n")
        assert chunk_document(doc) == []

    def test_chunk_ids_are_deterministic(self):
        content = "# Title\n\n" + "Some content. " * 200
        doc = _make_doc(content)
        chunks1 = chunk_document(doc)
        chunks2 = chunk_document(doc)
        assert [c.id for c in chunks1] == [c.id for c in chunks2]

    def test_chunk_ids_unique_within_doc(self):
        content = "# Title\n\n" + "Some content. " * 200
        doc = _make_doc(content)
        chunks = chunk_document(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_is_sequential(self):
        content = "# Title\n\n" + "Some content. " * 200
        doc = _make_doc(content)
        chunks = chunk_document(doc)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_section_from_markdown_headers(self):
        doc = _make_doc(
            "# Main Title\n\n"
            "## Subsection A\n\n"
            "Content about A.\n\n"
            "## Subsection B\n\n"
            "Content about B.\n"
        )
        chunks = chunk_document(doc)
        sections = {c.section for c in chunks if c.section}
        assert any("Subsection A" in s for s in sections)
        assert any("Subsection B" in s for s in sections)

    def test_embedding_is_none_before_embedder(self):
        doc = _make_doc("# Title\n\nContent.\n")
        chunks = chunk_document(doc)
        assert all(c.embedding is None for c in chunks)

    def test_to_payload_excludes_id_and_embedding(self):
        doc = _make_doc("# Title\n\nContent.\n")
        chunks = chunk_document(doc)
        payload = chunks[0].to_payload()
        assert "id" not in payload
        assert "embedding" not in payload
        assert "content" in payload
        assert "title" in payload
        assert "parent_doc_id" in payload

    def test_long_doc_produces_multiple_chunks(self):
        # ~600 tokens of content should produce >1 chunk at 512 target.
        content = "# Title\n\n" + ("The quick brown fox jumps over the lazy dog. " * 80)
        doc = _make_doc(content)
        chunks = chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_size_and_overlap_constants(self):
        assert CHUNK_SIZE_TOKENS == 512
        assert CHUNK_OVERLAP_TOKENS == 64

    def test_different_parent_ids_produce_different_chunk_ids(self):
        content = "# Title\n\nContent here.\n"
        chunks_a = chunk_document(_make_doc(content, "doc-a"))
        chunks_b = chunk_document(_make_doc(content, "doc-b"))
        assert chunks_a[0].id != chunks_b[0].id
