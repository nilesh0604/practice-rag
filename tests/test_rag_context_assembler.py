"""Tests for the context assembler (rag/context_assembler.py)."""

from __future__ import annotations

from schemas.documents import RetrievedDoc

from rag.context_assembler import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_CHUNKS,
    EMPTY_CONTEXT,
    ContextAssembler,
)


def _make_doc(title="Doc", content="content", section=None, score=0.9):
    return RetrievedDoc(
        id="x",
        content=content,
        title=title,
        source_url="https://example.com/x",
        section=section,
        score=score,
    )


class TestContextAssembler:
    def test_empty_docs_returns_placeholder(self):
        ca = ContextAssembler()
        assert ca.assemble([]) == EMPTY_CONTEXT

    def test_single_doc_numbered(self):
        ca = ContextAssembler()
        doc = _make_doc(title="Query Parameters - FastAPI", content="You can declare query params.")
        result = ca.assemble([doc])
        assert "[1] Query Parameters - FastAPI" in result
        assert "You can declare query params." in result

    def test_multiple_docs_numbered_sequentially(self):
        ca = ContextAssembler()
        docs = [
            _make_doc(title="A", content="content a"),
            _make_doc(title="B", content="content b"),
            _make_doc(title="C", content="content c"),
        ]
        result = ca.assemble(docs)
        assert "[1] A" in result
        assert "[2] B" in result
        assert "[3] C" in result

    def test_section_in_header(self):
        ca = ContextAssembler()
        doc = _make_doc(title="FastAPI Docs", content="body", section="tutorial")
        result = ca.assemble([doc])
        assert "[1] FastAPI Docs — tutorial" in result

    def test_no_section_no_dash(self):
        ca = ContextAssembler()
        doc = _make_doc(title="FastAPI Docs", content="body", section=None)
        result = ca.assemble([doc])
        assert "—" not in result.split("\n")[0]

    def test_max_chunks_limit(self):
        ca = ContextAssembler(max_chunks=2)
        docs = [_make_doc(title=f"D{i}", content=f"c{i}") for i in range(5)]
        result = ca.assemble(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" not in result

    def test_chunks_separated_by_double_newline(self):
        ca = ContextAssembler()
        docs = [_make_doc(title="A", content="aaa"), _make_doc(title="B", content="bbb")]
        result = ca.assemble(docs)
        assert "\n\n" in result

    def test_content_stripped(self):
        ca = ContextAssembler()
        doc = _make_doc(title="T", content="  spaced content  \n")
        result = ca.assemble([doc])
        assert "spaced content" in result
        # Leading/trailing whitespace from content should be stripped
        assert "  spaced" not in result

    def test_max_chars_soft_cap(self):
        ca = ContextAssembler(max_chunks=10, max_chars=50)
        docs = [_make_doc(title="A", content="x" * 40) for _ in range(5)]
        result = ca.assemble(docs)
        # First chunk (~42 chars) fits; second would exceed 50 → only 1 included
        assert "[1]" in result
        assert "[2]" not in result

    def test_max_chars_allows_first_chunk_even_if_over(self):
        """If even the first chunk exceeds max_chars, it's still included (never empty)."""
        ca = ContextAssembler(max_chunks=5, max_chars=10)
        doc = _make_doc(title="Big", content="x" * 100)
        result = ca.assemble([doc])
        assert "[1] Big" in result

    def test_empty_context_constant(self):
        assert "No relevant context" in EMPTY_CONTEXT

    def test_default_max_chunks_is_five(self):
        assert DEFAULT_MAX_CHUNKS == 5

    def test_default_max_chars_positive(self):
        assert DEFAULT_MAX_CHARS > 0
