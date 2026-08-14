"""Tests for the RAG orchestrator (rag/orchestrator.py).

All collaborators are mocked. Verifies the full flow: rewrite → retrieve →
assemble → generate → post-process, and that tokens are yielded before the
final PostProcessResult.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.orchestrator import RAGOrchestrator
from rag.post_processor import PostProcessResult
from schemas.documents import RetrievedDoc


def _make_doc(title="Doc", content="content", score=0.9):
    return RetrievedDoc(
        id="x",
        content=content,
        title=title,
        source_url="https://example.com/x",
        section=None,
        score=score,
    )


def _build_orchestrator(tokens=None, docs=None, confidence=0.9):
    """Build an orchestrator with all mocked collaborators."""
    if tokens is None:
        tokens = ["Hello", " world"]
    if docs is None:
        docs = [_make_doc(title="FastAPI Docs", content="fastapi content")]

    retriever = MagicMock()
    retriever.retrieve.return_value = docs

    assembler = MagicMock()
    assembler.assemble.return_value = "assembled context"

    generator = MagicMock()
    generator.stream.return_value = iter(tokens)

    pp = MagicMock()
    pp.post_process.return_value = PostProcessResult(
        answer="".join(tokens),
        citations=[],
        confidence=confidence,
    )

    rewriter = MagicMock()
    rewriter.rewrite.return_value = "rewritten query"

    return RAGOrchestrator(retriever, assembler, generator, pp, rewriter), {
        "retriever": retriever,
        "assembler": assembler,
        "generator": generator,
        "pp": pp,
        "rewriter": rewriter,
    }


class TestStreamAnswer:
    def test_yields_tokens_then_result(self):
        orch, _ = _build_orchestrator(tokens=["a", "b", "c"])
        items = list(orch.stream_answer("query"))
        # First items are strings (tokens), last is PostProcessResult
        assert items[0] == "a"
        assert items[1] == "b"
        assert items[2] == "c"
        assert isinstance(items[-1], PostProcessResult)

    def test_token_count_matches_generator(self):
        orch, _ = _build_orchestrator(tokens=["t1", "t2", "t3", "t4"])
        items = list(orch.stream_answer("query"))
        tokens = [i for i in items if isinstance(i, str)]
        assert len(tokens) == 4

    def test_exactly_one_result_at_end(self):
        orch, _ = _build_orchestrator(tokens=["x"])
        items = list(orch.stream_answer("query"))
        results = [i for i in items if isinstance(i, PostProcessResult)]
        assert len(results) == 1
        assert items[-1] is results[0]

    def test_rewriter_called_with_query_and_history(self):
        orch, mocks = _build_orchestrator()
        list(orch.stream_answer("my query", "my history"))
        mocks["rewriter"].rewrite.assert_called_once_with("my query", "my history")

    def test_retriever_called_with_rewritten_query(self):
        orch, mocks = _build_orchestrator()
        mocks["rewriter"].rewrite.return_value = "rewritten!"
        list(orch.stream_answer("original"))
        mocks["retriever"].retrieve.assert_called_once_with("rewritten!")

    def test_assembler_called_with_retrieved_docs(self):
        orch, mocks = _build_orchestrator()
        docs = [_make_doc(title="A", content="a")]
        mocks["retriever"].retrieve.return_value = docs
        list(orch.stream_answer("q"))
        mocks["assembler"].assemble.assert_called_once_with(docs)

    def test_generator_called_with_rewritten_query_context_history(self):
        orch, mocks = _build_orchestrator()
        mocks["rewriter"].rewrite.return_value = "rw"
        mocks["assembler"].assemble.return_value = "ctx"
        list(orch.stream_answer("q", "hist"))
        mocks["generator"].stream.assert_called_once_with("rw", "ctx", "hist")

    def test_post_processor_called_with_full_answer_and_docs(self):
        orch, mocks = _build_orchestrator(tokens=["Hello", " ", "World"])
        docs = [_make_doc(title="D", content="c")]
        mocks["retriever"].retrieve.return_value = docs
        list(orch.stream_answer("q"))
        mocks["pp"].post_process.assert_called_once_with("Hello World", docs)

    def test_default_rewriter_is_passthrough(self):
        """When no rewriter is given, the query passes through unchanged."""
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        assembler = MagicMock()
        assembler.assemble.return_value = "ctx"
        generator = MagicMock()
        generator.stream.return_value = iter(["tok"])
        pp = MagicMock()
        pp.post_process.return_value = PostProcessResult(answer="tok", confidence=0.5)

        orch = RAGOrchestrator(retriever, assembler, generator, pp)
        list(orch.stream_answer("original query"))
        # Passthrough → retriever gets the original query
        retriever.retrieve.assert_called_once_with("original query")

    def test_empty_token_stream(self):
        orch, _ = _build_orchestrator(tokens=[])
        items = list(orch.stream_answer("query"))
        # No tokens, but still a result
        results = [i for i in items if isinstance(i, PostProcessResult)]
        assert len(results) == 1
        assert items == results


class TestAnswerMethod:
    def test_returns_answer_string(self):
        orch, _ = _build_orchestrator(tokens=["Hello", " world"])
        answer, result, docs = orch.answer("query")
        assert answer == "Hello world"

    def test_returns_post_process_result(self):
        orch, _ = _build_orchestrator(tokens=["a"], confidence=0.8)
        answer, result, docs = orch.answer("query")
        assert isinstance(result, PostProcessResult)
        assert result.confidence == 0.8

    def test_returns_retrieved_docs(self):
        docs = [_make_doc(title="X", content="x")]
        orch, _ = _build_orchestrator(tokens=["a"], docs=docs)
        answer, result, returned_docs = orch.answer("query")
        assert returned_docs == docs

    def test_answer_matches_concatenated_tokens(self):
        orch, _ = _build_orchestrator(tokens=["Fast", "API", " is", " great"])
        answer, _, _ = orch.answer("q")
        assert answer == "FastAPI is great"
