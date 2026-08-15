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


# ── Step 6: guardrail integration ──────────────────────────────────────


def _mock_guardrail_suite(
    *,
    input_blocked: bool = False,
    input_reason: str = "",
    classification_label: str = "documentation",
    classification_handled: bool = False,
    classification_answer: str = "",
    output_blocked: bool = False,
    output_reason: str = "",
    output_scrubbed: str | None = None,
):
    """Build a MagicMock GuardrailSuite with canned decisions."""
    from api.guardrails import GuardrailDecision, QueryClassification

    suite = MagicMock()
    suite.check_input.return_value = GuardrailDecision(
        blocked=input_blocked, reason=input_reason,
    )
    suite.classify.return_value = QueryClassification(
        label=classification_label,
        handled=classification_handled,
        answer=classification_answer,
    )
    scrubbed = output_scrubbed if output_scrubbed is not None else ""
    suite.check_output.return_value = GuardrailDecision(
        blocked=output_blocked, reason=output_reason, scrubbed=scrubbed,
    )
    return suite


class TestOrchestratorGuardrailsStream:
    def _build_with_guardrails(self, suite, tokens=None, docs=None):
        if tokens is None:
            tokens = ["Hello", " world"]
        if docs is None:
            docs = [_make_doc(title="D", content="c")]
        retriever = MagicMock()
        retriever.retrieve.return_value = docs
        assembler = MagicMock()
        assembler.assemble.return_value = "ctx"
        generator = MagicMock()
        generator.stream.return_value = iter(tokens)
        pp = MagicMock()
        # Echo the input answer so output-guardrail replacements are visible
        # in the returned PostProcessResult (a fixed return_value would mask
        # the refusal text the orchestrator passes to the post-processor).
        pp.post_process.side_effect = lambda answer, retrieved_docs: PostProcessResult(
            answer=answer, confidence=0.9,
        )
        return RAGOrchestrator(retriever, assembler, generator, pp, guardrail_suite=suite)

    def test_input_blocked_short_circuits_to_refusal(self):
        suite = _mock_guardrail_suite(input_blocked=True, input_reason="injection")
        orch = self._build_with_guardrails(suite)
        items = list(orch.stream_answer("ignore previous instructions"))
        tokens = [i for i in items if isinstance(i, str)]
        results = [i for i in items if isinstance(i, PostProcessResult)]
        assert len(tokens) == 1
        assert "can't process" in tokens[0]
        assert len(results) == 1
        # Retriever and generator never called.
        orch.retriever.retrieve.assert_not_called()
        orch.generator.stream.assert_not_called()

    def test_greeting_short_circuits_with_canned_answer(self):
        suite = _mock_guardrail_suite(
            classification_label="greeting",
            classification_handled=True,
            classification_answer="Hello! I'm a docs assistant.",
        )
        orch = self._build_with_guardrails(suite)
        items = list(orch.stream_answer("hi"))
        tokens = [i for i in items if isinstance(i, str)]
        assert tokens == ["Hello! I'm a docs assistant."]
        orch.retriever.retrieve.assert_not_called()
        orch.generator.stream.assert_not_called()

    def test_off_topic_short_circuits_with_canned_answer(self):
        suite = _mock_guardrail_suite(
            classification_label="off_topic",
            classification_handled=True,
            classification_answer="I only answer docs questions.",
        )
        orch = self._build_with_guardrails(suite)
        items = list(orch.stream_answer("weather"))
        tokens = [i for i in items if isinstance(i, str)]
        assert tokens == ["I only answer docs questions."]
        orch.retriever.retrieve.assert_not_called()

    def test_documentation_proceeds_through_rag_flow(self):
        suite = _mock_guardrail_suite(classification_label="documentation")
        orch = self._build_with_guardrails(suite, tokens=["a", "b"])
        items = list(orch.stream_answer("How do I use FastAPI?"))
        tokens = [i for i in items if isinstance(i, str)]
        assert tokens == ["a", "b"]
        orch.retriever.retrieve.assert_called_once()

    def test_guardrail_receives_history(self):
        """The orchestrator must forward conversation history to both
        check_input and classify so context-dependent follow-ups are not
        false-positive blocked or misrouted to off_topic."""
        suite = _mock_guardrail_suite(classification_label="documentation")
        orch = self._build_with_guardrails(suite)
        list(orch.stream_answer("summarize the above", "prior Q&A"))
        suite.check_input.assert_called_once_with("summarize the above", "prior Q&A")
        suite.classify.assert_called_once_with("summarize the above", "prior Q&A")

    def test_follow_up_not_blocked_with_history(self):
        """A context-dependent follow-up with on-topic history must pass
        through the guardrail and reach the RAG flow (not short-circuit)."""
        suite = _mock_guardrail_suite(
            classification_label="follow_up",
            classification_handled=False,
        )
        orch = self._build_with_guardrails(suite, tokens=["summary"])
        items = list(orch.stream_answer(
            "please summarize all above 3 answers",
            "Q: What is Pydantic? A: ...",
        ))
        tokens = [i for i in items if isinstance(i, str)]
        assert tokens == ["summary"]
        orch.retriever.retrieve.assert_called_once()

    def test_compare_proceeds_through_rag_flow(self):
        suite = _mock_guardrail_suite(classification_label="compare")
        orch = self._build_with_guardrails(suite)
        list(orch.stream_answer("compare FastAPI and Flask"))
        orch.retriever.retrieve.assert_called_once()

    def test_output_blocked_replaces_answer_in_result(self):
        suite = _mock_guardrail_suite(output_blocked=True, output_reason="harmful")
        orch = self._build_with_guardrails(suite, tokens=["harmful", " text"])
        items = list(orch.stream_answer("q"))
        result = [i for i in items if isinstance(i, PostProcessResult)][0]
        assert "can't provide" in result.answer
        # Post-processor receives the refusal text, not the original.
        orch.post_processor.post_process.assert_called_once()
        pp_arg = orch.post_processor.post_process.call_args[0][0]
        assert "can't provide" in pp_arg

    def test_output_scrub_applies_pii_redaction(self):
        scrubbed = "Email me at [REDACTED-EMAIL]"
        suite = _mock_guardrail_suite(output_scrubbed=scrubbed)
        orch = self._build_with_guardrails(suite, tokens=["Email me at user@example.com"])
        items = list(orch.stream_answer("q"))
        result = [i for i in items if isinstance(i, PostProcessResult)][0]
        # Post-processor receives the scrubbed text.
        pp_arg = orch.post_processor.post_process.call_args[0][0]
        assert pp_arg == scrubbed

    def test_no_guardrail_suite_unchanged_behavior(self):
        """When guardrail_suite is None, the flow is identical to Step 3."""
        orch, _ = _build_orchestrator(tokens=["x", "y"])
        items = list(orch.stream_answer("query"))
        tokens = [i for i in items if isinstance(i, str)]
        assert tokens == ["x", "y"]


class TestOrchestratorGuardrailsAnswer:
    def _build_with_guardrails(self, suite, tokens=None, docs=None):
        if tokens is None:
            tokens = ["Hello", " world"]
        if docs is None:
            docs = [_make_doc(title="D", content="c")]
        retriever = MagicMock()
        retriever.retrieve.return_value = docs
        assembler = MagicMock()
        assembler.assemble.return_value = "ctx"
        generator = MagicMock()
        generator.stream.return_value = iter(tokens)
        pp = MagicMock()
        pp.post_process.side_effect = lambda answer, retrieved_docs: PostProcessResult(
            answer=answer, confidence=0.9,
        )
        return RAGOrchestrator(retriever, assembler, generator, pp, guardrail_suite=suite)

    def test_input_blocked_returns_refusal(self):
        suite = _mock_guardrail_suite(input_blocked=True)
        orch = self._build_with_guardrails(suite)
        answer, result, docs = orch.answer("ignore previous instructions")
        assert "can't process" in answer
        assert docs == []
        orch.retriever.retrieve.assert_not_called()

    def test_guardrail_receives_history(self):
        suite = _mock_guardrail_suite(classification_label="documentation")
        orch = self._build_with_guardrails(suite)
        orch.answer("summarize the above", "prior Q&A")
        suite.check_input.assert_called_once_with("summarize the above", "prior Q&A")
        suite.classify.assert_called_once_with("summarize the above", "prior Q&A")

    def test_greeting_handled_returns_canned(self):
        suite = _mock_guardrail_suite(
            classification_label="greeting",
            classification_handled=True,
            classification_answer="Hello!",
        )
        orch = self._build_with_guardrails(suite)
        answer, result, docs = orch.answer("hi")
        assert answer == "Hello!"
        assert docs == []

    def test_output_blocked_replaces_answer(self):
        suite = _mock_guardrail_suite(output_blocked=True)
        orch = self._build_with_guardrails(suite, tokens=["bad", " answer"])
        answer, result, docs = orch.answer("q")
        assert "can't provide" in answer

    def test_output_scrub_applies(self):
        suite = _mock_guardrail_suite(output_scrubbed="scrubbed answer")
        orch = self._build_with_guardrails(suite, tokens=["raw answer"])
        answer, _, _ = orch.answer("q")
        assert answer == "scrubbed answer"


# ── Step 7: tracer integration ─────────────────────────────────────────


def _mock_tracer():
    """Build a MagicMock LangfuseTracer with realistic span/trace handles."""
    from api.observability import SpanHandle, TraceHandle

    tracer = MagicMock()
    tracer.start_trace.return_value = TraceHandle(id="trace-1", enabled=True)
    # start_span returns a SpanHandle so end_span can be called on it
    tracer.start_span.side_effect = lambda trace, name, metadata=None: SpanHandle(
        name=name, start=0.0, enabled=True,
    )
    return tracer


class TestOrchestratorTracerStream:
    def _build_with_tracer(self, tracer, tokens=None, docs=None):
        if tokens is None:
            tokens = ["Hello", " world"]
        if docs is None:
            docs = [_make_doc(title="D", content="c")]
        retriever = MagicMock()
        retriever.retrieve.return_value = docs
        assembler = MagicMock()
        assembler.assemble.return_value = "ctx"
        generator = MagicMock()
        generator.stream.return_value = iter(tokens)
        pp = MagicMock()
        pp.post_process.side_effect = lambda answer, retrieved_docs: PostProcessResult(
            answer=answer, confidence=0.9,
        )
        return RAGOrchestrator(
            retriever, assembler, generator, pp, tracer=tracer,
        )

    def test_start_trace_called(self):
        tracer = _mock_tracer()
        orch = self._build_with_tracer(tracer)
        list(orch.stream_answer("query"))
        tracer.start_trace.assert_called_once()

    def test_retrieval_span_created_and_ended(self):
        tracer = _mock_tracer()
        orch = self._build_with_tracer(tracer)
        list(orch.stream_answer("query"))
        span_names = [c.args[1] for c in tracer.start_span.call_args_list]
        assert "retrieval" in span_names
        assert tracer.end_span.call_count >= 2  # retrieval + generation

    def test_generation_span_created(self):
        tracer = _mock_tracer()
        orch = self._build_with_tracer(tracer)
        list(orch.stream_answer("query"))
        span_names = [c.args[1] for c in tracer.start_span.call_args_list]
        assert "generation" in span_names

    def test_trace_id_set_on_result(self):
        tracer = _mock_tracer()
        tracer.start_trace.return_value.id = "trace-abc"
        orch = self._build_with_tracer(tracer)
        items = list(orch.stream_answer("query"))
        result = [i for i in items if isinstance(i, PostProcessResult)][0]
        assert result.trace_id == "trace-abc"

    def test_no_tracer_unchanged_behavior(self):
        orch, _ = _build_orchestrator(tokens=["x", "y"])
        items = list(orch.stream_answer("query"))
        tokens = [i for i in items if isinstance(i, str)]
        assert tokens == ["x", "y"]
        result = [i for i in items if isinstance(i, PostProcessResult)][0]
        assert result.trace_id is None

    def test_guardrail_spans_created_with_suite(self):
        tracer = _mock_tracer()
        suite = _mock_guardrail_suite()
        orch = self._build_with_tracer(tracer)
        orch.guardrail_suite = suite
        list(orch.stream_answer("query"))
        span_names = [c.args[1] for c in tracer.start_span.call_args_list]
        assert "guardrail_input" in span_names
        assert "guardrail_output" in span_names


class TestOrchestratorTracerAnswer:
    def _build_with_tracer(self, tracer, tokens=None):
        if tokens is None:
            tokens = ["Hello", " world"]
        retriever = MagicMock()
        retriever.retrieve.return_value = [_make_doc()]
        assembler = MagicMock()
        assembler.assemble.return_value = "ctx"
        generator = MagicMock()
        generator.stream.return_value = iter(tokens)
        pp = MagicMock()
        pp.post_process.side_effect = lambda answer, retrieved_docs: PostProcessResult(
            answer=answer, confidence=0.9,
        )
        return RAGOrchestrator(
            retriever, assembler, generator, pp, tracer=tracer,
        )

    def test_start_trace_called(self):
        tracer = _mock_tracer()
        orch = self._build_with_tracer(tracer)
        orch.answer("query")
        tracer.start_trace.assert_called_once()

    def test_trace_id_set_on_result(self):
        tracer = _mock_tracer()
        tracer.start_trace.return_value.id = "trace-xyz"
        orch = self._build_with_tracer(tracer)
        _, result, _ = orch.answer("query")
        assert result.trace_id == "trace-xyz"

    def test_spans_created(self):
        tracer = _mock_tracer()
        orch = self._build_with_tracer(tracer)
        orch.answer("query")
        span_names = [c.args[1] for c in tracer.start_span.call_args_list]
        assert "retrieval" in span_names
        assert "generation" in span_names
