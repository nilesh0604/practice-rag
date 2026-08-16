"""RAG orchestrator — ties the Step 3 components into one query → answer flow.

Wires the query rewriter, hybrid retriever, context assembler, streaming
generator, and post-processor into a single ``stream_answer`` generator that:

1. Rewrites the query (passthrough by default).
2. Retrieves top-K chunks via hybrid dense + sparse + RRF.
3. Assembles the chunks into the system-prompt CONTEXT block.
4. Streams answer tokens from the Ollama generator.
5. Post-processes the full answer (citations + groundedness score).

``stream_answer`` yields tokens (``str``) as they arrive, then yields a final
``PostProcessResult`` object. The caller distinguishes by type::

    for item in orchestrator.stream_answer(query, history):
        if isinstance(item, str):
            # SSE: event: delta\ndata: {item}
        else:
            # PostProcessResult — persist, emit sources + metadata + done

This keeps the orchestrator stateless and unit-testable without FastAPI.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Union

from rag.context_assembler import ContextAssembler
from rag.generator import Generator
from rag.post_processor import PostProcessor, PostProcessResult
from rag.query_clarifier import GENERIC_CLARIFICATION, PassthroughQueryClarifier, QueryClarifier
from rag.query_decomposer import PassthroughQueryDecomposer, QueryDecomposer
from rag.query_rewriter import PassthroughQueryRewriter, QueryRewriter
from rag.retriever import HybridRetriever
from schemas.documents import RetrievedDoc

# Runtime import is safe: api.guardrails imports nothing from rag (the
# TYPE_CHECKING guard below is purely defensive for the heavier types).
from api.guardrails import CLASS_AMBIGUOUS, CLASS_COMPARE, CLASS_SENSITIVE

if TYPE_CHECKING:
    # Avoid a runtime circular import: api.guardrails imports nothing from
    # rag, but rag is imported widely; keeping this under TYPE_CHECKING
    # means the orchestrator has no hard dependency on the api package.
    from api.guardrails import GuardrailSuite
    from api.observability import LangfuseTracer
    from rag.nim_reranker import NIMReranker

logger = logging.getLogger(__name__)

# Type alias for the mixed stream: tokens are str, the final item is a result.
StreamItem = Union[str, PostProcessResult]

OUTPUT_REFUSAL: str = (
    "I can't provide that answer. Please ask a question about "
    "FastAPI, Pydantic v2, or SQLModel."
)
"""Canned refusal substituted for a streamed answer when the output guardrail
blocks it. Also surfaced via ``PostProcessResult.guardrail_replacement`` so the
SSE layer can emit a ``guardrail_replacement`` event and the frontend can swap
the already-streamed (one-way) tokens."""


class RAGOrchestrator:
    """Coordinates the full RAG flow: rewrite → retrieve → assemble → generate → post-process.

    Each collaborator is injected so the orchestrator is fully mockable in
    unit tests. The default query rewriter is a passthrough (no LLM call).

    An optional ``guardrail_suite`` (Step 6) adds three checks:

    1. **Input guardrail** — before retrieval, the query is scanned for
       prompt injection. A blocked query short-circuits to a refusal
       answer (no retrieval, no generation).
    2. **Query classifier** — the query is routed to one of
       ``documentation`` / ``greeting`` / ``off_topic`` / ``compare``.
       ``greeting`` and ``off_topic`` are handled with a canned answer
       (no retrieval, no generation); ``documentation`` and ``compare``
       proceed through the full RAG flow.
    3. **Output guardrail** — after generation, the answer is PII-scrubbed
       and (optionally) judged for harmful content. A harmful answer is
       replaced with a refusal. The scrubbed text is what gets post-
       processed and persisted. Because the tokens were already streamed
       (SSE is one-way), the refusal is also surfaced via
       ``PostProcessResult.guardrail_replacement`` so the SSE layer can
       emit a ``guardrail_replacement`` event and the frontend can swap
       the visible message.

    An optional ``query_decomposer`` splits a query the classifier labeled
    ``compare`` into one sub-query per compared subject. Retrieval runs once
    per sub-query and the results are merged + deduplicated (by doc id, max
    score kept) so the generator sees balanced context for both sides of the
    comparison. When the decomposer returns a single sub-query (the original
    query — either because the query is not a comparison or the passthrough
    decomposer is configured), the retrieval flow is identical to the
    non-decomposed path.

    An optional ``query_clarifier`` handles queries the classifier labeled
    ``ambiguous``. Instead of running retrieval + generation (which would
    likely produce a wrong or vague answer), the orchestrator asks the user
    to clarify. The clarifier generates a specific clarification prompt
    (ideally listing candidate interpretations); when it returns ``None``
    (passthrough clarifier, or the LLM/heuristic found no candidates), a
    generic clarification prompt is used so the user is always asked to
    disambiguate. No retrieval, no generation.

    Queries the classifier labels ``sensitive`` proceed through the full
    RAG flow (rewrite → retrieve → assemble → generate) but with generation
    **buffered** instead of streamed. The output guardrail runs on the
    complete answer buffer before any token is yielded, and only the
    guardrailed result (refusal if blocked, scrubbed answer otherwise) is
    delivered as a single token. This prevents partial exposure of content
    the output guardrail would later block — the user never sees un-
    guardrailed tokens for sensitive-topic queries. No
    ``guardrail_replacement`` is set on the result because nothing was
    streamed to swap (the single delta IS the final answer).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        context_assembler: ContextAssembler,
        generator: Generator,
        post_processor: PostProcessor,
        query_rewriter: QueryRewriter | None = None,
        guardrail_suite: "GuardrailSuite | None" = None,
        tracer: "LangfuseTracer | None" = None,
        reranker: "NIMReranker | None" = None,
        query_decomposer: QueryDecomposer | None = None,
        query_clarifier: QueryClarifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.context_assembler = context_assembler
        self.generator = generator
        self.post_processor = post_processor
        self.query_rewriter = query_rewriter or PassthroughQueryRewriter()
        self.guardrail_suite = guardrail_suite
        self.tracer = tracer
        self.reranker = reranker
        self.query_decomposer = query_decomposer or PassthroughQueryDecomposer()
        self.query_clarifier = query_clarifier or PassthroughQueryClarifier()

    def stream_answer(
        self,
        query: str,
        history: str = "",
    ) -> Iterator[StreamItem]:
        """Run the full RAG flow, yielding tokens then a final PostProcessResult.

        Args:
            query: The user's question.
            history: Formatted conversation history string (last-10-turns window).

        Yields:
            ``str`` tokens as they stream from the generator, followed by a
            single ``PostProcessResult`` with citations + confidence.
        """
        # ── Step 7: start a Langfuse trace for this request ────────────
        trace = None
        if self.tracer is not None:
            trace = self.tracer.start_trace("chat", metadata={"query": query})

        try:
            # ── Step 6: input guardrail + query classification ─────────
            if self.guardrail_suite is not None:
                gr_span = (
                    self.tracer.start_span(trace, "guardrail_input")
                    if trace is not None else None
                )
                decision = self.guardrail_suite.check_input(query, history)
                if decision.blocked:
                    logger.info("Orchestrator: input blocked — %s", decision.reason)
                    if gr_span is not None:
                        self.tracer.end_span(gr_span, metadata={"blocked": True, "reason": decision.reason})
                    yield from self._refusal_stream(
                        "I can't process that request. Please ask a question about "
                        "FastAPI, Pydantic v2, or SQLModel.",
                    )
                    return

                classification = self.guardrail_suite.classify(query, history)
                if gr_span is not None:
                    self.tracer.end_span(
                        gr_span,
                        metadata={"blocked": False, "class": classification.label},
                    )
                if classification.handled:
                    logger.info("Orchestrator: query classified as %s — short-circuit", classification.label)
                    yield from self._refusal_stream(classification.answer)
                    return
                classify_label = classification.label
            else:
                classify_label = ""

            # ── Ambiguous query → clarification prompt (no retrieval/generation) ──
            if classify_label == CLASS_AMBIGUOUS:
                clarification = self.query_clarifier.clarify(query, history)
                if not clarification:
                    clarification = GENERIC_CLARIFICATION
                logger.info("Orchestrator: ambiguous query — clarification prompt")
                yield from self._refusal_stream(clarification)
                return

            rewritten = self.query_rewriter.rewrite(query, history)
            logger.debug("Orchestrator: query=%r → rewritten=%r", query, rewritten)

            # ── Step 7: retrieval span ─────────────────────────────────
            ret_span = (
                self.tracer.start_span(trace, "retrieval", metadata={"rewritten": rewritten})
                if trace is not None else None
            )
            docs = self._retrieve(rewritten, classify_label)
            if ret_span is not None:
                self.tracer.end_span(ret_span, metadata={"num_docs": len(docs)})

            # ── Phase 4: rerank retrieved docs (if a reranker is configured) ──
            if self.reranker is not None:
                rr_span = (
                    self.tracer.start_span(trace, "rerank", metadata={"candidates": len(docs)})
                    if trace is not None else None
                )
                docs = self.reranker.rerank(rewritten, docs)
                if rr_span is not None:
                    self.tracer.end_span(rr_span, metadata={"reranked": len(docs)})

            context = self.context_assembler.assemble(docs)

            # ── Step 7: generation span (wraps the streaming loop) ──────
            gen_span = (
                self.tracer.start_span(trace, "generation")
                if trace is not None else None
            )

            # ── Sensitive-topic queries: buffer the full generation, run the
            # output guardrail on the complete buffer, and only THEN deliver
            # the (possibly refused) answer as a single token. This prevents
            # partial exposure of content the output guardrail would later
            # block — the user never sees un-guardrailed tokens. Non-sensitive
            # queries stream normally (tokens yielded as they arrive).
            output_blocked = False
            sensitive_buffered = classify_label == CLASS_SENSITIVE
            if sensitive_buffered:
                logger.info("Orchestrator: sensitive query — buffered (non-streaming) generation")
                answer = "".join(self.generator.stream(rewritten, context, history))
                if gen_span is not None:
                    self.tracer.end_span(
                        gen_span, metadata={"answer_len": len(answer), "buffered": True},
                    )
                # Output guardrail runs on the FULL buffer before any token
                # is yielded — no partial exposure.
                if self.guardrail_suite is not None:
                    out_span = (
                        self.tracer.start_span(trace, "guardrail_output")
                        if trace is not None else None
                    )
                    out_decision = self.guardrail_suite.check_output(answer)
                    if out_decision.blocked:
                        logger.info("Orchestrator: output blocked (sensitive) — %s", out_decision.reason)
                        answer = OUTPUT_REFUSAL
                        output_blocked = True
                    else:
                        answer = out_decision.scrubbed or answer
                    if out_span is not None:
                        self.tracer.end_span(
                            out_span,
                            metadata={"blocked": out_decision.blocked, "reason": out_decision.reason},
                        )
                # Deliver the guardrailed answer as a single token. No
                # ``guardrail_replacement`` is set — nothing was streamed to
                # swap; the single delta IS the final (safe) answer.
                yield answer
            else:
                answer = ""
                for token in self.generator.stream(rewritten, context, history):
                    answer += token
                    yield token
                if gen_span is not None:
                    self.tracer.end_span(
                        gen_span, metadata={"answer_len": len(answer)},
                    )
                # ── Step 6: output guardrail (PII scrub + harmful-content judge) ──
                if self.guardrail_suite is not None:
                    out_span = (
                        self.tracer.start_span(trace, "guardrail_output")
                        if trace is not None else None
                    )
                    out_decision = self.guardrail_suite.check_output(answer)
                    if out_decision.blocked:
                        logger.info("Orchestrator: output blocked — %s", out_decision.reason)
                        # Replace the streamed answer with a refusal. The already-
                        # streamed tokens are not un-sent (SSE is one-way), but the
                        # persisted + post-processed answer is the refusal so the
                        # session store and history endpoint reflect the block.
                        # ``guardrail_replacement`` is set on the result so the SSE
                        # layer can emit a swap event and the frontend can replace
                        # the visible (already-streamed) message with the refusal.
                        answer = OUTPUT_REFUSAL
                        output_blocked = True
                    else:
                        answer = out_decision.scrubbed or answer
                    if out_span is not None:
                        self.tracer.end_span(
                            out_span,
                            metadata={"blocked": out_decision.blocked, "reason": out_decision.reason},
                        )

            result = self.post_processor.post_process(answer, docs)
            # Only the streaming path sets ``guardrail_replacement`` — the
            # sensitive buffered path never streamed the harmful tokens, so
            # there is nothing for the frontend to swap (the single delta IS
            # the final, guardrailed answer).
            if output_blocked and not sensitive_buffered:
                result.guardrail_replacement = OUTPUT_REFUSAL
            if trace is not None:
                result.trace_id = trace.id
            yield result
        finally:
            # ── Step 7: end the Langfuse root span (closes the trace) ───
            if trace is not None:
                self.tracer.end_trace(trace)

    def _refusal_stream(self, answer: str) -> Iterator[StreamItem]:
        """Yield a canned answer as one token + an empty PostProcessResult.

        Used for input-blocked and short-circuited (greeting / off_topic)
        queries so the caller sees the same token-then-result shape as a
        real generation, without invoking the retriever or generator.
        """
        yield answer
        yield PostProcessResult(answer=answer)

    # ── retrieval (single + multi-query) ─────────────────────────────────

    def _retrieve(self, rewritten: str, classify_label: str) -> list[RetrievedDoc]:
        """Run retrieval, decomposing into sub-queries for ``compare`` queries.

        When a query decomposer is configured AND the classifier labeled the
        query ``compare``, the rewritten query is split into sub-queries and
        retrieval runs once per sub-query. The per-sub-query result lists are
        merged and deduplicated (by doc id, max score kept) so the generator
        sees balanced context for both sides of a comparison. For all other
        labels (or when decomposition yields a single sub-query), a single
        retrieval pass runs — identical to the non-decomposed path.
        """
        if classify_label != CLASS_COMPARE:
            return self.retriever.retrieve(rewritten)
        sub_queries = self.query_decomposer.decompose(rewritten)
        if len(sub_queries) <= 1:
            return self.retriever.retrieve(rewritten)
        logger.info(
            "Orchestrator: compare query decomposed into %d sub-queries", len(sub_queries),
        )
        merged: list[RetrievedDoc] = []
        for sq in sub_queries:
            merged = self._merge_docs(merged, self.retriever.retrieve(sq))
        return merged

    @staticmethod
    def _merge_docs(
        acc: list[RetrievedDoc],
        new: list[RetrievedDoc],
    ) -> list[RetrievedDoc]:
        """Merge two retrieved-doc lists, deduplicating by doc id.

        When a doc appears in both lists, the higher RRF score is kept. The
        merged list is sorted by score (descending) so the most relevant
        chunks surface first to the context assembler / reranker.
        """
        by_id: dict[str, RetrievedDoc] = {}
        for doc in (*acc, *new):
            existing = by_id.get(doc.id)
            if existing is None or doc.score > existing.score:
                by_id[doc.id] = doc
        return sorted(by_id.values(), key=lambda d: d.score, reverse=True)

    # ── convenience: non-streaming single-shot ─────────────────────────

    def answer(
        self,
        query: str,
        history: str = "",
    ) -> tuple[str, PostProcessResult, list[RetrievedDoc]]:
        """Non-streaming convenience method — collects all tokens and returns the result.

        Useful for tests, the eval pipeline, and the (future) non-streaming
        history endpoint. Applies the same guardrail checks as
        ``stream_answer``. Returns ``(answer_text, post_process_result, retrieved_docs)``.
        """
        # ── Step 7: start a Langfuse trace for this request ────────────
        trace = None
        if self.tracer is not None:
            trace = self.tracer.start_trace("chat", metadata={"query": query})

        try:
            # ── Step 6: input guardrail + query classification ─────────
            if self.guardrail_suite is not None:
                gr_span = (
                    self.tracer.start_span(trace, "guardrail_input")
                    if trace is not None else None
                )
                decision = self.guardrail_suite.check_input(query, history)
                if decision.blocked:
                    refusal = (
                        "I can't process that request. Please ask a question about "
                        "FastAPI, Pydantic v2, or SQLModel."
                    )
                    if gr_span is not None:
                        self.tracer.end_span(gr_span, metadata={"blocked": True})
                    return refusal, PostProcessResult(answer=refusal), []

                classification = self.guardrail_suite.classify(query, history)
                if gr_span is not None:
                    self.tracer.end_span(
                        gr_span, metadata={"blocked": False, "class": classification.label},
                    )
                if classification.handled:
                    return classification.answer, PostProcessResult(answer=classification.answer), []
                classify_label = classification.label
            else:
                classify_label = ""

            # ── Ambiguous query → clarification prompt (no retrieval/generation) ──
            if classify_label == CLASS_AMBIGUOUS:
                clarification = self.query_clarifier.clarify(query, history)
                if not clarification:
                    clarification = GENERIC_CLARIFICATION
                return clarification, PostProcessResult(answer=clarification), []

            rewritten = self.query_rewriter.rewrite(query, history)

            ret_span = (
                self.tracer.start_span(trace, "retrieval", metadata={"rewritten": rewritten})
                if trace is not None else None
            )
            docs = self._retrieve(rewritten, classify_label)
            if ret_span is not None:
                self.tracer.end_span(ret_span, metadata={"num_docs": len(docs)})

            # ── Phase 4: rerank retrieved docs (if a reranker is configured) ──
            if self.reranker is not None:
                rr_span = (
                    self.tracer.start_span(trace, "rerank", metadata={"candidates": len(docs)})
                    if trace is not None else None
                )
                docs = self.reranker.rerank(rewritten, docs)
                if rr_span is not None:
                    self.tracer.end_span(rr_span, metadata={"reranked": len(docs)})

            context = self.context_assembler.assemble(docs)

            gen_span = (
                self.tracer.start_span(trace, "generation")
                if trace is not None else None
            )
            answer = "".join(self.generator.stream(rewritten, context, history))
            if gen_span is not None:
                self.tracer.end_span(gen_span, metadata={"answer_len": len(answer)})

            # ── Step 6: output guardrail (PII scrub + harmful-content judge) ──
            output_blocked = False
            if self.guardrail_suite is not None:
                out_span = (
                    self.tracer.start_span(trace, "guardrail_output")
                    if trace is not None else None
                )
                out_decision = self.guardrail_suite.check_output(answer)
                if out_decision.blocked:
                    answer = OUTPUT_REFUSAL
                    output_blocked = True
                else:
                    answer = out_decision.scrubbed or answer
                if out_span is not None:
                    self.tracer.end_span(
                        out_span, metadata={"blocked": out_decision.blocked},
                    )

            result = self.post_processor.post_process(answer, docs)
            if output_blocked:
                result.guardrail_replacement = OUTPUT_REFUSAL
            if trace is not None:
                result.trace_id = trace.id
            return answer, result, docs
        finally:
            # ── Step 7: end the Langfuse root span (closes the trace) ───
            if trace is not None:
                self.tracer.end_trace(trace)
