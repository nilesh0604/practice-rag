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

from rag.bias_monitor import BiasAssessment, BiasMonitor, PassthroughBiasMonitor
from rag.context_assembler import ContextAssembler
from rag.drift_monitor import DriftMonitor, PassthroughDriftMonitor
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

LOW_CONFIDENCE_REFUSAL: str = (
    "I'm not confident this answer is grounded in the retrieved sources. "
    "Please rephrase the question or consult the documentation directly."
)
"""Canned refusal substituted for an answer whose post-processor groundedness
score falls below ``CONFIDENCE_THRESHOLD`` when the orchestrator is built with
``block_low_confidence=True``. Surfaced via
``PostProcessResult.guardrail_replacement`` (same swap mechanism as the output
guardrail) so the SSE layer can replace the already-streamed tokens. The real
(low) confidence score is preserved on the result so metadata + metrics still
capture the grounding failure."""

BIAS_REFUSAL: str = (
    "I can't provide that answer because it may contain biased or "
    "non-inclusive language. Please rephrase the question."
)
"""Canned refusal substituted for an answer the bias & fairness monitor
flags as biased when the orchestrator is built with ``block_biased=True``
(Responsible AI). Surfaced via ``PostProcessResult.guardrail_replacement``
(same swap mechanism as the output guardrail + hallucination block) so the
SSE layer can replace the already-streamed tokens. The real
``BiasAssessment`` (categories, evidence, score) is preserved on the result
so metadata + metrics capture the bias detection."""


class RAGOrchestrator:
    """Coordinates the full RAG flow: rewrite → retrieve → assemble → generate → post-process.

    Each collaborator is injected so the orchestrator is fully mockable in
    unit tests. The default query rewriter is a passthrough (no LLM call).

    An optional ``guardrail_suite`` (Step 6) adds three checks:

    1. **Input guardrail** — before retrieval, the query is scanned for
       prompt injection (regex + LLM judge), PII-scrubbed (emails/phones
       redacted), and checked for general content safety (LLM judge). A
       blocked query short-circuits to a refusal answer (no retrieval, no
       generation). The PII-scrubbed query is used for all downstream
       steps (classify, rewrite, retrieve, generate) so no raw PII reaches
       any LLM prompt or the retriever.
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
        block_low_confidence: bool = False,
        bias_monitor: BiasMonitor | None = None,
        block_biased: bool = False,
        drift_monitor: DriftMonitor | None = None,
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
        self.block_low_confidence = block_low_confidence
        """When True, an answer whose post-processor groundedness score is
        below ``CONFIDENCE_THRESHOLD`` is replaced with ``LOW_CONFIDENCE_REFUSAL``
        (a hallucination block). The refusal is persisted + post-processed
        exactly like an output-guardrail block: ``guardrail_replacement`` is
        set on the streaming path so the SSE layer swaps the already-streamed
        tokens. The real (low) confidence is preserved on the result for
        metadata/metrics. Default ``False`` preserves the warn-only behavior."""
        self.bias_monitor = bias_monitor or PassthroughBiasMonitor()
        """Bias & fairness monitor (Responsible AI). Runs after the output
        guardrail + hallucination block. When ``block_biased`` is True and
        the monitor flags the answer as biased, the answer is replaced with
        ``BIAS_REFUSAL`` (same swap mechanism). The real ``BiasAssessment``
        is preserved on the result so metadata + metrics capture the
        detection. Default is the passthrough (no-op) monitor."""
        self.drift_monitor = drift_monitor or PassthroughDriftMonitor()
        """Model drift monitor (Responsible AI). Records the final answer's
        features in a rolling window and compares the most recent window to
        the preceding one. A ``DriftReport`` is attached to the result so
        metrics capture drift alerts. Default is the passthrough (no-op)
        monitor."""
        self.block_biased = block_biased
        """When True, a biased answer is replaced with ``BIAS_REFUSAL``.
        Default ``False`` preserves the monitor-only behavior (the
        assessment is still recorded on the result for metrics)."""

    def stream_answer(
        self,
        query: str,
        history: str = "",
        department: str | None = None,
        role: str | None = None,
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

                # Use the PII-scrubbed query for the rest of the flow so no
                # raw PII reaches the classifier, rewriter, retriever, or
                # generator. When no PII was found, ``scrubbed`` equals the
                # original query (scrub_pii is idempotent on clean text).
                if decision.scrubbed:
                    query = decision.scrubbed

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
            docs = self._retrieve(rewritten, classify_label, department, role)
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
            # ── Hallucination block: when enabled, an answer whose
            # groundedness score is below CONFIDENCE_THRESHOLD is replaced
            # with a low-confidence refusal. Mirrors the output-guardrail
            # block: the refusal is persisted + post-processed, and
            # ``guardrail_replacement`` is set on the streaming path so the
            # SSE layer swaps the already-streamed tokens. The real (low)
            # confidence is preserved on the result so metadata + metrics
            # capture the grounding failure. Skipped when the output
            # guardrail already blocked (output block takes precedence).
            low_conf_blocked = False
            if (
                self.block_low_confidence
                and not output_blocked
                and result.low_confidence
            ):
                logger.info(
                    "Orchestrator: low-confidence block — confidence=%.3f",
                    result.confidence,
                )
                low_conf_blocked = True
                result.answer = LOW_CONFIDENCE_REFUSAL
                result.citations = []
            # ── Bias & fairness monitor (Responsible AI): runs after the
            # output guardrail + hallucination block on the final answer.
            # When ``block_biased`` is True and the monitor flags the
            # answer as biased, the answer is replaced with ``BIAS_REFUSAL``
            # (same swap mechanism). The real ``BiasAssessment`` is
            # preserved on the result so metadata + metrics capture the
            # detection. Skipped when the output guardrail or hallucination
            # block already replaced the answer (those take precedence — a
            # canned refusal is not biased content worth blocking).
            bias_blocked = False
            if not output_blocked and not low_conf_blocked:
                bias_span = (
                    self.tracer.start_span(trace, "bias_monitor")
                    if trace is not None else None
                )
                bias_assessment = self.bias_monitor.assess(result.answer)
                result.bias = bias_assessment
                if bias_span is not None:
                    self.tracer.end_span(
                        bias_span,
                        metadata={
                            "biased": bias_assessment.biased,
                            "categories": bias_assessment.categories,
                            "score": bias_assessment.score,
                        },
                    )
                if self.block_biased and bias_assessment.biased:
                    logger.info(
                        "Orchestrator: bias block — categories=%s score=%.2f",
                        bias_assessment.categories,
                        bias_assessment.score,
                    )
                    bias_blocked = True
                    result.answer = BIAS_REFUSAL
                    result.citations = []
                    result.bias_blocked = True
            # ── Model drift monitor (Responsible AI): record the final
            # answer and compare the current rolling window to the baseline.
            # The report is attached to the result so the chat route can
            # record drift metrics and the snapshot can surface alerts.
            drift_span = (
                self.tracer.start_span(trace, "drift_monitor")
                if trace is not None else None
            )
            drift_report = self.drift_monitor.assess(
                result,
                blocked=output_blocked or low_conf_blocked or bias_blocked,
            )
            result.drift = drift_report
            result.drift_alert = drift_report.drift_detected
            if drift_span is not None:
                self.tracer.end_span(
                    drift_span,
                    metadata={
                        "drift_detected": drift_report.drift_detected,
                        "total_samples": drift_report.total_samples,
                        "drifted_features": [
                            name for name, r in drift_report.features.items() if r.drifted
                        ],
                    },
                )
            # Only the streaming path sets ``guardrail_replacement`` — the
            # sensitive buffered path never streamed the harmful tokens, so
            # there is nothing for the frontend to swap (the single delta IS
            # the final, guardrailed answer).
            if (output_blocked or low_conf_blocked or bias_blocked) and not sensitive_buffered:
                result.guardrail_replacement = result.answer
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

    def _retrieve(
        self,
        rewritten: str,
        classify_label: str,
        department: str | None = None,
        role: str | None = None,
    ) -> list[RetrievedDoc]:
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
            if department or role:
                return self.retriever.retrieve(rewritten, department, role)
            return self.retriever.retrieve(rewritten)
        sub_queries = self.query_decomposer.decompose(rewritten)
        if len(sub_queries) <= 1:
            if department or role:
                return self.retriever.retrieve(rewritten, department, role)
            return self.retriever.retrieve(rewritten)
        logger.info(
            "Orchestrator: compare query decomposed into %d sub-queries", len(sub_queries),
        )
        merged: list[RetrievedDoc] = []
        for sq in sub_queries:
            if department or role:
                merged = self._merge_docs(
                    merged,
                    self.retriever.retrieve(sq, department, role),
                )
            else:
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
        department: str | None = None,
        role: str | None = None,
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

                # Use the PII-scrubbed query for the rest of the flow.
                if decision.scrubbed:
                    query = decision.scrubbed

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
            docs = self._retrieve(rewritten, classify_label, department, role)
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
            # ── Hallucination block (non-streaming path) — see stream_answer
            # for the full rationale. The returned ``answer`` is the refusal
            # so callers (eval, history) see the blocked text; the real low
            # confidence is preserved on the result.
            low_conf_blocked = False
            if (
                self.block_low_confidence
                and not output_blocked
                and result.low_confidence
            ):
                logger.info(
                    "Orchestrator: low-confidence block (answer) — confidence=%.3f",
                    result.confidence,
                )
                low_conf_blocked = True
                result.answer = LOW_CONFIDENCE_REFUSAL
                result.citations = []
                answer = LOW_CONFIDENCE_REFUSAL
            # ── Bias & fairness monitor (non-streaming path) — see
            # stream_answer for the full rationale. The returned ``answer``
            # is the refusal so callers (eval, history) see the blocked
            # text; the real ``BiasAssessment`` is preserved on the result.
            bias_blocked = False
            if not output_blocked and not low_conf_blocked:
                bias_span = (
                    self.tracer.start_span(trace, "bias_monitor")
                    if trace is not None else None
                )
                bias_assessment = self.bias_monitor.assess(result.answer)
                result.bias = bias_assessment
                if bias_span is not None:
                    self.tracer.end_span(
                        bias_span,
                        metadata={
                            "biased": bias_assessment.biased,
                            "categories": bias_assessment.categories,
                            "score": bias_assessment.score,
                        },
                    )
                if self.block_biased and bias_assessment.biased:
                    logger.info(
                        "Orchestrator: bias block (answer) — categories=%s score=%.2f",
                        bias_assessment.categories,
                        bias_assessment.score,
                    )
                    bias_blocked = True
                    result.answer = BIAS_REFUSAL
                    result.citations = []
                    result.bias_blocked = True
                    answer = BIAS_REFUSAL
            # ── Model drift monitor (non-streaming path) — see stream_answer
            # for the full rationale. The report is attached to the result.
            drift_report = self.drift_monitor.assess(
                result,
                blocked=output_blocked or low_conf_blocked or bias_blocked,
            )
            result.drift = drift_report
            result.drift_alert = drift_report.drift_detected
            if output_blocked or low_conf_blocked or bias_blocked:
                result.guardrail_replacement = result.answer
            if trace is not None:
                result.trace_id = trace.id
            return answer, result, docs
        finally:
            # ── Step 7: end the Langfuse root span (closes the trace) ───
            if trace is not None:
                self.tracer.end_trace(trace)
