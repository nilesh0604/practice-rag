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
            # SSE: data: {item}
        else:
            # PostProcessResult — persist, emit [DONE]

This keeps the orchestrator stateless and unit-testable without FastAPI.
"""

from __future__ import annotations

import logging
from typing import Iterator, Union

from rag.context_assembler import ContextAssembler
from rag.generator import Generator
from rag.post_processor import PostProcessor, PostProcessResult
from rag.query_rewriter import PassthroughQueryRewriter, QueryRewriter
from rag.retriever import HybridRetriever
from schemas.documents import RetrievedDoc

logger = logging.getLogger(__name__)

# Type alias for the mixed stream: tokens are str, the final item is a result.
StreamItem = Union[str, PostProcessResult]


class RAGOrchestrator:
    """Coordinates the full RAG flow: rewrite → retrieve → assemble → generate → post-process.

    Each collaborator is injected so the orchestrator is fully mockable in
    unit tests. The default query rewriter is a passthrough (no LLM call).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        context_assembler: ContextAssembler,
        generator: Generator,
        post_processor: PostProcessor,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        self.retriever = retriever
        self.context_assembler = context_assembler
        self.generator = generator
        self.post_processor = post_processor
        self.query_rewriter = query_rewriter or PassthroughQueryRewriter()

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
        rewritten = self.query_rewriter.rewrite(query, history)
        logger.debug("Orchestrator: query=%r → rewritten=%r", query, rewritten)

        docs = self.retriever.retrieve(rewritten)
        context = self.context_assembler.assemble(docs)

        answer = ""
        for token in self.generator.stream(rewritten, context, history):
            answer += token
            yield token

        result = self.post_processor.post_process(answer, docs)
        yield result

    # ── convenience: non-streaming single-shot ─────────────────────────

    def answer(
        self,
        query: str,
        history: str = "",
    ) -> tuple[str, PostProcessResult, list[RetrievedDoc]]:
        """Non-streaming convenience method — collects all tokens and returns the result.

        Useful for tests and for the (future) non-streaming history endpoint.
        Returns ``(answer_text, post_process_result, retrieved_docs)``.
        """
        rewritten = self.query_rewriter.rewrite(query, history)
        docs = self.retriever.retrieve(rewritten)
        context = self.context_assembler.assemble(docs)

        answer = "".join(self.generator.stream(rewritten, context, history))
        result = self.post_processor.post_process(answer, docs)
        return answer, result, docs
