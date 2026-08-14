"""Query rewriter — optional query reformulation before retrieval.

The architecture doc's orchestrator pseudocode includes a query-rewriting step
(step 3, marked optional) that reformulates the user's question using
conversation history before embedding. For a local 3B model this adds latency
and can introduce errors, so the default implementation is a **passthrough**
that returns the query unchanged.

An optional LLM-backed rewriter (``LLMQueryRewriter``) is provided for when a
stronger model is available. It sends a short reformulation prompt to Ollama
and returns the rewritten query. This is wired but not used by the default
orchestrator — enable it by passing an ``LLMQueryRewriter`` instance to
``RAGOrchestrator``.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

REWRITE_MODEL: str = "llama3.2:3b"
"""Model for LLM-based query rewriting (lightweight, 10-token-ish output)."""

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"

REWRITE_PROMPT_TEMPLATE: str = """\
Rewrite the following user question as a standalone search query for a \
documentation search engine. Use the conversation history only for context \
(coreference resolution). Output ONLY the rewritten query, nothing else.

Conversation history:
{history}

User question: {query}

Standalone query:"""


class QueryRewriter(Protocol):
    """Protocol for query rewriters — passthrough or LLM-backed."""

    def rewrite(self, query: str, history: str = "") -> str:
        """Return a reformulated query for retrieval."""
        ...


class PassthroughQueryRewriter:
    """Default rewriter — returns the query unchanged (no LLM call).

    Zero latency, zero risk of introducing errors. Used by the default
    orchestrator. Swap for ``LLMQueryRewriter`` when a stronger model is
    available and query rewriting is worth the extra round-trip.
    """

    def rewrite(self, query: str, history: str = "") -> str:
        return query


class LLMQueryRewriter:
    """LLM-backed query rewriter using Ollama.

    Sends a short reformulation prompt and returns the first non-empty line
    of the response as the rewritten query. Falls back to the original query
    if the LLM returns nothing or errors.
    """

    def __init__(
        self,
        model: str = REWRITE_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.ollama_url)
        return self._client

    def rewrite(self, query: str, history: str = "") -> str:
        prompt = REWRITE_PROMPT_TEMPLATE.format(history=history, query=query)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"num_predict": 64, "temperature": 0.0},
            )
            text = response.get("message", {}).get("content", "").strip()
            rewritten = text.splitlines()[0].strip() if text else ""
            if rewritten:
                logger.debug("Rewrote %r → %r", query, rewritten)
                return rewritten
        except Exception:
            logger.warning("Query rewrite failed; falling back to original query", exc_info=True)
        return query

    def close(self) -> None:
        self._client = None
