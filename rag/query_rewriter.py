"""Query rewriter — optional query reformulation before retrieval.

The architecture doc's orchestrator pseudocode includes a query-rewriting step
(step 3, marked optional) that reformulates the user's question using
conversation history before embedding. For a local 3B model this adds latency
and can introduce errors, so the default implementation is a **passthrough**
that returns the query unchanged.

An optional LLM-backed rewriter (``LLMQueryRewriter``) is provided for when a
stronger model is available. It sends a short reformulation prompt to Ollama
and returns the rewritten query. ``LLMQueryRewriter`` is wired into the live
app via ``api.deps.get_orchestrator()`` and runs on every query — but a
latency-skip heuristic (``_needs_rewrite``) short-circuits the LLM round-trip
for self-contained queries that don't need coreference resolution (no
anaphoric pronouns, or no conversation history to resolve against).
"""

from __future__ import annotations

import logging
import re
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

# Anaphoric pronouns / phrase markers that signal the query depends on
# conversation history for coreference resolution. Matched as whole words
# (case-insensitive) so substrings like "it" in "title" do NOT match.
# Conservative — false negatives (rewriting when not needed) only waste
# ~1-3s of latency; false positives (skipping when rewriting was needed)
# hurt retrieval quality. So the list errs toward inclusion.
_ANAPHORA_MARKERS: re.Pattern[str] = re.compile(
    r"\b(?:"
    r"it|its|itself|"
    r"this|these|"
    r"that|those|"
    r"they|them|their|theirs|themselves|"
    r"he|him|his|himself|"
    r"she|her|hers|herself|"
    r"the above|the aforementioned|"
    r"the former|the latter"
    r")\b",
    re.IGNORECASE,
)


def _needs_rewrite(query: str, history: str) -> bool:
    """Heuristic: does this query need LLM coreference resolution?

    Returns ``False`` (skip the rewrite round-trip) when:

    - ``history`` is empty/whitespace — no context to resolve against, or
    - the query contains no anaphoric pronouns / phrase markers.

    Returns ``True`` (proceed with the LLM rewrite) only when there is
    conversation history AND the query contains a marker that likely
    refers back to it.
    """
    if not history or not history.strip():
        return False
    return bool(_ANAPHORA_MARKERS.search(query))


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

    A latency-skip heuristic (``_needs_rewrite``) short-circuits the LLM
    round-trip for self-contained queries — those with no anaphoric pronouns
    or no conversation history to resolve against — returning the original
    query immediately without calling Ollama.
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
        # Latency-skip heuristic: avoid the ~1-3s Ollama round-trip for
        # self-contained queries that don't need coreference resolution.
        if not _needs_rewrite(query, history):
            logger.debug("Skipping rewrite (no anaphora or no history): %r", query)
            return query
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
