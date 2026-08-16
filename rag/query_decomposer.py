"""Query decomposer — splits complex/multi-part queries into sub-queries.

When the query classifier routes a query to ``compare`` (e.g. "Compare FastAPI
and Flask for building REST APIs"), a single retrieval pass may miss chunks
that are strongly relevant to only one of the compared subjects. Decomposing
the query into one sub-query per subject and running retrieval for each
broadens the candidate pool so the generator sees balanced context for both
sides of the comparison.

The decomposer is optional and injected into the orchestrator. When it
returns a single sub-query (the original query unchanged), the orchestrator's
retrieval flow is identical to the non-decomposed path — so simple
``documentation`` queries pay no extra cost. Decomposition only runs when the
classifier labels the query ``compare``.

Two implementations:

- ``PassthroughQueryDecomposer`` — returns ``[query]`` (no split, no LLM
  call). Zero latency, zero risk. Used when no decomposer is configured.
- ``LLMQueryDecomposer`` — sends a short split prompt to Ollama and parses
  the response into one sub-query per non-empty line. Falls back to a
  deterministic heuristic splitter (``_heuristic_decompose``) when Ollama is
  unavailable, returns nothing useful, or errors — so decomposition still
  works without the LLM.

All LLM-backed checks share the lazy-client pattern from ``Generator`` /
``Embedder`` / ``LLMQueryRewriter`` so unit tests inject mocks and never
touch the network.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

DECOMPOSE_MODEL: str = "llama3.2:3b"
"""Ollama model for query decomposition (lightweight, short output)."""

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"

DECOMPOSE_PROMPT_TEMPLATE: str = """\
Split the following user question into one standalone sub-question per subject \
being compared or per distinct part. Output ONLY the sub-questions, one per \
line, with no numbering, prefixes, or extra text. If the question is not a \
comparison or multi-part question, output it unchanged on a single line.

User question: {query}

Sub-questions:"""

# Heuristic split connectors (used when Ollama is unavailable). Ordered so
# stronger comparison markers are tried before the generic "and". Each
# splitter takes the first match only (maxsplit=1) so a two-way comparison
# yields two sub-queries; multi-way comparisons are handled by the LLM path.
_HEURISTIC_SPLITTERS: list[re.Pattern[str]] = [
    re.compile(r"\s+vs\.?\s+", re.IGNORECASE),
    re.compile(r"\s+versus\s+", re.IGNORECASE),
    re.compile(r"\s+compared\s+to\s+", re.IGNORECASE),
    re.compile(r"\s+difference\s+between\s+", re.IGNORECASE),
    re.compile(r"\s+and\s+", re.IGNORECASE),
]

# Strip leading numbering / bullet prefixes the LLM may add despite the
# "no numbering" instruction (e.g. "1. FastAPI ...", "- Flask ...").
_NUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s+", re.IGNORECASE)


def _strip_prefix(line: str) -> str:
    """Remove a leading numbering/bullet prefix from a sub-query line."""
    return _NUM_PREFIX_RE.sub("", line)


def _heuristic_decompose(query: str) -> list[str]:
    """Split a comparison query on the first matching connector.

    Returns ``[query]`` (no split) when no connector is found, or when the
    split produces an empty/whitespace-only part. Only the first connector
    match is used (``maxsplit=1``) so a two-way comparison yields two
    sub-queries.
    """
    stripped = query.strip()
    if not stripped:
        return [query]
    for pattern in _HEURISTIC_SPLITTERS:
        parts = pattern.split(stripped, maxsplit=1)
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            if left and right:
                return [left, right]
    return [query]


class QueryDecomposer(Protocol):
    """Protocol for query decomposers — passthrough, heuristic, or LLM-backed."""

    def decompose(self, query: str, history: str = "") -> list[str]:
        """Return a list of sub-queries (always ≥ 1 element)."""
        ...


class PassthroughQueryDecomposer:
    """Default decomposer — returns ``[query]`` (no split, no LLM call).

    Zero latency, zero risk. Used when no decomposer is configured so the
    orchestrator's retrieval flow is unchanged.
    """

    def decompose(self, query: str, history: str = "") -> list[str]:
        return [query]


class LLMQueryDecomposer:
    """LLM-backed query decomposer using Ollama, with a heuristic fallback.

    Sends a short split prompt and parses the response into one sub-query per
    non-empty line (numbering/bullet prefixes stripped). Falls back to the
    heuristic splitter (``_heuristic_decompose``) when the LLM returns
    nothing, a single line equal to the original (no decomposition), or
    errors — so decomposition still works without the LLM.
    """

    def __init__(
        self,
        model: str = DECOMPOSE_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        use_llm: bool = True,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.use_llm = use_llm
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.ollama_url)
        return self._client

    def decompose(self, query: str, history: str = "") -> list[str]:
        if not self.use_llm:
            return _heuristic_decompose(query)
        sub_queries = self._llm_decompose(query)
        if not sub_queries:
            logger.debug("LLM decompose returned nothing; heuristic fallback")
            return _heuristic_decompose(query)
        return sub_queries

    def _llm_decompose(self, query: str) -> list[str]:
        prompt = DECOMPOSE_PROMPT_TEMPLATE.format(query=query)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"num_predict": 128, "temperature": 0.0},
            )
            text = response.get("message", {}).get("content", "")
        except Exception:
            logger.warning("Query decompose LLM failed; heuristic fallback", exc_info=True)
            return []
        sub_queries = [_strip_prefix(line.strip()) for line in text.splitlines()]
        sub_queries = [sq for sq in sub_queries if sq]
        # If the LLM returned a single line equal to the original, treat it
        # as no decomposition and let the heuristic fallback try a split.
        if len(sub_queries) == 1 and sub_queries[0].lower() == query.strip().lower():
            return []
        return sub_queries

    def close(self) -> None:
        self._client = None
