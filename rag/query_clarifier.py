"""Query clarifier — generates a clarification prompt for ambiguous queries.

When the query classifier routes a query to ``ambiguous`` (e.g. "how do I use
validators?" without specifying Pydantic vs FastAPI dependency validators, or
a bare pronoun reference like "how do I configure it?" with no disambiguating
context), the orchestrator short-circuits the RAG flow and instead asks the
user to clarify. The clarifier produces that clarification prompt — ideally
listing the candidate interpretations so the user can pick one (e.g. "Do you
mean Pydantic validators or FastAPI dependency validators?").

The clarifier is optional and injected into the orchestrator. When it returns
``None`` (no clarification could be generated — passthrough clarifier, or the
LLM returned nothing useful), the orchestrator falls back to a generic
clarification prompt so the user is still asked to disambiguate rather than
receiving a potentially wrong answer.

Two implementations:

- ``PassthroughQueryClarifier`` — returns ``None`` (no clarification, no LLM
  call). Zero latency, zero risk. Used when no clarifier is configured.
- ``LLMQueryClarifier`` — sends a short clarify prompt to Ollama and parses
  the response into a clarification question. Falls back to a deterministic
  heuristic clarifier (``_heuristic_clarify``) when Ollama is unavailable,
  returns nothing useful, or errors — so clarification still works without
  the LLM.

All LLM-backed checks share the lazy-client pattern from ``Generator`` /
``Embedder`` / ``LLMQueryRewriter`` / ``LLMQueryDecomposer`` so unit tests
inject mocks and never touch the network.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

CLARIFY_MODEL: str = "llama3.2:3b"
"""Ollama model for query clarification (lightweight, short output)."""

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"

CLARIFY_PROMPT_TEMPLATE: str = """\
The user asked a question about a technical documentation chatbot (FastAPI, \
Pydantic v2, SQLModel) but the question is ambiguous — it could refer to \
multiple concepts. Identify the most likely candidate interpretations and \
write a single concise clarification question that asks the user to pick one. \
Output ONLY the clarification question, no preamble, no numbering, no extra \
text.

User question: {query}

Clarification question:"""

GENERIC_CLARIFICATION: str = (
    "Your question is a bit ambiguous. Could you please clarify which "
    "specific library or concept you're asking about (FastAPI, Pydantic v2, "
    "or SQLModel)?"
)
"""Fallback clarification prompt when no specific clarification could be
generated (LLM unavailable / returned nothing / heuristic found no
candidate). Used by the orchestrator so an ambiguous query always produces a
clarification rather than a potentially wrong answer."""

# Shared-concept nouns that appear across the documented libraries. When a
# short ambiguous query contains one of these without a library qualifier,
# the heuristic clarifier proposes the candidate interpretations.
_SHARED_CONCEPTS: dict[str, list[str]] = {
    "validator": ["Pydantic validators", "FastAPI dependency validators"],
    "validators": ["Pydantic validators", "FastAPI dependency validators"],
    "model": ["Pydantic models", "SQLModel models"],
    "models": ["Pydantic models", "SQLModel models"],
    "field": ["Pydantic fields", "SQLModel columns"],
    "fields": ["Pydantic fields", "SQLModel columns"],
    "cache": ["FastAPI response caching", "client-side caching"],
    "schema": ["Pydantic schemas", "SQLModel table schemas"],
    "schemas": ["Pydantic schemas", "SQLModel table schemas"],
    "session": ["FastAPI session middleware", "SQLModel database sessions"],
    "dependency": ["FastAPI dependencies (Depends)", "Pydantic model dependencies"],
    "dependencies": ["FastAPI dependencies (Depends)", "Pydantic model dependencies"],
}

# Strip leading labels the LLM may add despite the "no preamble" instruction
# (e.g. "Clarification: ...", "Q: ...").
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:clarification|question|q)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _strip_label_prefix(text: str) -> str:
    """Remove a leading ``Clarification:`` / ``Q:`` label from the response."""
    return _LABEL_PREFIX_RE.sub("", text).strip()


def _heuristic_clarify(query: str) -> str | None:
    """Generate a clarification prompt from shared-concept keywords.

    Scans the query for a known shared-concept noun and, if found, produces a
    "Do you mean X or Y?" prompt from the pre-registered candidates. Returns
    ``None`` when no shared concept is detected so the caller can fall back
    to :data:`GENERIC_CLARIFICATION`.
    """
    lowered = query.strip().lower()
    if not lowered:
        return None
    for concept, candidates in _SHARED_CONCEPTS.items():
        if re.search(r"\b" + re.escape(concept) + r"\b", lowered):
            opts = " or ".join(candidates)
            return f"Your question could refer to more than one concept. Do you mean {opts}?"
    return None


class QueryClarifier(Protocol):
    """Protocol for query clarifiers — passthrough, heuristic, or LLM-backed."""

    def clarify(self, query: str, history: str = "") -> str | None:
        """Return a clarification prompt string, or ``None`` when no
        specific clarification could be generated (caller falls back to
        :data:`GENERIC_CLARIFICATION`)."""
        ...


class PassthroughQueryClarifier:
    """Default clarifier — returns ``None`` (no clarification, no LLM call).

    Zero latency, zero risk. Used when no clarifier is configured so the
    orchestrator falls back to :data:`GENERIC_CLARIFICATION` for ambiguous
    queries.
    """

    def clarify(self, query: str, history: str = "") -> str | None:
        return None


class LLMQueryClarifier:
    """LLM-backed query clarifier using Ollama, with a heuristic fallback.

    Sends a short clarify prompt and parses the response into a single
    clarification question (leading label prefixes stripped). Falls back to
    the heuristic clarifier (``_heuristic_clarify``) when the LLM returns
    nothing, whitespace-only text, or errors — so clarification still works
    without the LLM. When the heuristic also finds no shared concept, the
    caller falls back to :data:`GENERIC_CLARIFICATION`.
    """

    def __init__(
        self,
        model: str = CLARIFY_MODEL,
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

    def clarify(self, query: str, history: str = "") -> str | None:
        if not self.use_llm:
            return _heuristic_clarify(query)
        clarification = self._llm_clarify(query)
        if not clarification:
            logger.debug("LLM clarify returned nothing; heuristic fallback")
            return _heuristic_clarify(query)
        return clarification

    def _llm_clarify(self, query: str) -> str | None:
        prompt = CLARIFY_PROMPT_TEMPLATE.format(query=query)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"num_predict": 128, "temperature": 0.0},
            )
            text = response.get("message", {}).get("content", "")
        except Exception:
            logger.warning("Query clarify LLM failed; heuristic fallback", exc_info=True)
            return None
        cleaned = _strip_label_prefix(text)
        if not cleaned:
            return None
        return cleaned

    def close(self) -> None:
        self._client = None
