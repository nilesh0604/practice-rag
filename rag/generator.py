"""Generator — Ollama streaming LLM client for answer generation.

Wraps the Ollama ``chat`` API with the architecture doc's generation settings
and system prompt. Yields answer tokens one at a time so the FastAPI layer can
stream them as SSE events.

Generation settings (from the architecture doc's Component Deep-Dive):

    temperature  = 0.1    (low creativity, maximize grounding)
    num_ctx      = 8192   (5 chunks × 512 tokens + system prompt + history)
    top_p        = 0.9    (slight diversity without hallucination)
    num_predict  = 512    (concise answers, not too long for a local model)

The model defaults to ``llama3.2:3b`` (the Phase 0 dev substitute, per
deviation D2). The architecture doc's production target is ``llama3.1:8b``;
swapping is a one-line change (``Generator(model="llama3.1:8b")``).

The system prompt template is the one documented in the architecture doc,
with ``{context}`` and ``{history}`` placeholders filled by the caller.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from api.observability import CircuitBreaker

logger = logging.getLogger(__name__)

# ── generation constants ───────────────────────────────────────────────

GENERATION_MODEL: str = "llama3.2:3b"
"""Ollama generation model. Dev substitute for the doc's llama3.1:8b (D2)."""

GENERATION_MODEL_TARGET: str = "llama3.1:8b"
"""The architecture doc's documented production generation model."""

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
"""Default Ollama base URL (host Ollama, not containerized — see D1)."""

TEMPERATURE: float = 0.1
NUM_CTX: int = 8192
TOP_P: float = 0.9
NUM_PREDICT: int = 512

# ── system prompt template (from the architecture doc) ─────────────────

SYSTEM_PROMPT_TEMPLATE: str = """\
You are a technical documentation assistant for FastAPI, Pydantic v2, and SQLModel.
Use ONLY the provided context to answer.

RULES:
1. If the context does not contain the answer, say: "I don't have enough information to answer that."
2. Cite the source title in [Source: title] format.
3. Keep answers concise — 2-4 sentences for simple questions.
4. Never fabricate API names, version numbers, or code signatures.
5. You may use the CONVERSATION HISTORY to synthesize a follow-up answer \
(such as a summary of prior turns) when the user references earlier answers. \
Do not invent facts not present in the context or history; if history is needed \
but empty, say: "I don't have enough information to answer that."

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}\
"""


def build_system_prompt(context: str, history: str = "") -> str:
    """Fill the system prompt template with context and conversation history."""
    return SYSTEM_PROMPT_TEMPLATE.format(context=context, history=history)


class Generator:
    """Streaming Ollama chat client for RAG answer generation.

    The Ollama client is lazily constructed (like the ``Embedder``) so unit
    tests can inject a mock without hitting the network.
    """

    def __init__(
        self,
        model: str = GENERATION_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = TEMPERATURE,
        num_ctx: int = NUM_CTX,
        top_p: float = TOP_P,
        num_predict: int = NUM_PREDICT,
        circuit_breaker: "CircuitBreaker | None" = None,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.top_p = top_p
        self.num_predict = num_predict
        self.circuit_breaker = circuit_breaker
        self._client = None

    @property
    def client(self):
        """Lazily import and construct the Ollama client."""
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.ollama_url)
        return self._client

    def stream(
        self,
        query: str,
        context: str,
        history: str = "",
    ) -> Iterator[str]:
        """Stream answer tokens for a query given assembled context + history.

        Yields content strings (tokens). Empty content chunks (e.g. tool-call
        metadata) are skipped so the caller only receives visible text.

        When a ``circuit_breaker`` is configured (Step 7), the initial Ollama
        ``chat`` call is routed through it: if the circuit is open
        ``CircuitOpenError`` is raised before any network call, and a
        connection failure increments the breaker's failure count.
        """
        system_prompt = build_system_prompt(context, history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
        }
        logger.debug("Streaming generation for query %r (model=%s)", query, self.model)

        def _do_chat():
            return self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=options,
            )

        if self.circuit_breaker is not None:
            chat_iter = self.circuit_breaker.call(_do_chat)
        else:
            chat_iter = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=options,
            )
        for chunk in chat_iter:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def close(self) -> None:
        """Release the Ollama client if one was constructed."""
        self._client = None
