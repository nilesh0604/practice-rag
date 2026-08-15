"""NVIDIA NIM reranker (Phase 4 of the NIM integration plan — net-new capability).

Adds a reranking stage between retrieval and generation. The reranker takes the
query + the retrieved candidate chunks and reorders them by relevance using the
NVIDIA NIM ``llama-nemotron-rerank-1b-v2`` model — a purpose-built cross-encoder
that scores query/passage pairs far more accurately than the bi-encoder dense
similarity used at retrieval time.

This is a **net-new capability** (the project has no reranker today), not a
swap. The standard RAG quality lever is: retrieve a larger candidate pool
(e.g. top-20) → rerank to top-5 → feed to the generator. The reranker's
cross-encoder relevance signal is stronger than RRF-fused bi-encoder scores,
so the top-5 after reranking is typically more relevant than the raw top-5.

Design notes (from ``docs/NVIDIA_NIM_INTEGRATION_PLAN.md`` §2.6 + §3.4):

- NIM is **opt-in** (``NIM_ENABLED=true``). A second flag
  ``NIM_RERANK_ENABLED`` (default ``true`` when ``NIM_ENABLED`` is true) allows
  disabling just the reranker while keeping the generator/guardrails on NIM —
  useful for the Phase 4 A/B comparison (answer quality with vs. without
  reranking).
- The NIM reranking API is **not** the OpenAI-compatible ``/chat/completions``
  or ``/embeddings`` endpoint. It is a dedicated ranking endpoint at
  ``https://ai.api.nvidia.com/v1/retrieval/{model}/reranking`` (note the
  different host — ``ai.api.nvidia.com`` vs. ``integrate.api.nvidia.com``).
  The request shape is ``{"model", "query": {"text"}, "passages":
  [{"text"}, ...], "truncate": "END"}`` and the response is
  ``{"rankings": [{"index", "logit"}, ...]}`` sorted by relevance (highest
  logit first). The ``index`` refers to the position in the original
  ``passages`` list.
- The reranker gets its **own** ``CircuitBreaker`` (separate from the
  generator's, guardrails', and embedder's NIM breakers and the Ollama
  breaker) so a reranker outage does not trip other breakers.
- **No fallback model** — there is no local reranker. On any failure (circuit
  open, 429, 404, timeout, connection error, malformed response), the reranker
  degrades gracefully by returning the original retrieved order truncated to
  ``top_n``. This is the status quo (no reranking) — a failed rerank degrades
  to the current behavior, which is acceptable because reranking is a quality
  enhancement, not a correctness requirement.
- The reranker **reorders** docs but does **not** replace their RRF retrieval
  scores — ``RetrievedDoc.score`` is kept as the fused retrieval score (the
  reranker's logits are unbounded and not comparable to the [0, 1] RRF score).
  The post-processor's groundedness score is computed independently from the
  doc content, so it is unaffected by the reorder.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from schemas.documents import RetrievedDoc

if TYPE_CHECKING:
    from api.observability import CircuitBreaker

logger = logging.getLogger(__name__)

# ── NIM constants ──────────────────────────────────────────────────────

NIM_RERANK_BASE_URL: str = "https://ai.api.nvidia.com/v1"
"""NVIDIA NIM reranking API base URL (free tier).

Different from the chat/embeddings base URL (``https://integrate.api.nvidia.com
/v1``) — the reranking service lives on ``ai.api.nvidia.com``. The full
endpoint is ``{base_url}/retrieval/{model}/reranking``."""

NIM_RERANK_MODEL: str = "nvidia/llama-nemotron-rerank-1b-v2"
"""NIM reranking model (plan §2.6).

Purpose-built cross-encoder reranker (1B parameter class). Scores query/
passage pairs for relevance — a stronger signal than the bi-encoder dense
similarity used at retrieval time. Override with ``NIM_RERANK_MODEL``."""

NIM_RERANK_TIMEOUT: float = 30.0
"""Per-request timeout for NIM reranking calls (seconds).

Shorter than the generator's 60s because reranking is a single non-streaming
call (like the guardrail calls), though it processes multiple passages."""

DEFAULT_RERANK_TOP_N: int = 5
"""Number of docs to keep after reranking. Matches the context assembler's
``max_chunks=5`` and the retriever's default ``top_k=5``."""

DEFAULT_RERANK_CANDIDATE_K: int = 20
"""Number of candidate docs to retrieve before reranking. Matches the
retriever's ``PREFETCH_LIMIT=20`` — fetch a larger pool, rerank down to
``top_n``. Override with ``NIM_RERANK_CANDIDATE_K``."""

TRUNCATE_END: str = "END"
"""Truncate passages that exceed the model's context window (default API
behavior). The alternative ``"NONE"`` returns an error for over-long inputs."""


# ── exceptions ─────────────────────────────────────────────────────────


class NIMRerankError(Exception):
    """Raised when a NIM reranking API call fails.

    Caught internally by ``rerank()`` to trigger graceful degradation (return
    the original retrieved order). Never propagates to the caller — a failed
    rerank degrades to no-reranking, not an error.
    """


# ── NIM reranker ───────────────────────────────────────────────────────


class NIMReranker:
    """NVIDIA NIM cross-encoder reranker for RAG retrieval.

    Takes a query + a list of ``RetrievedDoc`` candidates, sends the query and
    each doc's content to the NIM reranking endpoint, and returns the docs
    reordered by the model's relevance scores (top ``top_n``). The httpx
    client is lazily constructed (like the generator/guardrails/embedder) so
    unit tests inject a mock without hitting the network.

    Owns a dedicated ``CircuitBreaker`` (separate from all other breakers).
    When the circuit is open, ``rerank()`` skips the network call and returns
    the original order (graceful degradation).

    On **any** failure (circuit open, 429, 404, timeout, connection error,
    malformed response), ``rerank()`` returns the original docs truncated to
    ``top_n`` — the status quo (no reranking). The exception is logged but
    never raised, because reranking is a quality enhancement, not a
    correctness requirement.
    """

    def __init__(
        self,
        model: str = NIM_RERANK_MODEL,
        api_key: str | None = None,
        base_url: str = NIM_RERANK_BASE_URL,
        timeout: float = NIM_RERANK_TIMEOUT,
        top_n: int = DEFAULT_RERANK_TOP_N,
        truncate: str = TRUNCATE_END,
        circuit_breaker: "CircuitBreaker | None" = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.top_n = top_n
        self.truncate = truncate
        self.circuit_breaker = circuit_breaker
        self._client = None

    @property
    def client(self):
        """Lazily construct the httpx client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _endpoint_url(self) -> str:
        """Build the model-specific reranking endpoint URL.

        The NIM reranking API embeds the model name in the URL path:
        ``{base_url}/retrieval/{model}/reranking``.
        """
        return f"{self.base_url}/retrieval/{self.model}/reranking"

    def _build_payload(self, query: str, docs: list[RetrievedDoc]) -> dict:
        return {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": doc.content} for doc in docs],
            "truncate": self.truncate,
        }

    def _do_request(self, query: str, docs: list[RetrievedDoc]) -> list[dict]:
        """POST to the NIM reranking endpoint and return the rankings list.

        Raises ``NIMRerankError`` for any failure (429, 404, connection error,
        timeout, malformed response).
        """
        import httpx

        url = self._endpoint_url()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = self._build_payload(query, docs)
        try:
            resp = self.client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                raise NIMRerankError(
                    f"NIM rerank rate limited (429) for model {self.model}",
                )
            if resp.status_code == 404:
                raise NIMRerankError(
                    f"NIM rerank model not found (404): {self.model}",
                )
            if resp.status_code >= 400:
                body = resp.text[:200]
                raise NIMRerankError(
                    f"NIM rerank HTTP {resp.status_code}: {body}",
                )
            data = resp.json()
            rankings = data.get("rankings", [])
            if not rankings:
                raise NIMRerankError("NIM rerank returned no rankings")
            # Validate that every index is in range; a malformed response
            # could reference passages we did not send.
            n = len(docs)
            for entry in rankings:
                idx = entry.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= n:
                    raise NIMRerankError(
                        f"NIM rerank returned out-of-range index {idx} "
                        f"for {n} passages",
                    )
            return rankings
        except httpx.TimeoutException as exc:
            raise NIMRerankError(f"NIM rerank timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise NIMRerankError(
                f"NIM rerank connection error: {exc}",
            ) from exc
        except NIMRerankError:
            raise
        except Exception as exc:  # noqa: BLE001 — map unknown errors
            raise NIMRerankError(f"NIM rerank unexpected error: {exc}") from exc

    def rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """Rerank retrieved docs by NIM cross-encoder relevance → top ``top_n``.

        Args:
            query: The (rewritten) user query.
            docs: Candidate docs from the retriever (typically ~20).

        Returns:
            Docs reordered by the reranker's relevance scores, truncated to
            ``top_n``. On any failure (circuit open, 429, 404, timeout,
            malformed response), returns the original docs truncated to
            ``top_n`` — graceful degradation to no-reranking.

        The original RRF retrieval scores are preserved (the reranker
        reorders but does not rescore — its logits are unbounded and not
        comparable to the [0, 1] RRF score).
        """
        if not docs:
            return []
        # If we already have fewer than top_n candidates, reranking is
        # pointless — return as-is.
        if len(docs) <= self.top_n:
            return list(docs)

        try:
            if self.circuit_breaker is not None:
                rankings = self.circuit_breaker.call(
                    self._do_request, query, docs,
                )
            else:
                rankings = self._do_request(query, docs)
        except Exception as exc:  # noqa: BLE001 — NIMRerankError / CircuitOpenError
            logger.info(
                "NIM rerank failed — degrading to original order: %s", exc,
            )
            return list(docs[: self.top_n])

        # Reorder docs by the reranker's ranking (rankings are sorted by
        # relevance — highest logit first).
        reordered = [docs[entry["index"]] for entry in rankings]
        result = reordered[: self.top_n]
        logger.info(
            "NIM rerank %r → %d candidates, reranked to top %d",
            query,
            len(docs),
            len(result),
        )
        return result

    def close(self) -> None:
        """Release the httpx client if one was constructed."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None


# ── factory used by api/deps.py ────────────────────────────────────────


def build_reranker() -> "NIMReranker | None":
    """Build the reranker based on env vars.

    - ``NIM_ENABLED=true`` **and** ``NIM_RERANK_ENABLED`` not ``false``
      (case-insensitive) → ``NIMReranker`` with a dedicated circuit breaker.
      ``NIM_RERANK_TOP_N`` overrides the number of docs to keep (default 5).
    - Otherwise → ``None`` (no reranking — the current behavior).

    Returns ``None`` (not a no-op reranker) so the orchestrator can skip the
    rerank step entirely with a simple ``is not None`` check — no extra
    hosted call, no latency, no rate-limit consumption when disabled.
    """
    nim_enabled = os.getenv("NIM_ENABLED", "").strip().lower() in ("true", "1", "yes")
    if not nim_enabled:
        return None

    rerank_enabled = os.getenv("NIM_RERANK_ENABLED", "true").strip().lower()
    if rerank_enabled in ("false", "0", "no"):
        return None

    from api.observability import CircuitBreaker

    top_n = int(os.getenv("NIM_RERANK_TOP_N", str(DEFAULT_RERANK_TOP_N)))
    nim_breaker = CircuitBreaker(threshold=3, timeout=30.0)
    logger.info("NIM reranker enabled — cross-encoder reranking active")
    return NIMReranker(top_n=top_n, circuit_breaker=nim_breaker)


def get_rerank_candidate_k() -> int:
    """Return the retriever ``top_k`` to use when reranking is active.

    When a reranker is enabled, the retriever fetches a larger candidate pool
    (default 20) so the reranker has enough signal to reorder meaningfully.
    Override with ``NIM_RERANK_CANDIDATE_K``. Falls back to the standard
    ``DEFAULT_RERANK_CANDIDATE_K`` (20) which matches the retriever's
    ``PREFETCH_LIMIT``.
    """
    return int(os.getenv("NIM_RERANK_CANDIDATE_K", str(DEFAULT_RERANK_CANDIDATE_K)))
