"""NVIDIA NIM generator + fallback generator (Phase 1 of the NIM integration plan).

Provides two classes that implement the same ``stream(query, context, history)``
interface as ``rag.generator.Generator`` so the orchestrator can use them
interchangeably:

- **``NIMGenerator``** — a streaming OpenAI-compatible chat client for the
  NVIDIA NIM free tier (``https://integrate.api.nvidia.com/v1``). Uses
  ``httpx`` (already a project dependency) to POST to ``/chat/completions``
  with ``stream=True`` and parses the Server-Sent Events response. The
  system prompt template and generation settings are shared with the Ollama
  generator so answers are directly comparable.

- **``FallbackGenerator``** — wraps a *primary* generator and a *fallback*
  generator with a ``CircuitBreaker``. When the primary fails (rate limit,
  model retired, network error, or the circuit is open), it transparently
  falls back to the secondary. If the secondary also fails, it yields a
  canned refusal so the caller never receives a raw exception.

Design notes (from ``docs/NVIDIA_NIM_INTEGRATION_PLAN.md``):

- NIM is **opt-in** (``NIM_ENABLED=true``). When disabled, the orchestrator
  uses the plain Ollama ``Generator`` — no change to the default path.
- The NIM generator gets its **own** ``CircuitBreaker`` (separate from the
  Ollama breaker) so an NIM outage does not trip the Ollama breaker.
- NIM-specific HTTP errors (429 rate limit, 404 model retired) are mapped
  to fallback, **not** retried — retrying a 429 against a shared global
  limit would waste the budget.
- If the primary has already started yielding tokens and then fails
  mid-stream, the exception propagates (we do not attempt a mid-stream
  fallback, which would produce a confusing duplicate answer). Fallback
  only happens when the primary fails **before** yielding any tokens.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Iterator

from rag.generator import (
    NUM_PREDICT,
    TEMPERATURE,
    TOP_P,
    Generator,
    build_system_prompt,
)

if TYPE_CHECKING:
    from api.observability import CircuitBreaker

logger = logging.getLogger(__name__)

# ── NIM constants ──────────────────────────────────────────────────────

NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
"""NVIDIA NIM OpenAI-compatible API base URL (free tier)."""

NIM_GENERATION_MODEL: str = "meta/llama-3.1-8b-instruct"
"""Default NIM generation model.

Originally ``moonshotai/kimi-k2.6`` per integration plan §2.1, but that
model was retired/undeployed on the NIM free tier (returns 404 on
``/chat/completions`` as of 2026-08-15 despite still being listed in
``GET /v1/models``). Switched to ``meta/llama-3.1-8b-instruct`` — the
plan's own production target — which is reliably available on the free
tier (~0.35–1s latency) and matches the Ollama ``llama3.1:8b`` for
apples-to-apples comparison. Override with ``NIM_MODEL`` if needed.
"""

NIM_REQUEST_TIMEOUT: float = 60.0
"""Per-request timeout for NIM chat completions (seconds)."""

# Canned refusal when both primary and fallback fail.
_REFUSAL: str = "I don't have enough information to answer that."


# ── exceptions ─────────────────────────────────────────────────────────


class NIMError(Exception):
    """Raised when a NIM API call fails in a way that should trigger fallback."""


# ── NIM generator ──────────────────────────────────────────────────────


class NIMGenerator:
    """Streaming OpenAI-compatible chat client for NVIDIA NIM.

    The httpx client is lazily constructed (like ``Generator``'s Ollama
    client) so unit tests inject a mock without hitting the network.

    Note: this class does **not** own a ``CircuitBreaker``. Because
    ``_stream_request`` is a generator function, calling it returns a
    generator object without executing any code — so a breaker wrapping
    the call would never see HTTP failures (they happen during iteration).
    Instead, the ``FallbackGenerator`` owns the breaker and uses it to
    decide whether to even attempt the primary, and to record failures
    when the primary fails before yielding any tokens.
    """

    def __init__(
        self,
        model: str = NIM_GENERATION_MODEL,
        api_key: str | None = None,
        base_url: str = NIM_BASE_URL,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        max_tokens: int = NUM_PREDICT,
        timeout: float = NIM_REQUEST_TIMEOUT,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        """Lazily construct the httpx streaming client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _build_payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

    def _stream_request(self, messages: list[dict]) -> Iterator[str]:
        """POST to NIM ``/chat/completions`` and yield content deltas.

        Raises ``NIMError`` for any failure that should trigger fallback
        (429, 404, connection error, timeout).
        """
        import httpx

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = self._build_payload(messages)
        try:
            with self.client.stream(
                "POST", url, json=payload, headers=headers,
            ) as resp:
                if resp.status_code == 429:
                    raise NIMError(f"NIM rate limited (429) for model {self.model}")
                if resp.status_code == 404:
                    raise NIMError(f"NIM model not found (404): {self.model}")
                if resp.status_code >= 400:
                    body = resp.read().decode(errors="replace")[:200]
                    raise NIMError(
                        f"NIM HTTP {resp.status_code}: {body}",
                    )
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
        except httpx.TimeoutException as exc:
            raise NIMError(f"NIM request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise NIMError(f"NIM connection error: {exc}") from exc
        except NIMError:
            raise
        except Exception as exc:  # noqa: BLE001 — map unknown errors to fallback
            raise NIMError(f"NIM unexpected error: {exc}") from exc

    def stream(
        self,
        query: str,
        context: str,
        history: str = "",
    ) -> Iterator[str]:
        """Stream answer tokens from NIM for a query given context + history.

        Yields content strings (tokens). Raises ``NIMError`` for any
        failure (429, 404, timeout, connection error) that the
        ``FallbackGenerator`` catches to trigger fallback.
        """
        system_prompt = build_system_prompt(context, history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        logger.debug("Streaming NIM generation for query %r (model=%s)", query, self.model)
        yield from self._stream_request(messages)

    def close(self) -> None:
        """Release the httpx client if one was constructed."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None


# ── fallback generator ─────────────────────────────────────────────────


class FallbackGenerator:
    """Primary → fallback generator with circuit-breaker-driven failover.

    Wraps a *primary* generator (e.g. ``NIMGenerator``) and a *fallback*
    generator (e.g. Ollama ``Generator``). When the primary fails before
    yielding any tokens — due to ``CircuitOpenError``, ``NIMError``, or any
    other exception — the fallback generator is invoked instead. If the
    fallback also fails, a canned refusal string is yielded so the caller
    never receives a raw exception.

    If the primary has already started yielding tokens and then fails
    mid-stream, the exception propagates (a mid-stream fallback would
    produce a confusing duplicate/partial answer).
    """

    def __init__(
        self,
        primary,
        fallback,
        circuit_breaker: "CircuitBreaker | None" = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.circuit_breaker = circuit_breaker

    def stream(
        self,
        query: str,
        context: str,
        history: str = "",
    ) -> Iterator[str]:
        """Stream from the primary, falling back on failure before first token.

        If a ``circuit_breaker`` is configured and the circuit is open,
        the primary is skipped entirely and the fallback is used
        immediately (no network round-trip wasted on a known-bad
        dependency). When the primary fails before yielding any tokens,
        the failure is recorded on the breaker.
        """
        # ── check circuit breaker before attempting primary ──────────
        if self.circuit_breaker is not None and self.circuit_breaker.is_open():
            logger.info("Circuit open — skipping primary, using fallback")
        else:
            yielded_any = False
            try:
                for token in self.primary.stream(query, context, history):
                    yielded_any = True
                    yield token
                return  # primary succeeded — done
            except Exception as exc:
                if yielded_any:
                    # Primary started streaming then failed mid-stream — do not
                    # attempt a fallback (would produce a confusing answer).
                    logger.warning(
                        "Primary generator failed mid-stream after yielding "
                        "tokens — propagating: %s", exc,
                    )
                    raise
                logger.info(
                    "Primary generator failed before first token — falling "
                    "back: %s", exc,
                )
                if self.circuit_breaker is not None:
                    self.circuit_breaker.record_failure()

        # ── fallback phase ────────────────────────────────────────────
        try:
            for token in self.fallback.stream(query, context, history):
                yield token
        except Exception as exc:
            logger.error("Fallback generator also failed: %s", exc)
            yield _REFUSAL

    def close(self) -> None:
        """Release both generators' clients."""
        for gen in (self.primary, self.fallback):
            close = getattr(gen, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass


# ── factory used by api/deps.py ────────────────────────────────────────


def build_generator(circuit_breaker: "CircuitBreaker | None" = None) -> object:
    """Build the generator stack based on the ``NIM_ENABLED`` env var.

    - ``NIM_ENABLED=true`` (case-insensitive) → ``FallbackGenerator``
      with ``NIMGenerator`` primary + Ollama ``Generator`` fallback.
      The NIM generator gets its own ``CircuitBreaker`` (separate from
      the Ollama breaker) so an NIM outage does not trip Ollama.
    - Otherwise (default) → plain Ollama ``Generator`` with the shared
      circuit breaker. No change to the existing default path.
    """
    nim_enabled = os.getenv("NIM_ENABLED", "").strip().lower() in ("true", "1", "yes")
    if not nim_enabled:
        return Generator(circuit_breaker=circuit_breaker)

    from api.observability import CircuitBreaker

    nim_breaker = CircuitBreaker(threshold=3, timeout=30.0)
    nim_gen = NIMGenerator()
    ollama_gen = Generator(circuit_breaker=circuit_breaker)
    logger.info("NIM enabled — FallbackGenerator(NIM → Ollama) active")
    return FallbackGenerator(
        primary=nim_gen,
        fallback=ollama_gen,
        circuit_breaker=nim_breaker,
    )
