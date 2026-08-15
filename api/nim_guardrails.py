"""NVIDIA NIM guardrail variants + 3-tier fallback (Phase 2 of the NIM integration plan).

Extends the existing 2-tier guardrail design (Ollama LLM → regex/keyword) to a
3-tier design when ``NIM_ENABLED=true``:

    NIM nemoguard (hosted)  →  Ollama llama3.2:3b judge  →  regex/keyword

Three subclasses override only the LLM-judge / LLM-classify method of the
existing guardrail classes, so the regex/keyword tiers and the
``GuardrailDecision`` / ``QueryClassification`` contracts are inherited
unchanged:

- **``NIMInputGuardrail(InputGuardrail)``** — overrides ``_llm_judge`` to call
  the NIM content-safety model first; on NIM failure (rate limit, model
  retired, network error, circuit open) or an unparseable verdict, falls back
  to the parent Ollama injection judge. The regex injection tier (tier 1) is
  inherited and always runs first.
- **``NIMOutputGuardrail(OutputGuardrail)``** — overrides ``_llm_judge`` to
  call the NIM content-safety model first; on failure falls back to the parent
  Ollama harmful-content judge. The PII regex scrub (tier 1) is inherited and
  always runs first.
- **``NIMQueryClassifier(QueryClassifier)``** — overrides ``_llm_classify`` to
  call the NIM topic-control model first (binary on-topic/off-topic). When NIM
  says "off-topic" the query is routed to ``off_topic`` directly. When NIM says
  "on-topic" (or fails / is unparseable), it falls through to the parent Ollama
  5-way classifier for fine-grained routing. The keyword fallback
  (``classify_keywords``) is inherited as the final tier.

Design notes (from ``docs/NVIDIA_NIM_INTEGRATION_PLAN.md`` §2.2–2.4, §3.2):

- NIM is **opt-in** (``NIM_ENABLED=true``). When disabled, the orchestrator
  uses the plain Ollama ``GuardrailSuite`` — no change to the default path.
- The NIM guardrail client gets its **own** ``CircuitBreaker`` (separate from
  the generator's NIM breaker and the Ollama breaker) so a NIM guardrail
  outage does not trip other breakers.
- NIM-specific HTTP errors (429 rate limit, 404 model retired) are mapped to
  fallback, **not** retried — retrying a 429 against a shared global limit
  would waste the budget.
- Guardrail failures **never block** legitimate traffic — the existing
  graceful-degradation contract is preserved (NIM fail → Ollama fail →
  regex/keyword, which never raises).
- The hosted PII model (``nvidia/gliner-pii``, plan §2.4) is **not** wired
  here. It is in the plan's "check provider" list (uncertain availability),
  carries a chicken-and-egg data-egress problem, and F500 item #5 flags PII
  scrubbing upstream of hosted models as a blocker. The regex ``scrub_pii``
  (inherited, always-on) remains the PII tier. The hosted PII refinement is
  deferred to a sub-phase after availability is probed — see
  ``docs/F500_ENTERPRISE_ACTION_ITEMS.md`` item #8, action item #9.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from api.guardrails import (
    CLASS_OFF_TOPIC,
    HARMFUL_JUDGE_PROMPT,
    INJECTION_JUDGE_PROMPT,
    GuardrailSuite,
    InputGuardrail,
    OutputGuardrail,
    QueryClassifier,
    _extract_label,
)

if TYPE_CHECKING:
    from api.observability import CircuitBreaker

logger = logging.getLogger(__name__)

# ── NIM constants ──────────────────────────────────────────────────────

NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
"""NVIDIA NIM OpenAI-compatible API base URL (free tier).

Duplicated from ``rag.nim_generator`` to keep the guardrail module decoupled
from the generator module (no cross-rag/api import for a constant).
"""

NIM_CONTENT_SAFETY_MODEL: str = "nvidia/llama-3.1-nemoguard-8b-content-safety"
"""NIM content-safety guard model (plan §2.2).

Purpose-built for harmful-content classification (hate / violence / self-harm
/ sexual / illegal). Used here as the primary LLM judge for both the input and
output guardrails. On failure or unparseable verdict, the existing Ollama
judge (injection / harmful) is the fallback.
"""

NIM_TOPIC_CONTROL_MODEL: str = "nvidia/llama-3.1-nemoguard-8b-topic-control"
"""NIM topic-control guard model (plan §2.3).

Purpose-built binary on-topic/off-topic guard. Used here as the primary
classifier for off-topic rejection; on-topic queries fall through to the
Ollama 5-way classifier for fine-grained routing.
"""

NIM_GUARD_TIMEOUT: float = 30.0
"""Per-request timeout for NIM guardrail calls (seconds).

Shorter than the generator's 60s because guardrail calls are single-verdict
(non-streaming) and run on every request (2 calls per chat: input + output).
"""

# ── NIM topic-control prompt (binary, purpose-built for the guard model) ──

NIM_TOPIC_CONTROL_PROMPT: str = """\
You are a topic-control guard for a technical documentation chatbot. The \
chatbot may ONLY answer questions about FastAPI, Pydantic v2, and SQLModel \
documentation (including follow-up questions that reference prior turns about \
those libraries).

Decide whether the user's question is on-topic (about those libraries or a \
follow-up to prior on-topic conversation) or off-topic (unrelated to the \
documented libraries).

Reply with ONLY one word: "safe" (on-topic) or "unsafe" (off-topic).

Conversation history:
{history}

User question:
{query}"""


# ── exceptions ─────────────────────────────────────────────────────────


class NIMGuardrailError(Exception):
    """Raised when a NIM guardrail API call fails in a way that triggers fallback."""


# ── NIM guardrail client (non-streaming) ───────────────────────────────


class NIMGuardrailClient:
    """Non-streaming OpenAI-compatible chat client for NIM guardrail verdicts.

    Guardrail calls are single-verdict (``safe`` / ``unsafe`` or a label), so
    unlike the generator this client uses a plain (non-SSE) POST to
    ``/chat/completions`` with ``stream=False`` and returns the full content
    string. The httpx client is lazily constructed (like the generator) so
    unit tests inject a mock without hitting the network.

    Owns a dedicated ``CircuitBreaker`` (separate from the generator's NIM
    breaker and the Ollama breaker). When the circuit is open, ``judge()``
    raises ``CircuitOpenError`` immediately (no network round-trip) so the
    guardrail falls back to Ollama without wasting a call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = NIM_BASE_URL,
        timeout: float = NIM_GUARD_TIMEOUT,
        circuit_breaker: "CircuitBreaker | None" = None,
    ) -> None:
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker
        self._client = None

    @property
    def client(self):
        """Lazily construct the httpx client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _do_request(self, model: str, messages: list[dict]) -> str:
        """POST to NIM ``/chat/completions`` (non-streaming) and return content.

        Raises ``NIMGuardrailError`` for any failure that should trigger
        fallback (429, 404, connection error, timeout).
        """
        import httpx

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 8,
            "stream": False,
        }
        try:
            resp = self.client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                raise NIMGuardrailError(
                    f"NIM guardrail rate limited (429) for model {model}",
                )
            if resp.status_code == 404:
                raise NIMGuardrailError(
                    f"NIM guardrail model not found (404): {model}",
                )
            if resp.status_code >= 400:
                body = resp.text[:200]
                raise NIMGuardrailError(
                    f"NIM guardrail HTTP {resp.status_code}: {body}",
                )
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise NIMGuardrailError("NIM guardrail returned no choices")
            return choices[0].get("message", {}).get("content", "")
        except httpx.TimeoutException as exc:
            raise NIMGuardrailError(f"NIM guardrail timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise NIMGuardrailError(
                f"NIM guardrail connection error: {exc}",
            ) from exc
        except NIMGuardrailError:
            raise
        except Exception as exc:  # noqa: BLE001 — map unknown errors to fallback
            raise NIMGuardrailError(f"NIM guardrail unexpected error: {exc}") from exc

    def judge(self, model: str, messages: list[dict]) -> str:
        """Call NIM for a guardrail verdict, through the circuit breaker.

        Returns the model's content string. Raises ``NIMGuardrailError`` on
        any failure (429, 404, timeout, connection error) or
        ``CircuitOpenError`` when the breaker is open — both are caught by the
        guardrail subclasses to trigger the Ollama fallback.
        """
        if self.circuit_breaker is not None:
            return self.circuit_breaker.call(self._do_request, model, messages)
        return self._do_request(model, messages)

    def close(self) -> None:
        """Release the httpx client if one was constructed."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None


# ── NIM input guardrail (3-tier: regex → NIM → Ollama → regex-only) ─────


class NIMInputGuardrail(InputGuardrail):
    """Input guardrail with a NIM content-safety judge as the primary LLM tier.

    The regex injection scan (tier 1) is inherited and always runs first.
    ``_llm_judge`` is overridden to call the NIM content-safety model; on NIM
    failure (``NIMGuardrailError`` / ``CircuitOpenError``) or an unparseable
    verdict, it falls back to the parent Ollama injection judge. If Ollama
    also fails, the parent returns ``None`` and the guardrail degrades to
    regex-only (not blocked) — the inherited contract.
    """

    def __init__(
        self,
        nim_client: NIMGuardrailClient,
        model: str = NIM_CONTENT_SAFETY_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.nim_client = nim_client
        self.nim_model = model

    def _llm_judge(self, message: str, history: str = "") -> str | None:
        prompt = INJECTION_JUDGE_PROMPT.format(
            message=message, history=history or "(none)",
        )
        try:
            text = self.nim_client.judge(
                self.nim_model,
                [{"role": "user", "content": prompt}],
            )
            label = _extract_label(text, frozenset({"safe", "unsafe"}))
            if label is not None:
                return label
            logger.info(
                "NIM input judge returned no label (%r) — falling back to Ollama",
                text[:40],
            )
        except Exception as exc:  # noqa: BLE001 — NIMGuardrailError / CircuitOpenError
            logger.info("NIM input judge failed — falling back to Ollama: %s", exc)
        return super()._llm_judge(message, history)


# ── NIM output guardrail (3-tier: PII scrub → NIM → Ollama → scrub-only) ──


class NIMOutputGuardrail(OutputGuardrail):
    """Output guardrail with a NIM content-safety judge as the primary LLM tier.

    The PII regex scrub (tier 1) is inherited and always runs first.
    ``_llm_judge`` is overridden to call the NIM content-safety model; on
    failure or unparseable verdict, it falls back to the parent Ollama
    harmful-content judge. If Ollama also fails, the parent returns ``None``
    and the guardrail degrades to scrub-only (not blocked) — the inherited
    contract.
    """

    def __init__(
        self,
        nim_client: NIMGuardrailClient,
        model: str = NIM_CONTENT_SAFETY_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.nim_client = nim_client
        self.nim_model = model

    def _llm_judge(self, answer: str) -> str | None:
        prompt = HARMFUL_JUDGE_PROMPT.format(answer=answer)
        try:
            text = self.nim_client.judge(
                self.nim_model,
                [{"role": "user", "content": prompt}],
            )
            label = _extract_label(text, frozenset({"safe", "unsafe"}))
            if label is not None:
                return label
            logger.info(
                "NIM output judge returned no label (%r) — falling back to Ollama",
                text[:40],
            )
        except Exception as exc:  # noqa: BLE001 — NIMGuardrailError / CircuitOpenError
            logger.info("NIM output judge failed — falling back to Ollama: %s", exc)
        return super()._llm_judge(answer)


# ── NIM query classifier (3-tier: NIM topic-control → Ollama 5-way → keyword) ──


class NIMQueryClassifier(QueryClassifier):
    """Query classifier with a NIM topic-control guard as the primary tier.

    ``_llm_classify`` is overridden to call the NIM topic-control model first
    (binary on-topic/off-topic). When NIM says "off-topic" (``unsafe``), the
    query is routed to ``off_topic`` directly — the topic-control model's
    specialty. When NIM says "on-topic" (``safe``), or fails / is unparseable,
    it falls through to the parent Ollama 5-way classifier for fine-grained
    routing (documentation / greeting / compare / follow_up). If Ollama also
    fails, the parent returns ``None`` and ``classify()`` falls back to
    ``classify_keywords`` — the inherited contract.
    """

    def __init__(
        self,
        nim_client: NIMGuardrailClient,
        model: str = NIM_TOPIC_CONTROL_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.nim_client = nim_client
        self.nim_model = model

    def _llm_classify(self, query: str, history: str = "") -> str | None:
        prompt = NIM_TOPIC_CONTROL_PROMPT.format(
            query=query, history=history or "(none)",
        )
        try:
            text = self.nim_client.judge(
                self.nim_model,
                [{"role": "user", "content": prompt}],
            )
            verdict = _extract_label(text, frozenset({"safe", "unsafe"}))
            if verdict == "unsafe":
                logger.info("NIM topic-control routed query to off_topic")
                return CLASS_OFF_TOPIC
            if verdict == "safe":
                # On-topic per NIM — fall through to Ollama for fine-grained
                # routing (documentation / greeting / compare / follow_up).
                logger.debug("NIM topic-control says on-topic — deferring to Ollama 5-way")
            else:
                logger.info(
                    "NIM topic-control returned no label (%r) — falling back to Ollama",
                    text[:40],
                )
        except Exception as exc:  # noqa: BLE001 — NIMGuardrailError / CircuitOpenError
            logger.info("NIM topic-control failed — falling back to Ollama: %s", exc)
        return super()._llm_classify(query, history)


# ── factory used by api/deps.py ────────────────────────────────────────


def build_guardrail_suite() -> GuardrailSuite:
    """Build the guardrail suite based on the ``NIM_ENABLED`` env var.

    - ``NIM_ENABLED=true`` (case-insensitive) → ``GuardrailSuite`` with the
      three NIM-augmented subclasses sharing one ``NIMGuardrailClient`` (one
      API key + one rate-limit budget + one dedicated circuit breaker).
    - Otherwise (default) → plain Ollama ``GuardrailSuite`` with the shared
      Ollama circuit breaker. No change to the existing default path.
    """
    nim_enabled = os.getenv("NIM_ENABLED", "").strip().lower() in ("true", "1", "yes")
    if not nim_enabled:
        return GuardrailSuite(
            input_guardrail=InputGuardrail(),
            output_guardrail=OutputGuardrail(),
            classifier=QueryClassifier(),
        )

    from api.observability import CircuitBreaker

    nim_breaker = CircuitBreaker(threshold=3, timeout=30.0)
    nim_client = NIMGuardrailClient(circuit_breaker=nim_breaker)
    logger.info("NIM guardrails enabled — 3-tier fallback (NIM → Ollama → regex)")
    return GuardrailSuite(
        input_guardrail=NIMInputGuardrail(nim_client),
        output_guardrail=NIMOutputGuardrail(nim_client),
        classifier=NIMQueryClassifier(nim_client),
    )
