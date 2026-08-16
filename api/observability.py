"""Observability & hardening — Langfuse tracing, metrics, circuit breaker (Step 7).

Implements build-order items 39–41 (Phase 6 — Monitoring & Hardening):

**39. Langfuse traces** — ``LangfuseTracer`` wraps the langfuse Python SDK
(v3/v4 OpenTelemetry-style API) so the orchestrator can emit a trace per
chat request with child spans for retrieval, generation, and guardrail
phases, plus a feedback score event. The tracer degrades gracefully: when
the ``langfuse`` package is absent or ``LANGFUSE_BASE_URL`` is unset, every
method is a safe no-op that still logs span lifecycle events via
``structlog``/stdlib — this is the doc's "Langfuse-down fallback to
structlog" mitigation. The lazy-client pattern (like ``Generator`` /
``Embedder``) keeps unit tests network-free.

**40. Metrics** — ``MetricsCollector`` is a thread-safe in-memory store for
online serving metrics: TTFT (time-to-first-token) samples, request count,
cache hits/misses, and error count. ``snapshot()`` returns a dict with mean
/ p50 / p95 TTFT and the cache hit rate — surfaced via ``GET /api/v1/metrics``.
``evaluate_ttft_slo()`` is the **threshold gate** that asserts the doc's
"TTFT < 800ms" SLO against the p95 of the retained sample window (status:
``met`` / ``breached`` / ``no_data``); the result is included in the
snapshot under the ``slo`` key. The offline metrics (retrieval recall@5,
faithfulness, answer relevancy) come from the Step 6 eval gate; this
collector covers the *online* half.

**41. Resilience** — ``CircuitBreaker`` is the in-process breaker from the
doc's "Circuit Breaker (Optional)" snippet, wrapped around Ollama calls so
a flaky/unreachable Ollama does not stall every request. After
``threshold`` consecutive failures the circuit opens for ``timeout``
seconds; calls during that window raise ``CircuitOpenError`` instead of
hitting the network. A success resets the failure count.

All three components are independently injectable so the orchestrator,
generator, and routes stay unit-testable with mocks.
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, ContextManager, Iterator, Protocol

from rag.drift_monitor import DriftReport

logger = logging.getLogger(__name__)

# ── Langfuse constants ─────────────────────────────────────────────────

DEFAULT_LANGFUSE_BASE_URL: str = "https://us.cloud.langfuse.com"
"""Default Langfuse base URL (Langfuse Cloud SaaS, US region)."""

MAX_TTFT_SAMPLES: int = 200
"""Cap on retained TTFT samples for percentile computation (memory-bounded)."""

TTFT_SLO_TARGET_S: float = 0.8
"""TTFT SLO target in seconds (the doc's "TTFT < 800ms" online-serving SLO)."""

TTFT_SLO_PERCENTILE: float = 95.0
"""Percentile the TTFT SLO is evaluated against (p95 over the sample window)."""


# ════════════════════════════════════════════════════════════════════════
# Circuit breaker (build-order item 41)
# ════════════════════════════════════════════════════════════════════════


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit breaker is open."""


class CircuitBreaker:
    """In-process circuit breaker around an unreliable dependency (Ollama).

    After ``threshold`` consecutive failures the circuit *opens*: subsequent
    ``call()`` invocations raise ``CircuitOpenError`` immediately, without
    hitting the network. After ``timeout`` seconds the circuit *half-opens*:
    one call is allowed through; if it succeeds the circuit closes, if it
    fails the open window restarts. A success at any point resets the
    failure counter to zero.

    Thread-safe via a ``threading.Lock``. This is the practice-project
    equivalent of the doc's "Circuit Breaker (Optional)" snippet — simple,
    in-process, no external state.
    """

    def __init__(self, threshold: int = 3, timeout: float = 30.0) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self.threshold = threshold
        self.timeout = timeout
        self._failures = 0
        self._last_failure: float | None = None
        self._lock = threading.Lock()

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def state(self) -> str:
        """``"closed"``, ``"open"``, or ``"half_open"``."""
        with self._lock:
            if self._failures < self.threshold:
                return "closed"
            if self._last_failure is None:
                return "closed"
            if time.time() - self._last_failure >= self.timeout:
                return "half_open"
            return "open"

    def is_open(self) -> bool:
        """True when the circuit is open (calls should be rejected)."""
        return self.state == "open"

    def call(self, fn, *args, **kwargs):
        """Invoke ``fn(*args, **kwargs)`` through the breaker.

        Raises ``CircuitOpenError`` if the circuit is open. On success the
        failure count resets; on exception the count increments and the
        circuit may open.
        """
        with self._lock:
            if self._failures >= self.threshold:
                if (
                    self._last_failure is not None
                    and time.time() - self._last_failure < self.timeout
                ):
                    raise CircuitOpenError(
                        f"Circuit breaker is open "
                        f"({self._failures} failures, "
                        f"last {time.time() - self._last_failure:.1f}s ago)"
                    )
                # Half-open: allow one probe call through.
                logger.info("Circuit breaker half-open — probing with one call")
        try:
            result = fn(*args, **kwargs)
        except CircuitOpenError:
            raise
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result

    def record_success(self) -> None:
        """Reset the failure count (call succeeded)."""
        with self._lock:
            if self._failures > 0:
                logger.debug("Circuit breaker reset (was %d failures)", self._failures)
            self._failures = 0
            self._last_failure = None

    def record_failure(self) -> None:
        """Increment the failure count, possibly opening the circuit."""
        with self._lock:
            self._failures += 1
            self._last_failure = time.time()
            if self._failures >= self.threshold:
                logger.warning(
                    "Circuit breaker opened after %d failures", self._failures,
                )

    def reset(self) -> None:
        """Force-reset the breaker to the closed state (for tests / admin)."""
        with self._lock:
            self._failures = 0
            self._last_failure = None


# ════════════════════════════════════════════════════════════════════════
# Metrics collector (build-order item 40)
# ════════════════════════════════════════════════════════════════════════


@dataclass
class _MetricsState:
    """Mutable counters + TTFT sample ring buffer."""

    requests: int = 0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    ttft_samples: deque = field(default_factory=lambda: deque(maxlen=MAX_TTFT_SAMPLES))
    bias_checks: int = 0
    biased_answers: int = 0
    bias_blocks: int = 0
    bias_category_counts: dict[str, int] = field(default_factory=dict)
    drift_checks: int = 0
    drift_alerts: int = 0
    drift_feature_counts: dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe in-memory collector for online serving metrics.

    Tracks request count, error count, cache hit/miss, and TTFT (time-to-
    first-token) samples. ``snapshot()`` computes mean / p50 / p95 TTFT and
    the cache hit rate. Intended for a single-process FastAPI deployment;
    for multi-process or multi-instance, export these via Prometheus instead.
    """

    def __init__(self) -> None:
        self._state = _MetricsState()
        self._lock = threading.Lock()

    def record_request(self) -> None:
        with self._lock:
            self._state.requests += 1

    def record_error(self) -> None:
        with self._lock:
            self._state.errors += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._state.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._state.cache_misses += 1

    def record_ttft(self, seconds: float) -> None:
        """Record a time-to-first-token sample (seconds)."""
        if seconds < 0:
            return
        with self._lock:
            self._state.ttft_samples.append(seconds)

    def record_bias_check(
        self,
        biased: bool,
        categories: list[str] | None = None,
        blocked: bool = False,
    ) -> None:
        """Record a bias & fairness check result (Responsible AI).

        Increments the bias-check counter, the biased-answer counter when
        ``biased`` is True, the bias-block counter when ``blocked`` is True,
        and the per-category counts for each category in ``categories``.
        Surfaces the bias monitoring metrics in ``GET /api/v1/metrics``.
        """
        with self._lock:
            self._state.bias_checks += 1
            if biased:
                self._state.biased_answers += 1
            if blocked:
                self._state.bias_blocks += 1
            for cat in categories or []:
                self._state.bias_category_counts[cat] = (
                    self._state.bias_category_counts.get(cat, 0) + 1
                )

    def record_drift_check(
        self,
        report: DriftReport | None = None,
        alert: bool = False,
    ) -> None:
        """Record a model drift detection result (Responsible AI).

        Increments the drift-check counter, the drift-alert counter when
        the report indicates ``drift_detected`` or ``alert`` is True, and
        the per-feature counts for each feature that triggered a drift.
        Surfaces the drift monitoring metrics in ``GET /api/v1/metrics``.
        """
        is_alert = alert or (report is not None and report.drift_detected)
        with self._lock:
            self._state.drift_checks += 1
            if is_alert:
                self._state.drift_alerts += 1
            if report is not None:
                for name, feature in report.features.items():
                    if feature.drifted:
                        self._state.drift_feature_counts[name] = (
                            self._state.drift_feature_counts.get(name, 0) + 1
                        )

    def evaluate_ttft_slo(
        self,
        target_s: float = TTFT_SLO_TARGET_S,
        percentile: float = TTFT_SLO_PERCENTILE,
    ) -> dict[str, Any]:
        """Evaluate the TTFT SLO gate against the retained sample window.

        Computes the configured percentile of the TTFT samples and compares
        it to ``target_s``. Returns a JSON-serialisable dict:

        - ``target_s`` — the SLO threshold (seconds).
        - ``percentile`` — the percentile used (e.g. 95 for p95).
        - ``value_s`` — the computed percentile TTFT (0.0 when no samples).
        - ``status`` — ``"met"`` (value < target), ``"breached"`` (value >=
          target), or ``"no_data"`` (no samples recorded yet).
        - ``met`` — ``True`` / ``False`` / ``None`` (``None`` for ``no_data``).
        - ``samples`` — number of TTFT samples in the window.

        This is the threshold gate that turns the collected TTFT samples
        into an asserted SLO (build-order item 40 / doc "TTFT < 800ms").
        For a multi-process deployment the gate should be evaluated over a
        Prometheus-style rolling window instead of the in-memory ring buffer.
        """
        with self._lock:
            samples = list(self._state.ttft_samples)
        return _ttft_slo(samples, target_s, percentile)

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time metrics snapshot as a JSON-serialisable dict."""
        with self._lock:
            samples = list(self._state.ttft_samples)
            hits = self._state.cache_hits
            misses = self._state.cache_misses
            total_cache = hits + misses
            bias_checks = self._state.bias_checks
            biased = self._state.biased_answers
            drift_checks = self._state.drift_checks
            drift_alerts = self._state.drift_alerts
            return {
                "requests": self._state.requests,
                "errors": self._state.errors,
                "cache": {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": (hits / total_cache) if total_cache else 0.0,
                },
                "ttft": {
                    "count": len(samples),
                    "mean_s": statistics.fmean(samples) if samples else 0.0,
                    "p50_s": _percentile(samples, 50) if samples else 0.0,
                    "p95_s": _percentile(samples, 95) if samples else 0.0,
                    "min_s": min(samples) if samples else 0.0,
                    "max_s": max(samples) if samples else 0.0,
                },
                "slo": _ttft_slo(samples, TTFT_SLO_TARGET_S, TTFT_SLO_PERCENTILE),
                "bias": {
                    "checks": bias_checks,
                    "biased_answers": biased,
                    "blocks": self._state.bias_blocks,
                    "bias_rate": (biased / bias_checks) if bias_checks else 0.0,
                    "categories": dict(self._state.bias_category_counts),
                },
                "drift": {
                    "checks": drift_checks,
                    "alerts": drift_alerts,
                    "alert_rate": (drift_alerts / drift_checks) if drift_checks else 0.0,
                    "features": dict(self._state.drift_feature_counts),
                },
            }

    def reset(self) -> None:
        """Clear all counters and samples (for tests)."""
        with self._lock:
            self._state = _MetricsState()


def _percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolation percentile of a sorted sample list."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _ttft_slo(
    samples: list[float],
    target_s: float,
    percentile: float,
) -> dict[str, Any]:
    """Compute the TTFT SLO gate result from a sample list (lock-free).

    Returns ``status="no_data"`` (``met=None``) when there are no samples,
    ``"met"`` when the percentile TTFT is below ``target_s``, else
    ``"breached"``. Pure function so ``snapshot()`` can call it while
    already holding the collector lock.
    """
    if not samples:
        return {
            "target_s": target_s,
            "percentile": percentile,
            "value_s": 0.0,
            "status": "no_data",
            "met": None,
            "samples": 0,
        }
    value = _percentile(samples, percentile)
    met = value < target_s
    return {
        "target_s": target_s,
        "percentile": percentile,
        "value_s": value,
        "status": "met" if met else "breached",
        "met": met,
        "samples": len(samples),
    }


# ════════════════════════════════════════════════════════════════════════
# Langfuse tracer (build-order item 39)
# ════════════════════════════════════════════════════════════════════════


@dataclass
class TraceHandle:
    """Opaque handle for a Langfuse trace.

    Carries a correlation ``id`` (the Langfuse trace id) so the feedback
    endpoint can attach a score to the right trace even when Langfuse is
    disabled (the id is still logged via structlog for correlation).
    ``_lf_root`` is the root Langfuse observation (span) created by
    ``start_trace``; it is ``None`` for handles constructed by the feedback
    route (which only carries an ``id``) and for disabled traces.
    """

    id: str
    enabled: bool = True
    _lf_root: Any = None  # root LangfuseSpan (None when disabled / feedback-only)


@dataclass
class SpanHandle:
    """Opaque handle for a Langfuse span."""

    name: str
    start: float
    enabled: bool = True
    _lf_span: Any = None  # the real langfuse span object (None when disabled)


class LangfuseTracer:
    """Thin wrapper around the Langfuse v3/v4 SDK with graceful degradation.

    When Langfuse is available (``langfuse`` importable + ``LANGFUSE_BASE_URL``
    set + public/secret keys configured), traces and spans are sent to
    Langfuse Cloud (https://us.cloud.langfuse.com). When any of those
    conditions fail, every method is a safe no-op that still emits a
    structured log line — this is the doc's "Langfuse-down fallback to
    structlog" mitigation. The orchestrator calls tracer methods
    unconditionally; the tracer decides whether to ship to Langfuse or just
    log.

    The Langfuse client is lazily constructed (like ``Generator``) so unit
    tests inject a mock client or simply construct a disabled tracer.

    SDK mapping (Langfuse >=3, OpenTelemetry-style API):

    - ``start_trace`` → ``client.create_trace_id()`` + ``client.start_observation(as_type="span", trace_context=...)`` to create the root span.
    - ``start_span``  → ``root.start_observation(as_type="span", ...)`` for a nested child.
    - ``end_span``    → ``span.update(metadata=...)`` then ``span.end()``.
    - ``end_trace``   → ``root.update(metadata=...)`` then ``root.end()``.
    - ``record_score``→ ``client.create_score(trace_id=..., name=..., value=..., comment=...)``. Uses the trace id directly so the feedback route (which only has an id) works without the root span object.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        client: Any = None,
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._client = client
        self._host = host or os.getenv("LANGFUSE_BASE_URL", DEFAULT_LANGFUSE_BASE_URL)
        self._public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self._secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        if enabled is None:
            enabled = self._detect_enabled()
        self._enabled = enabled

    def _detect_enabled(self) -> bool:
        """True only if langfuse is importable AND host + keys are configured.

        Checks the *raw* env vars (not the defaulted ``self._host``) so that
        an unset ``LANGFUSE_BASE_URL`` disables tracing even though the host
        attribute has a cloud default for client construction.
        """
        if (
            not os.getenv("LANGFUSE_BASE_URL")
            or not self._public_key
            or not self._secret_key
        ):
            return False
        try:
            import langfuse  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def client(self):
        """Lazily construct the Langfuse client (or return the injected one)."""
        if not self._enabled:
            return None
        if self._client is None:
            from langfuse import Langfuse

            self._client = Langfuse(
                host=self._host,
                public_key=self._public_key,
                secret_key=self._secret_key,
            )
        return self._client

    # ── trace + span lifecycle ──────────────────────────────────────────

    def start_trace(self, name: str, metadata: dict | None = None) -> TraceHandle:
        """Begin a new trace for a chat request.

        Creates a Langfuse trace id and a root span (observation) on that
        trace. The root span is ended by ``end_trace``; child spans are
        created via ``start_span`` and ended via ``end_span``.
        """
        if not self._enabled:
            trace_id = _gen_id()
            logger.info("trace.start name=%s id=%s meta=%s", name, trace_id, metadata)
            return TraceHandle(id=trace_id, enabled=False)
        try:
            trace_id = self.client.create_trace_id()
            lf_root = self.client.start_observation(
                name=name,
                as_type="span",
                trace_context={"trace_id": trace_id},
                metadata=metadata or {},
            )
        except Exception:  # noqa: BLE001 — Langfuse must never break serving
            trace_id = _gen_id()
            logger.warning("Langfuse trace creation failed — falling back to log")
            return TraceHandle(id=trace_id, enabled=False)
        return TraceHandle(id=trace_id, enabled=True, _lf_root=lf_root)

    def start_span(
        self,
        trace: TraceHandle,
        name: str,
        metadata: dict | None = None,
    ) -> SpanHandle:
        """Begin a child span nested under the trace's root span."""
        start = time.time()
        if not trace.enabled or not self._enabled or trace._lf_root is None:
            logger.debug("span.start name=%s trace=%s meta=%s", name, trace.id, metadata)
            return SpanHandle(name=name, start=start, enabled=False)
        try:
            lf_span = trace._lf_root.start_observation(
                name=name,
                as_type="span",
                metadata=metadata or {},
            )
        except Exception:  # noqa: BLE001
            logger.warning("Langfuse span creation failed — falling back to log")
            return SpanHandle(name=name, start=start, enabled=False)
        return SpanHandle(name=name, start=start, enabled=True, _lf_span=lf_span)

    def end_span(self, span: SpanHandle, metadata: dict | None = None) -> None:
        """End a span, recording its duration and optional metadata."""
        duration = time.time() - span.start
        if not span.enabled or not self._enabled:
            logger.debug(
                "span.end name=%s duration=%.4fs meta=%s",
                span.name, duration, metadata,
            )
            return
        try:
            if metadata:
                span._lf_span.update(metadata=metadata)
            span._lf_span.end()
        except Exception:  # noqa: BLE001
            logger.warning("Langfuse span end failed")

    def end_trace(self, trace: TraceHandle, metadata: dict | None = None) -> None:
        """End the root span of a trace (call via try/finally in the orchestrator)."""
        if not trace.enabled or not self._enabled or trace._lf_root is None:
            logger.debug("trace.end id=%s meta=%s", trace.id, metadata)
            return
        try:
            if metadata:
                trace._lf_root.update(metadata=metadata)
            trace._lf_root.end()
        except Exception:  # noqa: BLE001
            logger.warning("Langfuse trace end failed")

    def span(
        self,
        trace: TraceHandle,
        name: str,
        metadata: dict | None = None,
    ) -> ContextManager[SpanHandle]:
        """Context-manager helper for a span that auto-ends on exit.

        Convenient for retrieval / guardrail spans that wrap a single call.
        The generation span (which wraps a streaming loop) uses explicit
        ``start_span`` / ``end_span`` instead.
        """
        return _SpanContext(self, trace, name, metadata)

    def record_score(
        self,
        trace: TraceHandle,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """Attach a score (e.g. user feedback thumbs up/down) to a trace.

        Uses ``client.create_score(trace_id=...)`` so this works for handles
        that only carry an ``id`` (the feedback route) as well as for full
        traces created by ``start_trace``.
        """
        if not trace.enabled or not self._enabled:
            logger.info(
                "score name=%s value=%s trace=%s comment=%s",
                name, value, trace.id, comment,
            )
            return
        try:
            self.client.create_score(
                trace_id=trace.id,
                name=name,
                value=value,
                comment=comment or "",
            )
        except Exception:  # noqa: BLE001
            logger.warning("Langfuse score recording failed")

    def flush(self) -> None:
        """Flush any buffered events to Langfuse (call at shutdown)."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception:  # noqa: BLE001
            logger.warning("Langfuse flush failed")

    def close(self) -> None:
        """Release the Langfuse client if one was constructed."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                pass
        self._client = None


class _SpanContext:
    """Context manager returned by ``LangfuseTracer.span()``."""

    def __init__(
        self,
        tracer: LangfuseTracer,
        trace: TraceHandle,
        name: str,
        metadata: dict | None,
    ) -> None:
        self._tracer = tracer
        self._trace = trace
        self._name = name
        self._metadata = metadata
        self._span: SpanHandle | None = None

    def __enter__(self) -> SpanHandle:
        self._span = self._tracer.start_span(self._trace, self._name, self._metadata)
        return self._span

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._span is not None:
            end_meta = dict(self._metadata or {})
            if exc is not None:
                end_meta["error"] = str(exc)
            self._tracer.end_span(self._span, end_meta)


# ── helpers ────────────────────────────────────────────────────────────


def _gen_id() -> str:
    """Generate a short unique trace id (hex timestamp + counter)."""
    return f"{int(time.time() * 1000):x}-{threading.get_ident() & 0xFFFF:x}"


# ════════════════════════════════════════════════════════════════════════
# Ollama warm-up + readiness (build-order item 41)
# ════════════════════════════════════════════════════════════════════════


def warm_up_ollama(
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b",
    timeout: float = 10.0,
) -> bool:
    """Ping Ollama and pre-load ``model`` so the first real query is fast.

    Sends a tiny one-token generation request to force the model into
    memory (the doc's "warm-up call on startup" mitigation). Returns True
    on success, False if Ollama is unreachable — the app starts either way
    (degrades to circuit-breaker / error handling at query time).
    """
    import urllib.error
    import urllib.request
    import json as _json

    # 1. Liveness ping: GET /api/tags
    try:
        with urllib.request.urlopen(
            f"{ollama_url}/api/tags", timeout=timeout,
        ) as resp:
            if resp.status != 200:
                logger.warning("Ollama warm-up: /api/tags returned %s", resp.status)
                return False
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Ollama warm-up: unreachable at %s — %s", ollama_url, exc)
        return False

    # 2. Model pre-load: a minimal generate call (num_predict=1).
    payload = _json.dumps({"model": model, "prompt": "hi", "stream": False, "options": {"num_predict": 1}}).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning("Ollama warm-up: generate returned %s", resp.status)
                return False
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Ollama warm-up: model pre-load failed for %s — %s", model, exc)
        return False

    logger.info("Ollama warm-up complete (model=%s)", model)
    return True


def check_ollama(ollama_url: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    """Lightweight liveness check for Ollama (GET /api/tags)."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def check_qdrant(qdrant_url: str = "http://localhost:6333", timeout: float = 2.0) -> bool:
    """Lightweight liveness check for Qdrant (GET /healthz)."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{qdrant_url}/healthz", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False
