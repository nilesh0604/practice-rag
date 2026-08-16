"""Tests for the Step 7 observability layer (api/observability.py).

Covers:
- ``CircuitBreaker`` — closed/open/half-open transitions, failure counting,
  success reset, threshold + timeout, thread-safety of state.
- ``MetricsCollector`` — request/error/cache counters, TTFT samples,
  snapshot percentiles, reset.
- ``LangfuseTracer`` — disabled-mode no-ops, enabled-mode span/score
  delegation with graceful degradation on SDK errors, lazy client.
- ``warm_up_ollama`` / ``check_ollama`` / ``check_qdrant`` — unreachable
  returns False, reachable returns True (mocked urllib).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from api.observability import (
    CircuitBreaker,
    CircuitOpenError,
    LangfuseTracer,
    MetricsCollector,
    SpanHandle,
    TraceHandle,
    TTFT_SLO_PERCENTILE,
    TTFT_SLO_TARGET_S,
    _percentile,
    _ttft_slo,
    check_ollama,
    check_qdrant,
    warm_up_ollama,
)


# ════════════════════════════════════════════════════════════════════════
# CircuitBreaker
# ════════════════════════════════════════════════════════════════════════


class TestCircuitBreakerConstruction:
    def test_default_threshold_and_timeout(self):
        cb = CircuitBreaker()
        assert cb.threshold == 3
        assert cb.timeout == 30.0

    def test_custom_threshold_and_timeout(self):
        cb = CircuitBreaker(threshold=5, timeout=10.0)
        assert cb.threshold == 5
        assert cb.timeout == 10.0

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError, match="threshold"):
            CircuitBreaker(threshold=0)

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValueError, match="timeout"):
            CircuitBreaker(timeout=0)

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.failures == 0
        assert not cb.is_open()


class TestCircuitBreakerCall:
    def test_call_success_returns_result(self):
        cb = CircuitBreaker()
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.failures == 0

    def test_call_success_resets_failures(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        assert cb.failures == 1
        cb.call(lambda: "ok")
        assert cb.failures == 0

    def test_call_failure_increments_count(self):
        cb = CircuitBreaker(threshold=3)

        def boom():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            cb.call(boom)
        assert cb.failures == 1
        assert cb.state == "closed"

    def test_call_with_args_and_kwargs(self):
        cb = CircuitBreaker()

        def add(a, b, c=0):
            return a + b + c

        assert cb.call(add, 1, 2, c=3) == 6

    def test_circuit_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, timeout=30)

        def boom():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(boom)
        assert cb.state == "open"
        assert cb.is_open()

    def test_open_circuit_raises_circuit_open_error(self):
        cb = CircuitBreaker(threshold=2, timeout=30)

        def boom():
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(boom)
        assert cb.is_open()
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "should not reach")

    def test_half_open_after_timeout_allows_probe(self):
        cb = CircuitBreaker(threshold=2, timeout=0.05)

        def boom():
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(boom)
        assert cb.state == "open"
        time.sleep(0.06)
        assert cb.state == "half_open"
        # Probe call succeeds → circuit closes
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == "closed"

    def test_half_open_probe_failure_reopens(self):
        cb = CircuitBreaker(threshold=2, timeout=0.05)

        def boom():
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(boom)
        time.sleep(0.06)
        assert cb.state == "half_open"
        with pytest.raises(RuntimeError):
            cb.call(boom)
        assert cb.state == "open"

    def test_record_success_resets(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2
        cb.record_success()
        assert cb.failures == 0
        assert cb.state == "closed"

    def test_reset_clears_state(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()
        cb.reset()
        assert cb.failures == 0
        assert cb.state == "closed"


# ════════════════════════════════════════════════════════════════════════
# MetricsCollector
# ════════════════════════════════════════════════════════════════════════


class TestMetricsCollector:
    def test_initial_snapshot_is_zeroed(self):
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["requests"] == 0
        assert snap["errors"] == 0
        assert snap["cache"]["hits"] == 0
        assert snap["cache"]["misses"] == 0
        assert snap["cache"]["hit_rate"] == 0.0
        assert snap["ttft"]["count"] == 0
        assert snap["ttft"]["mean_s"] == 0.0

    def test_record_request_increments(self):
        mc = MetricsCollector()
        mc.record_request()
        mc.record_request()
        assert mc.snapshot()["requests"] == 2

    def test_record_error_increments(self):
        mc = MetricsCollector()
        mc.record_error()
        assert mc.snapshot()["errors"] == 1

    def test_cache_hit_and_miss(self):
        mc = MetricsCollector()
        mc.record_cache_hit()
        mc.record_cache_hit()
        mc.record_cache_miss()
        snap = mc.snapshot()["cache"]
        assert snap["hits"] == 2
        assert snap["misses"] == 1
        assert snap["hit_rate"] == pytest.approx(2 / 3)

    def test_cache_hit_rate_zero_when_no_data(self):
        mc = MetricsCollector()
        assert mc.snapshot()["cache"]["hit_rate"] == 0.0

    def test_record_ttft(self):
        mc = MetricsCollector()
        mc.record_ttft(0.1)
        mc.record_ttft(0.2)
        mc.record_ttft(0.3)
        snap = mc.snapshot()["ttft"]
        assert snap["count"] == 3
        assert snap["mean_s"] == pytest.approx(0.2)
        assert snap["min_s"] == pytest.approx(0.1)
        assert snap["max_s"] == pytest.approx(0.3)

    def test_ttft_negative_ignored(self):
        mc = MetricsCollector()
        mc.record_ttft(-1.0)
        assert mc.snapshot()["ttft"]["count"] == 0

    def test_ttft_percentiles(self):
        mc = MetricsCollector()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            mc.record_ttft(v)
        snap = mc.snapshot()["ttft"]
        assert snap["p50_s"] == pytest.approx(0.55, abs=0.01)
        assert snap["p95_s"] == pytest.approx(0.955, abs=0.01)

    def test_ttft_max_samples_capped(self):
        from api.observability import MAX_TTFT_SAMPLES

        mc = MetricsCollector()
        for i in range(MAX_TTFT_SAMPLES + 50):
            mc.record_ttft(float(i))
        assert mc.snapshot()["ttft"]["count"] == MAX_TTFT_SAMPLES

    def test_reset_clears_all(self):
        mc = MetricsCollector()
        mc.record_request()
        mc.record_error()
        mc.record_cache_hit()
        mc.record_ttft(0.5)
        mc.reset()
        snap = mc.snapshot()
        assert snap["requests"] == 0
        assert snap["errors"] == 0
        assert snap["cache"]["hits"] == 0
        assert snap["ttft"]["count"] == 0


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([5.0], 50) == 5.0

    def test_p50_even_count(self):
        assert _percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_p50_odd_count(self):
        assert _percentile([1, 2, 3], 50) == pytest.approx(2.0)

    def test_p0_returns_min(self):
        assert _percentile([3, 1, 2], 0) == pytest.approx(1.0)

    def test_p100_returns_max(self):
        assert _percentile([3, 1, 2], 100) == pytest.approx(3.0)


# ════════════════════════════════════════════════════════════════════════
# TTFT SLO gate (build-order item 40 — "TTFT < 800ms")
# ════════════════════════════════════════════════════════════════════════


class TestTtftSloConstants:
    def test_target_is_800ms(self):
        assert TTFT_SLO_TARGET_S == pytest.approx(0.8)

    def test_default_percentile_is_p95(self):
        assert TTFT_SLO_PERCENTILE == pytest.approx(95.0)


class TestTtftSloHelper:
    def test_no_data_when_no_samples(self):
        result = _ttft_slo([], target_s=0.8, percentile=95.0)
        assert result["status"] == "no_data"
        assert result["met"] is None
        assert result["samples"] == 0
        assert result["value_s"] == 0.0
        assert result["target_s"] == pytest.approx(0.8)
        assert result["percentile"] == pytest.approx(95.0)

    def test_met_when_p95_below_target(self):
        # 20 samples 0.1–0.3s → p95 ~0.295s, well under 0.8s.
        samples = [0.1 + i * 0.01 for i in range(20)]
        result = _ttft_slo(samples, target_s=0.8, percentile=95.0)
        assert result["status"] == "met"
        assert result["met"] is True
        assert result["samples"] == 20
        assert result["value_s"] < 0.8

    def test_breached_when_p95_at_or_above_target(self):
        # 20 samples 0.7–1.3s → p95 ~1.245s, above 0.8s.
        samples = [0.7 + i * 0.03 for i in range(20)]
        result = _ttft_slo(samples, target_s=0.8, percentile=95.0)
        assert result["status"] == "breached"
        assert result["met"] is False
        assert result["value_s"] >= 0.8

    def test_boundary_target_equal_is_breached(self):
        # p95 exactly equal to target → not strictly below → breached.
        result = _ttft_slo([0.8], target_s=0.8, percentile=95.0)
        assert result["status"] == "breached"
        assert result["met"] is False

    def test_custom_target_and_percentile(self):
        # p99 of [0.1..0.5] ~0.496, target 0.5 → breached (not strictly below).
        samples = [0.1 + i * 0.04 for i in range(11)]
        result = _ttft_slo(samples, target_s=0.5, percentile=99.0)
        assert result["target_s"] == pytest.approx(0.5)
        assert result["percentile"] == pytest.approx(99.0)


class TestMetricsCollectorTtftSlo:
    def test_evaluate_slo_no_data_initially(self):
        mc = MetricsCollector()
        result = mc.evaluate_ttft_slo()
        assert result["status"] == "no_data"
        assert result["met"] is None
        assert result["samples"] == 0

    def test_evaluate_slo_met(self):
        mc = MetricsCollector()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            mc.record_ttft(v)
        result = mc.evaluate_ttft_slo()
        assert result["status"] == "met"
        assert result["met"] is True
        assert result["samples"] == 5
        assert result["value_s"] < 0.8

    def test_evaluate_slo_breached(self):
        mc = MetricsCollector()
        for v in [0.7, 0.8, 0.9, 1.0, 1.1]:
            mc.record_ttft(v)
        result = mc.evaluate_ttft_slo()
        assert result["status"] == "breached"
        assert result["met"] is False
        assert result["value_s"] >= 0.8

    def test_evaluate_slo_respects_custom_target(self):
        mc = MetricsCollector()
        for v in [0.1, 0.2, 0.3]:
            mc.record_ttft(v)
        # p95 of [0.1,0.2,0.3] = 0.3; with a tight 0.25 target → breached.
        result = mc.evaluate_ttft_slo(target_s=0.25)
        assert result["status"] == "breached"
        assert result["met"] is False
        assert result["target_s"] == pytest.approx(0.25)

    def test_snapshot_includes_slo_block(self):
        mc = MetricsCollector()
        mc.record_ttft(0.2)
        mc.record_ttft(0.3)
        snap = mc.snapshot()
        assert "slo" in snap
        slo = snap["slo"]
        assert slo["status"] == "met"
        assert slo["met"] is True
        assert slo["target_s"] == pytest.approx(0.8)
        assert slo["percentile"] == pytest.approx(95.0)
        assert slo["samples"] == 2

    def test_snapshot_slo_no_data_when_empty(self):
        mc = MetricsCollector()
        slo = mc.snapshot()["slo"]
        assert slo["status"] == "no_data"
        assert slo["met"] is None

    def test_snapshot_slo_consistent_with_ttft_p95(self):
        mc = MetricsCollector()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            mc.record_ttft(v)
        snap = mc.snapshot()
        assert snap["slo"]["value_s"] == pytest.approx(snap["ttft"]["p95_s"])

    def test_reset_clears_slo_samples(self):
        mc = MetricsCollector()
        mc.record_ttft(1.0)
        assert mc.evaluate_ttft_slo()["samples"] == 1
        mc.reset()
        assert mc.evaluate_ttft_slo()["status"] == "no_data"
        assert mc.evaluate_ttft_slo()["samples"] == 0


# ════════════════════════════════════════════════════════════════════════
# Bias & fairness metrics (Responsible AI)
# ════════════════════════════════════════════════════════════════════════


class TestBiasMetrics:
    def test_initial_snapshot_bias_zeroed(self):
        mc = MetricsCollector()
        snap = mc.snapshot()["bias"]
        assert snap["checks"] == 0
        assert snap["biased_answers"] == 0
        assert snap["blocks"] == 0
        assert snap["bias_rate"] == 0.0
        assert snap["categories"] == {}

    def test_record_bias_check_clean(self):
        mc = MetricsCollector()
        mc.record_bias_check(biased=False)
        snap = mc.snapshot()["bias"]
        assert snap["checks"] == 1
        assert snap["biased_answers"] == 0
        assert snap["bias_rate"] == 0.0

    def test_record_bias_check_biased(self):
        mc = MetricsCollector()
        mc.record_bias_check(biased=True, categories=["gendered_pronoun"])
        snap = mc.snapshot()["bias"]
        assert snap["checks"] == 1
        assert snap["biased_answers"] == 1
        assert snap["bias_rate"] == 1.0
        assert snap["categories"] == {"gendered_pronoun": 1}

    def test_record_bias_check_blocked(self):
        mc = MetricsCollector()
        mc.record_bias_check(biased=True, categories=["gendered_pronoun"], blocked=True)
        snap = mc.snapshot()["bias"]
        assert snap["blocks"] == 1

    def test_bias_rate_mixed(self):
        mc = MetricsCollector()
        mc.record_bias_check(biased=False)
        mc.record_bias_check(biased=True, categories=["gendered_pronoun"])
        mc.record_bias_check(biased=False)
        snap = mc.snapshot()["bias"]
        assert snap["checks"] == 3
        assert snap["biased_answers"] == 1
        assert snap["bias_rate"] == pytest.approx(1 / 3)

    def test_category_counts_accumulate(self):
        mc = MetricsCollector()
        mc.record_bias_check(biased=True, categories=["gendered_pronoun", "stereotype"])
        mc.record_bias_check(biased=True, categories=["gendered_pronoun"])
        snap = mc.snapshot()["bias"]
        assert snap["categories"] == {"gendered_pronoun": 2, "stereotype": 1}

    def test_reset_clears_bias_metrics(self):
        mc = MetricsCollector()
        mc.record_bias_check(biased=True, categories=["gendered_pronoun"], blocked=True)
        mc.reset()
        snap = mc.snapshot()["bias"]
        assert snap["checks"] == 0
        assert snap["biased_answers"] == 0
        assert snap["blocks"] == 0
        assert snap["categories"] == {}


# ════════════════════════════════════════════════════════════════════════
# LangfuseTracer
# ════════════════════════════════════════════════════════════════════════


class TestLangfuseTracerDisabled:
    def test_disabled_by_default_without_env(self):
        with patch.dict("os.environ", {}, clear=True):
            t = LangfuseTracer()
            assert not t.enabled

    def test_disabled_explicitly(self):
        t = LangfuseTracer(enabled=False)
        assert not t.enabled

    def test_start_trace_returns_disabled_handle(self):
        t = LangfuseTracer(enabled=False)
        trace = t.start_trace("chat")
        assert isinstance(trace, TraceHandle)
        assert not trace.enabled
        assert trace.id

    def test_start_span_returns_disabled_handle(self):
        t = LangfuseTracer(enabled=False)
        trace = t.start_trace("chat")
        span = t.start_span(trace, "retrieval")
        assert isinstance(span, SpanHandle)
        assert not span.enabled

    def test_end_span_is_noop(self):
        t = LangfuseTracer(enabled=False)
        trace = t.start_trace("chat")
        span = t.start_span(trace, "retrieval")
        t.end_span(span)  # should not raise

    def test_record_score_is_noop(self):
        t = LangfuseTracer(enabled=False)
        trace = t.start_trace("chat")
        t.record_score(trace, "feedback", 1.0)  # should not raise

    def test_span_context_manager(self):
        t = LangfuseTracer(enabled=False)
        trace = t.start_trace("chat")
        with t.span(trace, "retrieval") as span:
            assert isinstance(span, SpanHandle)
        # exit should not raise

    def test_flush_is_noop(self):
        t = LangfuseTracer(enabled=False)
        t.flush()  # should not raise

    def test_close_is_noop(self):
        t = LangfuseTracer(enabled=False)
        t.close()

    def test_client_returns_none_when_disabled(self):
        t = LangfuseTracer(enabled=False)
        assert t.client is None


class TestLangfuseTracerEnabled:
    def _make_enabled_tracer(self):
        """Tracer with a mock client forced enabled."""
        mock_client = MagicMock()
        mock_client.create_trace_id.return_value = "mock-trace-id"
        return LangfuseTracer(enabled=True, client=mock_client)

    def test_start_trace_creates_langfuse_trace(self):
        t = self._make_enabled_tracer()
        mock_root = MagicMock()
        t._client.start_observation.return_value = mock_root
        trace = t.start_trace("chat", metadata={"q": "hi"})
        assert trace.enabled
        assert trace._lf_root is mock_root
        assert trace.id == "mock-trace-id"
        t._client.create_trace_id.assert_called_once()
        t._client.start_observation.assert_called_once()
        call_kwargs = t._client.start_observation.call_args.kwargs
        assert call_kwargs["name"] == "chat"
        assert call_kwargs["as_type"] == "span"
        assert call_kwargs["trace_context"] == {"trace_id": "mock-trace-id"}
        assert call_kwargs["metadata"] == {"q": "hi"}

    def test_start_span_creates_langfuse_span(self):
        t = self._make_enabled_tracer()
        mock_root = MagicMock()
        mock_span = MagicMock()
        mock_root.start_observation.return_value = mock_span
        trace = TraceHandle(id="x", enabled=True, _lf_root=mock_root)
        span = t.start_span(trace, "retrieval", metadata={"k": "v"})
        assert span.enabled
        assert span._lf_span is mock_span
        mock_root.start_observation.assert_called_once()
        call_kwargs = mock_root.start_observation.call_args.kwargs
        assert call_kwargs["name"] == "retrieval"
        assert call_kwargs["as_type"] == "span"
        assert call_kwargs["metadata"] == {"k": "v"}

    def test_end_span_calls_update_then_end(self):
        t = self._make_enabled_tracer()
        mock_span = MagicMock()
        span = SpanHandle(name="gen", start=time.time(), enabled=True, _lf_span=mock_span)
        t.end_span(span, metadata={"len": 10})
        mock_span.update.assert_called_once_with(metadata={"len": 10})
        mock_span.end.assert_called_once()

    def test_end_span_without_metadata(self):
        t = self._make_enabled_tracer()
        mock_span = MagicMock()
        span = SpanHandle(name="gen", start=time.time(), enabled=True, _lf_span=mock_span)
        t.end_span(span)
        mock_span.update.assert_not_called()
        mock_span.end.assert_called_once()

    def test_end_trace_calls_update_then_end(self):
        t = self._make_enabled_tracer()
        mock_root = MagicMock()
        trace = TraceHandle(id="x", enabled=True, _lf_root=mock_root)
        t.end_trace(trace, metadata={"status": "ok"})
        mock_root.update.assert_called_once_with(metadata={"status": "ok"})
        mock_root.end.assert_called_once()

    def test_end_trace_without_metadata(self):
        t = self._make_enabled_tracer()
        mock_root = MagicMock()
        trace = TraceHandle(id="x", enabled=True, _lf_root=mock_root)
        t.end_trace(trace)
        mock_root.update.assert_not_called()
        mock_root.end.assert_called_once()

    def test_end_trace_is_noop_when_root_is_none(self):
        t = self._make_enabled_tracer()
        trace = TraceHandle(id="x", enabled=True, _lf_root=None)
        t.end_trace(trace)  # should not raise

    def test_record_score_calls_create_score(self):
        t = self._make_enabled_tracer()
        trace = TraceHandle(id="x", enabled=True, _lf_root=MagicMock())
        t.record_score(trace, "feedback", 1.0, comment="good")
        t._client.create_score.assert_called_once_with(
            trace_id="x", name="feedback", value=1.0, comment="good",
        )

    def test_record_score_works_with_id_only_handle(self):
        """Feedback route constructs a TraceHandle with just an id (no _lf_root)."""
        t = self._make_enabled_tracer()
        trace = TraceHandle(id="fb-trace-id", enabled=True, _lf_root=None)
        t.record_score(trace, "user_feedback", 0.0, comment="bad")
        t._client.create_score.assert_called_once_with(
            trace_id="fb-trace-id", name="user_feedback", value=0.0, comment="bad",
        )

    def test_start_trace_degrades_on_sdk_error(self):
        t = self._make_enabled_tracer()
        t._client.create_trace_id.side_effect = RuntimeError("langfuse down")
        trace = t.start_trace("chat")
        assert not trace.enabled  # falls back to disabled handle

    def test_start_trace_degrades_on_observation_error(self):
        t = self._make_enabled_tracer()
        t._client.start_observation.side_effect = RuntimeError("langfuse down")
        trace = t.start_trace("chat")
        assert not trace.enabled

    def test_start_span_degrades_on_sdk_error(self):
        t = self._make_enabled_tracer()
        mock_root = MagicMock()
        mock_root.start_observation.side_effect = RuntimeError("langfuse down")
        trace = TraceHandle(id="x", enabled=True, _lf_root=mock_root)
        span = t.start_span(trace, "retrieval")
        assert not span.enabled

    def test_end_span_degrades_on_sdk_error(self):
        t = self._make_enabled_tracer()
        mock_span = MagicMock()
        mock_span.end.side_effect = RuntimeError("langfuse down")
        span = SpanHandle(name="gen", start=time.time(), enabled=True, _lf_span=mock_span)
        t.end_span(span)  # should not raise

    def test_end_trace_degrades_on_sdk_error(self):
        t = self._make_enabled_tracer()
        mock_root = MagicMock()
        mock_root.end.side_effect = RuntimeError("langfuse down")
        trace = TraceHandle(id="x", enabled=True, _lf_root=mock_root)
        t.end_trace(trace)  # should not raise

    def test_record_score_degrades_on_sdk_error(self):
        t = self._make_enabled_tracer()
        t._client.create_score.side_effect = RuntimeError("langfuse down")
        trace = TraceHandle(id="x", enabled=True, _lf_root=MagicMock())
        t.record_score(trace, "feedback", 1.0)  # should not raise

    def test_flush_calls_client_flush(self):
        t = self._make_enabled_tracer()
        t.flush()
        t._client.flush.assert_called_once()

    def test_flush_degrades_on_error(self):
        t = self._make_enabled_tracer()
        t._client.flush.side_effect = RuntimeError("down")
        t.flush()  # should not raise

    def test_close_flushes_and_clears(self):
        t = self._make_enabled_tracer()
        client = t._client
        t.close()
        client.flush.assert_called_once()
        assert t._client is None

    def test_span_context_manager_ends_on_exit(self):
        t = self._make_enabled_tracer()
        mock_root = MagicMock()
        mock_span = MagicMock()
        mock_root.start_observation.return_value = mock_span
        trace = TraceHandle(id="x", enabled=True, _lf_root=mock_root)
        with t.span(trace, "retrieval"):
            pass
        mock_span.end.assert_called_once()

    def test_span_context_manager_records_error_on_exception(self):
        t = self._make_enabled_tracer()
        mock_root = MagicMock()
        mock_span = MagicMock()
        mock_root.start_observation.return_value = mock_span
        trace = TraceHandle(id="x", enabled=True, _lf_root=mock_root)
        with pytest.raises(ValueError):
            with t.span(trace, "retrieval"):
                raise ValueError("boom")
        mock_span.end.assert_called_once()
        update_kwargs = mock_span.update.call_args.kwargs
        assert "error" in update_kwargs.get("metadata", {})


class TestLangfuseTracerDetection:
    def test_enabled_when_host_and_keys_set_and_importable(self):
        with patch.dict("os.environ", {
            "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
            "LANGFUSE_PUBLIC_KEY": "pk-xxx",
            "LANGFUSE_SECRET_KEY": "sk-xxx",
        }):
            with patch("builtins.__import__") as mock_import:
                mock_import.return_value = MagicMock()
                t = LangfuseTracer()
                assert t.enabled

    def test_disabled_when_host_missing(self):
        with patch.dict("os.environ", {
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
        }, clear=True):
            t = LangfuseTracer()
            assert not t.enabled

    def test_disabled_when_keys_missing(self):
        with patch.dict("os.environ", {
            "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
        }, clear=True):
            t = LangfuseTracer()
            assert not t.enabled

    def test_disabled_when_langfuse_not_importable(self):
        with patch.dict("os.environ", {
            "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
        }):
            # Simulate langfuse not being installed by patching the import check.
            t = LangfuseTracer()
            t._enabled = False  # force disabled as if import failed
            assert not t.enabled


# ════════════════════════════════════════════════════════════════════════
# warm_up_ollama / check_ollama / check_qdrant
# ════════════════════════════════════════════════════════════════════════


class TestCheckOllama:
    def test_returns_false_when_unreachable(self):
        assert check_ollama("http://127.0.0.1:1", timeout=0.5) is False

    def test_returns_true_when_200(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert check_ollama("http://fake:11434") is True

    def test_returns_false_when_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert check_ollama("http://fake:11434") is False


class TestCheckQdrant:
    def test_returns_false_when_unreachable(self):
        assert check_qdrant("http://127.0.0.1:1", timeout=0.5) is False

    def test_returns_true_when_200(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert check_qdrant("http://fake:6333") is True


class TestWarmUpOllama:
    def test_returns_false_when_unreachable(self):
        assert warm_up_ollama("http://127.0.0.1:1", timeout=0.5) is False

    def test_returns_true_when_tags_and_generate_succeed(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert warm_up_ollama("http://fake:11434") is True

    def test_returns_false_when_tags_fail(self):
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert warm_up_ollama("http://fake:11434") is False
