"""Tests for the model drift monitor (rag/drift_monitor.py)."""

from __future__ import annotations

import pytest

from rag.drift_monitor import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    DriftFeatureReport,
    DriftMonitor,
    DriftReport,
    PassthroughDriftMonitor,
    _mean,
    _relative_change,
)


class SampleResult:
    """Stand-in for a PostProcessResult-like object."""

    def __init__(
        self,
        answer: str = "",
        citations: list | None = None,
        confidence: float = 0.0,
        bias=None,
        guardrail_replacement: str | None = None,
    ) -> None:
        self.answer = answer
        self.citations = citations or []
        self.confidence = confidence
        self.bias = bias
        self.guardrail_replacement = guardrail_replacement


class TestHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_relative_change(self):
        assert _relative_change(1.2, 1.0) == pytest.approx(0.2)

    def test_relative_change_zero_baseline(self):
        # Avoids division by zero; relative change uses EPSILON.
        assert _relative_change(0.5, 0.0) > 1.0


class TestDriftMonitorBasics:
    def test_initial_assess_no_data(self):
        monitor = DriftMonitor()
        result = SampleResult(answer="hello", confidence=0.8)
        report = monitor.assess(result)
        assert report.drift_detected is False
        assert report.total_samples == 1

    def test_passthrough_returns_clean(self):
        monitor = PassthroughDriftMonitor()
        result = SampleResult(answer="hello", confidence=0.8)
        report = monitor.assess(result)
        assert report.drift_detected is False
        assert report.features == {}

    def test_records_total_samples(self):
        monitor = DriftMonitor()
        for i in range(5):
            monitor.assess(SampleResult(answer=f"answer {i}", confidence=0.8))
        report = monitor.assess(SampleResult(answer="answer 5", confidence=0.8))
        assert report.total_samples == 6

    def test_no_drift_when_stable(self):
        monitor = DriftMonitor(window_size=10, threshold=0.5)
        for _ in range(30):
            monitor.assess(SampleResult(answer="same answer", confidence=0.8))
        report = monitor.assess(SampleResult(answer="same answer", confidence=0.8))
        assert report.drift_detected is False
        for feature in report.features.values():
            assert feature.drifted is False

    def test_detects_confidence_drift(self):
        monitor = DriftMonitor(window_size=10, threshold=0.15, min_samples=2)
        # Baseline window: high confidence.
        for _ in range(10):
            monitor.assess(SampleResult(answer="answer", confidence=0.8))
        # Current window: low confidence triggers drift.
        for _ in range(10):
            monitor.assess(SampleResult(answer="answer", confidence=0.4))
        report = monitor.assess(SampleResult(answer="answer", confidence=0.4))
        assert report.drift_detected is True
        assert report.features["confidence"].drifted is True

    def test_detects_citation_count_drift(self):
        monitor = DriftMonitor(window_size=5, threshold=0.1, min_samples=2)
        for _ in range(5):
            monitor.assess(SampleResult(answer="answer", confidence=0.8, citations=[1, 2]))
        for _ in range(5):
            monitor.assess(SampleResult(answer="answer", confidence=0.8, citations=[]))
        report = monitor.assess(SampleResult(answer="answer", confidence=0.8, citations=[]))
        assert report.drift_detected is True
        assert report.features["citation_count"].drifted is True

    def test_blocked_flag_included(self):
        monitor = DriftMonitor(window_size=5, threshold=0.1, min_samples=2)
        for _ in range(5):
            monitor.assess(SampleResult(answer="answer", confidence=0.8), blocked=False)
        for _ in range(5):
            monitor.assess(SampleResult(answer="answer", confidence=0.8), blocked=True)
        report = monitor.assess(SampleResult(answer="answer", confidence=0.8), blocked=True)
        assert report.features["blocked_flag"].drifted is True

    def test_guardrail_replacement_counts_as_blocked(self):
        monitor = DriftMonitor(window_size=5, threshold=0.1, min_samples=2)
        for _ in range(5):
            monitor.assess(SampleResult(answer="answer"))
        for _ in range(5):
            monitor.assess(SampleResult(answer="answer", guardrail_replacement="refusal"))
        report = monitor.assess(SampleResult(answer="answer", guardrail_replacement="refusal"))
        assert report.features["blocked_flag"].drifted is True

    def test_bias_score_extracted(self):
        class FakeBias:
            score = 0.5

        monitor = DriftMonitor(window_size=5, threshold=0.1, min_samples=2)
        for _ in range(5):
            monitor.assess(SampleResult(answer="answer", bias=FakeBias()))
        for _ in range(5):
            monitor.assess(SampleResult(answer="answer"))
        report = monitor.assess(SampleResult(answer="answer"))
        assert report.features["bias_score"].drifted is True

    def test_reset_clears_window(self):
        monitor = DriftMonitor()
        monitor.assess(SampleResult(answer="answer"))
        monitor.reset()
        report = monitor.assess(SampleResult(answer="answer"))
        assert report.total_samples == 1

    def test_assess_none_is_noop(self):
        monitor = DriftMonitor()
        report = monitor.assess(None)
        assert report.total_samples == 0

    def test_window_size_default(self):
        monitor = DriftMonitor()
        assert monitor.window_size == DEFAULT_WINDOW_SIZE

    def test_threshold_default(self):
        monitor = DriftMonitor()
        assert monitor.threshold == DEFAULT_DRIFT_THRESHOLD


class TestDriftReport:
    def test_defaults(self):
        report = DriftReport()
        assert report.drift_detected is False
        assert report.features == {}
        assert report.total_samples == 0

    def test_feature_report(self):
        feature = DriftFeatureReport(
            current_mean=0.4,
            baseline_mean=0.8,
            relative_change=0.5,
            drifted=True,
        )
        assert feature.drifted is True
        assert feature.relative_change == pytest.approx(0.5)
