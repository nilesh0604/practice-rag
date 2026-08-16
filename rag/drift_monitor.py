"""Model drift detection — post-generation output-distribution monitoring (Responsible AI).

Tracks a small set of online answer features (groundedness confidence,
citation count, answer length, guardrail/bias block rate, bias score) in a
rolling window and compares the most recent ``window_size`` samples against
the preceding ``window_size`` samples. When the relative change in the mean
of any feature exceeds ``threshold``, the monitor reports a drift alert.

This is a lightweight, in-process substitute for an enterprise model-
monitoring service (Prometheus/Grafana, Evidently, WhyLabs, etc.). It is
intended to surface *early warning* of output distribution shifts — for
example, a sudden drop in groundedness confidence or a spike in guardrail
blocks after a model or prompt change — without requiring external
infrastructure.

The monitor is intentionally statistical, not causal: a drift alert does not
prove the model changed, only that the observed answer distribution has
shifted enough to warrant review. In a production deployment this would be
replaced by a proper statistical drift test (Kolmogorov-Smirnov, PSI, Wasserstein)
against a held-out reference dataset, with alerting and a promotion gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

DEFAULT_WINDOW_SIZE: int = 50
"""Number of samples in each of the two windows (current + baseline)."""

DEFAULT_DRIFT_THRESHOLD: float = 0.20
"""Relative mean change that triggers a drift alert (20%)."""

DEFAULT_MIN_SAMPLES: int = 10
"""Minimum samples before any current/baseline mean can be reported."""

EPSILON: float = 1e-9
"""Smoothing term to avoid divide-by-zero in relative-change calculation."""

# Features surfaced in the drift report + metrics.
FEATURE_CONFIDENCE: str = "confidence"
FEATURE_CITATION_COUNT: str = "citation_count"
FEATURE_ANSWER_LENGTH: str = "answer_length"
FEATURE_BIAS_SCORE: str = "bias_score"
FEATURE_BLOCKED_FLAG: str = "blocked_flag"


# ── result dataclasses ─────────────────────────────────────────────────


@dataclass
class DriftFeatureReport:
    """Per-feature drift comparison for a single detection pass."""

    current_mean: float = 0.0
    baseline_mean: float = 0.0
    relative_change: float = 0.0
    drifted: bool = False


@dataclass
class DriftReport:
    """Result of a model drift detection pass.

    ``drift_detected`` is True when at least one tracked feature's mean
    changed by more than ``threshold`` between the baseline and current
    windows. ``features`` carries the per-feature comparison. ``window_size``
    is the configured window size and ``total_samples`` is how many answers
    have been observed so far.
    """

    drift_detected: bool = False
    features: dict[str, DriftFeatureReport] = field(default_factory=dict)
    window_size: int = DEFAULT_WINDOW_SIZE
    total_samples: int = 0


# ── sample dataclass (pure data, no I/O) ───────────────────────────────


@dataclass
class _Sample:
    """A single answer observation."""

    confidence: float = 0.0
    citation_count: int = 0
    answer_length: int = 0
    bias_score: float = 0.0
    blocked_flag: int = 0


# ── helpers (pure functions, no I/O) ───────────────────────────────────


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _relative_change(current: float, baseline: float) -> float:
    """Relative change between two means, capped to avoid huge spikes."""
    denom = max(abs(baseline), EPSILON)
    return abs(current - baseline) / denom


def _build_feature_report(
    current_values: list[float],
    baseline_values: list[float],
    threshold: float,
) -> DriftFeatureReport:
    """Compare one feature's current and baseline windows."""
    current = _mean(current_values)
    baseline = _mean(baseline_values)
    change = _relative_change(current, baseline)
    return DriftFeatureReport(
        current_mean=current,
        baseline_mean=baseline,
        relative_change=change,
        drifted=change >= threshold,
    )


# ── result-shaped protocol (mirrors BiasMonitor) ───────────────────────


class _HasDriftStats(Protocol):
    """Minimal shape for an object that can feed a drift sample."""

    answer: str
    citations: list
    confidence: float
    bias: object | None
    guardrail_replacement: str | None


# ── monitor ────────────────────────────────────────────────────────────


class DriftMonitor:
    """Rolling-window output-distribution drift detector.

    Records a small feature vector for each completed answer and compares
    the last ``window_size`` samples to the ``window_size`` samples before
    them. When the relative mean change for any feature exceeds
    ``threshold``, a ``DriftReport`` with ``drift_detected=True`` is
    returned. The monitor never raises; errors are logged and a no-alert
    report is returned.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        threshold: float = DEFAULT_DRIFT_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> None:
        self.window_size = max(1, window_size)
        self.threshold = max(0.0, threshold)
        self.min_samples = max(0, min_samples)
        self._samples: list[_Sample] = []

    def assess(
        self,
        result: _HasDriftStats | None,
        blocked: bool = False,
    ) -> DriftReport:
        """Record a new answer and return the current drift report.

        ``result`` is any object with ``answer``, ``citations``,
        ``confidence``, ``bias``, and ``guardrail_replacement`` attributes.
        ``blocked`` is True when the answer was replaced by a guardrail /
        hallucination / bias refusal, which is recorded as a binary feature.
        """
        if result is not None:
            try:
                self._record(result, blocked)
            except Exception:
                logger.warning("DriftMonitor: failed to record sample", exc_info=True)
        return self._detect()

    def _record(self, result: _HasDriftStats, blocked: bool) -> None:
        """Append one sample to the rolling window."""
        bias_score = 0.0
        if result.bias is not None:
            try:
                bias_score = float(result.bias.score)
            except Exception:
                bias_score = 0.0

        sample = _Sample(
            confidence=float(result.confidence),
            citation_count=len(result.citations) if result.citations else 0,
            answer_length=len(result.answer) if result.answer else 0,
            bias_score=bias_score,
            blocked_flag=1 if (blocked or result.guardrail_replacement is not None) else 0,
        )
        self._samples.append(sample)

    def _detect(self) -> DriftReport:
        """Compare the current and baseline windows and build a report."""
        total = len(self._samples)
        if total < self.min_samples:
            return DriftReport(window_size=self.window_size, total_samples=total)

        # Use the most recent ``window_size`` samples as the current window
        # and the ``window_size`` samples before that as the baseline.
        current_window = self._samples[-self.window_size:]
        baseline_start = max(0, total - 2 * self.window_size)
        baseline_window = self._samples[baseline_start : baseline_start + self.window_size]

        if not baseline_window:
            return DriftReport(window_size=self.window_size, total_samples=total)

        feature_map = {
            FEATURE_CONFIDENCE: ([s.confidence for s in current_window], [s.confidence for s in baseline_window]),
            FEATURE_CITATION_COUNT: (
                [float(s.citation_count) for s in current_window],
                [float(s.citation_count) for s in baseline_window],
            ),
            FEATURE_ANSWER_LENGTH: (
                [float(s.answer_length) for s in current_window],
                [float(s.answer_length) for s in baseline_window],
            ),
            FEATURE_BIAS_SCORE: ([s.bias_score for s in current_window], [s.bias_score for s in baseline_window]),
            FEATURE_BLOCKED_FLAG: (
                [float(s.blocked_flag) for s in current_window],
                [float(s.blocked_flag) for s in baseline_window],
            ),
        }

        features: dict[str, DriftFeatureReport] = {}
        drift_detected = False
        for name, (current_values, baseline_values) in feature_map.items():
            report = _build_feature_report(current_values, baseline_values, self.threshold)
            features[name] = report
            if report.drifted:
                drift_detected = True

        if drift_detected:
            logger.info(
                "DriftMonitor: drift detected — features=%s",
                [name for name, r in features.items() if r.drifted],
            )
        return DriftReport(
            drift_detected=drift_detected,
            features=features,
            window_size=self.window_size,
            total_samples=total,
        )

    def reset(self) -> None:
        """Clear the rolling window (for tests / admin)."""
        self._samples = []


class PassthroughDriftMonitor:
    """No-op drift monitor — ``assess`` always returns a clean report."""

    def assess(self, result: _HasDriftStats | None, blocked: bool = False) -> DriftReport:  # noqa: D401
        return DriftReport()

    def reset(self) -> None:
        """No-op reset."""
