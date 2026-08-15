"""Tests for the eval pipeline (eval/run_eval.py) — Step 6.

Tests the pure-logic parts that don't require Ollama or Qdrant:
- ``load_dataset`` — golden dataset JSON parsing.
- ``check_gate`` — threshold gate logic.
- ``mark_passes`` — per-question pass/fail marking.
- ``aggregate`` — summary aggregation.
- ``write_csv`` — CSV report generation.
- ``GoldenExample`` / ``EvalResult`` / ``EvalSummary`` dataclasses.

The ragas/local-judge backends (which call Ollama) are not unit-tested
here — they require a live Ollama + Qdrant stack and are exercised by
running ``python eval/run_eval.py --limit 1`` manually.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eval.run_eval import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    DEFAULT_THRESHOLD_FAITHFULNESS,
    DEFAULT_THRESHOLD_RECALL,
    DEFAULT_THRESHOLD_RELEVANCY,
    EvalResult,
    EvalSummary,
    GoldenExample,
    aggregate,
    check_gate,
    load_dataset,
    mark_passes,
    write_csv,
)


# ── load_dataset ───────────────────────────────────────────────────────


class TestLoadDataset:
    def test_loads_default_dataset(self):
        examples = load_dataset()
        assert len(examples) == 36
        assert all(isinstance(ex, GoldenExample) for ex in examples)

    def test_every_example_has_required_fields(self):
        examples = load_dataset()
        for ex in examples:
            assert ex.question
            assert ex.ground_truth
            assert ex.ground_truth_context
            assert ex.source_title

    def test_questions_are_unique(self):
        examples = load_dataset()
        questions = [ex.question for ex in examples]
        assert len(questions) == len(set(questions))

    def test_covers_all_source_titles(self):
        """The 35 questions should cover all 7 corpus documents."""
        examples = load_dataset()
        titles = {ex.source_title for ex in examples}
        expected = {
            "Path Parameters - FastAPI",
            "Query Parameters - FastAPI",
            "Dependency Injection - FastAPI",
            "Models - Pydantic v2",
            "Types - Pydantic v2",
            "Introduction to SQLModel",
            "Relationships - SQLModel",
        }
        assert expected.issubset(titles)

    def test_loads_from_custom_path(self, tmp_path: Path):
        data = {
            "examples": [
                {
                    "question": "test question?",
                    "ground_truth": "test answer",
                    "ground_truth_context": "test context",
                    "source_title": "Test Source",
                },
            ],
        }
        path = tmp_path / "custom-dataset.json"
        path.write_text(json.dumps(data))
        examples = load_dataset(path)
        assert len(examples) == 1
        assert examples[0].question == "test question?"

    def test_missing_examples_key_returns_empty(self, tmp_path: Path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"description": "empty"}))
        assert load_dataset(path) == []

    def test_missing_source_title_defaults_to_empty(self, tmp_path: Path):
        path = tmp_path / "no-source.json"
        path.write_text(
            json.dumps(
                {
                    "examples": [
                        {
                            "question": "q",
                            "ground_truth": "a",
                            "ground_truth_context": "c",
                        },
                    ],
                },
            ),
        )
        examples = load_dataset(path)
        assert examples[0].source_title == ""


# ── check_gate ─────────────────────────────────────────────────────────


class TestCheckGate:
    def test_all_above_threshold_passes(self):
        summary = EvalSummary(
            total=10,
            mean_faithfulness=0.85,
            mean_relevancy=0.80,
            mean_recall=0.75,
        )
        assert check_gate(summary) is True
        assert summary.gate_passed is True

    def test_faithfulness_below_threshold_fails(self):
        summary = EvalSummary(
            mean_faithfulness=0.70,
            mean_relevancy=0.80,
            mean_recall=0.75,
        )
        assert check_gate(summary) is False

    def test_recall_below_threshold_fails(self):
        summary = EvalSummary(
            mean_faithfulness=0.80,
            mean_relevancy=0.80,
            mean_recall=0.65,
        )
        assert check_gate(summary) is False

    def test_relevancy_below_threshold_fails(self):
        summary = EvalSummary(
            mean_faithfulness=0.80,
            mean_relevancy=0.60,
            mean_recall=0.75,
        )
        assert check_gate(summary) is False

    def test_custom_thresholds(self):
        summary = EvalSummary(
            mean_faithfulness=0.80,
            mean_relevancy=0.80,
            mean_recall=0.80,
        )
        # With higher thresholds, 0.80 fails faithfulness (0.85 required).
        assert check_gate(summary, threshold_faithfulness=0.85) is False
        # With lower thresholds, 0.80 passes.
        assert check_gate(
            summary,
            threshold_faithfulness=0.70,
            threshold_recall=0.70,
            threshold_relevancy=0.70,
        ) is True

    def test_exact_threshold_passes(self):
        """A mean exactly at the threshold passes (>= comparison)."""
        summary = EvalSummary(
            mean_faithfulness=DEFAULT_THRESHOLD_FAITHFULNESS,
            mean_relevancy=DEFAULT_THRESHOLD_RELEVANCY,
            mean_recall=DEFAULT_THRESHOLD_RECALL,
        )
        assert check_gate(summary) is True


# ── mark_passes ────────────────────────────────────────────────────────


class TestMarkPasses:
    def test_all_passing(self):
        results = [
            EvalResult(
                question="q1", answer="a", faithfulness=0.9,
                answer_relevancy=0.9, context_recall=0.9, latency_s=1.0,
                source_title="S",
            ),
            EvalResult(
                question="q2", answer="a", faithfulness=0.8,
                answer_relevancy=0.8, context_recall=0.8, latency_s=1.0,
                source_title="S",
            ),
        ]
        mark_passes(results)
        assert all(r.passed for r in results)

    def test_one_fails_on_faithfulness(self):
        results = [
            EvalResult(
                question="q", answer="a", faithfulness=0.5,
                answer_relevancy=0.9, context_recall=0.9, latency_s=1.0,
                source_title="S",
            ),
        ]
        mark_passes(results)
        assert results[0].passed is False

    def test_error_marks_as_failed(self):
        results = [
            EvalResult(
                question="q", answer="", faithfulness=0.9,
                answer_relevancy=0.9, context_recall=0.9, latency_s=1.0,
                source_title="S", error="connection refused",
            ),
        ]
        mark_passes(results)
        assert results[0].passed is False

    def test_custom_thresholds(self):
        results = [
            EvalResult(
                question="q", answer="a", faithfulness=0.75,
                answer_relevancy=0.75, context_recall=0.75, latency_s=1.0,
                source_title="S",
            ),
        ]
        # With higher thresholds, 0.75 fails.
        mark_passes(results, threshold_faithfulness=0.80, threshold_recall=0.80, threshold_relevancy=0.80)
        assert results[0].passed is False


# ── aggregate ──────────────────────────────────────────────────────────


class TestAggregate:
    def test_aggregates_means(self):
        results = [
            EvalResult(
                question="q1", answer="a", faithfulness=0.8,
                answer_relevancy=0.7, context_recall=0.9, latency_s=2.0,
                source_title="S",
            ),
            EvalResult(
                question="q2", answer="a", faithfulness=0.6,
                answer_relevancy=0.9, context_recall=0.5, latency_s=4.0,
                source_title="S",
            ),
        ]
        mark_passes(results)
        summary = aggregate(results)
        assert summary.total == 2
        assert summary.mean_faithfulness == pytest.approx(0.7)
        assert summary.mean_relevancy == pytest.approx(0.8)
        assert summary.mean_recall == pytest.approx(0.7)
        assert summary.mean_latency == pytest.approx(3.0)

    def test_excludes_errors_from_means(self):
        results = [
            EvalResult(
                question="q1", answer="a", faithfulness=0.9,
                answer_relevancy=0.9, context_recall=0.9, latency_s=1.0,
                source_title="S",
            ),
            EvalResult(
                question="q2", answer="", faithfulness=0.0,
                answer_relevancy=0.0, context_recall=0.0, latency_s=1.0,
                source_title="S", error="failed",
            ),
        ]
        summary = aggregate(results)
        assert summary.total == 2
        # Only the non-error result contributes to means.
        assert summary.mean_faithfulness == pytest.approx(0.9)
        assert summary.mean_relevancy == pytest.approx(0.9)
        assert summary.mean_recall == pytest.approx(0.9)
        # But latency includes all results.
        assert summary.mean_latency == pytest.approx(1.0)

    def test_passed_count(self):
        results = [
            EvalResult(
                question="q1", answer="a", faithfulness=0.9,
                answer_relevancy=0.9, context_recall=0.9, latency_s=1.0,
                source_title="S",
            ),
            EvalResult(
                question="q2", answer="a", faithfulness=0.5,
                answer_relevancy=0.9, context_recall=0.9, latency_s=1.0,
                source_title="S",
            ),
        ]
        mark_passes(results)
        summary = aggregate(results)
        assert summary.passed == 1

    def test_empty_results(self):
        summary = aggregate([])
        assert summary.total == 0
        assert summary.mean_latency == 0.0

    def test_all_errors(self):
        results = [
            EvalResult(
                question="q", answer="", faithfulness=0.0,
                answer_relevancy=0.0, context_recall=0.0, latency_s=1.0,
                source_title="S", error="err",
            ),
        ]
        summary = aggregate(results)
        # With all errors, means are 0 (n=1 guard prevents div-by-zero).
        assert summary.mean_faithfulness == 0.0


# ── write_csv ──────────────────────────────────────────────────────────


class TestWriteCsv:
    def test_writes_header_and_rows(self, tmp_path: Path):
        results = [
            EvalResult(
                question="q1", answer="a1", faithfulness=0.85,
                answer_relevancy=0.90, context_recall=0.78, latency_s=2.5,
                source_title="FastAPI Docs", passed=True,
            ),
            EvalResult(
                question="q2", answer="a2", faithfulness=0.60,
                answer_relevancy=0.70, context_recall=0.65, latency_s=3.0,
                source_title="Pydantic Docs", passed=False, error="timeout",
            ),
        ]
        path = tmp_path / "report.csv"
        write_csv(results, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["question"] == "q1"
        assert rows[0]["source_title"] == "FastAPI Docs"
        assert rows[0]["faithfulness"] == "0.8500"
        assert rows[0]["passed"] == "True"
        assert rows[1]["error"] == "timeout"
        assert rows[1]["passed"] == "False"

    def test_csv_has_all_columns(self, tmp_path: Path):
        results = [
            EvalResult(
                question="q", answer="a", faithfulness=0.5,
                answer_relevancy=0.5, context_recall=0.5, latency_s=1.0,
                source_title="S",
            ),
        ]
        path = tmp_path / "report.csv"
        write_csv(results, path)
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected = {
                "question", "source_title", "faithfulness",
                "answer_relevancy", "context_recall", "latency_s",
                "passed", "error",
            }
            assert set(reader.fieldnames) == expected

    def test_empty_results_writes_header_only(self, tmp_path: Path):
        path = tmp_path / "empty.csv"
        write_csv([], path)
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            rows = list(reader)
            assert rows == []
