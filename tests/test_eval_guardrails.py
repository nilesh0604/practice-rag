"""Tests for the guardrail eval pipeline (eval/run_guardrail_eval.py) — Phase 2.

Tests the pure-logic parts that don't require Ollama, NIM, or Qdrant:
- ``load_dataset`` — adversarial dataset JSON parsing.
- ``BinaryMetrics`` — precision/recall/F1/FPR/FNR arithmetic.
- ``ClassMetrics`` — per-class precision/recall/F1.
- ``aggregate`` — wiring of rows → per-check summaries.
- ``compute``-style helpers (``_bool_to_verdict``, ``_score_binary``).
- ``write_csv`` — CSV report generation.

The live ``run_example`` / ``build_suite`` paths (which call Ollama/NIM) are
not unit-tested here — they require a live stack and are exercised by running
``python eval/run_guardrail_eval.py --limit 5`` manually.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval.run_guardrail_eval import (
    DEFAULT_DATASET,
    GuardrailExample,
    GuardrailRow,
    GuardrailSummary,
    BinaryMetrics,
    ClassMetrics,
    _bool_to_verdict,
    _score_binary,
    aggregate,
    load_dataset,
    run_example,
    write_csv,
)


# ── load_dataset ───────────────────────────────────────────────────────


class TestLoadDataset:
    def test_loads_default_dataset(self):
        examples = load_dataset()
        assert len(examples) >= 40  # 48 examples in v1.0
        assert all(isinstance(ex, GuardrailExample) for ex in examples)

    def test_every_example_has_required_fields(self):
        for ex in load_dataset():
            assert ex.id
            assert ex.category
            assert ex.input
            assert ex.expected_input in ("block", "allow")

    def test_categories_are_balanced(self):
        examples = load_dataset()
        by_cat: dict[str, int] = {}
        for ex in examples:
            by_cat[ex.category] = by_cat.get(ex.category, 0) + 1
        # Each adversarial category needs enough samples for meaningful FPR/FNR.
        for cat in ("on_topic_safe", "off_topic", "prompt_injection", "harmful_output"):
            assert by_cat.get(cat, 0) >= 6, f"{cat} has only {by_cat.get(cat, 0)} samples"
        assert by_cat.get("borderline", 0) >= 4

    def test_ids_are_unique(self):
        examples = load_dataset()
        ids = [ex.id for ex in examples]
        assert len(ids) == len(set(ids))

    def test_harmful_output_examples_have_answer(self):
        for ex in load_dataset():
            if ex.category == "harmful_output" and ex.expected_output:
                assert ex.answer is not None, f"{ex.id} needs an answer to judge"

    def test_prompt_injection_examples_skip_output(self):
        for ex in load_dataset():
            if ex.category == "prompt_injection":
                assert ex.expected_output is None
                assert ex.expected_input == "block"


# ── BinaryMetrics ──────────────────────────────────────────────────────


class TestBinaryMetrics:
    def test_perfect_scores(self):
        m = BinaryMetrics(tp=10, fp=0, fn=0, tn=10)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.fpr == 0.0
        assert m.fnr == 0.0

    def test_all_false_positives(self):
        m = BinaryMetrics(tp=0, fp=5, fn=0, tn=5)
        assert m.precision == 0.0
        assert m.fpr == 5 / 10
        assert m.fnr == 0.0  # no attacks, so no false negatives

    def test_all_false_negatives(self):
        m = BinaryMetrics(tp=0, fp=0, fn=5, tn=0)
        assert m.recall == 0.0
        assert m.fnr == 1.0
        assert m.fpr == 0.0  # no legitimate examples

    def test_zero_denominators_are_safe(self):
        m = BinaryMetrics()
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.fpr == 0.0
        assert m.fnr == 0.0

    def test_f1_harmonic_mean(self):
        m = BinaryMetrics(tp=6, fp=4, fn=2, tn=8)
        assert m.precision == pytest.approx(0.6)
        assert m.recall == pytest.approx(0.75)
        assert m.f1 == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))


# ── ClassMetrics ───────────────────────────────────────────────────────


class TestClassMetrics:
    def test_perfect_class(self):
        cm = ClassMetrics(label="documentation", tp=10, fp=0, fn=0)
        assert cm.precision == 1.0
        assert cm.recall == 1.0
        assert cm.f1 == 1.0

    def test_zero_denominators_are_safe(self):
        cm = ClassMetrics(label="x")
        assert cm.precision == 0.0
        assert cm.f1 == 0.0


# ── _bool_to_verdict / _score_binary ───────────────────────────────────


class TestHelpers:
    def test_bool_to_verdict(self):
        assert _bool_to_verdict(True) == "block"
        assert _bool_to_verdict(False) == "allow"

    def test_score_binary_tp(self):
        m = BinaryMetrics()
        _score_binary(m, "block", "block")
        assert m.tp == 1

    def test_score_binary_fn(self):
        m = BinaryMetrics()
        _score_binary(m, "block", "allow")
        assert m.fn == 1

    def test_score_binary_fp(self):
        m = BinaryMetrics()
        _score_binary(m, "allow", "block")
        assert m.fp == 1

    def test_score_binary_tn(self):
        m = BinaryMetrics()
        _score_binary(m, "allow", "allow")
        assert m.tn == 1

    def test_score_binary_error(self):
        m = BinaryMetrics()
        _score_binary(m, "block", "allow", error=True)
        assert m.errors == 1
        assert m.tp == 0 and m.fn == 0


# ── aggregate ──────────────────────────────────────────────────────────


def _row(
    example_id: str,
    expected_input: str,
    actual_input: str,
    expected_class: str = "",
    actual_class: str = "",
    expected_output: str = "",
    actual_output: str = "",
    error: str = "",
) -> GuardrailRow:
    return GuardrailRow(
        backend="ollama",
        example_id=example_id,
        category="x",
        input_text="q",
        expected_input=expected_input,
        actual_input=actual_input,
        input_correct=(expected_input == actual_input) and not error,
        expected_class=expected_class,
        actual_class=actual_class,
        class_correct=bool(expected_class) and actual_class == expected_class,
        expected_output=expected_output,
        actual_output=actual_output,
        output_correct=bool(expected_output) and actual_output == expected_output,
        input_reason="",
        output_reason="",
        error=error,
    )


class TestAggregate:
    def test_input_metrics_accumulate(self):
        rows = [
            _row("1", "block", "block"),
            _row("2", "block", "allow"),  # fn
            _row("3", "allow", "block"),  # fp
            _row("4", "allow", "allow"),
        ]
        s = aggregate(rows, "ollama")
        assert s.input_metrics.tp == 1
        assert s.input_metrics.fn == 1
        assert s.input_metrics.fp == 1
        assert s.input_metrics.tn == 1

    def test_output_metrics_skip_null_expected(self):
        rows = [
            _row("1", "allow", "allow", expected_output="allow", actual_output="allow"),
            _row("2", "allow", "allow", expected_output="block", actual_output="allow"),
            _row("3", "allow", "allow"),  # no expected_output -> skipped
        ]
        s = aggregate(rows, "ollama")
        assert s.output_metrics.total == 2
        assert s.output_metrics.tn == 1
        assert s.output_metrics.fn == 1

    def test_topic_reject_binary_view(self):
        rows = [
            _row("1", "allow", "allow", expected_class="off_topic", actual_class="off_topic"),
            _row("2", "allow", "allow", expected_class="documentation", actual_class="off_topic"),
            _row("3", "allow", "allow", expected_class="off_topic", actual_class="documentation"),
            _row("4", "allow", "allow", expected_class="documentation", actual_class="documentation"),
        ]
        s = aggregate(rows, "ollama")
        # positive = rejected (off_topic)
        assert s.topic_reject_metrics.tp == 1  # off_topic expected + actual
        assert s.topic_reject_metrics.fp == 1  # documentation expected, off_topic actual
        assert s.topic_reject_metrics.fn == 1  # off_topic expected, documentation actual
        assert s.topic_reject_metrics.tn == 1  # documentation expected + actual

    def test_per_class_metrics(self):
        rows = [
            _row("1", "allow", "allow", expected_class="documentation", actual_class="documentation"),
            _row("2", "allow", "allow", expected_class="documentation", actual_class="off_topic"),
            _row("3", "allow", "allow", expected_class="off_topic", actual_class="off_topic"),
        ]
        s = aggregate(rows, "ollama")
        doc = s.class_metrics["documentation"]
        off = s.class_metrics["off_topic"]
        assert doc.tp == 1 and doc.fn == 1
        assert off.tp == 1 and off.fp == 1  # the misrouted doc became an off_topic fp


# ── run_example (mocked suite) ─────────────────────────────────────────


class TestRunExample:
    def _mock_suite(self, input_blocked=False, class_label="documentation", output_blocked=False):
        suite = MagicMock()
        suite.check_input.return_value = MagicMock(
            blocked=input_blocked, reason="r" if input_blocked else "",
        )
        suite.classify.return_value = MagicMock(label=class_label, handled=False)
        suite.check_output.return_value = MagicMock(
            blocked=output_blocked, reason="r" if output_blocked else "",
        )
        return suite

    def test_records_input_block(self):
        ex = GuardrailExample(
            id="x", category="prompt_injection", input="ignore prev",
            answer=None, history="", expected_input="block",
            expected_class=None, expected_output=None,
        )
        out = run_example(self._mock_suite(input_blocked=True), ex)
        assert out.input_blocked is True
        assert out.class_label == "documentation"

    def test_runs_output_when_answer_and_expected_present(self):
        ex = GuardrailExample(
            id="x", category="harmful_output", input="q",
            answer="bad answer", history="", expected_input="allow",
            expected_class="documentation", expected_output="block",
        )
        out = run_example(self._mock_suite(output_blocked=True), ex)
        assert out.output_blocked is True

    def test_skips_output_when_answer_none(self):
        ex = GuardrailExample(
            id="x", category="off_topic", input="weather",
            answer=None, history="", expected_input="allow",
            expected_class="off_topic", expected_output=None,
        )
        out = run_example(self._mock_suite(), ex)
        assert out.output_blocked is False
        suite = self._mock_suite()
        run_example(suite, ex)
        suite.check_output.assert_not_called()

    def test_records_error_without_aborting(self):
        suite = MagicMock()
        suite.check_input.side_effect = RuntimeError("boom")
        ex = GuardrailExample(
            id="x", category="x", input="q", answer=None, history="",
            expected_input="allow", expected_class=None, expected_output=None,
        )
        out = run_example(suite, ex)
        assert out.error == "boom"


# ── write_csv ──────────────────────────────────────────────────────────


class TestWriteCsv:
    def test_writes_header_and_rows(self, tmp_path: Path):
        path = tmp_path / "g.csv"
        rows = [
            _row("1", "block", "block", expected_class="off_topic", actual_class="off_topic"),
            _row("2", "allow", "allow", expected_output="allow", actual_output="allow"),
        ]
        write_csv(rows, path)
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 2
        assert data[0]["example_id"] == "1"
        assert data[0]["input_correct"] == "True"
        assert data[1]["expected_output"] == "allow"

    def test_appends_when_file_exists(self, tmp_path: Path):
        path = tmp_path / "g.csv"
        write_csv([_row("1", "block", "block")], path)
        write_csv([_row("2", "allow", "allow")], path)
        with path.open(encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        assert len(data) == 2
