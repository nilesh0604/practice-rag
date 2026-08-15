"""Tests for the retrieval eval pipeline (eval/run_retrieval_eval.py) — Phase 3.

Tests the pure-logic parts that don't require Ollama, NIM, or Qdrant:
- ``load_dataset`` — golden dataset → (question, source_title) pairs.
- ``compute_hit`` — recall@k hit/rank logic incl. substring tolerance.
- ``aggregate`` — recall@k + latency aggregation.
- ``write_csv`` — CSV report generation.

The live ``run_collection`` / ``build_*_retriever`` paths (which call Qdrant)
are not unit-tested here — they require a live stack and are exercised by
running ``python eval/run_retrieval_eval.py --limit 5`` manually.
"""

from __future__ import annotations

import csv
from pathlib import Path

from eval.run_retrieval_eval import (
    DEFAULT_DATASET,
    RetrievalExample,
    RetrievalRow,
    aggregate,
    compute_hit,
    load_dataset,
    write_csv,
)


# ── load_dataset ───────────────────────────────────────────────────────


class TestLoadDataset:
    def test_loads_default_dataset(self):
        examples = load_dataset()
        assert len(examples) >= 30  # golden-dataset.json has 36
        assert all(isinstance(ex, RetrievalExample) for ex in examples)

    def test_every_example_has_question_and_title(self):
        for ex in load_dataset():
            assert ex.question
            assert ex.source_title

    def test_questions_match_golden(self):
        from eval.run_eval import load_dataset as load_golden

        golden = load_golden()
        retrieval = load_dataset()
        assert len(retrieval) == len(golden)
        assert [ex.question for ex in retrieval] == [ex.question for ex in golden]


# ── compute_hit ────────────────────────────────────────────────────────


class TestComputeHit:
    def test_exact_match_at_rank_1(self):
        hit, rank = compute_hit(["Path Parameters - FastAPI"], "Path Parameters - FastAPI")
        assert hit is True and rank == 1

    def test_substring_match_ground_truth_in_title(self):
        hit, rank = compute_hit(["Path Parameters - FastAPI"], "Path Parameters")
        assert hit is True and rank == 1

    def test_substring_match_title_in_ground_truth(self):
        hit, rank = compute_hit(["Models"], "Models - Pydantic v2")
        assert hit is True and rank == 1

    def test_case_insensitive(self):
        hit, rank = compute_hit(["path parameters - fastapi"], "Path Parameters - FastAPI")
        assert hit is True and rank == 1

    def test_miss_returns_rank_zero(self):
        hit, rank = compute_hit(["Query Parameters - FastAPI"], "Path Parameters - FastAPI")
        assert hit is False and rank == 0

    def test_rank_is_position_of_hit(self):
        titles = ["A", "B", "Path Parameters - FastAPI", "D", "E"]
        hit, rank = compute_hit(titles, "Path Parameters - FastAPI")
        assert hit is True and rank == 3

    def test_empty_titles(self):
        hit, rank = compute_hit([], "Path Parameters - FastAPI")
        assert hit is False and rank == 0

    def test_empty_ground_truth(self):
        hit, rank = compute_hit(["Path Parameters - FastAPI"], "")
        assert hit is False and rank == 0


# ── aggregate ──────────────────────────────────────────────────────────


def _row(question: str, hit: bool, rank: int, error: str = "") -> RetrievalRow:
    return RetrievalRow(
        backend="ollama",
        question=question,
        source_title="t",
        hit=hit,
        rank=rank,
        top_titles="",
        latency_s=0.1,
        error=error,
    )


class TestAggregate:
    def test_recall_is_hits_over_valid(self):
        rows = [
            _row("q1", True, 1),
            _row("q2", True, 3),
            _row("q3", False, 0),
            _row("q4", True, 5),
        ]
        s = aggregate(rows, "ollama", available=True)
        assert s.hits == 3
        assert s.recall_at_k == 3 / 4

    def test_errors_excluded_from_recall_denominator(self):
        rows = [
            _row("q1", True, 1),
            _row("q2", False, 0, error="boom"),
            _row("q3", True, 2),
        ]
        s = aggregate(rows, "ollama", available=True)
        assert s.errors == 1
        assert s.recall_at_k == 2 / 2  # only the two non-error rows

    def test_mean_latency_over_all_rows(self):
        rows = [
            _row("q1", True, 1),
            RetrievalRow("ollama", "q2", "t", False, 0, "", 0.3, error=""),
        ]
        s = aggregate(rows, "ollama", available=True)
        assert s.mean_latency == 0.2

    def test_empty_rows_safe(self):
        s = aggregate([], "ollama", available=True)
        assert s.recall_at_k == 0.0
        assert s.mean_latency == 0.0

    def test_unavailable_collection(self):
        s = aggregate([_row("q1", True, 1)], "nim", available=False)
        assert s.available is False


# ── write_csv ──────────────────────────────────────────────────────────


class TestWriteCsv:
    def test_writes_header_and_rows(self, tmp_path: Path):
        path = tmp_path / "r.csv"
        rows = [
            _row("q1", True, 1),
            _row("q2", False, 0),
        ]
        write_csv(rows, path)
        with path.open(encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        assert len(data) == 2
        assert data[0]["hit"] == "True"
        assert data[0]["rank"] == "1"
        assert data[1]["hit"] == "False"

    def test_appends_when_file_exists(self, tmp_path: Path):
        path = tmp_path / "r.csv"
        write_csv([_row("q1", True, 1)], path)
        write_csv([_row("q2", False, 0)], path)
        with path.open(encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        assert len(data) == 2
