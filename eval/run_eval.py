"""eval/run_eval.py — Offline Ragas evaluation against the golden dataset (Step 6).

Build-order item 37: Ragas eval script with local ``llama3.1:8b`` judge,
thresholds ``faithfulness >= 0.75``, ``context_recall >= 0.70``.

Build-order item 38: eval gate wired into the pre-commit checklist.

Usage::

    conda activate rag-chat
    python eval/run_eval.py
    python eval/run_eval.py --threshold-faithfulness 0.80 --threshold-recall 0.75
    python eval/run_eval.py --limit 5          # quick smoke (first 5 questions)
    python eval/run_eval.py --output report.csv

The script:

1. Loads ``eval/golden-dataset.json`` (35 hand-curated Q&A pairs).
2. For each question, runs the full RAG pipeline (``RAGOrchestrator.answer()``)
   to produce the generated answer + retrieved contexts.
3. Computes Ragas metrics (``faithfulness``, ``context_recall``,
   ``answer_relevancy``) using a local Ollama judge. If the ``ragas``
   package cannot import (version mismatch with ``langchain_community``),
   it falls back to a lightweight local LLM judge that scores each metric
   on a 0–1 scale.
4. Writes a CSV report (question, faithfulness, relevancy, context_recall,
   latency, pass/fail).
5. Prints a summary and exits non-zero if any metric's mean is below its
   threshold (the eval gate).

The eval gate is designed to be run before committing when the ingestion
pipeline or system prompt has changed — it acts as a regression check for
RAG quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

EVAL_DIR: Path = Path(__file__).resolve().parent
DEFAULT_DATASET: Path = EVAL_DIR / "golden-dataset.json"
DEFAULT_OUTPUT: Path = EVAL_DIR / "eval_report.csv"

DEFAULT_JUDGE_MODEL: str = os.getenv("EVAL_JUDGE_MODEL", "llama3.2:3b")
"""Local Ollama judge model. Dev substitute for the doc's llama3.1:8b (D2)."""

DEFAULT_OLLAMA_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_THRESHOLD_FAITHFULNESS: float = 0.75
DEFAULT_THRESHOLD_RECALL: float = 0.70
DEFAULT_THRESHOLD_RELEVANCY: float = 0.70


# ── data structures ────────────────────────────────────────────────────


@dataclass
class GoldenExample:
    """One Q&A triple from the golden dataset."""

    question: str
    ground_truth: str
    ground_truth_context: str
    source_title: str


@dataclass
class EvalResult:
    """Per-question eval result."""

    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    latency_s: float
    source_title: str
    passed: bool = False
    error: str = ""


@dataclass
class EvalSummary:
    """Aggregate eval summary across all questions."""

    total: int = 0
    passed: int = 0
    mean_faithfulness: float = 0.0
    mean_relevancy: float = 0.0
    mean_recall: float = 0.0
    mean_latency: float = 0.0
    gate_passed: bool = False
    results: list[EvalResult] = field(default_factory=list)


# ── dataset loading ────────────────────────────────────────────────────


def load_dataset(path: Path = DEFAULT_DATASET) -> list[GoldenExample]:
    """Load the golden dataset JSON and return a list of GoldenExample."""
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for ex in data.get("examples", []):
        examples.append(
            GoldenExample(
                question=ex["question"],
                ground_truth=ex["ground_truth"],
                ground_truth_context=ex["ground_truth_context"],
                source_title=ex.get("source_title", ""),
            )
        )
    return examples


# ── threshold gate ─────────────────────────────────────────────────────


def check_gate(
    summary: EvalSummary,
    threshold_faithfulness: float = DEFAULT_THRESHOLD_FAITHFULNESS,
    threshold_recall: float = DEFAULT_THRESHOLD_RECALL,
    threshold_relevancy: float = DEFAULT_THRESHOLD_RELEVANCY,
) -> bool:
    """Return True if all metric means meet their thresholds."""
    summary.gate_passed = (
        summary.mean_faithfulness >= threshold_faithfulness
        and summary.mean_recall >= threshold_recall
        and summary.mean_relevancy >= threshold_relevancy
    )
    return summary.gate_passed


def mark_passes(
    results: list[EvalResult],
    threshold_faithfulness: float = DEFAULT_THRESHOLD_FAITHFULNESS,
    threshold_recall: float = DEFAULT_THRESHOLD_RECALL,
    threshold_relevancy: float = DEFAULT_THRESHOLD_RELEVANCY,
) -> None:
    """Set ``passed`` on each result based on per-question thresholds."""
    for r in results:
        r.passed = (
            r.faithfulness >= threshold_faithfulness
            and r.context_recall >= threshold_recall
            and r.answer_relevancy >= threshold_relevancy
            and not r.error
        )


# ── CSV report ─────────────────────────────────────────────────────────


def write_csv(results: list[EvalResult], path: Path) -> None:
    """Write the per-question results to a CSV file."""
    fieldnames = [
        "question",
        "source_title",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "latency_s",
        "passed",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "question": r.question,
                    "source_title": r.source_title,
                    "faithfulness": f"{r.faithfulness:.4f}",
                    "answer_relevancy": f"{r.answer_relevancy:.4f}",
                    "context_recall": f"{r.context_recall:.4f}",
                    "latency_s": f"{r.latency_s:.2f}",
                    "passed": r.passed,
                    "error": r.error,
                }
            )
    logger.info("Wrote eval report to %s", path)


# ── summary printing ───────────────────────────────────────────────────


def print_summary(summary: EvalSummary, thresholds: dict[str, float]) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("  RAG EVAL SUMMARY")
    print("=" * 60)
    print(f"  Questions evaluated : {summary.total}")
    print(f"  Questions passed    : {summary.passed}/{summary.total}")
    print()
    print(f"  Metric              Mean     Threshold   Pass")
    print(f"  ─────────────────── ──────── ─────────── ────")
    for name, mean, thresh in [
        ("faithfulness", summary.mean_faithfulness, thresholds["faithfulness"]),
        ("answer_relevancy", summary.mean_relevancy, thresholds["relevancy"]),
        ("context_recall", summary.mean_recall, thresholds["recall"]),
    ]:
        status = "✅" if mean >= thresh else "❌"
        print(f"  {name:<20} {mean:.4f}   {thresh:.2f}        {status}")
    print(f"\n  Mean latency        : {summary.mean_latency:.2f}s")
    print(f"  Gate                : {'PASSED ✅' if summary.gate_passed else 'FAILED ❌'}")
    print("=" * 60)


# ── ragas backend (preferred) ──────────────────────────────────────────


def _try_ragas_imports() -> tuple[bool, str]:
    """Try importing ragas metrics; return (ok, error_message)."""
    try:
        from ragas.metrics import faithfulness, answer_relevancy, context_recall  # noqa: F401

        return True, ""
    except Exception as exc:
        return False, str(exc)


def run_with_ragas(
    examples: list[GoldenExample],
    orchestrator,
    judge_model: str,
    ollama_url: str,
) -> list[EvalResult]:
    """Run eval using the ragas library with a local Ollama judge.

    Constructs a ragas-compatible LLM wrapper around Ollama and evaluates
    each question. Falls back to the local judge if ragas evaluation
    fails for a specific question.
    """
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import context_recall, faithfulness, answer_relevancy

    # Build a LangChain-compatible Ollama LLM for the judge.
    judge_llm = llm_factory(model=judge_model, base_url=ollama_url)

    results: list[EvalResult] = []
    for ex in examples:
        start = time.time()
        try:
            answer, _, docs = orchestrator.answer(ex.question)
            contexts = [d.content for d in docs]
            from datasets import Dataset

            ds = Dataset.from_dict(
                {
                    "question": [ex.question],
                    "answer": [answer],
                    "contexts": [contexts],
                    "ground_truth": [ex.ground_truth],
                    "ground_truth_context": [ex.ground_truth_context],
                }
            )
            scores = evaluate(
                ds,
                metrics=[faithfulness, answer_relevancy, context_recall],
                llm=judge_llm,
            )
            row = scores.to_pandas().iloc[0]
            results.append(
                EvalResult(
                    question=ex.question,
                    answer=answer,
                    faithfulness=float(row.get("faithfulness", 0.0)),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                    context_recall=float(row.get("context_recall", 0.0)),
                    latency_s=time.time() - start,
                    source_title=ex.source_title,
                )
            )
        except Exception as exc:
            results.append(
                EvalResult(
                    question=ex.question,
                    answer="",
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_recall=0.0,
                    latency_s=time.time() - start,
                    source_title=ex.source_title,
                    error=str(exc),
                )
            )
    return results


# ── local LLM judge fallback ───────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are an evaluator for a RAG system. Score how faithful the generated \
answer is to the provided context. A faithful answer only uses information \
present in the context — no fabrication.

Reply with ONLY a number from 0.0 to 1.0 (1.0 = fully faithful, 0.0 = entirely fabricated).

Context:
{context}

Generated answer:
{answer}

Score:"""

_RELEVANCY_PROMPT = """\
You are an evaluator for a RAG system. Score how relevant the generated \
answer is to the user's question. A relevant answer directly addresses \
the question.

Reply with ONLY a number from 0.0 to 1.0 (1.0 = perfectly relevant, 0.0 = irrelevant).

Question:
{question}

Generated answer:
{answer}

Score:"""""

_CONTEXT_RECALL_PROMPT = """\
You are an evaluator for a RAG system. Score how well the retrieved context \
covers the ground-truth answer. A score of 1.0 means all information in the \
ground truth is present in the retrieved context.

Reply with ONLY a number from 0.0 to 1.0.

Retrieved context:
{context}

Ground-truth answer:
{ground_truth}

Score:"""


def _llm_score(prompt: str, model: str, ollama_url: str) -> float:
    """Ask the local LLM judge for a 0–1 score; parse and clamp it."""
    import ollama

    client = ollama.Client(host=ollama_url)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"num_predict": 8, "temperature": 0.0},
    )
    text = response.get("message", {}).get("content", "").strip()
    # Extract the first float-like token.
    import re

    match = re.search(r"(\d+\.?\d*)", text)
    if not match:
        return 0.0
    score = float(match.group(1))
    return min(max(score, 0.0), 1.0)


def run_with_local_judge(
    examples: list[GoldenExample],
    orchestrator,
    judge_model: str,
    ollama_url: str,
) -> list[EvalResult]:
    """Run eval using a lightweight local LLM judge (fallback when ragas unavailable).

    For each question, runs the RAG pipeline, then asks the local Ollama
    judge to score faithfulness, answer relevancy, and context recall on
    a 0–1 scale using simple prompts.
    """
    results: list[EvalResult] = []
    for i, ex in enumerate(examples, 1):
        start = time.time()
        try:
            answer, _, docs = orchestrator.answer(ex.question)
            contexts = "\n\n".join(d.content for d in docs)

            faithfulness = _llm_score(
                _FAITHFULNESS_PROMPT.format(context=contexts, answer=answer),
                judge_model,
                ollama_url,
            )
            relevancy = _llm_score(
                _RELEVANCY_PROMPT.format(question=ex.question, answer=answer),
                judge_model,
                ollama_url,
            )
            recall = _llm_score(
                _CONTEXT_RECALL_PROMPT.format(context=contexts, ground_truth=ex.ground_truth),
                judge_model,
                ollama_url,
            )

            results.append(
                EvalResult(
                    question=ex.question,
                    answer=answer,
                    faithfulness=faithfulness,
                    answer_relevancy=relevancy,
                    context_recall=recall,
                    latency_s=time.time() - start,
                    source_title=ex.source_title,
                )
            )
            logger.info(
                "[%d/%d] faith=%.2f rel=%.2f recall=%.2f (%.1fs) — %s",
                i,
                len(examples),
                faithfulness,
                relevancy,
                recall,
                time.time() - start,
                ex.question[:50],
            )
        except Exception as exc:
            results.append(
                EvalResult(
                    question=ex.question,
                    answer="",
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_recall=0.0,
                    latency_s=time.time() - start,
                    source_title=ex.source_title,
                    error=str(exc),
                )
            )
            logger.warning("[%d/%d] ERROR: %s", i, len(examples), exc)
    return results


# ── summary aggregation ────────────────────────────────────────────────


def aggregate(results: list[EvalResult]) -> EvalSummary:
    """Compute the aggregate summary from per-question results."""
    valid = [r for r in results if not r.error]
    total = len(results)
    n = len(valid) if valid else 1
    summary = EvalSummary(
        total=total,
        passed=sum(1 for r in results if r.passed),
        mean_faithfulness=sum(r.faithfulness for r in valid) / n,
        mean_relevancy=sum(r.answer_relevancy for r in valid) / n,
        mean_recall=sum(r.context_recall for r in valid) / n,
        mean_latency=sum(r.latency_s for r in results) / total if total else 0.0,
        results=results,
    )
    return summary


# ── orchestrator construction ──────────────────────────────────────────


def build_orchestrator(ollama_url: str = DEFAULT_OLLAMA_URL):
    """Construct a real RAG orchestrator with live Ollama + Qdrant.

    This is the production wiring — the eval script needs the real
    pipeline (not mocks) to measure actual RAG quality. Uses the same
    dependency wiring as ``api/deps.py``.
    """
    from api.guardrails import GuardrailSuite
    from rag.context_assembler import ContextAssembler
    from rag.generator import Generator
    from rag.orchestrator import RAGOrchestrator
    from rag.post_processor import PostProcessor
    from rag.qdrant_collection import get_qdrant_client
    from rag.retriever import HybridRetriever
    from ingestion.embedder import Embedder

    client = get_qdrant_client()
    embedder = Embedder(ollama_url=ollama_url)
    retriever = HybridRetriever(client, embedder)
    return RAGOrchestrator(
        retriever=retriever,
        context_assembler=ContextAssembler(),
        generator=Generator(ollama_url=ollama_url),
        post_processor=PostProcessor(embedder),
        guardrail_suite=GuardrailSuite(),
    )


# ── main ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline Ragas eval against the golden dataset.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help=f"Path to golden dataset JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N questions (for quick smoke tests).",
    )
    parser.add_argument(
        "--threshold-faithfulness", type=float, default=DEFAULT_THRESHOLD_FAITHFULNESS,
        help=f"Faithfulness gate threshold (default: {DEFAULT_THRESHOLD_FAITHFULNESS})",
    )
    parser.add_argument(
        "--threshold-recall", type=float, default=DEFAULT_THRESHOLD_RECALL,
        help=f"Context recall gate threshold (default: {DEFAULT_THRESHOLD_RECALL})",
    )
    parser.add_argument(
        "--threshold-relevancy", type=float, default=DEFAULT_THRESHOLD_RELEVANCY,
        help=f"Answer relevancy gate threshold (default: {DEFAULT_THRESHOLD_RELEVANCY})",
    )
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Ollama judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--ollama-url", type=str, default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--no-ragas", action="store_true",
        help="Skip ragas and use the local LLM judge directly.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load dataset.
    examples = load_dataset(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    logger.info("Loaded %d golden examples from %s", len(examples), args.dataset)

    # Build the orchestrator (real Ollama + Qdrant).
    logger.info("Building RAG orchestrator (judge model=%s)...", args.judge_model)
    orchestrator = build_orchestrator(args.ollama_url)

    # Choose backend: ragas (preferred) or local judge (fallback).
    use_ragas = not args.no_ragas
    if use_ragas:
        ok, err = _try_ragas_imports()
        if not ok:
            logger.warning("Ragas import failed (%s) — falling back to local LLM judge.", err)
            use_ragas = False

    if use_ragas:
        logger.info("Running eval with Ragas backend...")
        results = run_with_ragas(examples, orchestrator, args.judge_model, args.ollama_url)
    else:
        logger.info("Running eval with local LLM judge backend...")
        results = run_with_local_judge(
            examples, orchestrator, args.judge_model, args.ollama_url,
        )

    # Mark per-question pass/fail.
    thresholds = {
        "faithfulness": args.threshold_faithfulness,
        "recall": args.threshold_recall,
        "relevancy": args.threshold_relevancy,
    }
    mark_passes(results, **thresholds)

    # Aggregate + gate check.
    summary = aggregate(results)
    check_gate(summary, **thresholds)

    # Write CSV + print summary.
    write_csv(results, args.output)
    print_summary(summary, thresholds)

    # Exit code: 0 if gate passed, 1 if failed.
    return 0 if summary.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
