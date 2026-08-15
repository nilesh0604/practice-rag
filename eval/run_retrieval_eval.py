"""eval/run_retrieval_eval.py — Retrieval recall@5 comparison (Phase 3 of the NIM plan).

The existing ``eval/run_eval.py`` uses Ragas ``context_recall`` as a proxy for
retrieval quality, and it only queries the default Ollama ``docs-knowledge``
collection. Phase 3 of the NIM integration plan calls for a **true recall@5**
comparison between the Ollama collection and the separate NIM-embedded
``ctc_rag_nim`` collection — i.e. for each golden question, does the
ground-truth ``source_title`` appear in the top-5 retrieved chunks?

This script reuses ``eval/golden-dataset.json`` (each example carries a
``source_title`` ground truth) and, for each question, queries both
collections via ``HybridRetriever.retrieve()``, then checks whether the
ground-truth title is present in the top-5 titles. It reports:

- recall@5 per collection (Ollama vs NIM),
- per-question hit/miss for both collections,
- the delta (NIM − Ollama) so the Phase 3 uplift is measurable,
- a CSV report (one row per question per collection).

Usage::

    conda activate rag-chat
    # Compare both collections (requires the NIM collection to be ingested):
    NIM_ENABLED=true python eval/run_retrieval_eval.py
    # Only the default Ollama collection:
    python eval/run_retrieval_eval.py --no-nim
    # Only the NIM collection:
    python eval/run_retrieval_eval.py --only-nim
    python eval/run_retrieval_eval.py --limit 5 --top-k 5
    python eval/run_retrieval_eval.py --output eval/retrieval_report.csv

Prerequisite: the NIM collection must already be ingested via
``python -m ingestion.nim_embedder`` (``run_nim_ingestion``) with
``NIM_ENABLED=true``. If the collection is missing or empty, the NIM column
reports ``n/a`` and the script does not fail the gate on it.

Gate: exits non-zero if the Ollama recall@5 is below ``--threshold``
(default 0.70). The NIM recall@5 is reported for comparison but does not
fail the gate (it is a comparison metric, not a release gate — the live app
keeps using the Ollama collection).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

EVAL_DIR: Path = Path(__file__).resolve().parent
DEFAULT_DATASET: Path = EVAL_DIR / "golden-dataset.json"
DEFAULT_OUTPUT: Path = EVAL_DIR / "retrieval_report.csv"

DEFAULT_TOP_K: int = 5
DEFAULT_THRESHOLD: float = 0.70

OLLAMA_BACKEND: str = "ollama"
NIM_BACKEND: str = "nim"


# ── data structures ────────────────────────────────────────────────────


@dataclass
class RetrievalExample:
    """One question + ground-truth source title from the golden dataset."""

    question: str
    source_title: str


@dataclass
class RetrievalRow:
    """Per-question retrieval result for one collection."""

    backend: str
    question: str
    source_title: str
    hit: bool
    rank: int  # 1-based rank of the ground-truth title, 0 if not in top-k
    top_titles: str  # pipe-separated top-k titles
    latency_s: float
    error: str = ""


@dataclass
class RetrievalSummary:
    """Aggregate recall@k for one collection."""

    backend: str
    total: int = 0
    hits: int = 0
    errors: int = 0
    recall_at_k: float = 0.0
    mean_latency: float = 0.0
    available: bool = True
    rows: list[RetrievalRow] = field(default_factory=list)


# ── dataset loading ────────────────────────────────────────────────────


def load_dataset(path: Path = DEFAULT_DATASET) -> list[RetrievalExample]:
    """Load the golden dataset and keep only question + source_title."""
    data = json.loads(path.read_text(encoding="utf-8"))
    examples: list[RetrievalExample] = []
    for ex in data.get("examples", []):
        title = ex.get("source_title", "")
        if not title:
            continue
        examples.append(RetrievalExample(question=ex["question"], source_title=title))
    return examples


# ── recall@k computation ───────────────────────────────────────────────


def compute_hit(
    titles: list[str],
    ground_truth: str,
) -> tuple[bool, int]:
    """Return (hit, rank) for a ground-truth title against a ranked title list.

    ``rank`` is 1-based; 0 means the title was not found in the list. The
    match is case-insensitive and substring-tolerant because chunk titles can
    carry slight suffixes (e.g. "Path Parameters - FastAPI" vs a stored
    "Path Parameters").
    """
    needle = ground_truth.strip().lower()
    # First pass: exact match (case-insensitive).
    for rank, title in enumerate(titles, start=1):
        hay = (title or "").strip().lower()
        if needle and needle == hay:
            return True, rank
    # Second pass: substring match, but only when both sides are long enough
    # that a substring overlap is meaningful (avoids single-char false matches
    # like "a" being found inside "path parameters").
    min_len = 5
    for rank, title in enumerate(titles, start=1):
        hay = (title or "").strip().lower()
        if (
            needle
            and len(needle) >= min_len
            and len(hay) >= min_len
            and (needle in hay or hay in needle)
        ):
            return True, rank
    return False, 0


def aggregate(rows: list[RetrievalRow], backend: str, available: bool) -> RetrievalSummary:
    """Compute recall@k + latency from per-question rows."""
    valid = [r for r in rows if not r.error]
    total = len(rows)
    summary = RetrievalSummary(
        backend=backend,
        total=total,
        hits=sum(1 for r in valid if r.hit),
        errors=sum(1 for r in rows if r.error),
        available=available,
        rows=rows,
    )
    if valid:
        summary.recall_at_k = summary.hits / len(valid)
    if rows:
        summary.mean_latency = sum(r.latency_s for r in rows) / total
    return summary


# ── CSV report ─────────────────────────────────────────────────────────


def write_csv(rows: list[RetrievalRow], path: Path) -> None:
    """Write per-question rows to CSV (appends if the file exists)."""
    fieldnames = [
        "backend",
        "question",
        "source_title",
        "hit",
        "rank",
        "top_titles",
        "latency_s",
        "error",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "backend": r.backend,
                    "question": r.question,
                    "source_title": r.source_title,
                    "hit": r.hit,
                    "rank": r.rank,
                    "top_titles": r.top_titles,
                    "latency_s": f"{r.latency_s:.3f}",
                    "error": r.error,
                }
            )
    logger.info("Wrote retrieval report to %s", path)


# ── retriever construction ─────────────────────────────────────────────


def build_ollama_retriever(ollama_url: str, top_k: int):
    """Build a HybridRetriever over the default Ollama ``docs-knowledge`` collection."""
    from ingestion.embedder import Embedder
    from rag.qdrant_collection import COLLECTION_NAME, get_qdrant_client
    from rag.retriever import HybridRetriever

    return HybridRetriever(
        client=get_qdrant_client(),
        embedder=Embedder(ollama_url=ollama_url),
        collection_name=COLLECTION_NAME,
        top_k=top_k,
    )


def build_nim_retriever(top_k: int):
    """Build a HybridRetriever over the NIM ``ctc_rag_nim`` comparison collection.

    Delegates to ``ingestion.nim_embedder.build_nim_retriever`` (already in the
    codebase) and overrides ``top_k`` for the eval.
    """
    from ingestion.nim_embedder import build_nim_retriever as _build

    retriever = _build()
    retriever.top_k = top_k
    return retriever


def collection_available(client, collection_name: str) -> bool:
    """Return True if the Qdrant collection exists and has points."""
    try:
        from qdrant_client.http.models import CountRequest

        info = client.get_collection(collection_name)
        count = client.count(collection_name, exact=False).count
        return bool(info) and count > 0
    except Exception as exc:  # noqa: BLE001
        logger.info("Collection %r not available: %s", collection_name, exc)
        return False


# ── running ────────────────────────────────────────────────────────────


def run_collection(
    examples: list[RetrievalExample],
    backend: str,
    retriever,
    top_k: int,
) -> list[RetrievalRow]:
    """Run retrieval for all examples against one retriever."""
    import time

    rows: list[RetrievalRow] = []
    for i, ex in enumerate(examples, 1):
        start = time.time()
        try:
            docs = retriever.retrieve(ex.question)
            titles = [d.title for d in docs[:top_k]]
            hit, rank = compute_hit(titles, ex.source_title)
            rows.append(
                RetrievalRow(
                    backend=backend,
                    question=ex.question,
                    source_title=ex.source_title,
                    hit=hit,
                    rank=rank,
                    top_titles=" | ".join(titles),
                    latency_s=time.time() - start,
                )
            )
            logger.info(
                "[%s %d/%d] hit=%s rank=%d — %s",
                backend, i, len(examples), hit, rank, ex.question[:50],
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                RetrievalRow(
                    backend=backend,
                    question=ex.question,
                    source_title=ex.source_title,
                    hit=False,
                    rank=0,
                    top_titles="",
                    latency_s=time.time() - start,
                    error=str(exc),
                )
            )
            logger.warning("[%s %d/%d] ERROR: %s", backend, i, len(examples), exc)
    return rows


# ── summary printing ───────────────────────────────────────────────────


def print_summary(
    ollama: RetrievalSummary,
    nim: RetrievalSummary | None,
    top_k: int,
    threshold: float,
) -> bool:
    """Print the recall@k comparison; return True if the Ollama gate passed."""
    print("\n" + "=" * 70)
    print(f"  RETRIEVAL RECALL@{top_k} EVAL SUMMARY")
    print("=" * 70)

    def _line(name: str, s: RetrievalSummary) -> None:
        avail = "" if s.available else "  [collection n/a]"
        print(
            f"  {name:<10} recall@{top_k}={s.recall_at_k:.3f}"
            f"  hits={s.hits}/{s.total}  err={s.errors}"
            f"  mean_latency={s.mean_latency:.3f}s{avail}"
        )

    _line("Ollama", ollama)
    if nim is not None:
        _line("NIM", nim)
        if ollama.available and nim.available and ollama.total and nim.total:
            delta = nim.recall_at_k - ollama.recall_at_k
            sign = "+" if delta >= 0 else ""
            print(f"  Δ (NIM − Ollama)   = {sign}{delta:.3f}")

    gate_ok = ollama.available and ollama.recall_at_k >= threshold
    print(
        f"\n  Gate (Ollama recall@{top_k} >= {threshold:.2f}): "
        f"{'PASSED' if gate_ok else 'FAILED'}",
    )
    print("=" * 70)
    return gate_ok


# ── main ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run retrieval recall@k comparison (Ollama vs NIM collections).",
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
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"K for recall@K (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Ollama recall@k gate threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--no-nim", action="store_true",
        help="Skip the NIM collection (Ollama only).",
    )
    parser.add_argument(
        "--only-nim", action="store_true",
        help="Run only the NIM collection (no Ollama gate).",
    )
    parser.add_argument(
        "--ollama-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL for the Ollama embedder.",
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

    examples = load_dataset(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    logger.info("Loaded %d retrieval examples from %s", len(examples), args.dataset)

    # Reset the CSV so repeated runs don't accumulate stale rows.
    if args.output.exists():
        args.output.unlink()

    ollama_summary: RetrievalSummary | None = None
    nim_summary: RetrievalSummary | None = None

    if not args.only_nim:
        logger.info("Building Ollama retriever (docs-knowledge)...")
        from rag.qdrant_collection import COLLECTION_NAME, get_qdrant_client

        client = get_qdrant_client()
        ollama_available = collection_available(client, COLLECTION_NAME)
        retriever = build_ollama_retriever(args.ollama_url, args.top_k)
        rows = run_collection(examples, OLLAMA_BACKEND, retriever, args.top_k)
        write_csv(rows, args.output)
        ollama_summary = aggregate(rows, OLLAMA_BACKEND, ollama_available)

    if not args.no_nim:
        logger.info("Building NIM retriever (ctc_rag_nim)...")
        from rag.qdrant_collection import NIM_COLLECTION_NAME, get_qdrant_client

        client = get_qdrant_client()
        nim_available = collection_available(client, NIM_COLLECTION_NAME)
        if not nim_available:
            logger.warning(
                "NIM collection %r is missing/empty — skipping NIM retrieval. "
                "Ingest it first with `NIM_ENABLED=true python -m ingestion.nim_embedder`.",
                NIM_COLLECTION_NAME,
            )
            nim_summary = RetrievalSummary(
                backend=NIM_BACKEND, total=len(examples), available=False,
            )
        else:
            retriever = build_nim_retriever(args.top_k)
            rows = run_collection(examples, NIM_BACKEND, retriever, args.top_k)
            write_csv(rows, args.output)
            nim_summary = aggregate(rows, NIM_BACKEND, True)

    if ollama_summary is None:
        # --only-nim: no Ollama gate to fail.
        if nim_summary is not None:
            print_summary(
                RetrievalSummary(backend=OLLAMA_BACKEND, available=False),
                nim_summary, args.top_k, args.threshold,
            )
        return 0

    gate_ok = print_summary(ollama_summary, nim_summary, args.top_k, args.threshold)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
