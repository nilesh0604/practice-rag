"""Post-processor — citation extraction + groundedness/confidence scoring.

Runs after generation completes. Two responsibilities:

1. **Citation extraction** — parse ``[Source: title]`` markers from the
   generated answer and match them to retrieved documents by title
   (case-insensitive). Only citations that match a retrieved doc become
   ``Citation`` objects (with a clickable ``source_url``); hallucinated
   titles with no matching doc are dropped so the UI never links to a
   non-existent source.

2. **Groundedness score** — per the architecture doc's Confidence Score
   section, embed the answer and each retrieved chunk, then take the max
   cosine similarity between the answer embedding and any chunk embedding.
   This measures how well the answer is supported by the retrieved context.
   Below ``CONFIDENCE_THRESHOLD`` (0.65 per the doc) the UI shows a
   low-confidence warning.

The embedder is injected so the score is unit-testable with a mock.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from ingestion.embedder import Embedder
from rag.bias_monitor import BiasAssessment
from schemas.chat import Citation
from schemas.documents import RetrievedDoc

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD: float = 0.65
"""Below this groundedness score the UI shows a low-confidence warning
(from the architecture doc's Security & Responsible AI section)."""

SNIPPET_MAX_CHARS: int = 200
"""Maximum length of a citation snippet. The supporting chunk's content is
truncated to this many characters (on a word boundary when possible) so the
citation chip stays compact in the UI."""

_CITATION_RE = re.compile(r"\[Source:\s*(.+?)\]")
"""Matches ``[Source: title]`` markers in generated answers (non-greedy title)."""


@dataclass
class PostProcessResult:
    """Output of the post-processor — the final, persisted answer shape.

    The FastAPI layer adds ``session_id`` to produce a ``ChatResponse``.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    low_confidence: bool = False
    trace_id: str | None = None
    """Langfuse trace id (Step 7). Set by the orchestrator when tracing is
    enabled so the chat route can surface it in the ``ChatResponse`` and the
    frontend can pass it back with feedback for score correlation."""
    guardrail_replacement: str | None = None
    """Set by the orchestrator when the **output guardrail** blocks the
    streamed answer. Carries the refusal text that replaces the already-
    streamed tokens. The SSE layer emits an ``event: guardrail_replacement``
    frame from this so the frontend can swap the visible message (SSE is
    one-way — the original tokens cannot be un-sent). ``None`` means no
    output block occurred (or no guardrail suite is configured)."""
    bias: BiasAssessment | None = None
    """Set by the orchestrator after the bias & fairness monitor runs on
    the final answer (Responsible AI). Carries the ``BiasAssessment``
    (biased flag, categories, evidence, score) so the chat route can
    record bias metrics via ``MetricsCollector.record_bias_check``.
    ``None`` means no bias monitor is configured or the check was skipped
    (e.g. the output guardrail / hallucination block already replaced the
    answer with a canned refusal)."""
    bias_blocked: bool = False
    """Set by the orchestrator to True when the bias block fired
    (``block_biased=True`` and the monitor flagged the answer as biased).
    Distinct from ``bias.biased`` (which is the monitor's verdict) so the
    chat route can record the block count even when the answer was
    replaced with ``BIAS_REFUSAL``."""


class PostProcessor:
    """Extracts citations and computes a groundedness score for an answer."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def post_process(
        self,
        answer: str,
        retrieved_docs: list[RetrievedDoc],
    ) -> PostProcessResult:
        """Run citation extraction + groundedness scoring on a completed answer."""
        citations = extract_citations(answer, retrieved_docs)
        confidence = compute_groundedness(answer, retrieved_docs, self.embedder)
        result = PostProcessResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            low_confidence=confidence < CONFIDENCE_THRESHOLD,
        )
        logger.info(
            "Post-process: %d citations, confidence=%.3f (%s)",
            len(citations),
            confidence,
            "low" if result.low_confidence else "ok",
        )
        return result


# ── citation extraction (pure functions, no embedder needed) ───────────


def extract_citations(
    answer: str,
    retrieved_docs: list[RetrievedDoc],
) -> list[Citation]:
    """Parse ``[Source: title]`` markers and match them to retrieved docs.

    Returns deduplicated ``Citation`` objects preserving first-seen order.
    Only markers whose title matches a retrieved doc (case-insensitive,
    stripped) are included — unmatched markers are treated as hallucinated
    and dropped.
    """
    if not answer or not retrieved_docs:
        return []

    # Build a case-insensitive title → doc lookup.
    title_to_doc: dict[str, RetrievedDoc] = {}
    for doc in retrieved_docs:
        key = doc.title.strip().lower()
        if key:
            title_to_doc.setdefault(key, doc)

    citations: list[Citation] = []
    seen_urls: set[str] = set()
    for match in _CITATION_RE.finditer(answer):
        raw_title = match.group(1).strip()
        key = raw_title.lower()
        doc = title_to_doc.get(key)
        if doc is None:
            logger.debug("Citation %r did not match any retrieved doc — dropping", raw_title)
            continue
        url = str(doc.source_url)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        citations.append(
            Citation(
                title=doc.title,
                source_url=doc.source_url,
                snippet=_make_snippet(doc.content),
                relevanceScore=doc.score,
                lastModified=doc.last_modified,
            )
        )
    return citations


def _make_snippet(content: str) -> str:
    """Truncate chunk content to ``SNIPPET_MAX_CHARS`` on a word boundary.

    Collapses whitespace and appends an ellipsis when truncated so the
    citation chip shows a clean one-line excerpt.
    """
    text = " ".join(content.split())
    if len(text) <= SNIPPET_MAX_CHARS:
        return text
    cut = text[:SNIPPET_MAX_CHARS].rsplit(" ", 1)[0]
    return f"{cut}…" if cut else f"{text[:SNIPPET_MAX_CHARS]}…"


# ── groundedness score ─────────────────────────────────────────────────


def compute_groundedness(
    answer: str,
    docs: list[RetrievedDoc],
    embedder: Embedder,
) -> float:
    """Max cosine similarity between the answer embedding and any chunk embedding.

    Faithful implementation of the architecture doc's ``groundedness()``
    pseudocode: embed the answer, embed each retrieved chunk's content, and
    return the highest cosine similarity. Returns 0.0 if the answer or docs
    are empty.
    """
    if not answer.strip() or not docs:
        return 0.0

    answer_vec = embedder.embed_text(answer)
    doc_vecs = embedder.embed_texts([d.content for d in docs])
    if not doc_vecs:
        return 0.0

    best = max(_cosine_similarity(answer_vec, dv) for dv in doc_vecs)
    return min(max(best, 0.0), 1.0)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors. Returns 0.0 for zero vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
