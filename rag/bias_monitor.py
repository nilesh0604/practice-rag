"""Bias & fairness monitor — post-generation fairness check (Responsible AI).

Runs **after** generation completes (and after the output guardrail /
hallucination block) to detect potentially biased or non-inclusive language
in the assistant's answer. This is the practice-project implementation of
the architecture doc's "Bias & fairness monitoring" Responsible AI feature.

Two tiers, mirroring the guardrail pattern:

1. **Heuristic regex tier** (always runs, deterministic, zero latency) —
   scans the answer for:

   - **Gendered pronouns in generic context** — standalone ``he`` / ``she``
     / ``him`` / ``his`` / ``her`` / ``himself`` / ``herself`` used to refer
     to a generic developer/user (category ``gendered_pronoun``). Word-
     boundary matching avoids false positives inside other words
     (``the``, ``this``, ``shepherd``).
   - **Gendered address terms** — ``guys`` / ``gals`` / ``ladies`` /
     ``gentlemen`` / ``fellas`` / ``boys`` used as generic address
     (category ``gendered_address``).
   - **Stereotypical ability claims** — patterns like
     ``men are`` / ``women can't`` / ``boys should`` that assert a
     gendered ability/role stereotype (category ``stereotype``).

   The heuristic is **over-inclusive by design** (a monitoring/warn tool,
   not a hard classifier) — it flags any standalone gendered pronoun even
   when the referent is a specific named person. The LLM judge refines it.

2. **Optional LLM judge** (best-effort, degrades gracefully) — a short
   prompt asks the local Ollama model to classify the answer as ``biased``
   or ``neutral`` and, when biased, name the category. On any error or
   missing label the judge is skipped and only the heuristic result is
   used (never blocks on LLM failure — same invariant as the guardrails).

The orchestrator calls ``BiasMonitor.assess(answer)`` and uses the returned
``BiasAssessment`` to:

- record bias metrics (via ``PostProcessResult.bias`` → chat route →
  ``MetricsCollector.record_bias_check``), and
- optionally **block** a biased answer (``block_biased=True``) by replacing
  it with ``BIAS_REFUSAL`` — the same swap mechanism as the output
  guardrail and the hallucination block.

All LLM-backed checks share the lazy-client pattern from ``Generator`` /
``Embedder`` / ``InputGuardrail`` so unit tests inject mocks and never
touch the network.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

GUARD_MODEL: str = "llama3.2:3b"
"""Ollama model for the bias LLM judge (dev substitute; matches the
guardrail suite's ``GUARD_MODEL``)."""

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
"""Default Ollama base URL (host Ollama, not containerized — see D1)."""

# Bias categories surfaced in metrics + the assessment.
CATEGORY_GENDERED_PRONOUN: str = "gendered_pronoun"
CATEGORY_GENDERED_ADDRESS: str = "gendered_address"
CATEGORY_STEREOTYPE: str = "stereotype"
CATEGORY_LLM: str = "llm_flagged"
"""Category recorded when only the LLM judge flagged bias (no heuristic
match). Keeps the category breakdown distinguishable from regex hits."""

EVIDENCE_MAX_CHARS: int = 80
"""Maximum length of a single evidence snippet (truncated on a word
boundary) so the assessment stays compact for logging/metadata."""

# ── heuristic regex patterns (tier 1) ──────────────────────────────────
# Each tuple is (compiled pattern, category). Matches are case-insensitive.
# Word boundaries (\b) prevent false positives inside other words.

# Standalone gendered pronouns. \bhe\b does NOT match "the" (the leading
# 't' is a word char so there is no boundary before 'h'); \bhis\b does NOT
# match "this"; \bshe\b does NOT match "shepherd".
_GENDERED_PRONOUN_RE = re.compile(
    r"\b(he|she|him|his|her|hers|himself|herself)\b",
    re.IGNORECASE,
)

# Gendered address terms used as generic address to a group.
_GENDERED_ADDRESS_RE = re.compile(
    r"\b(guys|gals|ladies|gentlemen|gentleman|fellas|boys|girls)\b",
    re.IGNORECASE,
)

# Stereotypical ability/role claims: "<gender> are/can't/cannot/should/...
# <verb>". Captures assertions of a gendered stereotype rather than a
# neutral mention. The lookahead requires a verb-ish token after the
# gender noun so "men and women" (a neutral enumeration) is not flagged.
_STEREOTYPE_RE = re.compile(
    r"\b(men|women|boys|girls|males|females)\b"
    r"\s+(?:are|is|can(?:not|'t| not)?|cannot|should|must|always|never|tend to|"
    r"are not|aren't|were|was|will|won't|do not|don't|does not|doesn't)\b",
    re.IGNORECASE,
)

# ── LLM judge prompt (tier 2) ──────────────────────────────────────────

BIAS_JUDGE_PROMPT: str = """You are a fairness reviewer for a technical documentation assistant.
Decide whether the answer below contains biased, gendered, or non-inclusive language
that stereotypes or excludes people by gender, race, age, or other demographic group.

Reply with exactly one word: "biased" or "neutral".
Do not explain. Do not add punctuation.

Answer:
{answer}"""


# ── result dataclass ───────────────────────────────────────────────────


@dataclass
class BiasAssessment:
    """Result of a bias & fairness check on a generated answer.

    ``biased`` is True when the answer was flagged by the heuristic tier
    and/or the LLM judge. ``categories`` is a deduplicated list of bias
    category tags (``gendered_pronoun`` / ``gendered_address`` /
    ``stereotype`` / ``llm_flagged``). ``evidence`` is a list of short
    matched snippets (heuristic hits only; truncated to
    ``EVIDENCE_MAX_CHARS``). ``score`` is a 0.0–1.0 severity proxy:
    heuristic-derived when only the regex tier fired, 1.0 when the LLM
    judge confirmed bias, 0.0 when not biased.
    """

    biased: bool = False
    categories: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    score: float = 0.0


# ── LLM client protocol (mirrors api/guardrails._OllamaLike) ───────────


class _OllamaLike(Protocol):
    """Minimal protocol for the Ollama chat client (for typing/mocking)."""

    def chat(self, model: str, messages: list[dict], stream: bool = False, options: dict | None = None): ...


def _lazy_ollama_client(ollama_url: str) -> _OllamaLike:
    """Construct an Ollama client lazily (imported only when called)."""
    import ollama

    return ollama.Client(host=ollama_url)


def _extract_label(text: str, valid: frozenset[str]) -> str | None:
    """Pull the first valid label word out of an LLM judge response."""
    for token in re.split(r"\W+", text.strip().lower()):
        if token in valid:
            return token
    return None


# ── pure heuristic functions (no LLM, no I/O) ──────────────────────────


def detect_bias_heuristic(answer: str) -> tuple[list[str], list[str]]:
    """Regex scan for biased / non-inclusive language patterns.

    Returns ``(categories, evidence)`` where ``categories`` is a deduplicated
    list of bias category tags and ``evidence`` is a list of short matched
    snippets (one per match, truncated to ``EVIDENCE_MAX_CHARS``). Both are
    empty when no heuristic pattern matches.
    """
    if not answer:
        return [], []
    categories: list[str] = []
    evidence: list[str] = []
    seen_cats: set[str] = set()

    def _record(match: re.Match[str], category: str) -> None:
        if category not in seen_cats:
            categories.append(category)
            seen_cats.add(category)
        evidence.append(_make_evidence(answer, match.start(), match.end()))

    for match in _GENDERED_PRONOUN_RE.finditer(answer):
        _record(match, CATEGORY_GENDERED_PRONOUN)
    for match in _GENDERED_ADDRESS_RE.finditer(answer):
        _record(match, CATEGORY_GENDERED_ADDRESS)
    for match in _STEREOTYPE_RE.finditer(answer):
        _record(match, CATEGORY_STEREOTYPE)
    return categories, evidence


def _make_evidence(text: str, start: int, end: int) -> str:
    """Build a short snippet around a match, truncated on a word boundary.

    Centers a window of ``EVIDENCE_MAX_CHARS`` around the match so the
    evidence carries enough context to be meaningful in logs/metadata
    without retaining the full answer.
    """
    match_len = end - start
    pad = max(0, (EVIDENCE_MAX_CHARS - match_len) // 2)
    win_start = max(0, start - pad)
    win_end = min(len(text), end + pad)
    snippet = text[win_start:win_end].strip()
    snippet = " ".join(snippet.split())
    if len(snippet) > EVIDENCE_MAX_CHARS:
        snippet = snippet[:EVIDENCE_MAX_CHARS].rsplit(" ", 1)[0]
    return snippet


# ── monitor ────────────────────────────────────────────────────────────


class BiasMonitor:
    """Bias & fairness check: heuristic regex + optional LLM judge.

    The check runs in two tiers:

    1. **Heuristic regex scan** — fast, deterministic, always runs. Flags
       gendered pronouns in generic context, gendered address terms, and
       stereotypical ability claims.
    2. **LLM judge** — optional (``use_llm``), runs on the full answer.
       Classifies the answer as ``biased`` / ``neutral``. On error,
       degrades to heuristic-only (never blocks on LLM failure).

    When the LLM judge flags bias but the heuristic found nothing, the
    ``llm_flagged`` category is added so the category breakdown
    distinguishes regex hits from LLM-only flags. When both fire, the
    heuristic categories are kept (the LLM confirms but the regex already
    localized the evidence).
    """

    def __init__(
        self,
        model: str = GUARD_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        use_llm: bool = True,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.use_llm = use_llm
        self._client: _OllamaLike | None = None

    @property
    def client(self) -> _OllamaLike:
        if self._client is None:
            self._client = _lazy_ollama_client(self.ollama_url)
        return self._client

    def assess(self, answer: str) -> BiasAssessment:
        """Return a ``BiasAssessment`` for the generated answer.

        The heuristic tier always runs. The LLM judge runs only when
        ``use_llm`` is True and Ollama is reachable; on error it is
        skipped so the assessment degrades to heuristic-only.
        """
        categories, evidence = detect_bias_heuristic(answer)
        biased = bool(categories)
        # Heuristic severity: more matches → higher score, capped at 0.75
        # so an LLM confirmation (1.0) always ranks above heuristic-only.
        score = min(len(evidence) * 0.25, 0.75) if biased else 0.0

        if self.use_llm and answer.strip():
            verdict = self._llm_judge(answer)
            if verdict == "biased":
                biased = True
                score = 1.0
                # Only add the llm_flagged category when the heuristic
                # found nothing — otherwise the regex categories already
                # describe the localized bias.
                if not categories:
                    categories = [CATEGORY_LLM]
            elif verdict == "neutral" and not categories:
                # LLM says neutral and heuristic found nothing → clean.
                biased = False
                score = 0.0
            # verdict is None (LLM error) → keep heuristic result only.

        assessment = BiasAssessment(
            biased=biased,
            categories=categories,
            evidence=evidence,
            score=score,
        )
        logger.info(
            "BiasMonitor: biased=%s score=%.2f categories=%s evidence=%d",
            biased,
            score,
            categories,
            len(evidence),
        )
        return assessment

    def _llm_judge(self, answer: str) -> str | None:
        """Return ``"biased"`` / ``"neutral"`` / ``None`` (on error/no label)."""
        prompt = BIAS_JUDGE_PROMPT.format(answer=answer)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"num_predict": 8, "temperature": 0.0},
            )
            text = response.get("message", {}).get("content", "")
            return _extract_label(text, frozenset({"biased", "neutral"}))
        except Exception:
            logger.warning("Bias LLM judge failed; heuristic-only", exc_info=True)
            return None

    def close(self) -> None:
        self._client = None


class PassthroughBiasMonitor:
    """No-op bias monitor — ``assess`` always returns a clean assessment.

    Used as the default when no bias monitor is configured so the
    orchestrator code path stays uniform (same as the passthrough query
    rewriter / decomposer / clarifier).
    """

    def assess(self, answer: str) -> BiasAssessment:  # noqa: D401
        return BiasAssessment()

    def close(self) -> None:
        pass
