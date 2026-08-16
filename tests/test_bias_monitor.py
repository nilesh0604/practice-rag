"""Tests for the bias & fairness monitor (rag/bias_monitor.py).

Covers:
- ``detect_bias_heuristic`` — regex tier: gendered pronouns, gendered
  address, stereotypes, false-positive avoidance (``the``/``this``/
  ``shepherd``), empty input.
- ``_make_evidence`` — snippet truncation on a word boundary.
- ``BiasMonitor.assess`` — heuristic-only (use_llm=False), LLM judge
  confirmation, LLM-only flag (``llm_flagged`` category), LLM neutral
  override, LLM error degrades to heuristic-only, score scaling.
- ``PassthroughBiasMonitor`` — always returns a clean assessment.
- ``BiasAssessment`` — default fields.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.bias_monitor import (
    BIAS_JUDGE_PROMPT,
    BiasAssessment,
    BiasMonitor,
    CATEGORY_GENDERED_ADDRESS,
    CATEGORY_GENDERED_PRONOUN,
    CATEGORY_LLM,
    CATEGORY_STEREOTYPE,
    EVIDENCE_MAX_CHARS,
    PassthroughBiasMonitor,
    _extract_label,
    _make_evidence,
    detect_bias_heuristic,
)


# ════════════════════════════════════════════════════════════════════════
# detect_bias_heuristic (pure regex tier)
# ════════════════════════════════════════════════════════════════════════


class TestDetectBiasHeuristic:
    def test_empty_answer_returns_empty(self):
        cats, ev = detect_bias_heuristic("")
        assert cats == []
        assert ev == []

    def test_gendered_pronoun_he_flagged(self):
        cats, ev = detect_bias_heuristic("When he deploys the app, he should test.")
        assert CATEGORY_GENDERED_PRONOUN in cats
        assert len(ev) >= 1

    def test_gendered_pronoun_she_flagged(self):
        cats, ev = detect_bias_heuristic("She configures the server herself.")
        assert CATEGORY_GENDERED_PRONOUN in cats

    def test_gendered_pronoun_his_flagged(self):
        cats, _ = detect_bias_heuristic("His configuration is correct.")
        assert CATEGORY_GENDERED_PRONOUN in cats

    def test_the_not_flagged_as_he(self):
        """``the`` must NOT match ``\\bhe\\b`` — no boundary before 'h'."""
        cats, _ = detect_bias_heuristic("the server is running")
        assert cats == []

    def test_this_not_flagged_as_his(self):
        """``this`` must NOT match ``\\bhis\\b``."""
        cats, _ = detect_bias_heuristic("this is the way")
        assert cats == []

    def test_shepherd_not_flagged_as_she(self):
        """``shepherd`` must NOT match ``\\bshe\\b``."""
        cats, _ = detect_bias_heuristic("the shepherd tends the flock")
        assert cats == []

    def test_gendered_address_guys_flagged(self):
        cats, _ = detect_bias_heuristic("Hey guys, let's deploy!")
        assert CATEGORY_GENDERED_ADDRESS in cats

    def test_gendered_address_ladies_flagged(self):
        cats, _ = detect_bias_heuristic("Listen ladies, the config goes here.")
        assert CATEGORY_GENDERED_ADDRESS in cats

    def test_stereotype_men_are_flagged(self):
        cats, _ = detect_bias_heuristic("Men are better at coding.")
        assert CATEGORY_STEREOTYPE in cats

    def test_stereotype_women_cant_flagged(self):
        cats, _ = detect_bias_heuristic("Women can't understand servers.")
        assert CATEGORY_STEREOTYPE in cats

    def test_neutral_enumeration_not_flagged(self):
        """``men and women`` (neutral enumeration, no verb after) is not a
        stereotype."""
        cats, _ = detect_bias_heuristic("Both men and women use FastAPI.")
        assert CATEGORY_STEREOTYPE not in cats

    def test_multiple_categories_deduplicated(self):
        cats, ev = detect_bias_heuristic(
            "He tells the guys that men are always right."
        )
        assert CATEGORY_GENDERED_PRONOUN in cats
        assert CATEGORY_GENDERED_ADDRESS in cats
        assert CATEGORY_STEREOTYPE in cats
        # Categories are deduplicated even with multiple matches per cat.
        assert len(cats) == len(set(cats))

    def test_clean_answer_no_flags(self):
        cats, ev = detect_bias_heuristic(
            "FastAPI is a modern web framework for building APIs with Python."
        )
        assert cats == []
        assert ev == []

    def test_evidence_is_truncated(self):
        long_answer = "When he " + "x " * 200 + "deploys"
        _, ev = detect_bias_heuristic(long_answer)
        assert len(ev) >= 1
        assert all(len(s) <= EVIDENCE_MAX_CHARS for s in ev)


# ════════════════════════════════════════════════════════════════════════
# _make_evidence
# ════════════════════════════════════════════════════════════════════════


class TestMakeEvidence:
    def test_short_text_returns_full_window(self):
        text = "he deploys the app"
        ev = _make_evidence(text, 0, 2)
        assert "he" in ev
        assert len(ev) <= EVIDENCE_MAX_CHARS

    def test_long_text_truncated(self):
        text = "prefix " * 50 + "he" + " suffix" * 50
        ev = _make_evidence(text, 49 * 7, 49 * 7 + 2)
        assert len(ev) <= EVIDENCE_MAX_CHARS

    def test_collapses_whitespace(self):
        text = "he   \n  deploys"
        ev = _make_evidence(text, 0, 2)
        assert "  " not in ev  # whitespace collapsed


# ════════════════════════════════════════════════════════════════════════
# _extract_label
# ════════════════════════════════════════════════════════════════════════


class TestExtractLabel:
    def test_finds_biased(self):
        assert _extract_label("biased", frozenset({"biased", "neutral"})) == "biased"

    def test_finds_neutral(self):
        assert _extract_label("neutral", frozenset({"biased", "neutral"})) == "neutral"

    def test_case_insensitive(self):
        assert _extract_label("BIASED", frozenset({"biased", "neutral"})) == "biased"

    def test_extracts_from_sentence(self):
        assert _extract_label("The answer is biased.", frozenset({"biased", "neutral"})) == "biased"

    def test_no_valid_label_returns_none(self):
        assert _extract_label("maybe", frozenset({"biased", "neutral"})) is None

    def test_empty_returns_none(self):
        assert _extract_label("", frozenset({"biased", "neutral"})) is None


# ════════════════════════════════════════════════════════════════════════
# BiasMonitor — heuristic-only (use_llm=False)
# ════════════════════════════════════════════════════════════════════════


class TestBiasMonitorHeuristicOnly:
    def test_clean_answer_not_biased(self):
        monitor = BiasMonitor(use_llm=False)
        assessment = monitor.assess("FastAPI is a web framework.")
        assert assessment.biased is False
        assert assessment.categories == []
        assert assessment.score == 0.0

    def test_gendered_pronoun_flagged(self):
        monitor = BiasMonitor(use_llm=False)
        assessment = monitor.assess("He should configure the server.")
        assert assessment.biased is True
        assert CATEGORY_GENDERED_PRONOUN in assessment.categories
        assert 0 < assessment.score <= 0.75

    def test_multiple_matches_increase_score(self):
        monitor = BiasMonitor(use_llm=False)
        assessment = monitor.assess("He and she deploy the app.")
        # Two pronoun matches → 0.25 each = 0.5
        assert assessment.score == pytest.approx(0.5)

    def test_score_capped_at_075(self):
        monitor = BiasMonitor(use_llm=False)
        # 4+ matches → 0.25 * 4 = 1.0, capped at 0.75
        assessment = monitor.assess("He she him her guys men are always right.")
        assert assessment.score == pytest.approx(0.75)

    def test_evidence_populated(self):
        monitor = BiasMonitor(use_llm=False)
        assessment = monitor.assess("He deploys the app.")
        assert len(assessment.evidence) >= 1
        assert all(len(s) <= EVIDENCE_MAX_CHARS for s in assessment.evidence)


# ════════════════════════════════════════════════════════════════════════
# BiasMonitor — LLM judge integration
# ════════════════════════════════════════════════════════════════════════


def _mock_client(verdict: str | None):
    """Build a mock Ollama client returning the given verdict label."""
    client = MagicMock()
    if verdict is None:
        client.chat.side_effect = Exception("ollama unreachable")
    else:
        client.chat.return_value = {"message": {"content": verdict}}
    return client


class TestBiasMonitorLlmJudge:
    def test_llm_confirms_heuristic_bias(self):
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client("biased")
        assessment = monitor.assess("He deploys the app.")
        assert assessment.biased is True
        assert assessment.score == 1.0  # LLM confirmation → 1.0
        # Heuristic categories kept (regex localized the evidence).
        assert CATEGORY_GENDERED_PRONOUN in assessment.categories

    def test_llm_only_flag_adds_llm_category(self):
        """When the heuristic finds nothing but the LLM flags bias, the
        ``llm_flagged`` category is added."""
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client("biased")
        assessment = monitor.assess("This code has subtle exclusionary framing.")
        assert assessment.biased is True
        assert assessment.score == 1.0
        assert CATEGORY_LLM in assessment.categories
        # No heuristic categories since regex found nothing.
        assert CATEGORY_GENDERED_PRONOUN not in assessment.categories

    def test_llm_neutral_with_no_heuristic_means_clean(self):
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client("neutral")
        assessment = monitor.assess("FastAPI is a web framework.")
        assert assessment.biased is False
        assert assessment.score == 0.0

    def test_llm_neutral_does_not_override_heuristic(self):
        """When the heuristic flags bias but the LLM says neutral, the
        heuristic result stands (the regex already localized the evidence)."""
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client("neutral")
        assessment = monitor.assess("He deploys the app.")
        # Heuristic found a pronoun → still biased.
        assert assessment.biased is True
        assert CATEGORY_GENDERED_PRONOUN in assessment.categories

    def test_llm_error_degrades_to_heuristic_only(self):
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client(None)  # raises
        assessment = monitor.assess("He deploys the app.")
        assert assessment.biased is True
        assert CATEGORY_GENDERED_PRONOUN in assessment.categories
        # Score is heuristic-derived, not 1.0 (LLM didn't confirm).
        assert assessment.score < 1.0

    def test_llm_error_on_clean_answer_means_clean(self):
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client(None)  # raises
        assessment = monitor.assess("FastAPI is a web framework.")
        assert assessment.biased is False
        assert assessment.score == 0.0

    def test_llm_prompt_contains_answer(self):
        """The judge prompt is formatted with the answer text."""
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client("neutral")
        monitor.assess("some answer text")
        call = monitor._client.chat.call_args
        prompt = call.kwargs["messages"][0]["content"]
        assert "some answer text" in prompt

    def test_llm_uses_guard_model_and_low_tokens(self):
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client("neutral")
        monitor.assess("text")
        call = monitor._client.chat.call_args
        assert call.kwargs["model"] == "llama3.2:3b"
        assert call.kwargs["options"]["num_predict"] == 8
        assert call.kwargs["options"]["temperature"] == 0.0

    def test_empty_answer_skips_llm(self):
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client("biased")
        assessment = monitor.assess("")
        assert assessment.biased is False
        monitor._client.chat.assert_not_called()

    def test_close_resets_client(self):
        monitor = BiasMonitor(use_llm=True)
        monitor._client = _mock_client("neutral")
        monitor.close()
        assert monitor._client is None


# ════════════════════════════════════════════════════════════════════════
# PassthroughBiasMonitor
# ════════════════════════════════════════════════════════════════════════


class TestPassthroughBiasMonitor:
    def test_always_returns_clean(self):
        monitor = PassthroughBiasMonitor()
        assessment = monitor.assess("He tells the guys that men are always right.")
        assert assessment.biased is False
        assert assessment.categories == []
        assert assessment.score == 0.0

    def test_close_is_noop(self):
        monitor = PassthroughBiasMonitor()
        monitor.close()  # should not raise


# ════════════════════════════════════════════════════════════════════════
# BiasAssessment dataclass
# ════════════════════════════════════════════════════════════════════════


class TestBiasAssessment:
    def test_defaults(self):
        a = BiasAssessment()
        assert a.biased is False
        assert a.categories == []
        assert a.evidence == []
        assert a.score == 0.0

    def test_construction(self):
        a = BiasAssessment(
            biased=True,
            categories=[CATEGORY_GENDERED_PRONOUN],
            evidence=["he deploys"],
            score=1.0,
        )
        assert a.biased is True
        assert a.categories == [CATEGORY_GENDERED_PRONOUN]
        assert a.evidence == ["he deploys"]
        assert a.score == 1.0
