"""Tests for the query decomposer (rag/query_decomposer.py).

Covers:
- ``_heuristic_decompose`` — connector splits (vs/versus/compared to/
  difference between/and), no-connector passthrough, empty/whitespace,
  empty-part guard, case-insensitivity.
- ``_strip_prefix`` — numbering/bullet prefix removal.
- ``PassthroughQueryDecomposer`` — returns ``[query]`` unchanged.
- ``LLMQueryDecomposer`` — LLM multi-line split, single-line-equal-to-
  original triggers heuristic fallback, empty/missing response falls back,
  exception falls back, ``use_llm=False`` uses heuristic, numbering prefixes
  stripped, lazy client, close resets client.
- ``DECOMPOSE_PROMPT_TEMPLATE`` — placeholders + instruction text.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.query_decomposer import (
    DECOMPOSE_PROMPT_TEMPLATE,
    LLMQueryDecomposer,
    PassthroughQueryDecomposer,
    _heuristic_decompose,
    _strip_prefix,
)


# ── _heuristic_decompose ───────────────────────────────────────────────


class TestHeuristicDecompose:
    def test_vs_split(self):
        assert _heuristic_decompose("FastAPI vs Flask") == ["FastAPI", "Flask"]

    def test_vs_with_period_split(self):
        assert _heuristic_decompose("FastAPI vs. Flask") == ["FastAPI", "Flask"]

    def test_versus_split(self):
        assert _heuristic_decompose("Pydantic versus dataclasses") == [
            "Pydantic",
            "dataclasses",
        ]

    def test_compared_to_split(self):
        assert _heuristic_decompose("SQLModel compared to SQLAlchemy") == [
            "SQLModel",
            "SQLAlchemy",
        ]

    def test_difference_between_split(self):
        assert _heuristic_decompose("the difference between FastAPI and Flask") == [
            "the",
            "FastAPI and Flask",
        ]

    def test_and_split(self):
        assert _heuristic_decompose("FastAPI and Flask") == ["FastAPI", "Flask"]

    def test_no_connector_returns_original(self):
        query = "How do I use path parameters in FastAPI?"
        assert _heuristic_decompose(query) == [query]

    def test_empty_query(self):
        assert _heuristic_decompose("") == [""]

    def test_whitespace_query(self):
        assert _heuristic_decompose("   ") == ["   "]

    def test_case_insensitive_vs(self):
        assert _heuristic_decompose("FastAPI VS Flask") == ["FastAPI", "Flask"]

    def test_case_insensitive_versus(self):
        assert _heuristic_decompose("FastAPI VERSUS Flask") == ["FastAPI", "Flask"]

    def test_only_first_connector_splits(self):
        # maxsplit=1 → three-way "and" only splits once (two sub-queries).
        result = _heuristic_decompose("A and B and C")
        assert result == ["A", "B and C"]

    def test_empty_part_after_split_returns_original(self):
        # "vs " with nothing after → right part empty → no split.
        assert _heuristic_decompose("FastAPI vs ") == ["FastAPI vs "]

    def test_empty_part_before_split_returns_original(self):
        assert _heuristic_decompose(" vs Flask") == [" vs Flask"]

    def test_strips_outer_whitespace_before_split(self):
        assert _heuristic_decompose("  FastAPI vs Flask  ") == ["FastAPI", "Flask"]

    def test_vs_stronger_than_and(self):
        # "vs" is tried before "and", so "A vs B and C" splits on "vs".
        assert _heuristic_decompose("A vs B and C") == ["A", "B and C"]


# ── _strip_prefix ──────────────────────────────────────────────────────


class TestStripPrefix:
    def test_numbered_prefix(self):
        assert _strip_prefix("1. FastAPI question") == "FastAPI question"

    def test_paren_numbered_prefix(self):
        assert _strip_prefix("1) FastAPI question") == "FastAPI question"

    def test_dash_prefix(self):
        assert _strip_prefix("- FastAPI question") == "FastAPI question"

    def test_asterisk_prefix(self):
        assert _strip_prefix("* FastAPI question") == "FastAPI question"

    def test_bullet_prefix(self):
        assert _strip_prefix("• FastAPI question") == "FastAPI question"

    def test_no_prefix_unchanged(self):
        assert _strip_prefix("FastAPI question") == "FastAPI question"

    def test_leading_whitespace_with_prefix(self):
        assert _strip_prefix("  2. Flask question") == "Flask question"

    def test_double_digit_number(self):
        assert _strip_prefix("10. Tenth question") == "Tenth question"

    def test_does_not_strip_inline_number(self):
        # "FastAPI 1.0" — the 1. is not at the start, so no strip.
        assert _strip_prefix("FastAPI 1.0 features") == "FastAPI 1.0 features"


# ── PassthroughQueryDecomposer ─────────────────────────────────────────


class TestPassthroughQueryDecomposer:
    def test_returns_single_element_list(self):
        d = PassthroughQueryDecomposer()
        assert d.decompose("compare FastAPI and Flask") == ["compare FastAPI and Flask"]

    def test_ignores_history(self):
        d = PassthroughQueryDecomposer()
        assert d.decompose("query", "history") == ["query"]

    def test_empty_query(self):
        d = PassthroughQueryDecomposer()
        assert d.decompose("") == [""]


# ── LLMQueryDecomposer ─────────────────────────────────────────────────


def _mock_ollama(content: str) -> MagicMock:
    """Build a mock Ollama client whose chat() returns the given content."""
    client = MagicMock()
    client.chat.return_value = {"message": {"content": content}}
    return client


class TestLLMQueryDecomposer:
    def test_multi_line_split(self):
        d = LLMQueryDecomposer()
        d._client = _mock_ollama("How does FastAPI handle routing?\nHow does Flask handle routing?")
        result = d.decompose("compare FastAPI and Flask")
        assert result == ["How does FastAPI handle routing?", "How does Flask handle routing?"]

    def test_strips_numbering_prefixes(self):
        d = LLMQueryDecomposer()
        d._client = _mock_ollama("1. FastAPI routing\n2. Flask routing")
        result = d.decompose("compare FastAPI and Flask")
        assert result == ["FastAPI routing", "Flask routing"]

    def test_strips_bullet_prefixes(self):
        d = LLMQueryDecomposer()
        d._client = _mock_ollama("- FastAPI routing\n- Flask routing")
        result = d.decompose("compare FastAPI and Flask")
        assert result == ["FastAPI routing", "Flask routing"]

    def test_single_line_equal_to_original_triggers_heuristic_fallback(self):
        """If the LLM returns the original query unchanged on one line,
        fall back to the heuristic splitter (which may still split it)."""
        d = LLMQueryDecomposer()
        query = "FastAPI vs Flask"
        d._client = _mock_ollama(query)
        result = d.decompose(query)
        # Heuristic splits "vs" → two sub-queries.
        assert result == ["FastAPI", "Flask"]

    def test_single_line_no_connector_returns_original(self):
        """LLM returns original on one line, heuristic finds no connector
        → returns the original as a single-element list."""
        d = LLMQueryDecomposer()
        query = "How do I use FastAPI?"
        d._client = _mock_ollama(query)
        result = d.decompose(query)
        assert result == [query]

    def test_empty_response_falls_back_to_heuristic(self):
        d = LLMQueryDecomposer()
        d._client = _mock_ollama("")
        result = d.decompose("FastAPI vs Flask")
        assert result == ["FastAPI", "Flask"]

    def test_no_message_key_falls_back(self):
        d = LLMQueryDecomposer()
        d._client = MagicMock()
        d._client.chat.return_value = {}
        result = d.decompose("FastAPI vs Flask")
        assert result == ["FastAPI", "Flask"]

    def test_exception_falls_back_to_heuristic(self):
        d = LLMQueryDecomposer()
        client = MagicMock()
        client.chat.side_effect = ConnectionError("no ollama")
        d._client = client
        result = d.decompose("FastAPI vs Flask")
        assert result == ["FastAPI", "Flask"]

    def test_use_llm_false_uses_heuristic(self):
        d = LLMQueryDecomposer(use_llm=False)
        d._client = _mock_ollama("should not be used")
        result = d.decompose("FastAPI vs Flask")
        assert result == ["FastAPI", "Flask"]
        d._client.chat.assert_not_called()

    def test_use_llm_false_no_connector_returns_original(self):
        d = LLMQueryDecomposer(use_llm=False)
        query = "How do I use FastAPI?"
        result = d.decompose(query)
        assert result == [query]

    def test_whitespace_only_lines_dropped(self):
        d = LLMQueryDecomposer()
        d._client = _mock_ollama("FastAPI routing\n   \nFlask routing\n\n")
        result = d.decompose("compare FastAPI and Flask")
        assert result == ["FastAPI routing", "Flask routing"]

    def test_stream_false(self):
        d = LLMQueryDecomposer()
        d._client = _mock_ollama("a\nb")
        d.decompose("compare A and B")
        assert d._client.chat.call_args.kwargs["stream"] is False

    def test_query_in_prompt(self):
        d = LLMQueryDecomposer()
        d._client = _mock_ollama("a\nb")
        d.decompose("compare FastAPI and Flask")
        prompt = d._client.chat.call_args.kwargs["messages"][0]["content"]
        assert "compare FastAPI and Flask" in prompt

    def test_close_resets_client(self):
        d = LLMQueryDecomposer()
        d._client = MagicMock()
        d.close()
        assert d._client is None

    def test_lazy_client_construction(self):
        d = LLMQueryDecomposer()
        assert d._client is None
        with patch("ollama.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            _ = d.client
            MockClient.assert_called_once_with(host="http://localhost:11434")


# ── DECOMPOSE_PROMPT_TEMPLATE ──────────────────────────────────────────


class TestDecomposePromptTemplate:
    def test_has_query_placeholder(self):
        assert "{query}" in DECOMPOSE_PROMPT_TEMPLATE

    def test_has_sub_questions_instruction(self):
        assert "Sub-questions:" in DECOMPOSE_PROMPT_TEMPLATE

    def test_instructs_one_per_line(self):
        assert "one per line" in DECOMPOSE_PROMPT_TEMPLATE

    def test_instructs_no_numbering(self):
        assert "no numbering" in DECOMPOSE_PROMPT_TEMPLATE
