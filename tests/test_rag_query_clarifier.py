"""Tests for the query clarifier (rag/query_clarifier.py).

Covers:
- ``_heuristic_clarify`` — shared-concept detection (validator/model/field/
  cache/schema/session/dependency), no-concept returns None, empty/
  whitespace, word-boundary matching.
- ``_strip_label_prefix`` — ``Clarification:`` / ``Q:`` prefix removal.
- ``PassthroughQueryClarifier`` — returns None unchanged.
- ``LLMQueryClarifier`` — LLM clarification, empty/whitespace response falls
  back, missing message key falls back, exception falls back,
  ``use_llm=False`` uses heuristic, label prefixes stripped, lazy client,
  close resets client.
- ``CLARIFY_PROMPT_TEMPLATE`` — placeholders + instruction text.
- ``GENERIC_CLARIFICATION`` — non-empty fallback text.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.query_clarifier import (
    CLARIFY_PROMPT_TEMPLATE,
    GENERIC_CLARIFICATION,
    LLMQueryClarifier,
    PassthroughQueryClarifier,
    _heuristic_clarify,
    _strip_label_prefix,
)


# ── _heuristic_clarify ────────────────────────────────────────────────


class TestHeuristicClarify:
    def test_validator_concept(self):
        result = _heuristic_clarify("how do I use validators?")
        assert result is not None
        assert "Pydantic validators" in result
        assert "FastAPI dependency validators" in result

    def test_model_concept(self):
        result = _heuristic_clarify("how do I define a model?")
        assert result is not None
        assert "Pydantic models" in result
        assert "SQLModel models" in result

    def test_field_concept(self):
        result = _heuristic_clarify("how do I configure fields?")
        assert result is not None
        assert "Pydantic fields" in result

    def test_cache_concept(self):
        result = _heuristic_clarify("how do I set up the cache?")
        assert result is not None
        assert "caching" in result

    def test_schema_concept(self):
        result = _heuristic_clarify("how do I define schemas?")
        assert result is not None
        assert "Pydantic schemas" in result
        assert "SQLModel table schemas" in result

    def test_session_concept(self):
        result = _heuristic_clarify("how do I use a session?")
        assert result is not None
        assert "session" in result.lower()

    def test_dependency_concept(self):
        result = _heuristic_clarify("how do I use dependencies?")
        assert result is not None
        assert "FastAPI dependencies" in result

    def test_no_concept_returns_none(self):
        assert _heuristic_clarify("how do I use path parameters in FastAPI?") is None

    def test_empty_query_returns_none(self):
        assert _heuristic_clarify("") is None

    def test_whitespace_query_returns_none(self):
        assert _heuristic_clarify("   ") is None

    def test_word_boundary_validator(self):
        # "validatortwo" should NOT match "validator" (word boundary).
        assert _heuristic_clarify("how do I use validatortwo?") is None

    def test_case_insensitive(self):
        result = _heuristic_clarify("HOW DO I USE VALIDATORS?")
        assert result is not None
        assert "Pydantic validators" in result

    def test_plural_and_singular(self):
        assert _heuristic_clarify("how do I use a validator?") is not None
        assert _heuristic_clarify("how do I use validators?") is not None


# ── _strip_label_prefix ───────────────────────────────────────────────


class TestStripLabelPrefix:
    def test_clarification_prefix(self):
        assert _strip_label_prefix("Clarification: Do you mean X or Y?") == "Do you mean X or Y?"

    def test_question_prefix(self):
        assert _strip_label_prefix("Question: Do you mean X or Y?") == "Do you mean X or Y?"

    def test_q_prefix_colon(self):
        assert _strip_label_prefix("Q: Do you mean X or Y?") == "Do you mean X or Y?"

    def test_q_prefix_dash(self):
        assert _strip_label_prefix("Q- Do you mean X or Y?") == "Do you mean X or Y?"

    def test_no_prefix_unchanged(self):
        assert _strip_label_prefix("Do you mean X or Y?") == "Do you mean X or Y?"

    def test_leading_whitespace_with_prefix(self):
        assert _strip_label_prefix("  Clarification: Do you mean X?") == "Do you mean X?"

    def test_case_insensitive_prefix(self):
        assert _strip_label_prefix("clarification: Do you mean X?") == "Do you mean X?"

    def test_empty_string(self):
        assert _strip_label_prefix("") == ""

    def test_whitespace_only(self):
        assert _strip_label_prefix("   ") == ""


# ── PassthroughQueryClarifier ─────────────────────────────────────────


class TestPassthroughQueryClarifier:
    def test_returns_none(self):
        c = PassthroughQueryClarifier()
        assert c.clarify("how do I use validators?") is None

    def test_ignores_history(self):
        c = PassthroughQueryClarifier()
        assert c.clarify("query", "history") is None

    def test_empty_query(self):
        c = PassthroughQueryClarifier()
        assert c.clarify("") is None


# ── LLMQueryClarifier ─────────────────────────────────────────────────


def _mock_ollama(content: str) -> MagicMock:
    """Build a mock Ollama client whose chat() returns the given content."""
    client = MagicMock()
    client.chat.return_value = {"message": {"content": content}}
    return client


class TestLLMQueryClarifier:
    def test_llm_returns_clarification(self):
        c = LLMQueryClarifier()
        c._client = _mock_ollama("Do you mean Pydantic validators or FastAPI dependency validators?")
        result = c.clarify("how do I use validators?")
        assert result == "Do you mean Pydantic validators or FastAPI dependency validators?"

    def test_strips_label_prefix(self):
        c = LLMQueryClarifier()
        c._client = _mock_ollama("Clarification: Do you mean X or Y?")
        result = c.clarify("ambiguous query")
        assert result == "Do you mean X or Y?"

    def test_empty_response_falls_back_to_heuristic(self):
        c = LLMQueryClarifier()
        c._client = _mock_ollama("")
        result = c.clarify("how do I use validators?")
        assert result is not None
        assert "Pydantic validators" in result

    def test_whitespace_response_falls_back_to_heuristic(self):
        c = LLMQueryClarifier()
        c._client = _mock_ollama("   \n  \n")
        result = c.clarify("how do I use validators?")
        assert result is not None
        assert "Pydantic validators" in result

    def test_no_message_key_falls_back(self):
        c = LLMQueryClarifier()
        c._client = MagicMock()
        c._client.chat.return_value = {}
        result = c.clarify("how do I use validators?")
        assert result is not None
        assert "Pydantic validators" in result

    def test_exception_falls_back_to_heuristic(self):
        c = LLMQueryClarifier()
        client = MagicMock()
        client.chat.side_effect = ConnectionError("no ollama")
        c._client = client
        result = c.clarify("how do I use validators?")
        assert result is not None
        assert "Pydantic validators" in result

    def test_llm_and_heuristic_both_fail_returns_none(self):
        """When the LLM returns nothing AND the heuristic finds no shared
        concept, ``clarify`` returns None so the caller can use
        GENERIC_CLARIFICATION."""
        c = LLMQueryClarifier()
        c._client = _mock_ollama("")
        result = c.clarify("how do I use path parameters in FastAPI?")
        assert result is None

    def test_use_llm_false_uses_heuristic(self):
        c = LLMQueryClarifier(use_llm=False)
        c._client = _mock_ollama("should not be used")
        result = c.clarify("how do I use validators?")
        assert result is not None
        assert "Pydantic validators" in result
        c._client.chat.assert_not_called()

    def test_use_llm_false_no_concept_returns_none(self):
        c = LLMQueryClarifier(use_llm=False)
        result = c.clarify("how do I use path parameters in FastAPI?")
        assert result is None

    def test_stream_false(self):
        c = LLMQueryClarifier()
        c._client = _mock_ollama("Do you mean X or Y?")
        c.clarify("ambiguous query")
        assert c._client.chat.call_args.kwargs["stream"] is False

    def test_query_in_prompt(self):
        c = LLMQueryClarifier()
        c._client = _mock_ollama("Do you mean X or Y?")
        c.clarify("how do I use validators?")
        prompt = c._client.chat.call_args.kwargs["messages"][0]["content"]
        assert "how do I use validators?" in prompt

    def test_close_resets_client(self):
        c = LLMQueryClarifier()
        c._client = MagicMock()
        c.close()
        assert c._client is None

    def test_lazy_client_construction(self):
        c = LLMQueryClarifier()
        assert c._client is None
        with patch("ollama.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            _ = c.client
            MockClient.assert_called_once_with(host="http://localhost:11434")


# ── CLARIFY_PROMPT_TEMPLATE ───────────────────────────────────────────


class TestClarifyPromptTemplate:
    def test_has_query_placeholder(self):
        assert "{query}" in CLARIFY_PROMPT_TEMPLATE

    def test_has_clarification_instruction(self):
        assert "Clarification question:" in CLARIFY_PROMPT_TEMPLATE

    def test_instructs_only_clarification(self):
        assert "ONLY the clarification question" in CLARIFY_PROMPT_TEMPLATE

    def test_mentions_ambiguous(self):
        assert "ambiguous" in CLARIFY_PROMPT_TEMPLATE


# ── GENERIC_CLARIFICATION ─────────────────────────────────────────────


class TestGenericClarification:
    def test_non_empty(self):
        assert len(GENERIC_CLARIFICATION) > 0

    def test_mentions_libraries(self):
        lowered = GENERIC_CLARIFICATION.lower()
        assert "fastapi" in lowered
        assert "pydantic" in lowered
        assert "sqlmodel" in lowered

    def test_asks_to_clarify(self):
        assert "clarify" in GENERIC_CLARIFICATION.lower()
