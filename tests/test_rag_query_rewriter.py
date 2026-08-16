"""Tests for the query rewriter (rag/query_rewriter.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.query_rewriter import (
    LLMQueryRewriter,
    PassthroughQueryRewriter,
    REWRITE_PROMPT_TEMPLATE,
    _needs_rewrite,
)

# A query + history pair that passes the _needs_rewrite heuristic (has
# history AND an anaphoric pronoun) so the LLM call path is exercised.
_REWRITE_QUERY = "how does it work"
_REWRITE_HISTORY = "user asked about Depends"


class TestPassthroughQueryRewriter:
    def test_returns_query_unchanged(self):
        rw = PassthroughQueryRewriter()
        assert rw.rewrite("how to use FastAPI") == "how to use FastAPI"

    def test_ignores_history(self):
        rw = PassthroughQueryRewriter()
        assert rw.rewrite("query", "some history") == "query"

    def test_empty_query(self):
        rw = PassthroughQueryRewriter()
        assert rw.rewrite("") == ""


class TestNeedsRewriteHeuristic:
    """Tests for the latency-skip heuristic (_needs_rewrite)."""

    def test_no_history_skips(self):
        assert _needs_rewrite("how does it work", "") is False

    def test_whitespace_history_skips(self):
        assert _needs_rewrite("how does it work", "   \n  ") is False

    def test_none_history_skips(self):
        assert _needs_rewrite("how does it work", None) is False  # type: ignore[arg-type]

    def test_history_but_no_pronoun_skips(self):
        assert _needs_rewrite("how to use FastAPI Depends", "prior context") is False

    def test_pronoun_with_history_rewrites(self):
        assert _needs_rewrite("how does it work", "prior context") is True

    def test_that_with_history_rewrites(self):
        assert _needs_rewrite("what about that?", "prior context") is True

    def test_they_with_history_rewrites(self):
        assert _needs_rewrite("how do they compare", "prior context") is True

    def test_the_above_with_history_rewrites(self):
        assert _needs_rewrite("summarize the above", "prior context") is True

    def test_case_insensitive(self):
        assert _needs_rewrite("HOW DOES IT WORK", "prior context") is True
        assert _needs_rewrite("What About THAT?", "prior context") is True

    def test_word_boundary_no_false_positive_on_it(self):
        # "it" inside "title" must NOT match
        assert _needs_rewrite("what is a title field", "prior context") is False

    def test_word_boundary_no_false_positive_on_the(self):
        # "the" inside "these" is fine, but "the" alone is not a marker
        # (not in the list); "these" IS a marker though
        assert _needs_rewrite("what is the answer", "prior context") is False
        assert _needs_rewrite("what are these errors", "prior context") is True

    def test_word_boundary_no_false_positive_on_his(self):
        # "his" inside "history" must NOT match
        assert _needs_rewrite("what is the history field", "prior context") is False

    def test_word_boundary_no_false_positive_on_she(self):
        # "she" inside "washed" must NOT match
        assert _needs_rewrite("what is a washed dataset", "prior context") is False

    def test_self_contained_technical_query_skips(self):
        assert _needs_rewrite(
            "how to configure Pydantic v2 model validators", "prior context"
        ) is False

    def test_follow_up_with_pronoun_rewrites(self):
        assert _needs_rewrite(
            "can you explain that with an example", "user asked about Depends"
        ) is True


class TestLLMQueryRewriter:
    def test_returns_rewritten_query(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "standalone query about FastAPI"}}
        rw._client = mock_client

        result = rw.rewrite(_REWRITE_QUERY, _REWRITE_HISTORY)
        assert result == "standalone query about FastAPI"

    def test_takes_first_line_only(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "first line\nsecond line"}}
        rw._client = mock_client

        result = rw.rewrite(_REWRITE_QUERY, _REWRITE_HISTORY)
        assert result == "first line"

    def test_strips_whitespace(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "  trimmed query  "}}
        rw._client = mock_client

        assert rw.rewrite(_REWRITE_QUERY, _REWRITE_HISTORY) == "trimmed query"

    def test_empty_response_falls_back_to_original(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": ""}}
        rw._client = mock_client

        assert rw.rewrite(_REWRITE_QUERY, _REWRITE_HISTORY) == _REWRITE_QUERY

    def test_no_message_key_falls_back(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        rw._client = mock_client

        assert rw.rewrite(_REWRITE_QUERY, _REWRITE_HISTORY) == _REWRITE_QUERY

    def test_exception_falls_back_to_original(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("no ollama")
        rw._client = mock_client

        assert rw.rewrite(_REWRITE_QUERY, _REWRITE_HISTORY) == _REWRITE_QUERY

    def test_history_in_prompt(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "rewritten"}}
        rw._client = mock_client

        rw.rewrite("what about that?", "conversation context here")
        call = mock_client.chat.call_args
        prompt = call.kwargs["messages"][0]["content"]
        assert "conversation context here" in prompt
        assert "what about that?" in prompt

    def test_stream_false(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "r"}}
        rw._client = mock_client

        rw.rewrite(_REWRITE_QUERY, _REWRITE_HISTORY)
        assert mock_client.chat.call_args.kwargs["stream"] is False

    # ── latency-skip heuristic integration ────────────────────────────

    def test_skips_llm_call_when_no_history(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        rw._client = mock_client

        result = rw.rewrite("how does it work", "")
        assert result == "how does it work"
        mock_client.chat.assert_not_called()

    def test_skips_llm_call_when_no_pronoun(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        rw._client = mock_client

        result = rw.rewrite("how to use FastAPI Depends", "prior context")
        assert result == "how to use FastAPI Depends"
        mock_client.chat.assert_not_called()

    def test_calls_llm_when_pronoun_and_history(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "rewritten"}}
        rw._client = mock_client

        rw.rewrite("how does it work", "prior context")
        mock_client.chat.assert_called_once()

    def test_close_resets_client(self):
        rw = LLMQueryRewriter()
        rw._client = MagicMock()
        rw.close()
        assert rw._client is None

    def test_lazy_client_construction(self):
        rw = LLMQueryRewriter()
        assert rw._client is None
        with patch("ollama.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            _ = rw.client
            MockClient.assert_called_once_with(host="http://localhost:11434")


class TestRewritePromptTemplate:
    def test_has_placeholders(self):
        assert "{history}" in REWRITE_PROMPT_TEMPLATE
        assert "{query}" in REWRITE_PROMPT_TEMPLATE

    def test_outputs_only_query_instruction(self):
        assert "ONLY the rewritten query" in REWRITE_PROMPT_TEMPLATE
