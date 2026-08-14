"""Tests for the query rewriter (rag/query_rewriter.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.query_rewriter import (
    LLMQueryRewriter,
    PassthroughQueryRewriter,
    REWRITE_PROMPT_TEMPLATE,
)


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


class TestLLMQueryRewriter:
    def test_returns_rewritten_query(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "standalone query about FastAPI"}}
        rw._client = mock_client

        result = rw.rewrite("how does it work", "user asked about Depends")
        assert result == "standalone query about FastAPI"

    def test_takes_first_line_only(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "first line\nsecond line"}}
        rw._client = mock_client

        result = rw.rewrite("query")
        assert result == "first line"

    def test_strips_whitespace(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "  trimmed query  "}}
        rw._client = mock_client

        assert rw.rewrite("query") == "trimmed query"

    def test_empty_response_falls_back_to_original(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": ""}}
        rw._client = mock_client

        assert rw.rewrite("original query") == "original query"

    def test_no_message_key_falls_back(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        rw._client = mock_client

        assert rw.rewrite("original") == "original"

    def test_exception_falls_back_to_original(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("no ollama")
        rw._client = mock_client

        assert rw.rewrite("original query") == "original query"

    def test_history_in_prompt(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "rewritten"}}
        rw._client = mock_client

        rw.rewrite("my question", "conversation context here")
        call = mock_client.chat.call_args
        prompt = call.kwargs["messages"][0]["content"]
        assert "conversation context here" in prompt
        assert "my question" in prompt

    def test_stream_false(self):
        rw = LLMQueryRewriter()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "r"}}
        rw._client = mock_client

        rw.rewrite("q")
        assert mock_client.chat.call_args.kwargs["stream"] is False

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
