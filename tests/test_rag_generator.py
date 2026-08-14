"""Tests for the generator (rag/generator.py).

The Ollama client is mocked so no network call is made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.generator import (
    GENERATION_MODEL,
    NUM_CTX,
    NUM_PREDICT,
    SYSTEM_PROMPT_TEMPLATE,
    TEMPERATURE,
    TOP_P,
    Generator,
    build_system_prompt,
)


def _chat_chunks(tokens: list[str]):
    """Build fake Ollama streaming chunks (list of dicts with message.content)."""
    return [{"message": {"content": tok}} for tok in tokens]


class TestBuildSystemPrompt:
    def test_context_inserted(self):
        prompt = build_system_prompt("my context here")
        assert "my context here" in prompt

    def test_history_inserted(self):
        prompt = build_system_prompt("ctx", "user asked about X")
        assert "user asked about X" in prompt

    def test_empty_history(self):
        prompt = build_system_prompt("ctx", "")
        assert "CONTEXT:\nctx" in prompt

    def test_rules_present(self):
        prompt = build_system_prompt("ctx")
        assert "RULES:" in prompt
        assert "[Source: title]" in prompt

    def test_template_has_placeholders(self):
        assert "{context}" in SYSTEM_PROMPT_TEMPLATE
        assert "{history}" in SYSTEM_PROMPT_TEMPLATE


class TestGeneratorStream:
    def test_yields_tokens(self):
        gen = Generator()
        mock_client = MagicMock()
        mock_client.chat.return_value = iter(_chat_chunks(["Hello", " ", "world"]))
        gen._client = mock_client

        tokens = list(gen.stream("query", "context"))
        assert tokens == ["Hello", " ", "world"]

    def test_skips_empty_content(self):
        gen = Generator()
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([
            {"message": {"content": "a"}},
            {"message": {"content": ""}},
            {"message": {"content": "b"}},
        ])
        gen._client = mock_client

        tokens = list(gen.stream("query", "context"))
        assert tokens == ["a", "b"]

    def test_passes_system_and_user_messages(self):
        gen = Generator()
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([])
        gen._client = mock_client

        list(gen.stream("my question", "my context", "my history"))
        call = mock_client.chat.call_args
        messages = call.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "my context" in messages[0]["content"]
        assert "my history" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "my question"

    def test_stream_true_passed(self):
        gen = Generator()
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([])
        gen._client = mock_client

        list(gen.stream("q", "c"))
        assert mock_client.chat.call_args.kwargs["stream"] is True

    def test_options_contain_generation_settings(self):
        gen = Generator()
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([])
        gen._client = mock_client

        list(gen.stream("q", "c"))
        opts = mock_client.chat.call_args.kwargs["options"]
        assert opts["temperature"] == TEMPERATURE
        assert opts["num_ctx"] == NUM_CTX
        assert opts["top_p"] == TOP_P
        assert opts["num_predict"] == NUM_PREDICT

    def test_model_passed(self):
        gen = Generator(model="llama3.1:8b")
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([])
        gen._client = mock_client

        list(gen.stream("q", "c"))
        assert mock_client.chat.call_args.kwargs["model"] == "llama3.1:8b"

    def test_default_model(self):
        assert GENERATION_MODEL == "llama3.2:3b"

    def test_lazy_client_construction(self):
        gen = Generator()
        assert gen._client is None
        with patch("ollama.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            _ = gen.client
            MockClient.assert_called_once_with(host="http://localhost:11434")
        assert gen._client is mock_instance

    def test_close_resets_client(self):
        gen = Generator()
        gen._client = MagicMock()
        gen.close()
        assert gen._client is None

    def test_custom_settings(self):
        gen = Generator(temperature=0.5, num_predict=256)
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([])
        gen._client = mock_client

        list(gen.stream("q", "c"))
        opts = mock_client.chat.call_args.kwargs["options"]
        assert opts["temperature"] == 0.5
        assert opts["num_predict"] == 256

    def test_concatenated_tokens_form_answer(self):
        gen = Generator()
        mock_client = MagicMock()
        mock_client.chat.return_value = iter(_chat_chunks(["Fast", "API", " uses ", "Depends"]))
        gen._client = mock_client

        answer = "".join(gen.stream("q", "c"))
        assert answer == "FastAPI uses Depends"
