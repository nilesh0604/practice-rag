"""Tests for the NIM generator + fallback generator (rag/nim_generator.py).

The httpx client is mocked so no network call is made. The fallback
generator tests use mock primary/fallback objects with the same
``stream(query, context, history)`` interface as ``Generator``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rag.generator import Generator
from rag.nim_generator import (
    NIM_BASE_URL,
    NIM_GENERATION_MODEL,
    NIMError,
    NIMGenerator,
    FallbackGenerator,
    build_generator,
)


# ── helpers ────────────────────────────────────────────────────────────


def _sse_lines(deltas: list[str]) -> list[str]:
    """Build fake SSE response lines from OpenAI-style content deltas."""
    lines = []
    for delta in deltas:
        chunk = {"choices": [{"delta": {"content": delta}}]}
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return lines


class _MockGen:
    """Minimal mock generator with ``stream`` + ``close``."""

    def __init__(self, tokens: list[str] | Exception):
        self.tokens = tokens
        self.stream_called = False
        self.closed = False

    def stream(self, query, context, history=""):
        self.stream_called = True
        if isinstance(self.tokens, Exception):
            raise self.tokens
        yield from self.tokens

    def close(self):
        self.closed = True


# ── NIMGenerator: streaming ────────────────────────────────────────────


class TestNIMGeneratorStream:
    def test_yields_tokens_from_sse(self):
        gen = NIMGenerator(api_key="test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(_sse_lines(["Hello", " ", "world"]))
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        tokens = list(gen.stream("query", "context"))
        assert tokens == ["Hello", " ", "world"]

    def test_skips_empty_content(self):
        gen = NIMGenerator(api_key="test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': 'a'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': ''}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'b'}}]})}",
            "data: [DONE]",
        ]
        mock_response.iter_lines.return_value = iter(lines)
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        tokens = list(gen.stream("q", "c"))
        assert tokens == ["a", "b"]

    def test_passes_system_and_user_messages(self):
        gen = NIMGenerator(api_key="test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(_sse_lines(["x"]))
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        list(gen.stream("my question", "my context", "my history"))
        call_args = mock_client.stream.call_args
        payload = call_args.kwargs["json"]
        assert payload["messages"][0]["role"] == "system"
        assert "my context" in payload["messages"][0]["content"]
        assert "my history" in payload["messages"][0]["content"]
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "my question"

    def test_stream_true_in_payload(self):
        gen = NIMGenerator(api_key="test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(_sse_lines(["x"]))
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        list(gen.stream("q", "c"))
        payload = mock_client.stream.call_args.kwargs["json"]
        assert payload["stream"] is True

    def test_authorization_header(self):
        gen = NIMGenerator(api_key="my-secret-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(_sse_lines(["x"]))
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        list(gen.stream("q", "c"))
        headers = mock_client.stream.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-secret-key"

    def test_default_model(self):
        assert NIM_GENERATION_MODEL == "meta/llama-3.1-8b-instruct"

    def test_default_base_url(self):
        assert NIM_BASE_URL == "https://integrate.api.nvidia.com/v1"

    def test_url_built_from_base_url(self):
        gen = NIMGenerator(api_key="k")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(_sse_lines(["x"]))
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        list(gen.stream("q", "c"))
        url = mock_client.stream.call_args.args[1]
        assert url == "https://integrate.api.nvidia.com/v1/chat/completions"

    def test_lazy_client_construction(self):
        gen = NIMGenerator(api_key="k")
        assert gen._client is None
        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            _ = gen.client
            MockClient.assert_called_once()
        assert gen._client is mock_instance

    def test_close_resets_client(self):
        gen = NIMGenerator(api_key="k")
        gen._client = MagicMock()
        gen.close()
        assert gen._client is None


# ── NIMGenerator: error handling ───────────────────────────────────────


class TestNIMGeneratorErrors:
    def test_429_raises_nim_error(self):
        gen = NIMGenerator(api_key="k")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.read.return_value = b"rate limited"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        with pytest.raises(NIMError, match="429"):
            list(gen.stream("q", "c"))

    def test_404_raises_nim_error(self):
        gen = NIMGenerator(api_key="k")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.read.return_value = b"not found"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        with pytest.raises(NIMError, match="404"):
            list(gen.stream("q", "c"))

    def test_500_raises_nim_error(self):
        gen = NIMGenerator(api_key="k")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.read.return_value = b"internal server error"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response
        gen._client = mock_client

        with pytest.raises(NIMError, match="500"):
            list(gen.stream("q", "c"))

    def test_timeout_raises_nim_error(self):
        import httpx

        gen = NIMGenerator(api_key="k")
        mock_client = MagicMock()
        mock_client.stream.side_effect = httpx.TimeoutException("timed out")
        gen._client = mock_client

        with pytest.raises(NIMError, match="timed out"):
            list(gen.stream("q", "c"))

    def test_connect_error_raises_nim_error(self):
        import httpx

        gen = NIMGenerator(api_key="k")
        mock_client = MagicMock()
        mock_client.stream.side_effect = httpx.ConnectError("conn refused")
        gen._client = mock_client

        with pytest.raises(NIMError, match="conn"):
            list(gen.stream("q", "c"))


# ── NIMGenerator: no circuit breaker (owned by FallbackGenerator) ───────


class TestNIMGeneratorNoBreaker:
    def test_nim_generator_has_no_circuit_breaker_attr(self):
        gen = NIMGenerator(api_key="k")
        assert not hasattr(gen, "circuit_breaker")


# ── FallbackGenerator ──────────────────────────────────────────────────


class TestFallbackGenerator:
    def test_primary_success_no_fallback(self):
        primary = _MockGen(["a", "b", "c"])
        fallback = _MockGen(["fallback"])
        fg = FallbackGenerator(primary, fallback)

        tokens = list(fg.stream("q", "c"))
        assert tokens == ["a", "b", "c"]
        assert primary.stream_called
        assert not fallback.stream_called

    def test_primary_fails_fallback_used(self):
        primary = _MockGen(NIMError("NIM down"))
        fallback = _MockGen(["fallback", "tokens"])
        fg = FallbackGenerator(primary, fallback)

        tokens = list(fg.stream("q", "c"))
        assert tokens == ["fallback", "tokens"]
        assert primary.stream_called
        assert fallback.stream_called

    def test_both_fail_yields_refusal(self):
        primary = _MockGen(NIMError("NIM down"))
        fallback = _MockGen(ConnectionError("Ollama down"))
        fg = FallbackGenerator(primary, fallback)

        tokens = list(fg.stream("q", "c"))
        assert tokens == ["I don't have enough information to answer that."]

    def test_mid_stream_failure_propagates(self):
        class _MidStreamFail:
            def stream(self, q, c, h=""):
                yield "partial"
                raise NIMError("mid-stream failure")
            def close(self): pass

        primary = _MidStreamFail()
        fallback = _MockGen(["fallback"])
        fg = FallbackGenerator(primary, fallback)

        with pytest.raises(NIMError):
            list(fg.stream("q", "c"))

    def test_circuit_open_triggers_fallback(self):
        from api.observability import CircuitBreaker

        cb = CircuitBreaker(threshold=2, timeout=30)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()

        class _CBPrimary:
            def __init__(self):
                self.stream_called = False
            def stream(self, q, c, h=""):
                self.stream_called = True
                yield "should not reach"
            def close(self): pass

        primary = _CBPrimary()
        fallback = _MockGen(["fallback"])
        fg = FallbackGenerator(primary, fallback, circuit_breaker=cb)

        tokens = list(fg.stream("q", "c"))
        assert tokens == ["fallback"]
        assert not primary.stream_called

    def test_primary_failure_records_breaker_failure(self):
        from api.observability import CircuitBreaker

        cb = CircuitBreaker(threshold=3)
        primary = _MockGen(NIMError("NIM down"))
        fallback = _MockGen(["fallback"])
        fg = FallbackGenerator(primary, fallback, circuit_breaker=cb)

        list(fg.stream("q", "c"))
        assert cb.failures == 1

    def test_close_closes_both(self):
        primary = _MockGen(["a"])
        fallback = _MockGen(["b"])
        fg = FallbackGenerator(primary, fallback)
        fg.close()
        assert primary.closed
        assert fallback.closed

    def test_close_handles_missing_close_method(self):
        class _NoClose:
            def stream(self, q, c, h=""):
                yield "x"

        primary = _NoClose()
        fallback = _MockGen(["y"])
        fg = FallbackGenerator(primary, fallback)
        fg.close()  # should not raise
        assert fallback.closed


# ── build_generator factory ────────────────────────────────────────────


class TestBuildGenerator:
    def test_default_returns_ollama_generator(self, monkeypatch):
        monkeypatch.delenv("NIM_ENABLED", raising=False)
        from rag.generator import Generator

        gen = build_generator()
        assert isinstance(gen, Generator)

    def test_nim_disabled_returns_ollama_generator(self, monkeypatch):
        monkeypatch.setenv("NIM_ENABLED", "false")
        from rag.generator import Generator

        gen = build_generator()
        assert isinstance(gen, Generator)

    def test_nim_enabled_returns_fallback_generator(self, monkeypatch):
        monkeypatch.setenv("NIM_ENABLED", "true")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

        gen = build_generator()
        assert isinstance(gen, FallbackGenerator)
        assert isinstance(gen.primary, NIMGenerator)
        assert isinstance(gen.fallback, Generator)

    def test_nim_enabled_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("NIM_ENABLED", "TRUE")
        monkeypatch.setenv("NVIDIA_API_KEY", "k")

        gen = build_generator()
        assert isinstance(gen, FallbackGenerator)

    def test_nim_enabled_accepts_1(self, monkeypatch):
        monkeypatch.setenv("NIM_ENABLED", "1")
        monkeypatch.setenv("NVIDIA_API_KEY", "k")

        gen = build_generator()
        assert isinstance(gen, FallbackGenerator)
