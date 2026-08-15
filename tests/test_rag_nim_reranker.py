"""Tests for the NIM reranker (rag/nim_reranker.py) + orchestrator integration.

The httpx client is mocked so no network call is made. The reranker tests
verify request shape, reordering, graceful degradation on every error path,
and circuit-breaker behavior. The orchestrator integration tests verify the
rerank step is inserted between retrieval and context assembly when a
reranker is configured, and skipped when it is not.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rag.nim_reranker import (
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANK_TOP_N,
    NIM_RERANK_BASE_URL,
    NIM_RERANK_MODEL,
    NIMRerankError,
    NIMReranker,
    TRUNCATE_END,
    build_reranker,
    get_rerank_candidate_k,
)
from schemas.documents import RetrievedDoc


# ── helpers ────────────────────────────────────────────────────────────


def _make_doc(title="Doc", content="content", score=0.5, idx=0):
    return RetrievedDoc(
        id=f"doc-{idx}",
        content=content,
        title=title,
        source_url="https://example.com/x",
        section=None,
        score=score,
    )


def _make_docs(n: int) -> list[RetrievedDoc]:
    return [
        _make_doc(title=f"Doc {i}", content=f"content {i}", score=0.5 - i * 0.01, idx=i)
        for i in range(n)
    ]


def _mock_response(status_code=200, rankings=None):
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if status_code < 400:
        resp.json.return_value = {"rankings": rankings or []}
    else:
        resp.text = "error body"
    return resp


def _rankings(indices: list[int]) -> list[dict]:
    """Build a rankings list from a list of indices (logits descending)."""
    return [
        {"index": idx, "logit": 10.0 - i}
        for i, idx in enumerate(indices)
    ]


# ── NIMReranker: endpoint URL ──────────────────────────────────────────


class TestEndpointURL:
    def test_url_embeds_model_in_path(self):
        rr = NIMReranker(api_key="k")
        url = rr._endpoint_url()
        assert url == f"{NIM_RERANK_BASE_URL}/retrieval/{NIM_RERANK_MODEL}/reranking"

    def test_url_strips_trailing_slash_from_base(self):
        rr = NIMReranker(api_key="k", base_url="https://ai.api.nvidia.com/v1/")
        assert rr._endpoint_url() == (
            f"https://ai.api.nvidia.com/v1/retrieval/{NIM_RERANK_MODEL}/reranking"
        )

    def test_url_uses_custom_model(self):
        rr = NIMReranker(api_key="k", model="nvidia/custom-rerank")
        assert "/retrieval/nvidia/custom-rerank/reranking" in rr._endpoint_url()

    def test_base_url_differs_from_chat_embeddings(self):
        """The reranking host is ai.api.nvidia.com, not integrate.api.nvidia.com."""
        assert NIM_RERANK_BASE_URL == "https://ai.api.nvidia.com/v1"


# ── NIMReranker: request shape ─────────────────────────────────────────


class TestRequestShape:
    def test_payload_has_model_query_passages_truncate(self):
        docs = _make_docs(3)
        rr = NIMReranker(api_key="k")
        payload = rr._build_payload("my query", docs)
        assert payload["model"] == NIM_RERANK_MODEL
        assert payload["query"] == {"text": "my query"}
        assert payload["truncate"] == TRUNCATE_END
        assert len(payload["passages"]) == 3
        assert payload["passages"][0] == {"text": "content 0"}

    def test_payload_uses_doc_content_not_title(self):
        docs = [_make_doc(title="Important Title", content="the actual text", idx=0)]
        rr = NIMReranker(api_key="k")
        payload = rr._build_payload("q", docs)
        assert payload["passages"][0]["text"] == "the actual text"

    def test_custom_truncate_none(self):
        rr = NIMReranker(api_key="k", truncate="NONE")
        payload = rr._build_payload("q", _make_docs(1))
        assert payload["truncate"] == "NONE"

    def test_custom_model_in_payload(self):
        rr = NIMReranker(api_key="k", model="nvidia/custom-rerank")
        payload = rr._build_payload("q", _make_docs(1))
        assert payload["model"] == "nvidia/custom-rerank"

    def test_authorization_header_sent(self):
        rr = NIMReranker(api_key="secret-key")
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            200, rankings=_rankings([0]),
        )
        rr._client = mock_client
        rr._do_request("q", _make_docs(1))
        _, kwargs = mock_client.post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer secret-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"


# ── NIMReranker: reordering ────────────────────────────────────────────


class TestRerank:
    def test_reorders_docs_by_rankings(self):
        docs = _make_docs(5)
        rr = NIMReranker(api_key="k", top_n=3)
        mock_client = MagicMock()
        # Rankings say: index 2 is best, then 0, then 4, then 1, then 3
        mock_client.post.return_value = _mock_response(
            200, rankings=_rankings([2, 0, 4, 1, 3]),
        )
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert len(result) == 3
        assert result[0].id == "doc-2"
        assert result[1].id == "doc-0"
        assert result[2].id == "doc-4"

    def test_truncates_to_top_n(self):
        docs = _make_docs(10)
        rr = NIMReranker(api_key="k", top_n=5)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            200, rankings=_rankings(list(range(10))),
        )
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert len(result) == 5

    def test_preserves_original_rrf_scores(self):
        docs = [
            _make_doc(title="A", content="a", score=0.9, idx=0),
            _make_doc(title="B", content="b", score=0.5, idx=1),
            _make_doc(title="C", content="c", score=0.3, idx=2),
        ]
        rr = NIMReranker(api_key="k", top_n=2)
        mock_client = MagicMock()
        # Reranker says index 2 is best (even though it had the lowest RRF score)
        mock_client.post.return_value = _mock_response(
            200, rankings=_rankings([2, 0, 1]),
        )
        rr._client = mock_client
        result = rr.rerank("query", docs)
        # The doc that was originally score=0.3 is now first, but its score
        # is still 0.3 (the reranker reorders but does not rescore).
        assert result[0].score == 0.3
        assert result[1].score == 0.9

    def test_returns_as_is_when_fewer_than_top_n(self):
        docs = _make_docs(3)
        rr = NIMReranker(api_key="k", top_n=5)
        mock_client = MagicMock()
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert result == docs
        # No API call when reranking is pointless
        mock_client.post.assert_not_called()

    def test_returns_empty_for_empty_docs(self):
        rr = NIMReranker(api_key="k")
        assert rr.rerank("query", []) == []

    def test_exactly_top_n_does_not_call_api(self):
        """When len(docs) == top_n, reranking is pointless — skip the call."""
        docs = _make_docs(5)
        rr = NIMReranker(api_key="k", top_n=5)
        mock_client = MagicMock()
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert len(result) == 5
        mock_client.post.assert_not_called()


# ── NIMReranker: error mapping + graceful degradation ──────────────────


class TestErrorMapping:
    def _setup(self, status_code=200, rankings=None, exc=None):
        rr = NIMReranker(api_key="k", top_n=3)
        mock_client = MagicMock()
        if exc is not None:
            mock_client.post.side_effect = exc
        else:
            mock_client.post.return_value = _mock_response(status_code, rankings)
        rr._client = mock_client
        return rr, mock_client

    def test_429_raises_nim_rerank_error(self):
        rr, _ = self._setup(status_code=429)
        with pytest.raises(NIMRerankError, match="429"):
            rr._do_request("q", _make_docs(5))

    def test_404_raises_nim_rerank_error(self):
        rr, _ = self._setup(status_code=404)
        with pytest.raises(NIMRerankError, match="404"):
            rr._do_request("q", _make_docs(5))

    def test_500_raises_nim_rerank_error(self):
        rr, _ = self._setup(status_code=500)
        with pytest.raises(NIMRerankError, match="500"):
            rr._do_request("q", _make_docs(5))

    def test_empty_rankings_raises(self):
        rr, _ = self._setup(rankings=[])
        with pytest.raises(NIMRerankError, match="no rankings"):
            rr._do_request("q", _make_docs(5))

    def test_out_of_range_index_raises(self):
        rr, _ = self._setup(rankings=[{"index": 10, "logit": 1.0}])
        with pytest.raises(NIMRerankError, match="out-of-range"):
            rr._do_request("q", _make_docs(5))

    def test_missing_index_field_raises(self):
        rr, _ = self._setup(rankings=[{"logit": 1.0}])
        with pytest.raises(NIMRerankError, match="out-of-range"):
            rr._do_request("q", _make_docs(5))

    def test_timeout_raises_nim_rerank_error(self):
        import httpx
        rr, _ = self._setup(exc=httpx.TimeoutException("timed out"))
        with pytest.raises(NIMRerankError, match="timed out"):
            rr._do_request("q", _make_docs(5))

    def test_connect_error_raises_nim_rerank_error(self):
        import httpx
        rr, _ = self._setup(exc=httpx.ConnectError("conn refused"))
        with pytest.raises(NIMRerankError, match="connection error"):
            rr._do_request("q", _make_docs(5))

    def test_unexpected_error_raises_nim_rerank_error(self):
        rr, _ = self._setup(exc=ValueError("boom"))
        with pytest.raises(NIMRerankError, match="unexpected"):
            rr._do_request("q", _make_docs(5))


class TestGracefulDegradation:
    """On any failure, rerank() returns original docs[:top_n] — never raises."""

    def test_429_degrades_to_original_order(self):
        docs = _make_docs(10)
        rr = NIMReranker(api_key="k", top_n=3)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(429)
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert len(result) == 3
        assert result[0].id == "doc-0"
        assert result[1].id == "doc-1"
        assert result[2].id == "doc-2"

    def test_404_degrades_to_original_order(self):
        docs = _make_docs(10)
        rr = NIMReranker(api_key="k", top_n=3)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(404)
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert [d.id for d in result] == ["doc-0", "doc-1", "doc-2"]

    def test_timeout_degrades_to_original_order(self):
        import httpx
        docs = _make_docs(10)
        rr = NIMReranker(api_key="k", top_n=3)
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("slow")
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert len(result) == 3
        assert result[0].id == "doc-0"

    def test_connect_error_degrades_to_original_order(self):
        import httpx
        docs = _make_docs(10)
        rr = NIMReranker(api_key="k", top_n=3)
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("nope")
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert len(result) == 3

    def test_malformed_response_degrades(self):
        docs = _make_docs(10)
        rr = NIMReranker(api_key="k", top_n=3)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(200, rankings=[])
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert [d.id for d in result] == ["doc-0", "doc-1", "doc-2"]

    def test_rerank_never_raises(self):
        """The public rerank() method must never raise — it always degrades."""
        import httpx
        docs = _make_docs(10)
        rr = NIMReranker(api_key="k", top_n=3)
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("dead")
        rr._client = mock_client
        # Should not raise
        result = rr.rerank("query", docs)
        assert len(result) == 3


# ── NIMReranker: circuit breaker ───────────────────────────────────────


class TestCircuitBreaker:
    def test_open_circuit_degrades_gracefully(self):
        from api.observability import CircuitBreaker, CircuitOpenError
        docs = _make_docs(10)
        breaker = CircuitBreaker(threshold=1, timeout=30.0)
        # Trip the breaker with one failure
        breaker.record_failure()
        assert breaker.is_open()
        rr = NIMReranker(api_key="k", top_n=3, circuit_breaker=breaker)
        mock_client = MagicMock()
        rr._client = mock_client
        result = rr.rerank("query", docs)
        # Circuit open → no API call → original order
        mock_client.post.assert_not_called()
        assert [d.id for d in result] == ["doc-0", "doc-1", "doc-2"]

    def test_failure_recorded_on_error(self):
        from api.observability import CircuitBreaker
        docs = _make_docs(10)
        breaker = CircuitBreaker(threshold=3, timeout=30.0)
        rr = NIMReranker(api_key="k", top_n=3, circuit_breaker=breaker)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(429)
        rr._client = mock_client
        rr.rerank("query", docs)
        assert breaker.failures == 1

    def test_success_resets_breaker(self):
        from api.observability import CircuitBreaker
        docs = _make_docs(10)
        breaker = CircuitBreaker(threshold=3, timeout=30.0)
        breaker.record_failure()
        rr = NIMReranker(api_key="k", top_n=3, circuit_breaker=breaker)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            200, rankings=_rankings([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )
        rr._client = mock_client
        rr.rerank("query", docs)
        assert breaker.failures == 0

    def test_breaker_wraps_the_request(self):
        """When a breaker is configured, _do_request goes through breaker.call."""
        from api.observability import CircuitBreaker
        docs = _make_docs(10)
        breaker = CircuitBreaker(threshold=3, timeout=30.0)
        rr = NIMReranker(api_key="k", top_n=3, circuit_breaker=breaker)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            200, rankings=_rankings([2, 0, 1, 3, 4, 5, 6, 7, 8, 9]),
        )
        rr._client = mock_client
        result = rr.rerank("query", docs)
        assert result[0].id == "doc-2"
        assert breaker.failures == 0


# ── NIMReranker: lazy client + close ───────────────────────────────────


class TestLazyClient:
    def test_client_is_lazy(self):
        rr = NIMReranker(api_key="k")
        assert rr._client is None
        with patch("httpx.Client") as mock_httpx:
            _ = rr.client
            mock_httpx.assert_called_once()

    def test_client_reused(self):
        rr = NIMReranker(api_key="k")
        c1 = rr.client
        c2 = rr.client
        assert c1 is c2

    def test_close_releases_client(self):
        rr = NIMReranker(api_key="k")
        mock_client = MagicMock()
        rr._client = mock_client
        rr.close()
        mock_client.close.assert_called_once()
        assert rr._client is None

    def test_close_when_no_client(self):
        rr = NIMReranker(api_key="k")
        rr.close()  # should not raise
        assert rr._client is None

    def test_close_swallows_errors(self):
        rr = NIMReranker(api_key="k")
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("boom")
        rr._client = mock_client
        rr.close()  # should not raise
        assert rr._client is None


# ── build_reranker factory ─────────────────────────────────────────────


class TestBuildReranker:
    def test_returns_none_when_nim_disabled(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("NIM_ENABLED", None)
            assert build_reranker() is None

    def test_returns_none_when_nim_enabled_false(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "false"}):
            assert build_reranker() is None

    def test_returns_none_when_rerank_disabled(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "true", "NIM_RERANK_ENABLED": "false"}):
            assert build_reranker() is None

    def test_returns_reranker_when_enabled(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "true", "NVIDIA_API_KEY": "k"}):
            rr = build_reranker()
            assert rr is not None
            assert isinstance(rr, NIMReranker)
            assert rr.api_key == "k"

    def test_returns_reranker_when_rerank_enabled_explicitly(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "true", "NIM_RERANK_ENABLED": "true"}):
            rr = build_reranker()
            assert isinstance(rr, NIMReranker)

    def test_rerank_enabled_defaults_true_when_nim_enabled(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "true"}, clear=False):
            import os
            os.environ.pop("NIM_RERANK_ENABLED", None)
            rr = build_reranker()
            assert isinstance(rr, NIMReranker)

    def test_top_n_override(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "true", "NIM_RERANK_TOP_N": "7"}):
            rr = build_reranker()
            assert rr.top_n == 7

    def test_top_n_default(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "true"}):
            rr = build_reranker()
            assert rr.top_n == DEFAULT_RERANK_TOP_N

    def test_dedicated_circuit_breaker(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "true"}):
            rr = build_reranker()
            assert rr.circuit_breaker is not None
            assert rr.circuit_breaker is not None
            # The breaker should be separate (fresh instance)
            assert rr.circuit_breaker.threshold == 3

    def test_uses_default_model(self):
        with patch.dict("os.environ", {"NIM_ENABLED": "true"}):
            rr = build_reranker()
            assert rr.model == NIM_RERANK_MODEL


# ── get_rerank_candidate_k ─────────────────────────────────────────────


class TestGetRerankCandidateK:
    def test_default_is_20(self):
        import os
        os.environ.pop("NIM_RERANK_CANDIDATE_K", None)
        assert get_rerank_candidate_k() == DEFAULT_RERANK_CANDIDATE_K
        assert DEFAULT_RERANK_CANDIDATE_K == 20

    def test_env_override(self):
        with patch.dict("os.environ", {"NIM_RERANK_CANDIDATE_K": "50"}):
            assert get_rerank_candidate_k() == 50


# ── orchestrator integration ───────────────────────────────────────────


class TestOrchestratorRerankIntegration:
    """Verify the orchestrator inserts the rerank step when configured."""

    def _build_orchestrator(self, docs=None, tokens=None, reranker=None):
        from rag.orchestrator import RAGOrchestrator
        from rag.post_processor import PostProcessResult

        if docs is None:
            docs = _make_docs(5)
        if tokens is None:
            tokens = ["answer"]

        retriever = MagicMock()
        retriever.retrieve.return_value = docs

        assembler = MagicMock()
        assembler.assemble.return_value = "context"

        generator = MagicMock()
        generator.stream.return_value = iter(tokens)

        pp = MagicMock()
        pp.post_process.return_value = PostProcessResult(answer="".join(tokens))

        rewriter = MagicMock()
        rewriter.rewrite.return_value = "rewritten"

        return RAGOrchestrator(
            retriever, assembler, generator, pp, rewriter, reranker=reranker,
        ), {
            "retriever": retriever,
            "assembler": assembler,
            "generator": generator,
            "pp": pp,
        }

    def test_reranker_called_when_configured(self):
        reranker = MagicMock()
        reranked = _make_docs(3)
        reranker.rerank.return_value = reranked
        orch, collab = self._build_orchestrator(docs=_make_docs(20), reranker=reranker)

        list(orch.stream_answer("query"))

        reranker.rerank.assert_called_once()
        # The reranker receives the rewritten query + retrieved docs
        call_args = reranker.rerank.call_args
        assert call_args[0][0] == "rewritten"
        assert len(call_args[0][1]) == 20

    def test_reranked_docs_passed_to_assembler(self):
        reranker = MagicMock()
        reranked = _make_docs(3)
        reranker.rerank.return_value = reranked
        orch, collab = self._build_orchestrator(docs=_make_docs(20), reranker=reranker)

        list(orch.stream_answer("query"))

        # The assembler should receive the reranked docs, not the original 20
        assemble_args = collab["assembler"].assemble.call_args
        assert assemble_args[0][0] is reranked

    def test_reranked_docs_passed_to_post_processor(self):
        reranker = MagicMock()
        reranked = _make_docs(3)
        reranker.rerank.return_value = reranked
        orch, collab = self._build_orchestrator(docs=_make_docs(20), reranker=reranker)

        list(orch.stream_answer("query"))

        pp_args = collab["pp"].post_process.call_args
        # post_process(answer, docs) — the docs should be the reranked ones
        assert pp_args[0][1] is reranked

    def test_no_reranker_skips_rerank(self):
        orch, collab = self._build_orchestrator(docs=_make_docs(5), reranker=None)
        list(orch.stream_answer("query"))
        # Assembler receives the original docs directly
        assemble_args = collab["assembler"].assemble.call_args
        assert len(assemble_args[0][0]) == 5

    def test_reranker_called_in_answer_method_too(self):
        reranker = MagicMock()
        reranked = _make_docs(3)
        reranker.rerank.return_value = reranked
        orch, _ = self._build_orchestrator(docs=_make_docs(20), reranker=reranker)

        orch.answer("query")

        reranker.rerank.assert_called_once()

    def test_reranker_failure_does_not_break_orchestrator(self):
        """If the reranker raises (it shouldn't), the orchestrator should
        still work — but NIMReranker.rerank never raises, so this tests
        that a mock reranker that returns degraded docs works fine."""
        reranker = MagicMock()
        # Simulate graceful degradation: reranker returns original[:5]
        original = _make_docs(20)
        reranker.rerank.return_value = original[:5]
        orch, collab = self._build_orchestrator(docs=original, reranker=reranker)

        items = list(orch.stream_answer("query"))
        # Should still get tokens + a result
        assert any(isinstance(i, str) for i in items)

    def test_rerank_span_traced(self):
        """When a tracer is configured, a 'rerank' span is emitted."""
        from rag.orchestrator import RAGOrchestrator
        from rag.post_processor import PostProcessResult

        retriever = MagicMock()
        retriever.retrieve.return_value = _make_docs(20)
        assembler = MagicMock()
        assembler.assemble.return_value = "ctx"
        generator = MagicMock()
        generator.stream.return_value = iter(["t"])
        pp = MagicMock()
        pp.post_process.return_value = PostProcessResult(answer="t")
        rewriter = MagicMock()
        rewriter.rewrite.return_value = "rw"
        tracer = MagicMock()
        trace = MagicMock()
        tracer.start_trace.return_value = trace
        # Give each span a unique mock so we can check end_span calls
        span_objs = {}
        def fake_start_span(t, name, metadata=None):
            span_objs[name] = MagicMock()
            return span_objs[name]
        tracer.start_span.side_effect = fake_start_span
        reranker = MagicMock()
        reranker.rerank.return_value = _make_docs(3)

        orch = RAGOrchestrator(
            retriever, assembler, generator, pp, rewriter,
            tracer=tracer, reranker=reranker,
        )
        list(orch.stream_answer("query"))

        # A span named "rerank" should have been started and ended
        assert "rerank" in span_objs
        tracer.end_span.assert_any_call(
            span_objs["rerank"], metadata={"reranked": 3},
        )
