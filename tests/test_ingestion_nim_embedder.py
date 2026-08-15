"""Tests for the NIM embedder + NIM ingestion path (ingestion/nim_embedder.py).

The httpx client is mocked so no network call is made. The NIM collection
config + ensure helper tests mock the Qdrant client. The ``run_nim_ingestion``
tests mock the underlying ``run_ingestion`` + ``ensure_nim_collection``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestion.nim_embedder import (
    INPUT_TYPE_PASSAGE,
    INPUT_TYPE_QUERY,
    NIM_BASE_URL,
    NIMEmbedder,
    NIMEmbeddingError,
    build_nim_embedder,
    build_nim_retriever,
    run_nim_ingestion,
)
from rag.qdrant_collection import (
    COLLECTION_NAME,
    NIM_COLLECTION_NAME,
    NIM_VECTOR_SIZE,
    SPARSE_VECTOR_NAME,
    VECTOR_NAME,
    build_nim_collection_config,
)


# ── helpers ────────────────────────────────────────────────────────────


def _mock_embeddings_response(n_texts: int, dim: int = NIM_VECTOR_SIZE):
    """Build a fake OpenAI-shaped /embeddings response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": [
            {"embedding": [0.01 * (i + 1)] * dim, "index": i}
            for i in range(n_texts)
        ]
    }
    return resp


# ── NIMEmbedder: request shape ─────────────────────────────────────────


class TestNIMEmbedderRequest:
    def test_embed_single_text_passage(self):
        emb = NIMEmbedder(api_key="test-key")
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_embeddings_response(1)
        emb._client = mock_client

        vec = emb.embed_text("hello", input_type=INPUT_TYPE_PASSAGE)
        assert len(vec) == NIM_VECTOR_SIZE
        mock_client.post.assert_called_once()
        url = mock_client.post.call_args.args[0]
        assert url == f"{NIM_BASE_URL}/embeddings"

    def test_embed_single_text_query_default(self):
        emb = NIMEmbedder(api_key="test-key")
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_embeddings_response(1)
        emb._client = mock_client

        emb.embed_text("hello")
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["input_type"] == INPUT_TYPE_QUERY

    def test_embed_batch_passage(self):
        emb = NIMEmbedder(api_key="test-key")
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_embeddings_response(3)
        emb._client = mock_client

        vecs = emb.embed_texts(["a", "b", "c"], input_type=INPUT_TYPE_PASSAGE)
        assert len(vecs) == 3
        assert all(len(v) == NIM_VECTOR_SIZE for v in vecs)
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["input"] == ["a", "b", "c"]
        assert payload["input_type"] == INPUT_TYPE_PASSAGE
        assert payload["model"] == "nvidia/llama-nemotron-embed-1b-v2"

    def test_auth_header_sent(self):
        emb = NIMEmbedder(api_key="nvapi-secret")
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_embeddings_response(1)
        emb._client = mock_client

        emb.embed_text("hello")
        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer nvapi-secret"
        assert headers["Content-Type"] == "application/json"

    def test_empty_list_returns_empty(self):
        emb = NIMEmbedder(api_key="test-key")
        mock_client = MagicMock()
        emb._client = mock_client
        assert emb.embed_texts([]) == []
        mock_client.post.assert_not_called()

    def test_base_url_trailing_slash_stripped(self):
        emb = NIMEmbedder(api_key="k", base_url="https://integrate.api.nvidia.com/v1/")
        assert emb.base_url == "https://integrate.api.nvidia.com/v1"


# ── NIMEmbedder: dimensions / Matryoshka ───────────────────────────────


class TestNIMEmbedderDimensions:
    def test_default_dimensions_is_native_2048(self):
        emb = NIMEmbedder(api_key="k")
        assert emb.dimensions == NIM_VECTOR_SIZE

    def test_dimensions_param_overrides(self):
        emb = NIMEmbedder(api_key="k", dimensions=768)
        assert emb.dimensions == 768

    def test_dimensions_env_override(self, monkeypatch):
        monkeypatch.setenv("NIM_EMBEDDING_DIM", "1024")
        emb = NIMEmbedder(api_key="k")
        assert emb.dimensions == 1024

    def test_reduced_dimensions_sent_in_payload(self):
        emb = NIMEmbedder(api_key="k", dimensions=768)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_embeddings_response(1, dim=768)
        emb._client = mock_client

        emb.embed_text("hello")
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["dimensions"] == 768

    def test_native_dimensions_omitted_from_payload(self):
        emb = NIMEmbedder(api_key="k")  # 2048 native
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_embeddings_response(1)
        emb._client = mock_client

        emb.embed_text("hello")
        payload = mock_client.post.call_args.kwargs["json"]
        assert "dimensions" not in payload

    def test_dimension_mismatch_raises(self):
        emb = NIMEmbedder(api_key="k", dimensions=2048)
        mock_client = MagicMock()
        # Server returns 128-d despite expecting 2048
        mock_client.post.return_value = _mock_embeddings_response(1, dim=128)
        emb._client = mock_client

        with pytest.raises(NIMEmbeddingError, match="dimension mismatch"):
            emb.embed_text("hello")


# ── NIMEmbedder: error mapping ─────────────────────────────────────────


class TestNIMEmbedderErrors:
    def _emb_with_status(self, status_code: int, body: str = "err"):
        emb = NIMEmbedder(api_key="k")
        mock_client = MagicMock()
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = body
        mock_client.post.return_value = resp
        emb._client = mock_client
        return emb

    def test_429_rate_limited(self):
        emb = self._emb_with_status(429)
        with pytest.raises(NIMEmbeddingError, match="rate limited"):
            emb.embed_text("hello")

    def test_404_model_not_found(self):
        emb = self._emb_with_status(404)
        with pytest.raises(NIMEmbeddingError, match="not found"):
            emb.embed_text("hello")

    def test_500_generic_http_error(self):
        emb = self._emb_with_status(500, "internal error")
        with pytest.raises(NIMEmbeddingError, match="HTTP 500"):
            emb.embed_text("hello")

    def test_empty_data_raises(self):
        emb = NIMEmbedder(api_key="k")
        mock_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": []}
        mock_client.post.return_value = resp
        emb._client = mock_client
        with pytest.raises(NIMEmbeddingError, match="0 vectors"):
            emb.embed_text("hello")

    def test_count_mismatch_raises(self):
        emb = NIMEmbedder(api_key="k")
        mock_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        # Asked for 3, got 1
        resp.json.return_value = {"data": [{"embedding": [0.1] * NIM_VECTOR_SIZE}]}
        mock_client.post.return_value = resp
        emb._client = mock_client
        with pytest.raises(NIMEmbeddingError, match="1 vectors for 3 inputs"):
            emb.embed_texts(["a", "b", "c"])

    def test_timeout_mapped(self):
        import httpx

        emb = NIMEmbedder(api_key="k")
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        emb._client = mock_client
        with pytest.raises(NIMEmbeddingError, match="timed out"):
            emb.embed_text("hello")

    def test_connect_error_mapped(self):
        import httpx

        emb = NIMEmbedder(api_key="k")
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("no route")
        emb._client = mock_client
        with pytest.raises(NIMEmbeddingError, match="connection error"):
            emb.embed_text("hello")

    def test_unexpected_error_mapped(self):
        emb = NIMEmbedder(api_key="k")
        mock_client = MagicMock()
        mock_client.post.side_effect = RuntimeError("boom")
        emb._client = mock_client
        with pytest.raises(NIMEmbeddingError, match="unexpected error"):
            emb.embed_text("hello")


# ── NIMEmbedder: circuit breaker ───────────────────────────────────────


class TestNIMEmbedderCircuitBreaker:
    def test_circuit_open_skips_network(self):
        from api.observability import CircuitBreaker, CircuitOpenError

        breaker = CircuitBreaker(threshold=1, timeout=30.0)
        breaker.record_failure()  # open immediately
        emb = NIMEmbedder(api_key="k", circuit_breaker=breaker)
        mock_client = MagicMock()
        emb._client = mock_client

        with pytest.raises(CircuitOpenError):
            emb.embed_text("hello")
        mock_client.post.assert_not_called()

    def test_failure_recorded_on_error(self):
        from api.observability import CircuitBreaker

        breaker = CircuitBreaker(threshold=3, timeout=30.0)
        emb = NIMEmbedder(api_key="k", circuit_breaker=breaker)
        mock_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 429
        mock_client.post.return_value = resp
        emb._client = mock_client

        with pytest.raises(NIMEmbeddingError):
            emb.embed_text("hello")
        assert breaker._failures == 1

    def test_success_resets_breaker(self):
        from api.observability import CircuitBreaker

        breaker = CircuitBreaker(threshold=3, timeout=30.0)
        breaker.record_failure()
        emb = NIMEmbedder(api_key="k", circuit_breaker=breaker)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_embeddings_response(1)
        emb._client = mock_client

        emb.embed_text("hello")
        assert breaker._failures == 0

    def test_no_breaker_works_directly(self):
        emb = NIMEmbedder(api_key="k", circuit_breaker=None)
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_embeddings_response(1)
        emb._client = mock_client
        vec = emb.embed_text("hello")
        assert len(vec) == NIM_VECTOR_SIZE


# ── NIMEmbedder: lazy client + close ───────────────────────────────────


class TestNIMEmbedderLifecycle:
    def test_lazy_client_construction(self):
        emb = NIMEmbedder(api_key="k")
        assert emb._client is None
        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            _ = emb.client
            MockClient.assert_called_once()
        assert emb._client is mock_instance

    def test_close_resets_client(self):
        emb = NIMEmbedder(api_key="k")
        emb._client = MagicMock()
        emb.close()
        assert emb._client is None

    def test_close_is_safe_when_no_client(self):
        emb = NIMEmbedder(api_key="k")
        emb.close()  # no error
        assert emb._client is None


# ── build_nim_embedder factory ─────────────────────────────────────────


class TestBuildNimEmbedder:
    def test_default_model(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-key")
        emb = build_nim_embedder()
        assert emb.model == "nvidia/llama-nemotron-embed-1b-v2"
        assert emb.api_key == "nvapi-key"

    def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("NIM_EMBEDDING_MODEL", "baai/bge-m3")
        emb = build_nim_embedder()
        assert emb.model == "baai/bge-m3"

    def test_dimensions_env_override(self, monkeypatch):
        monkeypatch.setenv("NIM_EMBEDDING_DIM", "768")
        emb = build_nim_embedder()
        assert emb.dimensions == 768

    def test_circuit_breaker_attached(self):
        from api.observability import CircuitBreaker

        breaker = CircuitBreaker(threshold=3, timeout=30.0)
        emb = build_nim_embedder(circuit_breaker=breaker)
        assert emb.circuit_breaker is breaker


# ── NIM collection config ──────────────────────────────────────────────


class TestNimCollectionConfig:
    def test_default_config_uses_nim_collection_name(self):
        cfg = build_nim_collection_config()
        assert cfg.collection_name == NIM_COLLECTION_NAME
        assert cfg.collection_name != COLLECTION_NAME

    def test_default_config_uses_nim_vector_size(self):
        cfg = build_nim_collection_config()
        assert cfg.vector_size == NIM_VECTOR_SIZE
        assert cfg.vector_size != 768

    def test_reuses_same_vector_names_as_default(self):
        cfg = build_nim_collection_config()
        # Same named vectors so IndexWriter/HybridRetriever work unchanged.
        assert cfg.vector_name == VECTOR_NAME
        assert cfg.sparse_vector_name == SPARSE_VECTOR_NAME

    def test_collection_name_override(self):
        cfg = build_nim_collection_config(collection_name="custom_nim")
        assert cfg.collection_name == "custom_nim"

    def test_vector_size_override(self):
        cfg = build_nim_collection_config(vector_size=768)
        assert cfg.vector_size == 768

    def test_vector_size_env_override(self, monkeypatch):
        monkeypatch.setenv("NIM_EMBEDDING_DIM", "1024")
        cfg = build_nim_collection_config()
        assert cfg.vector_size == 1024

    def test_explicit_vector_size_beats_env(self, monkeypatch):
        monkeypatch.setenv("NIM_EMBEDDING_DIM", "1024")
        cfg = build_nim_collection_config(vector_size=512)
        assert cfg.vector_size == 512


# ── ensure_nim_collection ──────────────────────────────────────────────


class TestEnsureNimCollection:
    def test_creates_nim_collection_if_missing(self):
        from rag.qdrant_collection import ensure_nim_collection as _ensure

        mock_client = MagicMock()
        # No collections exist yet.
        mock_client.get_collections.return_value = MagicMock(collections=[])
        with patch(
            "rag.qdrant_collection.get_qdrant_client", return_value=mock_client
        ):
            client, cfg = _ensure()
        assert cfg.collection_name == NIM_COLLECTION_NAME
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == NIM_COLLECTION_NAME

    def test_skips_creation_if_exists(self):
        mock_client = MagicMock()
        existing = MagicMock()
        existing.name = NIM_COLLECTION_NAME
        mock_client.get_collections.return_value = MagicMock(
            collections=[existing]
        )
        from rag.qdrant_collection import ensure_nim_collection as _ensure

        with patch(
            "rag.qdrant_collection.get_qdrant_client", return_value=mock_client
        ):
            _ensure()
        mock_client.create_collection.assert_not_called()

    def test_does_not_touch_default_collection(self):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        from rag.qdrant_collection import ensure_nim_collection as _ensure

        with patch(
            "rag.qdrant_collection.get_qdrant_client", return_value=mock_client
        ):
            _ensure()
        # Only the NIM collection should be created, never docs-knowledge.
        if mock_client.create_collection.called:
            name = mock_client.create_collection.call_args.kwargs[
                "collection_name"
            ]
            assert name == NIM_COLLECTION_NAME
            assert name != COLLECTION_NAME


# ── build_nim_retriever ────────────────────────────────────────────────


class TestBuildNimRetriever:
    def test_returns_hybrid_retriever_pointed_at_nim_collection(self):
        from rag.retriever import HybridRetriever

        mock_client = MagicMock()
        with patch(
            "rag.qdrant_collection.get_qdrant_client", return_value=mock_client
        ), patch(
            "ingestion.nim_embedder.build_nim_embedder"
        ) as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb_fn.return_value = mock_emb
            retriever = build_nim_retriever()
        assert isinstance(retriever, HybridRetriever)
        assert retriever.collection_name == NIM_COLLECTION_NAME
        assert retriever.embedder is mock_emb

    def test_uses_nim_vector_names(self):
        mock_client = MagicMock()
        with patch(
            "rag.qdrant_collection.get_qdrant_client", return_value=mock_client
        ), patch(
            "ingestion.nim_embedder.build_nim_embedder",
            return_value=MagicMock(),
        ):
            retriever = build_nim_retriever()
        assert retriever.vector_name == VECTOR_NAME
        assert retriever.sparse_vector_name == SPARSE_VECTOR_NAME


# ── run_nim_ingestion ──────────────────────────────────────────────────


class TestRunNimIngestion:
    def test_refuses_without_nim_enabled(self, monkeypatch):
        monkeypatch.delenv("NIM_ENABLED", raising=False)
        with pytest.raises(RuntimeError, match="NIM_ENABLED=true"):
            run_nim_ingestion("data/corpus")

    def test_runs_with_nim_enabled(self, monkeypatch):
        monkeypatch.setenv("NIM_ENABLED", "true")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-key")

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        expected_summary = {"files_total": 1, "files_indexed": 1}

        with patch(
            "rag.qdrant_collection.get_qdrant_client", return_value=mock_client
        ), patch(
            "ingestion.run.run_ingestion", return_value=expected_summary
        ) as mock_run, patch(
            "ingestion.nim_embedder.build_nim_embedder"
        ) as mock_emb_fn:
            mock_emb_fn.return_value = MagicMock(model="m", dimensions=2048)
            result = run_nim_ingestion("data/corpus")

        assert result == expected_summary
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        # collection_config must be the NIM config
        assert kwargs["collection_config"].collection_name == NIM_COLLECTION_NAME
        # embedder + index_writer injected
        assert "embedder" in kwargs
        assert "index_writer" in kwargs

    def test_full_reindex_passed_through(self, monkeypatch):
        monkeypatch.setenv("NIM_ENABLED", "true")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-key")

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])

        with patch(
            "rag.qdrant_collection.get_qdrant_client", return_value=mock_client
        ), patch(
            "ingestion.run.run_ingestion", return_value={}
        ) as mock_run, patch(
            "ingestion.nim_embedder.build_nim_embedder",
            return_value=MagicMock(model="m", dimensions=2048),
        ):
            run_nim_ingestion("data/corpus", full_reindex=True)

        assert mock_run.call_args.kwargs["full_reindex"] is True

    def test_injected_embedder_used(self, monkeypatch):
        monkeypatch.setenv("NIM_ENABLED", "true")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-key")

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        custom_emb = MagicMock(model="custom", dimensions=2048)

        with patch(
            "rag.qdrant_collection.get_qdrant_client", return_value=mock_client
        ), patch(
            "ingestion.run.run_ingestion", return_value={}
        ) as mock_run:
            run_nim_ingestion("data/corpus", embedder=custom_emb)

        assert mock_run.call_args.kwargs["embedder"] is custom_emb
