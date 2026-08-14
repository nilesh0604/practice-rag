"""Tests for the embedder (ingestion/embedder.py).

Dense embedding tests mock the Ollama client. Sparse embedding tests
exercise the pure-Python tokenizer / hashing trick directly (no mocks).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import SparseVector

from ingestion.embedder import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    SPARSE_MAX_DIM,
    Embedder,
    sparse_embed_batch,
    sparse_embed_text,
    tokenize,
)


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("How to declare path parameters in FastAPI")
        assert "declare" in tokens
        assert "path" in tokens
        assert "parameters" in tokens
        assert "fastapi" in tokens

    def test_lowercase(self):
        tokens = tokenize("FastAPI Pydantic SQLModel")
        assert "fastapi" in tokens
        assert "pydantic" in tokens
        assert "sqlmodel" in tokens

    def test_stops_words_removed(self):
        tokens = tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens

    def test_hyphenated_tokens(self):
        tokens = tokenize("base-model query-params")
        assert "base-model" in tokens
        assert "query-params" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_stop_words(self):
        assert tokenize("the a an is at of") == []

    def test_code_identifiers(self):
        tokens = tokenize("BaseModel FastAPI Depends")
        assert "basemodel" in tokens
        assert "depends" in tokens


class TestSparseEmbedText:
    def test_returns_sparse_vector(self):
        sv = sparse_embed_text("FastAPI path parameters")
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) == len(sv.values)
        assert len(sv.indices) > 0

    def test_indices_within_bounds(self):
        sv = sparse_embed_text("some text content here")
        for idx in sv.indices:
            assert 0 <= idx < SPARSE_MAX_DIM

    def test_deterministic(self):
        sv1 = sparse_embed_text("FastAPI query parameters")
        sv2 = sparse_embed_text("FastAPI query parameters")
        assert sv1.indices == sv2.indices
        assert sv1.values == sv2.values

    def test_different_text_different_vectors(self):
        sv1 = sparse_embed_text("FastAPI path parameters")
        sv2 = sparse_embed_text("Pydantic model validation")
        assert sv1.indices != sv2.indices

    def test_empty_text_returns_empty_vector(self):
        sv = sparse_embed_text("")
        assert sv.indices == []
        assert sv.values == []

    def test_term_frequency_weighting(self):
        # "fastapi" appears 3 times → weight = 1 + log(3)
        import math

        sv = sparse_embed_text("fastapi fastapi fastapi")
        assert len(sv.values) == 1
        assert sv.values[0] == pytest.approx(1 + math.log(3))

    def test_indices_sorted(self):
        sv = sparse_embed_text("alpha beta gamma delta epsilon")
        assert sv.indices == sorted(sv.indices)

    def test_shared_token_same_index(self):
        # The same word in two different texts must hash to the same index.
        sv1 = sparse_embed_text("fastapi rules")
        sv2 = sparse_embed_text("i love fastapi")
        # Find the index for "fastapi" in both
        idx1 = sv1.indices
        idx2 = sv2.indices
        shared = set(idx1) & set(idx2)
        assert len(shared) >= 1  # "fastapi" should share an index


class TestSparseEmbedBatch:
    def test_batch_returns_list(self):
        result = sparse_embed_batch(["hello world", "fastapi docs"])
        assert len(result) == 2
        assert all(isinstance(sv, SparseVector) for sv in result)

    def test_empty_batch(self):
        assert sparse_embed_batch([]) == []


class TestEmbedderDense:
    def _mock_response(self, n_texts: int, dim: int = EMBEDDING_DIM):
        resp = MagicMock()
        resp.embeddings = [[0.01 * (i + 1)] * dim for i in range(n_texts)]
        return resp

    def test_embed_single_text(self):
        emb = Embedder()
        mock_client = MagicMock()
        mock_client.embed.return_value = self._mock_response(1)
        emb._client = mock_client

        vec = emb.embed_text("hello")
        assert len(vec) == EMBEDDING_DIM
        mock_client.embed.assert_called_once()
        call = mock_client.embed.call_args
        assert call.kwargs["model"] == EMBEDDING_MODEL

    def test_embed_batch(self):
        emb = Embedder()
        mock_client = MagicMock()
        mock_client.embed.return_value = self._mock_response(3)
        emb._client = mock_client

        vecs = emb.embed_texts(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == EMBEDDING_DIM for v in vecs)
        mock_client.embed.assert_called_once()
        call = mock_client.embed.call_args
        assert call.kwargs["input"] == ["a", "b", "c"]

    def test_embed_empty_list(self):
        emb = Embedder()
        assert emb.embed_texts([]) == []

    def test_dimension_mismatch_raises(self):
        emb = Embedder()
        mock_client = MagicMock()
        bad_resp = MagicMock()
        bad_resp.embeddings = [[0.1] * 128]  # wrong dim
        mock_client.embed.return_value = bad_resp
        emb._client = mock_client

        with pytest.raises(ValueError, match="dimension mismatch"):
            emb.embed_text("hello")

    def test_lazy_client_construction(self):
        emb = Embedder()
        assert emb._client is None
        with patch("ollama.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            _ = emb.client
            MockClient.assert_called_once_with(host="http://localhost:11434")
        assert emb._client is mock_instance

    def test_close_resets_client(self):
        emb = Embedder()
        emb._client = MagicMock()
        emb.close()
        assert emb._client is None
