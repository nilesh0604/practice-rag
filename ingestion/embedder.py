"""Embedder — generates dense and sparse vector representations for chunks.

**Dense** — Ollama ``nomic-embed-text`` (768-d). Batched via the Ollama
Python client's ``embed`` call which accepts a sequence of strings.

**Sparse** — a lightweight, dependency-free BM25-style sparse vector
using a hashing trick. Each token is lowercased, hashed to a deterministic
integer index via ``zlib.crc32`` modulo ``SPARSE_MAX_DIM``, and weighted
by its term frequency (``1 + log(tf)``). Qdrant's ``Modifier.IDF``
(collection-level) then applies inverse-document-frequency weighting at
query time, producing proper BM25-like scoring without a separate SPLADE
model or a global vocabulary.

The sparse tokenizer is shared between ingestion (index-time) and the
retriever (query-time) so the same token → index mapping is used on both
sides. It lives here so the retriever (Step 3) can import
``sparse_embed_text`` directly.
"""

from __future__ import annotations

import logging
import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)

# ── dense embedding constants ────────────────────────────────────────

EMBEDDING_MODEL: str = "nomic-embed-text"
"""Ollama model for dense embeddings (768-d)."""

EMBEDDING_DIM: int = 768
"""Dense embedding dimensionality — must match the Qdrant collection."""

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
"""Default Ollama base URL (host Ollama, not containerized)."""

# ── sparse embedding constants ───────────────────────────────────────

SPARSE_MAX_DIM: int = 1 << 20  # 1,048,576
"""Upper bound on sparse vector indices (hashing-trick space). Large enough
that collisions are negligible for a small documentation corpus."""

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
"""Token pattern: lowercase alphanumeric runs with internal hyphens/apostrophes
(e.g. ``base-model``, ``don't``). Matches code identifiers and prose."""

# Minimal English stop-word set — removes ultra-high-frequency tokens that
# add no discriminative value and would dominate the sparse vector.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "to", "was", "were", "will", "with", "you",
    "your", "i", "we", "they", "but", "not", "can", "do", "if", "so",
})


def tokenize(text: str) -> list[str]:
    """Lowercase, extract word tokens, remove stop words."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOP_WORDS and len(tok) > 1
    ]


def sparse_embed_text(text: str) -> SparseVector:
    """Generate a BM25-style sparse vector from a single text.

    Uses a deterministic hashing trick (``zlib.crc32``) so the same token
    always maps to the same index — no global vocabulary needed. Term
    frequency is log-normalized (``1 + log(tf)``).
    """
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    counts = Counter(tokens)
    indices: list[int] = []
    values: list[float] = []
    for token, tf in counts.items():
        idx = zlib.crc32(token.encode("utf-8")) % SPARSE_MAX_DIM
        indices.append(idx)
        values.append(float(1 + __import__("math").log(tf)))
    # Sort by index for deterministic ordering (Qdrant doesn't require it
    # but it makes tests and debugging easier).
    paired = sorted(zip(indices, values))
    return SparseVector(
        indices=[i for i, _ in paired],
        values=[v for _, v in paired],
    )


def sparse_embed_batch(texts: list[str]) -> list[SparseVector]:
    """Generate sparse vectors for a batch of texts."""
    return [sparse_embed_text(t) for t in texts]


# ── dense embedder ───────────────────────────────────────────────────


class Embedder:
    """Wraps the Ollama embedding client for batched dense embeddings.

    The sparse generation is a pure function (no Ollama call) so it is
    exposed at module level and used directly by the index writer.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self._client = None

    @property
    def client(self):
        """Lazily import and construct the Ollama client (avoids import at
        module load time so unit tests can mock it)."""
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.ollama_url)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts → list of 768-d float vectors.

        Ollama's ``embed`` call accepts a sequence of inputs and returns
        one embedding per input, preserving order.
        """
        if not texts:
            return []
        response = self.client.embed(model=self.model, input=texts)
        embeddings = response.embeddings
        # Sanity-check dimensionality on the first vector.
        if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                f"got {len(embeddings[0])} from model {self.model}"
            )
        return [list(e) for e in embeddings]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text → one 768-d vector."""
        return self.embed_texts([text])[0]

    def close(self) -> None:
        """Release the Ollama client if one was constructed."""
        self._client = None
