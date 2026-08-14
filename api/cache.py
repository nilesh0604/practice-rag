"""In-memory LRU response cache — Tier 1 exact query match.

Implements the architecture doc's Caching Layer > Tier 1: an in-memory
``functools.lru_cache``-style store keyed by a *normalized* query string.
On a cache hit the chat endpoint replays the cached answer as a single SSE
token plus the stored ``ChatResponse`` metadata, skipping retrieval +
generation entirely.

Normalization (``normalize_query``) lowercases, strips, and collapses
internal whitespace so ``"  How   do I ... "`` and ``"how do i ..."`` share
one entry. Conversation history is **not** part of the key — this is the
doc's "exact query match" tier; a semantic cache (Tier 2) would consider
embeddings + context, but that is out of scope for the single-user practice
path and is documented as a known simplification.

The cache stores ``ChatResponse`` objects. On replay the caller rebuilds a
``ChatResponse`` with the *current* session id (the cached one belongs to
the original session), so ``session_id`` is never leaked across sessions.

Thread-safe via a ``threading.Lock`` (FastAPI runs sync endpoints in a
threadpool). Bounded by ``max_size`` with LRU eviction.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict

from schemas.chat import ChatResponse

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE: int = 256
"""Default cache capacity. Plenty for a single-user practice project."""

_WS_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Normalize a query for use as a cache key.

    Lowercases, strips leading/trailing whitespace, and collapses internal
    whitespace runs to a single space. An empty/whitespace-only query
    normalizes to the empty string (which the cache treats as a miss).
    """
    if not query:
        return ""
    return _WS_RE.sub(" ", query.strip()).lower()


class LRUCache:
    """A bounded, thread-safe LRU cache mapping normalized query → ChatResponse.

    Uses an ``OrderedDict`` so move-to-end on access gives true LRU ordering
    and ``popitem(last=False)`` evicts the least-recently-used entry.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._store: OrderedDict[str, ChatResponse] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ── public API ─────────────────────────────────────────────────────

    def get(self, key: str) -> ChatResponse | None:
        """Return the cached response for a *raw* query key, or None on miss.

        The key is normalized internally. Accessing an entry marks it as
        most-recently-used.
        """
        norm = normalize_query(key)
        if not norm:
            return None
        with self._lock:
            response = self._store.get(norm)
            if response is None:
                self._misses += 1
                logger.debug("Cache miss for %r", key)
                return None
            self._store.move_to_end(norm)
            self._hits += 1
            logger.debug("Cache hit for %r", key)
            return response

    def put(self, key: str, response: ChatResponse) -> None:
        """Store a response under the normalized query key.

        If the key already exists its value is updated and marked
        most-recently-used. Evicts the LRU entry when over capacity.
        """
        norm = normalize_query(key)
        if not norm:
            return
        with self._lock:
            if norm in self._store:
                self._store[norm] = response
                self._store.move_to_end(norm)
                return
            self._store[norm] = response
            while len(self._store) > self.max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("Cache evicted LRU entry %r", evicted_key)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    # ── introspection ──────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> dict[str, int]:
        """Return hit/miss/size counters for observability (cache hit rate)."""
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
            }
