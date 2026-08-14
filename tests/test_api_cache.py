"""Tests for the in-memory LRU response cache (api/cache.py)."""

from __future__ import annotations

from schemas.chat import ChatResponse, Citation


def _make_response(answer="answer", session_id="sess_1", confidence=0.9):
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        citations=[Citation(title="FastAPI Docs", source_url="https://example.com/x")],
        confidence=confidence,
    )


class TestNormalizeQuery:
    def test_lowercases(self):
        from api.cache import normalize_query

        assert normalize_query("How Do I") == "how do i"

    def test_strips_leading_trailing(self):
        from api.cache import normalize_query

        assert normalize_query("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        from api.cache import normalize_query

        assert normalize_query("how   do   i") == "how do i"

    def test_empty_returns_empty(self):
        from api.cache import normalize_query

        assert normalize_query("") == ""
        assert normalize_query("   ") == ""


class TestLRUCacheGetPut:
    def test_miss_returns_none(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=4)
        assert cache.get("missing") is None

    def test_put_then_get(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=4)
        resp = _make_response()
        cache.put("How do I?", resp)
        assert cache.get("How do I?") is resp

    def test_normalized_key_match(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=4)
        cache.put("  How   DO i?  ", _make_response())
        # Different casing/whitespace normalizes to the same key.
        assert cache.get("how do i?") is not None

    def test_empty_key_not_stored(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=4)
        cache.put("", _make_response())
        assert cache.get("") is None
        assert len(cache) == 0

    def test_overwrite_existing_key(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=4)
        cache.put("q", _make_response(answer="old"))
        cache.put("q", _make_response(answer="new"))
        assert cache.get("q").answer == "new"
        assert len(cache) == 1


class TestLRUEviction:
    def test_evicts_lru_when_over_capacity(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=2)
        cache.put("a", _make_response(answer="a"))
        cache.put("b", _make_response(answer="b"))
        # Access "a" so "b" becomes LRU.
        cache.get("a")
        cache.put("c", _make_response(answer="c"))
        assert cache.get("a") is not None  # recently used, kept
        assert cache.get("b") is None  # evicted
        assert cache.get("c") is not None

    def test_size_never_exceeds_max(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=3)
        for i in range(10):
            cache.put(f"q{i}", _make_response(answer=str(i)))
        assert len(cache) == 3

    def test_max_size_one(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=1)
        cache.put("a", _make_response())
        cache.put("b", _make_response())
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_invalid_max_size_raises(self):
        from api.cache import LRUCache

        import pytest

        with pytest.raises(ValueError):
            LRUCache(max_size=0)


class TestLRUCacheStats:
    def test_hits_and_misses(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=4)
        cache.put("q", _make_response())
        cache.get("q")  # hit
        cache.get("q")  # hit
        cache.get("nope")  # miss
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["max_size"] == 4

    def test_clear_resets(self):
        from api.cache import LRUCache

        cache = LRUCache(max_size=4)
        cache.put("q", _make_response())
        cache.get("q")
        cache.clear()
        assert len(cache) == 0
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
