"""Tests for ``GET /api/v1/metrics`` endpoint (api/routes/metrics.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.cache import LRUCache
from api.deps import get_cache, get_metrics
from api.main import create_app
from api.observability import MetricsCollector


def _make_client(metrics=None, cache=None):
    app = create_app()
    mc = metrics or MetricsCollector()
    c = cache or LRUCache(max_size=4)
    app.dependency_overrides[get_metrics] = lambda: mc
    app.dependency_overrides[get_cache] = lambda: c
    return TestClient(app), mc, c


class TestMetricsEndpoint:
    def test_returns_200(self):
        client, _, _ = _make_client()
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_returns_json_with_expected_keys(self):
        client, _, _ = _make_client()
        body = client.get("/api/v1/metrics").json()
        assert "requests" in body
        assert "errors" in body
        assert "cache" in body
        assert "ttft" in body
        assert "cache_store" in body

    def test_reflects_recorded_metrics(self):
        client, mc, _ = _make_client(metrics=MetricsCollector())
        mc.record_request()
        mc.record_request()
        mc.record_error()
        mc.record_cache_hit()
        mc.record_cache_miss()
        mc.record_ttft(0.15)
        body = client.get("/api/v1/metrics").json()
        assert body["requests"] == 2
        assert body["errors"] == 1
        assert body["cache"]["hits"] == 1
        assert body["cache"]["misses"] == 1
        assert body["cache"]["hit_rate"] == 0.5
        assert body["ttft"]["count"] == 1
        assert body["ttft"]["mean_s"] == 0.15

    def test_includes_cache_store_stats(self):
        cache = LRUCache(max_size=4)
        client, _, _ = _make_client(cache=cache)
        body = client.get("/api/v1/metrics").json()
        assert body["cache_store"]["max_size"] == 4

    def test_empty_metrics_snapshot(self):
        client, _, _ = _make_client()
        body = client.get("/api/v1/metrics").json()
        assert body["requests"] == 0
        assert body["ttft"]["count"] == 0
        assert body["cache"]["hit_rate"] == 0.0
