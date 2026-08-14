"""Tests for ``GET /api/v1/health`` (api/routes/health.py)."""

from __future__ import annotations


class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.json()["status"] == "ok"

    def test_has_service_and_version(self, client):
        body = client.get("/api/v1/health").json()
        assert body["service"] == "practice-rag-api"
        assert "version" in body
