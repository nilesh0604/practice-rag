"""Tests for ``GET /api/v1/health/ready`` readiness endpoint (api/routes/health.py)."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


class TestReadinessEndpoint:
    def test_returns_200_when_both_deps_up(self):
        with patch("api.routes.health.check_ollama", return_value=True), \
             patch("api.routes.health.check_qdrant", return_value=True):
            resp = _client().get("/api/v1/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["dependencies"]["ollama"]["ok"] is True
        assert body["dependencies"]["qdrant"]["ok"] is True

    def test_returns_503_when_ollama_down(self):
        with patch("api.routes.health.check_ollama", return_value=False), \
             patch("api.routes.health.check_qdrant", return_value=True):
            resp = _client().get("/api/v1/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["dependencies"]["ollama"]["ok"] is False
        assert body["dependencies"]["qdrant"]["ok"] is True

    def test_returns_503_when_qdrant_down(self):
        with patch("api.routes.health.check_ollama", return_value=True), \
             patch("api.routes.health.check_qdrant", return_value=False):
            resp = _client().get("/api/v1/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["dependencies"]["qdrant"]["ok"] is False

    def test_returns_503_when_both_down(self):
        with patch("api.routes.health.check_ollama", return_value=False), \
             patch("api.routes.health.check_qdrant", return_value=False):
            resp = _client().get("/api/v1/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"

    def test_includes_service_version(self):
        with patch("api.routes.health.check_ollama", return_value=True), \
             patch("api.routes.health.check_qdrant", return_value=True):
            body = _client().get("/api/v1/health/ready").json()
        assert "version" in body

    def test_includes_dependency_urls(self):
        with patch("api.routes.health.check_ollama", return_value=True), \
             patch("api.routes.health.check_qdrant", return_value=True):
            body = _client().get("/api/v1/health/ready").json()
        assert "url" in body["dependencies"]["ollama"]
        assert "url" in body["dependencies"]["qdrant"]


class TestLivenessEndpoint:
    def test_health_returns_200(self):
        resp = _client().get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "practice-rag-api"

    def test_health_includes_version(self):
        body = _client().get("/api/v1/health").json()
        assert "version" in body
