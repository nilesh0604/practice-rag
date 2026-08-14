"""Tests for ``POST /api/v1/ingest`` (api/routes/ingest.py).

The ingestion pipeline itself is tested in ``test_ingestion_run.py``; here
we only verify the HTTP wrapper: that it delegates to ``run_ingestion``
with the right args and surfaces the summary or a 500 on failure.
"""

from __future__ import annotations

from unittest.mock import patch


_SUMMARY = {
    "files_total": 7,
    "files_indexed": 1,
    "files_skipped": 6,
    "chunks_upserted": 6,
    "chunks_deleted_docs": 0,
}


class TestIngestEndpoint:
    def test_incremental_default(self, client):
        with patch("ingestion.run.run_ingestion", return_value=_SUMMARY) as mock_run:
            resp = client.post("/api/v1/ingest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["summary"]["files_total"] == 7
        mock_run.assert_called_once_with("data/corpus", full_reindex=False)

    def test_full_reindex_flag(self, client):
        with patch("ingestion.run.run_ingestion", return_value=_SUMMARY) as mock_run:
            resp = client.post("/api/v1/ingest", json={"full_reindex": True})
        assert resp.status_code == 200
        mock_run.assert_called_once_with("data/corpus", full_reindex=True)

    def test_custom_corpus_dir(self, client):
        with patch("ingestion.run.run_ingestion", return_value=_SUMMARY) as mock_run:
            resp = client.post(
                "/api/v1/ingest", json={"corpus_dir": "data/other"}
            )
        assert resp.status_code == 200
        mock_run.assert_called_once_with("data/other", full_reindex=False)

    def test_pipeline_failure_returns_500(self, client):
        with patch(
            "ingestion.run.run_ingestion",
            side_effect=RuntimeError("qdrant down"),
        ):
            resp = client.post("/api/v1/ingest")
        assert resp.status_code == 500
        assert "qdrant down" in resp.json()["detail"]
