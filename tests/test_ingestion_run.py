"""Tests for the ingestion orchestrator (ingestion/run.py).

These are integration-style tests that mock the Ollama embedder client
and the Qdrant client, but exercise the real sync → parse → chunk →
embed → upsert pipeline against a temp corpus directory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.embedder import EMBEDDING_DIM
from ingestion.index_writer import IndexWriter
from ingestion.manifest import load_manifest
from ingestion.run import run_ingestion


def _make_corpus(tmp_path: Path) -> Path:
    """Create a tiny corpus with 2 markdown files."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "---\ntitle: Doc A\nsource_url: https://example.com/a\n---\n"
        "# Topic A\n\nSome content about FastAPI.\n"
    )
    (corpus / "b.md").write_text(
        "---\ntitle: Doc B\nsource_url: https://example.com/b\n---\n"
        "# Topic B\n\nContent about Pydantic models.\n"
    )
    return corpus


def _mock_embedder():
    """An Embedder whose Ollama client returns fixed 768-d vectors."""
    from ingestion.embedder import Embedder

    emb = Embedder()
    mock_client = MagicMock()
    def fake_embed(model, input, **kwargs):
        resp = MagicMock()
        resp.embeddings = [[0.01 * (i + 1)] * EMBEDDING_DIM for i in range(len(input))]
        return resp
    mock_client.embed.side_effect = fake_embed
    emb._client = mock_client
    return emb


def _mock_index_writer(client=None):
    """An IndexWriter backed by a mock QdrantClient."""
    mock_client = client or MagicMock()
    return IndexWriter(mock_client), mock_client


class TestRunIngestion:
    @patch("ingestion.run.ensure_collection")
    def test_full_ingestion_first_run(self, mock_ensure, tmp_path: Path):
        mock_client = MagicMock()
        mock_ensure.return_value = mock_client
        corpus = _make_corpus(tmp_path)
        emb = _mock_embedder()
        writer, _ = _mock_index_writer(mock_client)

        summary = run_ingestion(
            corpus,
            embedder=emb,
            index_writer=writer,
        )

        assert summary["files_total"] == 2
        assert summary["files_indexed"] == 2
        assert summary["files_skipped"] == 0
        assert summary["chunks_upserted"] > 0

    @patch("ingestion.run.ensure_collection")
    def test_incremental_skips_unchanged(self, mock_ensure, tmp_path: Path):
        mock_client = MagicMock()
        mock_ensure.return_value = mock_client
        corpus = _make_corpus(tmp_path)

        emb = _mock_embedder()
        writer, _ = _mock_index_writer(mock_client)

        # First run
        run_ingestion(corpus, embedder=emb, index_writer=writer)
        first_upserts = mock_client.upsert.call_count

        # Second run — nothing changed
        emb2 = _mock_embedder()
        writer2, mock_client2 = _mock_index_writer()
        # Re-patch ensure_collection to return the second mock
        mock_ensure.return_value = mock_client2
        summary = run_ingestion(corpus, embedder=emb2, index_writer=writer2)

        assert summary["files_indexed"] == 0
        assert summary["files_skipped"] == 2
        assert summary["chunks_upserted"] == 0
        assert mock_client2.upsert.call_count == 0

    @patch("ingestion.run.ensure_collection")
    def test_incremental_reindexes_changed_file(self, mock_ensure, tmp_path: Path):
        mock_client = MagicMock()
        mock_ensure.return_value = mock_client
        corpus = _make_corpus(tmp_path)

        emb = _mock_embedder()
        writer, _ = _mock_index_writer(mock_client)
        run_ingestion(corpus, embedder=emb, index_writer=writer)

        # Modify one file
        (corpus / "a.md").write_text(
            "---\ntitle: Doc A Updated\nsource_url: https://example.com/a\n---\n"
            "# Topic A Updated\n\nNew content about FastAPI path parameters.\n"
        )

        emb2 = _mock_embedder()
        writer2, mock_client2 = _mock_index_writer()
        mock_ensure.return_value = mock_client2
        summary = run_ingestion(corpus, embedder=emb2, index_writer=writer2)

        assert summary["files_indexed"] == 1
        assert summary["files_skipped"] == 1
        assert summary["chunks_upserted"] > 0
        # Should have deleted stale chunks for the changed doc
        mock_client2.delete.assert_called()

    @patch("ingestion.run.ensure_collection")
    def test_full_reindex_ignores_manifest(self, mock_ensure, tmp_path: Path):
        mock_client = MagicMock()
        mock_ensure.return_value = mock_client
        corpus = _make_corpus(tmp_path)

        emb = _mock_embedder()
        writer, _ = _mock_index_writer(mock_client)
        run_ingestion(corpus, embedder=emb, index_writer=writer)

        # Full re-index
        emb2 = _mock_embedder()
        writer2, mock_client2 = _mock_index_writer()
        mock_ensure.return_value = mock_client2
        summary = run_ingestion(
            corpus, full_reindex=True, embedder=emb2, index_writer=writer2
        )

        assert summary["files_indexed"] == 2
        assert summary["files_skipped"] == 0

    @patch("ingestion.run.ensure_collection")
    def test_manifest_persisted_after_run(self, mock_ensure, tmp_path: Path):
        mock_client = MagicMock()
        mock_ensure.return_value = mock_client
        corpus = _make_corpus(tmp_path)

        emb = _mock_embedder()
        writer, _ = _mock_index_writer(mock_client)
        run_ingestion(corpus, embedder=emb, index_writer=writer)

        manifest = load_manifest(corpus / "manifest.json")
        assert len(manifest.files) == 2
        assert "a.md" in manifest.files
        assert "b.md" in manifest.files
        assert manifest.get_entry("a.md").chunk_count > 0

    @patch("ingestion.run.ensure_collection")
    def test_empty_corpus(self, mock_ensure, tmp_path: Path):
        mock_client = MagicMock()
        mock_ensure.return_value = mock_client
        corpus = tmp_path / "empty"
        corpus.mkdir()

        emb = _mock_embedder()
        writer, _ = _mock_index_writer(mock_client)
        summary = run_ingestion(corpus, embedder=emb, index_writer=writer)

        assert summary["files_total"] == 0
        assert summary["files_indexed"] == 0

    @patch("ingestion.run.ensure_collection")
    def test_deleted_file_pruned_from_manifest(self, mock_ensure, tmp_path: Path):
        mock_client = MagicMock()
        mock_ensure.return_value = mock_client
        corpus = _make_corpus(tmp_path)

        emb = _mock_embedder()
        writer, _ = _mock_index_writer(mock_client)
        run_ingestion(corpus, embedder=emb, index_writer=writer)

        # Delete a file
        (corpus / "a.md").unlink()

        emb2 = _mock_embedder()
        writer2, mock_client2 = _mock_index_writer()
        mock_ensure.return_value = mock_client2
        run_ingestion(corpus, embedder=emb2, index_writer=writer2)

        manifest = load_manifest(corpus / "manifest.json")
        assert "a.md" not in manifest.files
        assert "b.md" in manifest.files
