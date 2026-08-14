"""Ingestion orchestrator — ties sync → parse → chunk → embed → upsert.

This is the entrypoint for both first-run (full index) and subsequent
runs (incremental). It:

1. Discovers source files in the corpus directory.
2. Compares each file's SHA-256 hash against ``manifest.json``.
3. For changed/new files: parse → chunk → embed (dense) → upsert (dense +
   sparse). Stale chunks for re-indexed docs are deleted first.
4. Removes manifest entries for deleted files.
5. Persists the updated manifest.

Usage::

    from ingestion.run import run_ingestion
    run_ingestion("data/corpus")               # incremental
    run_ingestion("data/corpus", full_reindex=True)  # force full re-index

Or as a script::

    python -m ingestion.run data/corpus [--full-reindex]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ingestion.chunker import chunk_document
from ingestion.embedder import Embedder
from ingestion.index_writer import IndexWriter
from ingestion.manifest import (
    Manifest,
    file_hash,
    load_manifest,
    save_manifest,
)
from ingestion.parser import parse_file
from ingestion.sync import discover_files
from rag.qdrant_collection import ensure_collection

logger = logging.getLogger(__name__)

MANIFEST_FILENAME: str = "manifest.json"
"""Default manifest file name (written inside the corpus dir)."""


def run_ingestion(
    corpus_dir: str | Path,
    *,
    full_reindex: bool = False,
    manifest_path: str | Path | None = None,
    embedder: Embedder | None = None,
    index_writer: IndexWriter | None = None,
) -> dict:
    """Run the full ingestion pipeline on ``corpus_dir``.

    Args:
        corpus_dir: directory containing source documents (.md/.html/.pdf).
        full_reindex: if True, ignore the manifest and re-index every file
            (drops and recreates the collection first).
        manifest_path: override for the manifest file location. Defaults
            to ``<corpus_dir>/manifest.json``.
        embedder: optional pre-constructed Embedder (for testing / reuse).
        index_writer: optional pre-constructed IndexWriter.

    Returns:
        A summary dict: ``{files_total, files_indexed, files_skipped,
        chunks_upserted, chunks_deleted}``.
    """
    corpus = Path(corpus_dir).resolve()
    manifest_file = Path(manifest_path) if manifest_path else corpus / MANIFEST_FILENAME

    # ── 1. discover files ─────────────────────────────────────────────
    files = discover_files(corpus)
    if not files:
        logger.warning("No supported source files found in %s", corpus)
        return _summary(0, 0, 0, 0, 0)

    rel_paths = {str(f.relative_to(corpus)) for f in files}

    # ── 2. ensure collection + load manifest ──────────────────────────
    if full_reindex:
        client = ensure_collection()
        # Recreate to clear all existing points.
        from rag.qdrant_collection import create_collection

        create_collection(client, recreate=True)
        manifest = Manifest()
    else:
        client = ensure_collection()
        manifest = load_manifest(manifest_file)

    emb = embedder or Embedder()
    writer = index_writer or IndexWriter(client)

    # ── 3. process changed files ──────────────────────────────────────
    files_indexed = 0
    files_skipped = 0
    chunks_upserted = 0
    chunks_deleted = 0

    for file_path in files:
        rel = str(file_path.relative_to(corpus))
        current_hash = file_hash(file_path)

        if not full_reindex and not manifest.is_changed(rel, current_hash):
            files_skipped += 1
            logger.debug("Skipping unchanged: %s", rel)
            continue

        # Parse → chunk
        parsed = parse_file(file_path, corpus_root=corpus)
        chunks = chunk_document(parsed)
        if not chunks:
            logger.warning("No chunks produced from %s; skipping", rel)
            continue

        # Delete stale chunks for this parent doc (incremental re-index).
        if not full_reindex:
            writer.delete_by_parent(parsed.parent_doc_id)
            chunks_deleted += 1  # counts docs refreshed, not individual points

        # Embed (dense) — batch all chunks of this file in one Ollama call.
        dense_vectors = emb.embed_texts([c.content for c in chunks])
        for chunk, vec in zip(chunks, dense_vectors):
            chunk.embedding = vec

        # Upsert (dense + sparse + payload).
        upserted = writer.upsert_chunks(chunks)
        chunks_upserted += upserted
        files_indexed += 1

        # Update manifest.
        manifest.update(rel, current_hash, chunk_count=len(chunks))

    # ── 4. prune deleted files from manifest ──────────────────────────
    stale = manifest.stale_paths(rel_paths)
    for rel in stale:
        # Best-effort: delete chunks for the stale doc.
        parent_id = rel.replace("/", "-").rsplit(".", 1)[0].lower()
        writer.delete_by_parent(parent_id)
        manifest.remove(rel)
        chunks_deleted += 1

    # ── 5. persist manifest ───────────────────────────────────────────
    save_manifest(manifest, manifest_file)

    summary = _summary(
        len(files), files_indexed, files_skipped, chunks_upserted, chunks_deleted
    )
    logger.info("Ingestion complete: %s", summary)
    return summary


def _summary(total, indexed, skipped, upserted, deleted) -> dict:
    return {
        "files_total": total,
        "files_indexed": indexed,
        "files_skipped": skipped,
        "chunks_upserted": upserted,
        "chunks_deleted_docs": deleted,
    }


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline.")
    parser.add_argument("corpus_dir", help="Path to the corpus directory.")
    parser.add_argument(
        "--full-reindex",
        action="store_true",
        help="Drop and recreate the collection; re-index every file.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    summary = run_ingestion(args.corpus_dir, full_reindex=args.full_reindex)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
