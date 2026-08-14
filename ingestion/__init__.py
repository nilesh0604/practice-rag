"""Offline ingestion pipeline — converts a documentation corpus into
embedded chunks stored in Qdrant.

Pipeline stages (each module is independently unit-testable):

    sync.py        → discover / download source documents
    parser.py      → parse MD / HTML / PDF into ParsedDocument
    chunker.py     → split into DocumentChunk (512 tok, 64 overlap)
    embedder.py    → nomic-embed-text dense + hashing-trick sparse
    index_writer.py→ Qdrant upsert (dense + sparse + payload)
    manifest.py    → file-hash manifest for incremental sync
    run.py         → orchestrator tying the stages together

The shared Pydantic contracts (DocumentChunk) live in `schemas/`.
"""

from ingestion.chunker import chunk_document
from ingestion.embedder import Embedder, sparse_embed_batch, sparse_embed_text, tokenize
from ingestion.index_writer import IndexWriter
from ingestion.manifest import Manifest, file_hash, load_manifest, save_manifest
from ingestion.parser import ParsedDocument, parse_file
from ingestion.run import run_ingestion
from ingestion.sync import discover_files

__all__ = [
    "Embedder",
    "IndexWriter",
    "Manifest",
    "ParsedDocument",
    "chunk_document",
    "discover_files",
    "file_hash",
    "load_manifest",
    "parse_file",
    "run_ingestion",
    "save_manifest",
    "sparse_embed_batch",
    "sparse_embed_text",
    "tokenize",
]
