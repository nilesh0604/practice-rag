"""Incremental-sync manifest — tracks file hashes so only changed files
are re-indexed on subsequent ingestion runs.

The manifest is a JSON file (``manifest.json``) mapping each source file's
relative path to its SHA-256 hash, last-indexed timestamp, and chunk count.
On a re-run, files whose hash is unchanged are skipped; changed files are
re-parsed/re-chunked/re-embedded, and their stale chunks are deleted from
Qdrant by ``parent_doc_id`` before the new chunks are upserted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FileEntry:
    """One file's ingestion state."""

    hash: str
    last_indexed: str  # ISO-8601 UTC
    chunk_count: int


@dataclass
class Manifest:
    """In-memory manifest — a dict of relative-path → FileEntry."""

    files: dict[str, FileEntry] = field(default_factory=dict)

    # ── queries ───────────────────────────────────────────────────────

    def is_changed(self, rel_path: str, current_hash: str) -> bool:
        """True if the file is new or its hash differs from the manifest."""
        entry = self.files.get(rel_path)
        return entry is None or entry.hash != current_hash

    def get_entry(self, rel_path: str) -> FileEntry | None:
        return self.files.get(rel_path)

    # ── mutations ─────────────────────────────────────────────────────

    def update(self, rel_path: str, file_hash: str, chunk_count: int) -> None:
        """Record (or replace) a file's entry after a successful index."""
        self.files[rel_path] = FileEntry(
            hash=file_hash,
            last_indexed=datetime.now(timezone.utc).isoformat(),
            chunk_count=chunk_count,
        )

    def remove(self, rel_path: str) -> None:
        """Drop a file from the manifest (e.g. it was deleted from disk)."""
        self.files.pop(rel_path, None)

    def stale_paths(self, current_paths: set[str]) -> set[str]:
        """Paths in the manifest but no longer on disk (deleted sources)."""
        return set(self.files) - current_paths


# ── file hashing ──────────────────────────────────────────────────────


def file_hash(path: str | Path) -> str:
    """SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── (de)serialization ────────────────────────────────────────────────


def save_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write the manifest to ``path`` as pretty-printed JSON."""
    data = {"files": {k: asdict(v) for k, v in manifest.files.items()}}
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: str | Path) -> Manifest:
    """Load a manifest from ``path``.

    Returns an empty manifest if the file does not exist (first run).
    """
    p = Path(path)
    if not p.exists():
        return Manifest()
    raw = json.loads(p.read_text(encoding="utf-8"))
    files = {
        rel: FileEntry(**entry) for rel, entry in raw.get("files", {}).items()
    }
    return Manifest(files=files)
