"""Tests for the incremental-sync manifest (ingestion/manifest.py)."""

from __future__ import annotations

import json
from pathlib import Path

from ingestion.manifest import (
    FileEntry,
    Manifest,
    file_hash,
    load_manifest,
    save_manifest,
)


class TestFileHash:
    def test_deterministic(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        h1 = file_hash(f)
        h2 = file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert file_hash(f1) != file_hash(f2)

    def test_large_file(self, tmp_path: Path):
        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * 100000)
        h = file_hash(f)
        assert len(h) == 64


class TestManifest:
    def test_is_changed_new_file(self):
        m = Manifest()
        assert m.is_changed("new.md", "abc123") is True

    def test_is_changed_same_hash(self):
        m = Manifest(files={"doc.md": FileEntry(hash="abc", last_indexed="t", chunk_count=3)})
        assert m.is_changed("doc.md", "abc") is False

    def test_is_changed_different_hash(self):
        m = Manifest(files={"doc.md": FileEntry(hash="abc", last_indexed="t", chunk_count=3)})
        assert m.is_changed("doc.md", "xyz") is True

    def test_update_creates_entry(self):
        m = Manifest()
        m.update("doc.md", "abc", 5)
        entry = m.get_entry("doc.md")
        assert entry is not None
        assert entry.hash == "abc"
        assert entry.chunk_count == 5
        assert "T" in entry.last_indexed  # ISO timestamp

    def test_update_replaces_entry(self):
        m = Manifest(files={"doc.md": FileEntry(hash="old", last_indexed="t0", chunk_count=1)})
        m.update("doc.md", "new", 9)
        entry = m.get_entry("doc.md")
        assert entry.hash == "new"
        assert entry.chunk_count == 9

    def test_remove(self):
        m = Manifest(files={"doc.md": FileEntry(hash="abc", last_indexed="t", chunk_count=1)})
        m.remove("doc.md")
        assert m.get_entry("doc.md") is None
        assert "doc.md" not in m.files

    def test_stale_paths(self):
        m = Manifest(files={
            "a.md": FileEntry(hash="1", last_indexed="t", chunk_count=1),
            "b.md": FileEntry(hash="2", last_indexed="t", chunk_count=2),
            "c.md": FileEntry(hash="3", last_indexed="t", chunk_count=3),
        })
        stale = m.stale_paths({"a.md", "b.md"})
        assert stale == {"c.md"}


class TestManifestSerialization:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        m = Manifest()
        m.update("a.md", "hash_a", 3)
        m.update("b.md", "hash_b", 7)
        path = tmp_path / "manifest.json"
        save_manifest(m, path)

        loaded = load_manifest(path)
        assert set(loaded.files.keys()) == {"a.md", "b.md"}
        assert loaded.get_entry("a.md").hash == "hash_a"
        assert loaded.get_entry("a.md").chunk_count == 3
        assert loaded.get_entry("b.md").chunk_count == 7

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        loaded = load_manifest(tmp_path / "nonexistent.json")
        assert loaded.files == {}

    def test_save_produces_valid_json(self, tmp_path: Path):
        m = Manifest()
        m.update("doc.md", "abc", 2)
        path = tmp_path / "manifest.json"
        save_manifest(m, path)
        raw = json.loads(path.read_text())
        assert "files" in raw
        assert "doc.md" in raw["files"]
