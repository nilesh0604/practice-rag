"""Tests for doc discovery (ingestion/sync.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.sync import SUPPORTED_EXTENSIONS, discover_files


class TestDiscoverFiles:
    def test_finds_markdown_files(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.md").write_text("# B")
        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"a.md", "b.md"}

    def test_finds_nested_files(self, tmp_path: Path):
        sub = tmp_path / "fastapi" / "tutorial"
        sub.mkdir(parents=True)
        (sub / "query-params.md").write_text("# Query Params")
        (tmp_path / "index.md").write_text("# Index")
        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert {"query-params.md", "index.md"} == names

    def test_skips_unsupported_extensions(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.txt").write_text("text")
        (tmp_path / "c.json").write_text("{}")
        files = discover_files(tmp_path)
        assert {f.name for f in files} == {"a.md"}

    def test_skips_hidden_and_vcs_dirs(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config.md").write_text("git config")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.md").write_text("lib")
        (tmp_path / "real.md").write_text("# Real")
        files = discover_files(tmp_path)
        assert {f.name for f in files} == {"real.md"}

    def test_finds_html_and_pdf(self, tmp_path: Path):
        (tmp_path / "page.html").write_text("<h1>Hi</h1>")
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
        files = discover_files(tmp_path)
        assert {f.name for f in files} == {"page.html", "doc.pdf"}

    def test_sorted_output(self, tmp_path: Path):
        for name in ("c.md", "a.md", "b.md"):
            (tmp_path / name).write_text(f"# {name}")
        files = discover_files(tmp_path)
        names = [f.name for f in files]
        assert names == ["a.md", "b.md", "c.md"]

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            discover_files("/nonexistent/path/that/does/not/exist")

    def test_empty_dir(self, tmp_path: Path):
        assert discover_files(tmp_path) == []

    def test_supported_extensions_includes_md(self):
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".html" in SUPPORTED_EXTENSIONS
