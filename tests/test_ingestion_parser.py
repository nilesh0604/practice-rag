"""Tests for the parser (ingestion/parser.py)."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pytest

from ingestion.parser import ParsedDocument, parse_file, parse_markdown


class TestParseMarkdown:
    def test_extracts_title_from_frontmatter(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("---\ntitle: My Title\nsource_url: https://example.com/page\n---\n# Body\n")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert doc.title == "My Title"
        assert str(doc.source_url) == "https://example.com/page"

    def test_extracts_title_from_h1_when_no_frontmatter(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello World\n\nSome content.\n")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert doc.title == "Hello World"

    def test_falls_back_to_filename_when_no_title(self, tmp_path: Path):
        f = tmp_path / "my-doc.md"
        f.write_text("Just some content without a heading.\n")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert "My Doc" in doc.title  # title-cased stem

    def test_parent_doc_id_is_deterministic(self, tmp_path: Path):
        sub = tmp_path / "fastapi"
        sub.mkdir()
        f = sub / "query-params.md"
        f.write_text("# Query Params\n\nContent.\n")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert doc.parent_doc_id == "fastapi-query-params"

    def test_parent_doc_id_without_corpus_root(self, tmp_path: Path):
        f = tmp_path / "query-params.md"
        f.write_text("# Query Params\n\nContent.\n")
        doc = parse_markdown(f)
        assert doc.parent_doc_id == "query-params"

    def test_last_modified_is_utc_aware(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# Doc\n")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert doc.last_modified.tzinfo is not None
        assert doc.last_modified.tzinfo == timezone.utc

    def test_source_url_from_frontmatter(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("---\nsource_url: https://fastapi.tiangolo.com/tutorial/path-params/\n---\n# Path\n")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert "fastapi.tiangolo.com" in str(doc.source_url)

    def test_source_url_heuristic_fastapi(self, tmp_path: Path):
        sub = tmp_path / "fastapi"
        sub.mkdir()
        f = sub / "query-params.md"
        f.write_text("# Query Params\n\nContent.\n")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert "fastapi.tiangolo.com" in str(doc.source_url)

    def test_content_strips_frontmatter(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("---\ntitle: T\n---\n# Heading\n\nBody text.\n")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert "---" not in doc.content
        assert "Body text" in doc.content

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.md"
        f.write_text("")
        doc = parse_markdown(f, corpus_root=tmp_path)
        assert doc.content == ""


class TestParseFileDispatch:
    def test_dispatches_markdown(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nContent.\n")
        doc = parse_file(f, corpus_root=tmp_path)
        assert isinstance(doc, ParsedDocument)
        assert doc.title == "Title"

    def test_dispatches_html(self, tmp_path: Path):
        f = tmp_path / "page.html"
        f.write_text("<html><head><title>Page</title></head><body><p>Hello</p></body></html>")
        doc = parse_file(f, corpus_root=tmp_path)
        assert doc.title == "Page"
        assert "Hello" in doc.content

    def test_unsupported_extension_raises(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("text")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_file(f)
