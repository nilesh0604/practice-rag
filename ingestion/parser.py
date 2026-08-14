"""Parser — converts source files (MD / HTML / PDF) into ParsedDocument.

A ParsedDocument carries the raw text plus the metadata that will become
the Qdrant payload (title, source_url, section, last_modified,
parent_doc_id). The chunker consumes ParsedDocument and produces
DocumentChunk objects.

Markdown is the primary format for this corpus (FastAPI / Pydantic /
SQLModel docs are Markdown source). HTML and PDF are supported for
completeness but the seed corpus is Markdown-only.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """A parsed source file ready for chunking."""

    file_path: str
    content: str
    title: str
    source_url: str
    last_modified: datetime
    parent_doc_id: str
    section: str | None = None
    extra_metadata: dict = field(default_factory=dict)


# ── helpers ───────────────────────────────────────────────────────────

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_URL_RE = re.compile(r"^\s*(?:source_url|url|canonical):\s*(.+)$", re.MULTILINE)
_TITLE_RE = re.compile(r"^\s*(?:title):\s*(.+)$", re.MULTILINE)


def _stable_doc_id(file_path: Path, corpus_root: Path | None = None) -> str:
    """Deterministic parent_doc_id from the file's relative path.

    When ``corpus_root`` is given, the path is made relative to it so the
    id is stable across machines. Without a corpus root, only the file
    name (stem) is used to avoid leaking absolute temp paths into ids.
    """
    if corpus_root is not None:
        try:
            rel = file_path.relative_to(corpus_root)
        except ValueError:
            rel = Path(file_path.name)
    else:
        rel = Path(file_path.name)
    stem = str(rel).replace("/", "-").removesuffix(file_path.suffix)
    return stem.lower()


def _extract_frontmatter(text: str) -> tuple[str, dict[str, str]]:
    """Split YAML-like frontmatter from the body. Returns (body, meta)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text, {}
    fm = match.group(1)
    body = text[match.end():]
    meta: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip("\"'")
    return body, meta


def _derive_source_url(file_path: Path, meta: dict[str, str]) -> str:
    """Resolve the canonical source URL from frontmatter or a docs-base heuristic."""
    if "source_url" in meta:
        return meta["source_url"]
    if "url" in meta:
        return meta["url"]
    # Heuristic: map data/corpus/fastapi/query-params.md → fastapi.tiangolo.com/...
    parts = file_path.parts
    for known in ("fastapi", "pydantic", "sqlmodel"):
        if known in parts:
            slug = file_path.stem
            bases = {
                "fastapi": f"https://fastapi.tiangolo.com/tutorial/{slug}/",
                "pydantic": f"https://docs.pydantic.dev/2/{slug}/",
                "sqlmodel": f"https://sqlmodel.tiangolo.com/{slug}/",
            }
            return bases[known]
    return f"https://docs.example.com/{file_path.stem}"


# ── format-specific parsers ──────────────────────────────────────────


def parse_markdown(file_path: Path, corpus_root: Path | None = None) -> ParsedDocument:
    """Parse a Markdown file — extracts title from frontmatter or first H1."""
    raw = file_path.read_text(encoding="utf-8")
    body, meta = _extract_frontmatter(raw)

    title = meta.get("title", "")
    if not title:
        h1 = _H1_RE.search(body)
        title = h1.group(1).strip() if h1 else file_path.stem.replace("-", " ").title()

    source_url = _derive_source_url(file_path, meta)
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    doc_id = _stable_doc_id(file_path, corpus_root)

    return ParsedDocument(
        file_path=str(file_path),
        content=body.strip(),
        title=title,
        source_url=source_url,
        last_modified=mtime,
        parent_doc_id=doc_id,
    )


def parse_html(file_path: Path, corpus_root: Path | None = None) -> ParsedDocument:
    """Parse an HTML file with BeautifulSoup — strips tags, keeps text."""
    from bs4 import BeautifulSoup

    raw = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else file_path.stem
    content = soup.get_text(separator="\n", strip=True)
    source_url = _derive_source_url(file_path, {})
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    doc_id = _stable_doc_id(file_path, corpus_root)

    return ParsedDocument(
        file_path=str(file_path),
        content=content,
        title=title,
        source_url=source_url,
        last_modified=mtime,
        parent_doc_id=doc_id,
    )


def parse_pdf(file_path: Path, corpus_root: Path | None = None) -> ParsedDocument:
    """Parse a PDF file with pypdf — extracts text page by page."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    content = "\n\n".join(pages).strip()
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else file_path.stem)
    source_url = _derive_source_url(file_path, {})
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    doc_id = _stable_doc_id(file_path, corpus_root)

    return ParsedDocument(
        file_path=str(file_path),
        content=content,
        title=title,
        source_url=source_url,
        last_modified=mtime,
        parent_doc_id=doc_id,
    )


# ── dispatch ─────────────────────────────────────────────────────────

_PARSERS = {
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
    ".pdf": parse_pdf,
}


def parse_file(file_path: str | Path, corpus_root: str | Path | None = None) -> ParsedDocument:
    """Dispatch to the correct parser based on file extension."""
    path = Path(file_path)
    root = Path(corpus_root) if corpus_root else None
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    logger.debug("Parsing %s as %s", path, path.suffix)
    return parser(path, root)
