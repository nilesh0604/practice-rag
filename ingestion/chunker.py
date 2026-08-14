"""Chunker — splits a ParsedDocument into DocumentChunk objects.

Two-stage splitting per the architecture doc's chunking strategy:

1. **MarkdownHeaderTextSplitter** — splits on H1 / H2 / H3 headers so each
   chunk carries its section path as metadata. Keeps header context
   attached to the content beneath it.

2. **RecursiveCharacterTextSplitter.from_tiktoken_encoder** — further
   splits long sections to a target of 512 tokens with 64-token overlap.
   Uses the ``cl100k_base`` encoding (tiktoken default) for token counting.

Chunk IDs are deterministic: ``sha256(parent_doc_id + ":" + chunk_index)``
truncated to 16 hex chars. This makes re-indexing idempotent — the same
source file always produces the same chunk IDs, so Qdrant upserts
overwrite rather than duplicate.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import TYPE_CHECKING

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

if TYPE_CHECKING:
    from ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)

# ── chunking constants (architecture doc) ────────────────────────────

CHUNK_SIZE_TOKENS: int = 512
"""Target chunk size in tokens (fits ~400 words; 5 chunks fit in 8K ctx)."""

CHUNK_OVERLAP_TOKENS: int = 64
"""Overlap between adjacent chunks to preserve sentences split at boundaries."""

ENCODING_NAME: str = "cl100k_base"
"""tiktoken encoding used for token counting (matches modern LLM tokenizers)."""

# H1 / H2 / H3 → metadata keys for the section path
_MARKDOWN_HEADERS: list[tuple[str, str]] = [
    ("#", "section_h1"),
    ("##", "section_h2"),
    ("###", "section_h3"),
]


def _build_splitters() -> tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
    """Build the two-stage splitter pipeline."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS,
        strip_headers=False,
    )
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    return header_splitter, token_splitter


def _chunk_id(parent_doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID chunk id from parent + index.

    Qdrant requires point IDs to be either unsigned integers or UUIDs.
    We hash ``parent_doc_id:chunk_index`` with SHA-256 and convert the
    first 32 hex chars to a UUID string — deterministic across re-indexes
    so upserts overwrite rather than duplicate.
    """
    raw = f"{parent_doc_id}:{chunk_index}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return str(uuid.UUID(hex=digest))


def _section_path(metadata: dict[str, str]) -> str | None:
    """Join non-empty section headers into a slash-separated path."""
    parts = [
        metadata.get("section_h1"),
        metadata.get("section_h2"),
        metadata.get("section_h3"),
    ]
    joined = " / ".join(p for p in parts if p)
    return joined or None


def chunk_document(doc: "ParsedDocument") -> list:
    """Split a ParsedDocument into a list of DocumentChunk objects.

    Returns an empty list if the document content is empty.
    """
    from schemas.documents import DocumentChunk

    if not doc.content.strip():
        logger.warning("Empty content for %s; skipping", doc.file_path)
        return []

    header_splitter, token_splitter = _build_splitters()

    # Stage 1: split by markdown headers → LangChain Document objects
    # with section metadata.
    try:
        header_chunks = header_splitter.split_text(doc.content)
    except Exception:
        # Non-markdown or malformed content — treat the whole doc as one
        # headerless chunk.
        from langchain_core.documents import Document

        header_chunks = [Document(page_content=doc.content, metadata={})]

    # Stage 2: token-based recursive split within each header section.
    final_texts: list[tuple[str, str | None]] = []
    for h_chunk in header_chunks:
        section = _section_path(h_chunk.metadata)
        sub_texts = token_splitter.split_text(h_chunk.page_content)
        for text in sub_texts:
            text = text.strip()
            if text:
                final_texts.append((text, section))

    if not final_texts:
        return []

    chunks = []
    for idx, (text, section) in enumerate(final_texts):
        chunks.append(
            DocumentChunk(
                id=_chunk_id(doc.parent_doc_id, idx),
                content=text,
                title=doc.title,
                source_url=doc.source_url,
                section=section or doc.section,
                last_modified=doc.last_modified,
                chunk_index=idx,
                parent_doc_id=doc.parent_doc_id,
            )
        )

    logger.info(
        "Chunked %s → %d chunks (target %d tok, overlap %d)",
        doc.file_path,
        len(chunks),
        CHUNK_SIZE_TOKENS,
        CHUNK_OVERLAP_TOKENS,
    )
    return chunks
