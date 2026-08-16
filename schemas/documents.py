"""Document contracts shared by the ingestion pipeline and the serving layer.

These models are the "single contract" called out in the architecture doc's
Single-Language Python Stack section: both the offline ingestion pipeline and
the online FastAPI server depend on them, so getting them right first prevents
rework in both layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class DocumentChunk(BaseModel):
    """A single chunk produced by the ingestion pipeline.

    This is the unit that gets embedded and upserted into Qdrant. The fields
    map 1:1 to the Qdrant payload documented in the architecture doc
    (title, source_url, section, last_modified, chunk_index, parent_doc_id),
    plus the chunk text and (optional) dense embedding vector.

    `embedding` is optional so the same model can represent a chunk before
    embedding (pipeline intermediate) and after embedding (ready to upsert).
    """

    model_config = ConfigDict(
        extra="forbid",
        ser_json_timedelta="iso8601",
    )

    id: str = Field(
        ...,
        description="Stable chunk identifier (deterministic hash of parent_doc_id + chunk_index).",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The chunk text. Code blocks preserved in backticks.",
    )
    title: str = Field(..., description="Document title, e.g. 'Query Parameters - FastAPI'.")
    source_url: HttpUrl = Field(..., description="Canonical URL of the source page.")
    section: str | None = Field(
        default=None,
        description="Logical section / heading path, e.g. 'tutorial'.",
    )
    last_modified: datetime = Field(
        ...,
        description="Last-modified timestamp of the source document (UTC).",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of this chunk within its parent document.",
    )
    parent_doc_id: str = Field(
        ...,
        description="Logical document id, e.g. 'fastapi-query-params'. "
        "Used to filter/delete all chunks of a stale document.",
    )
    department: str | None = Field(
        default=None,
        description="Department this chunk is intended for, e.g. 'engineering'.",
    )
    roles: list[str] | None = Field(
        default=None,
        description="Allowed roles for this chunk, e.g. ['manager', 'lead'].",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="768-d dense embedding from nomic-embed-text. "
        "None until the embedder step has run.",
    )

    @field_validator("last_modified")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        """Normalize naive datetimes to UTC so Qdrant payloads are unambiguous."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the Qdrant payload dict (no embedding).

        The embedding is upserted separately as the named vector; the payload
        holds only the metadata fields so retrievals can return them directly.
        """
        return self.model_dump(
            mode="json",
            exclude={"id", "embedding"},
        )


class RetrievedDoc(BaseModel):
    """A chunk returned by the hybrid retriever, with its fused score.

    Produced by the RAG orchestrator's retriever step (dense + sparse + RRF).
    Carries enough metadata to render a citation and to re-assemble context.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Qdrant point id / chunk id.")
    content: str = Field(..., description="Chunk text used to assemble the CONTEXT block.")
    title: str = Field(..., description="Document title, rendered in [Source: title].")
    source_url: HttpUrl = Field(..., description="Source URL for the citation link.")
    section: str | None = Field(default=None, description="Section the chunk belongs to.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fused retrieval score (RRF-normalized to [0, 1]).",
    )
    chunk_index: int | None = Field(
        default=None,
        description="Chunk position within the parent doc, if known.",
    )
    parent_doc_id: str | None = Field(
        default=None,
        description="Parent document id, if known.",
    )
    last_modified: datetime | None = Field(
        default=None,
        description="Last-modified timestamp of the source document (UTC), "
        "if present in the Qdrant payload. Used to populate citation "
        "``lastModified`` so the UI can show source freshness.",
    )
    department: str | None = Field(
        default=None,
        description="Department this chunk is intended for, if present.",
    )
    roles: list[str] | None = Field(
        default=None,
        description="Allowed roles for this chunk, if present.",
    )
