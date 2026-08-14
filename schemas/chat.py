"""Chat request/response contracts for the FastAPI serving layer.

The SSE stream emits raw tokens (`data: <token>\\n\\n`) followed by
`data: [DONE]`; `ChatResponse` is the non-streaming, post-processed
representation that gets persisted to the session store and returned by
`GET /api/v1/history/{session_id}`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ChatRequest(BaseModel):
    """Inbound chat message. Matches the architecture doc's `/api/v1/chat` body."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question. Bounded to deter prompt-injection payloads.",
    )
    session_id: str | None = Field(
        default=None,
        description="Existing session id. If omitted, the server creates a new one.",
    )


class Citation(BaseModel):
    """A single source citation rendered as `[Source: title]` -> link."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="Document title shown in the citation chip.")
    source_url: HttpUrl = Field(..., description="Clickable source URL.")


class ChatResponse(BaseModel):
    """Final, post-processed answer persisted to the session store.

    The streaming endpoint emits tokens as SSE; once generation completes the
    post-processor extracts citations and computes a groundedness/confidence
    score, producing this object for persistence and the history endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session id the answer belongs to.")
    answer: str = Field(..., description="Full generated answer text (concatenated tokens).")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Sources cited in the answer, deduplicated.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Groundedness/confidence score in [0, 1].",
    )

    @model_validator(mode="after")
    def _dedupe_citations(self) -> "ChatResponse":
        """Deduplicate citations by source_url, preserving first-seen order."""
        seen: set[str] = set()
        deduped: list[Citation] = []
        for cite in self.citations:
            key = str(cite.source_url)
            if key not in seen:
                seen.add(key)
                deduped.append(cite)
        self.citations = deduped
        return self
