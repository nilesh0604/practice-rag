"""``GET /api/v1/history/{session_id}`` — conversation history endpoint.

Returns the full chronological message list for a session, including
citations and confidence for assistant turns. Used by the frontend to
restore a prior conversation. Returns 404 for an unknown session id.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.conversation import ConversationStore
from api.deps import get_conversation_store

router = APIRouter()


class HistoryMessage(BaseModel):
    """A single message in the history response."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(..., description="'user' or 'assistant'.")
    content: str = Field(..., description="Message text.")
    citations: list[dict] | None = Field(
        default=None,
        description="Citations (assistant turns only), as JSON dicts.",
    )
    confidence: float | None = Field(
        default=None,
        description="Groundedness score (assistant turns only).",
    )
    created_at: str = Field(..., description="ISO-8601 UTC timestamp.")


class HistoryResponse(BaseModel):
    """Full history for a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    messages: list[HistoryMessage]


@router.get("/history/{session_id}", response_model=HistoryResponse)
def history(
    session_id: str,
    store: ConversationStore = Depends(get_conversation_store),
) -> HistoryResponse:
    """Return the full conversation history for a session."""
    if not store.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    rows = store.get_messages(session_id)
    messages = [
        HistoryMessage(
            role=r["role"],
            content=r["content"],
            citations=r["citations"],
            confidence=r["confidence"],
            created_at=r["created_at"],
        )
        for r in rows
    ]
    return HistoryResponse(session_id=session_id, messages=messages)
