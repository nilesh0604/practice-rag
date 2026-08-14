"""Feedback contract for the `POST /api/v1/feedback` endpoint.

Captures a thumbs up/down rating plus an optional comment for a specific
message in a session. Stored for offline eval correlation (Ragas + human
feedback) and surfaced in Langfuse as a score.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeedbackRequest(BaseModel):
    """Matches the architecture doc's `/api/v1/feedback` request body."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session the rated message belongs to.")
    message_index: int = Field(
        ...,
        ge=0,
        description="Zero-based index of the assistant message within the session.",
    )
    rating: Literal["up", "down"] = Field(
        ...,
        description="Thumbs up (positive) or thumbs down (negative).",
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional free-text feedback.",
    )
