"""``POST /api/v1/feedback`` — thumbs up/down + comment endpoint.

Records user feedback for a specific message in a session. The feedback is
stored for offline eval correlation (Ragas + human signal) and is surfaced
in Langfuse as a score (Step 7, build-order item 39). When the request
carries a ``trace_id`` (returned in the chat ``result`` frame), the score
is attached to that Langfuse trace; otherwise the score is logged via
structlog (the Langfuse-down fallback). Returns 404 for an unknown session.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from api.conversation import ConversationStore
from api.deps import get_conversation_store, get_tracer
from api.observability import LangfuseTracer, TraceHandle
from schemas.feedback import FeedbackRequest

router = APIRouter()


class FeedbackResponse(BaseModel):
    """Acknowledgement of a recorded feedback entry."""

    model_config = ConfigDict(extra="forbid")

    recorded: bool = True
    feedback_id: int


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(
    request: FeedbackRequest,
    store: ConversationStore = Depends(get_conversation_store),
    tracer: LangfuseTracer = Depends(get_tracer),
) -> FeedbackResponse:
    """Record a thumbs up/down rating for a message in a session."""
    if not store.session_exists(request.session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id!r} not found",
        )
    feedback_id = store.add_feedback(
        session_id=request.session_id,
        message_index=request.message_index,
        rating=request.rating,
        comment=request.comment,
    )
    # Step 7: attach the feedback as a Langfuse score on the chat trace.
    if request.trace_id:
        score_value = 1.0 if request.rating == "up" else 0.0
        handle = TraceHandle(id=request.trace_id, enabled=tracer.enabled)
        tracer.record_score(
            handle,
            name="user_feedback",
            value=score_value,
            comment=request.comment,
        )
    return FeedbackResponse(recorded=True, feedback_id=feedback_id)
