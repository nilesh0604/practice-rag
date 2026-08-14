"""``POST /api/v1/chat`` — SSE streaming RAG answer endpoint.

Wraps the Step 3 ``RAGOrchestrator.stream_answer`` generator in a
``StreamingResponse`` with ``text/event-stream`` media type. The SSE wire
format matches the architecture doc:

    data: <token>\\n\\n            # one per generated token
    event: result\\n               # exactly once, after the last token
    data: {ChatResponse json}\\n\\n
    data: [DONE]\\n\\n              # terminal sentinel

The ``event: result`` frame carries the post-processed ``ChatResponse``
(citations + confidence + session id) so the frontend can render citation
chips and the low-confidence warning without parsing the streamed text.

Cache (Tier 1): before invoking the orchestrator the normalized query is
looked up in the LRU cache. On a hit the cached answer is replayed as a
single ``data:`` frame followed by the ``result`` and ``[DONE]`` frames,
skipping retrieval + generation entirely. The replayed ``result`` frame is
rebuilt with the *current* session id so session ids never leak across
sessions.

The event-stream generator is factored into ``build_event_stream`` so it
can be unit-tested directly (collecting the yielded SSE strings) without
spinning up the HTTP layer.
"""

from __future__ import annotations

import logging
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.cache import LRUCache
from api.conversation import ConversationStore
from api.deps import get_cache, get_conversation_store, get_orchestrator
from rag.orchestrator import RAGOrchestrator
from rag.post_processor import PostProcessResult
from schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()

SSE_MEDIA_TYPE: str = "text/event-stream"
"""Media type for Server-Sent Events responses."""


def build_event_stream(
    orchestrator: RAGOrchestrator,
    query: str,
    history: str,
    session_id: str,
    store: ConversationStore,
    cache: LRUCache,
) -> Iterator[str]:
    """Build the SSE generator for a chat request.

    Yields SSE-formatted strings (``data: ...\\n\\n`` etc.). Side effects:
    records the user + assistant messages in the session store and stores
    the completed ``ChatResponse`` in the LRU cache.

    This is a sync generator — FastAPI's ``StreamingResponse`` runs sync
    iterators in a threadpool, which is exactly what we want for the
    blocking Ollama streaming call.
    """
    # Record the user's message before generation so the session is
    # persisted even if generation fails partway.
    store.add_message(session_id, "user", query)

    answer_text = ""
    result: PostProcessResult | None = None
    for item in orchestrator.stream_answer(query, history):
        if isinstance(item, str):
            answer_text += item
            yield f"data: {item}\n\n"
        else:
            result = item

    if result is None:
        # No PostProcessResult was yielded (e.g. empty stream) — synthesize
        # an empty result so the session + cache stay consistent.
        result = PostProcessResult(answer=answer_text)

    response = ChatResponse(
        session_id=session_id,
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
    )
    store.add_message(
        session_id,
        "assistant",
        result.answer,
        citations=result.citations,
        confidence=result.confidence,
    )
    cache.put(query, response)
    yield f"event: result\ndata: {response.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def build_replay_stream(
    cached: ChatResponse,
    session_id: str,
) -> Iterator[str]:
    """Replay a cached answer as SSE without invoking the orchestrator.

    The cached ``ChatResponse`` is rebuilt with the current ``session_id``
    so the frontend never sees another session's id.
    """
    replayed = ChatResponse(
        session_id=session_id,
        answer=cached.answer,
        citations=cached.citations,
        confidence=cached.confidence,
    )
    yield f"data: {cached.answer}\n\n"
    yield f"event: result\ndata: {replayed.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat")
def chat(
    request: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
    store: ConversationStore = Depends(get_conversation_store),
    cache: LRUCache = Depends(get_cache),
) -> StreamingResponse:
    """Stream a RAG answer for the user's message as Server-Sent Events."""
    query = request.message

    # Resolve / create the session.
    session_id = request.session_id or store.create_session()
    if request.session_id and not store.session_exists(request.session_id):
        # An unknown session id was supplied — reject rather than silently
        # creating a phantom session that the client thinks exists.
        raise HTTPException(status_code=404, detail=f"Session {request.session_id!r} not found")

    # Tier 1 cache: exact normalized query match.
    cached = cache.get(query)
    if cached is not None:
        logger.info("Cache hit — replaying cached answer for session %s", session_id)
        return StreamingResponse(
            build_replay_stream(cached, session_id),
            media_type=SSE_MEDIA_TYPE,
            headers={"X-Cache": "HIT"},
        )

    history = store.format_history(session_id)
    return StreamingResponse(
        build_event_stream(orchestrator, query, history, session_id, store, cache),
        media_type=SSE_MEDIA_TYPE,
        headers={"X-Cache": "MISS"},
    )
