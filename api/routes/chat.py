"""``POST /api/v1/chat`` — SSE streaming RAG answer endpoint.

Wraps the Step 3 ``RAGOrchestrator.stream_answer`` generator in a
``StreamingResponse`` with ``text/event-stream`` media type. The SSE wire
format uses named events:

    event: delta\\n                 # one per generated token
    data: <token>\\n\\n
    event: guardrail_replacement\\n # only when the output guardrail blocks
    data: {"answer": "<refusal>"}\\n\\n
    event: sources\\n               # exactly once, after the last token
    data: {"citations": [...]}\\n\\n
    event: metadata\\n              # exactly once, after sources
    data: {"session_id": ..., "confidence": ..., "trace_id": ...}\\n\\n
    event: done\\n                  # terminal sentinel
    data: [DONE]\\n\\n

The ``event: sources`` frame carries the post-processed citations so the
frontend can render citation chips without parsing the streamed text. The
``event: metadata`` frame carries the session id, groundedness/confidence
score, and Langfuse trace id so the frontend can set the session, render
the low-confidence warning, and pass the trace id back with feedback. The
full ``ChatResponse`` (including the concatenated ``answer``) is still
persisted to the session store and the LRU cache — only the wire format
drops the redundant ``answer`` (the frontend reconstructs it from deltas).

The optional ``event: guardrail_replacement`` frame is emitted **only** when
the output guardrail blocks the already-streamed answer. Because SSE is
one-way the original tokens cannot be un-sent, so this frame carries the
refusal text and the frontend swaps the visible message content for it. It
arrives after the last ``delta`` and before ``sources``.

Cache (Tier 1): before invoking the orchestrator the normalized query is
looked up in the LRU cache. On a hit the cached answer is replayed as a
single ``event: delta`` frame followed by the ``sources``, ``metadata``,
and ``done`` frames, skipping retrieval + generation entirely. The
replayed ``metadata`` frame is rebuilt with the *current* session id so
session ids never leak across sessions.

The event-stream generator is factored into ``build_event_stream`` so it
can be unit-tested directly (collecting the yielded SSE strings) without
spinning up the HTTP layer.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.cache import LRUCache
from api.conversation import ConversationStore
from api.deps import get_cache, get_conversation_store, get_metrics, get_orchestrator
from api.observability import MetricsCollector
from rag.orchestrator import RAGOrchestrator
from rag.post_processor import PostProcessResult
from schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()

SSE_MEDIA_TYPE: str = "text/event-stream"
"""Media type for Server-Sent Events responses."""


def _cache_key(query: str, department: str | None, role: str | None) -> str:
    """Build a cache key that includes department/role when present."""
    parts = [p for p in (department, role, query) if p]
    return " | ".join(parts) if len(parts) > 1 else query


def build_event_stream(
    orchestrator: RAGOrchestrator,
    query: str,
    history: str,
    session_id: str,
    store: ConversationStore,
    cache: LRUCache,
    metrics: MetricsCollector | None = None,
    department: str | None = None,
    role: str | None = None,
    cache_key: str | None = None,
) -> Iterator[str]:
    """Build the SSE generator for a chat request.

    Yields SSE-formatted strings (``data: ...\\n\\n`` etc.). Side effects:
    records the user + assistant messages in the session store and stores
    the completed ``ChatResponse`` in the LRU cache. When a ``metrics``
    collector is provided (Step 7), records the request count, TTFT (time
    to first token), and any errors.

    This is a sync generator — FastAPI's ``StreamingResponse`` runs sync
    iterators in a threadpool, which is exactly what we want for the
    blocking Ollama streaming call.
    """
    if metrics is not None:
        metrics.record_request()
    start = time.time()
    # Record the user's message before generation so the session is
    # persisted even if generation fails partway.
    store.add_message(session_id, "user", query)

    answer_text = ""
    result: PostProcessResult | None = None
    first_token_time: float | None = None
    try:
        for item in orchestrator.stream_answer(query, history, department, role):
            if isinstance(item, str):
                if first_token_time is None:
                    first_token_time = time.time()
                    if metrics is not None:
                        metrics.record_ttft(first_token_time - start)
                answer_text += item
                yield f"event: delta\ndata: {item}\n\n"
            else:
                result = item
    except Exception:
        if metrics is not None:
            metrics.record_error()
        raise

    if result is None:
        # No PostProcessResult was yielded (e.g. empty stream) — synthesize
        # an empty result so the session + cache stay consistent.
        result = PostProcessResult(answer=answer_text)

    # Step 7: surface the Langfuse trace id so the frontend can pass it
    # back with feedback (the score is then attached to the right trace).
    trace_id = getattr(result, "trace_id", None)

    response = ChatResponse(
        session_id=session_id,
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        trace_id=trace_id,
    )
    store.add_message(
        session_id,
        "assistant",
        result.answer,
        citations=result.citations,
        confidence=result.confidence,
    )
    cache.put(cache_key or query, response)
    # Responsible AI: record the bias & fairness check result (when a bias
    # monitor ran) so the ``GET /api/v1/metrics`` snapshot surfaces bias
    # monitoring metrics (checks, biased answers, blocks, per-category
    # counts). ``result.bias`` is None when no monitor is configured or the
    # check was skipped (output guardrail / hallucination block already
    # replaced the answer with a canned refusal).
    bias_assessment = getattr(result, "bias", None)
    if bias_assessment is not None and metrics is not None:
        metrics.record_bias_check(
            biased=bias_assessment.biased,
            categories=bias_assessment.categories,
            blocked=getattr(result, "bias_blocked", False),
        )
    # Responsible AI: record the model drift report (when a drift monitor
    # ran) so the ``GET /api/v1/metrics`` snapshot surfaces drift alerts.
    drift_report = getattr(result, "drift", None)
    if drift_report is not None and metrics is not None:
        metrics.record_drift_check(
            report=drift_report,
            alert=getattr(result, "drift_alert", False),
        )
    # If the output guardrail blocked the streamed answer, tell the frontend
    # to swap the already-streamed tokens for the refusal (SSE is one-way,
    # so the deltas cannot be un-sent). Emitted before `sources` so the UI
    # replacement lands before citations/metadata are applied.
    replacement = getattr(result, "guardrail_replacement", None)
    if replacement:
        replacement_json = json.dumps({"answer": response.answer})
        yield f"event: guardrail_replacement\ndata: {replacement_json}\n\n"
    # Split the post-processed result into two named events so the frontend
    # can render citation chips (sources) and request-level metadata
    # (session id / confidence / trace id) independently.
    citations_json = (
        response.model_dump_json(include={"citations"})
        if response.citations
        else '{"citations":[]}'
    )
    yield f"event: sources\ndata: {citations_json}\n\n"
    metadata_json = response.model_dump_json(
        include={"session_id", "confidence", "trace_id"}
    )
    yield f"event: metadata\ndata: {metadata_json}\n\n"
    yield "event: done\ndata: [DONE]\n\n"


def build_replay_stream(
    cached: ChatResponse,
    session_id: str,
) -> Iterator[str]:
    """Replay a cached answer as SSE without invoking the orchestrator.

    The cached ``ChatResponse`` is rebuilt with the current ``session_id``
    so the frontend never sees another session's id. Emits the same named
    events as ``build_event_stream``: one ``delta`` with the full cached
    answer, then ``sources`` + ``metadata`` + ``done``.
    """
    replayed = ChatResponse(
        session_id=session_id,
        answer=cached.answer,
        citations=cached.citations,
        confidence=cached.confidence,
    )
    yield f"event: delta\ndata: {cached.answer}\n\n"
    citations_json = (
        replayed.model_dump_json(include={"citations"})
        if replayed.citations
        else '{"citations":[]}'
    )
    yield f"event: sources\ndata: {citations_json}\n\n"
    metadata_json = replayed.model_dump_json(
        include={"session_id", "confidence", "trace_id"}
    )
    yield f"event: metadata\ndata: {metadata_json}\n\n"
    yield "event: done\ndata: [DONE]\n\n"


@router.post("/chat")
def chat(
    request: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
    store: ConversationStore = Depends(get_conversation_store),
    cache: LRUCache = Depends(get_cache),
    metrics: MetricsCollector = Depends(get_metrics),
) -> StreamingResponse:
    """Stream a RAG answer for the user's message as Server-Sent Events."""
    query = request.message
    department = request.department
    role = request.role
    cache_key = _cache_key(query, department, role)

    # Resolve / create the session.
    session_id = request.session_id or store.create_session()
    if request.session_id and not store.session_exists(request.session_id):
        # An unknown session id was supplied — reject rather than silently
        # creating a phantom session that the client thinks exists.
        raise HTTPException(status_code=404, detail=f"Session {request.session_id!r} not found")

    # Tier 1 cache: exact normalized query match (scoped by department/role).
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit — replaying cached answer for session %s", session_id)
        metrics.record_cache_hit()
        metrics.record_request()
        return StreamingResponse(
            build_replay_stream(cached, session_id),
            media_type=SSE_MEDIA_TYPE,
            headers={"X-Cache": "HIT"},
        )

    metrics.record_cache_miss()
    history = store.format_history(session_id)
    return StreamingResponse(
        build_event_stream(
            orchestrator,
            query,
            history,
            session_id,
            store,
            cache,
            metrics,
            department,
            role,
            cache_key=cache_key,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers={"X-Cache": "MISS"},
    )
