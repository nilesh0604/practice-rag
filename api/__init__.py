"""FastAPI serving layer — wraps the Step 3 RAG orchestrator in HTTP.

Exposes the architecture doc's five endpoints under ``/api/v1``:

    GET  /api/v1/health              — Docker Compose healthcheck
    POST /api/v1/chat                — SSE streaming RAG answer
    GET  /api/v1/history/{session_id} — conversation history
    POST /api/v1/feedback            — thumbs up/down + comment
    POST /api/v1/ingest              — trigger incremental re-index
"""
