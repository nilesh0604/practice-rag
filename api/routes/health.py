"""Health endpoints — liveness + readiness probes.

``GET /api/v1/health`` is a *liveness* probe (is the process up?) — cheap
and dependency-free so a slow downstream cannot flap the Docker Compose
healthcheck and cause restart loops. Used by the compose healthcheck.

``GET /api/v1/health/ready`` is a *readiness* probe (are Qdrant + Ollama
reachable?). It pings both downstreams with short timeouts and reports
their status. A 503 is returned when any dependency is down so a load
balancer / compose ``depends_on: condition: service_healthy`` can withhold
traffic — but the process itself stays up (liveness is independent of
readiness, per the doc's failure-mode table: "Qdrant down → FastAPI
returns 'Search unavailable'").
"""

from __future__ import annotations

import os

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from api.observability import check_ollama, check_qdrant

router = APIRouter()

SERVICE_VERSION: str = "0.7.0"
"""Service version surfaced in the health payload (Step 7 — Monitoring & Hardening)."""

DEFAULT_OLLAMA_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — always 200 while the process is serving."""
    return {"status": "ok", "service": "practice-rag-api", "version": SERVICE_VERSION}


@router.get("/health/ready")
def readiness() -> JSONResponse:
    """Readiness probe — checks Qdrant + Ollama reachability.

    Returns 200 when both dependencies are reachable, 503 otherwise. The
    body always includes per-dependency status so the operator can see
    *which* downstream is down without reading logs.
    """
    ollama_ok = check_ollama(DEFAULT_OLLAMA_URL)
    qdrant_ok = check_qdrant(DEFAULT_QDRANT_URL)
    body = {
        "status": "ready" if (ollama_ok and qdrant_ok) else "not_ready",
        "service": "practice-rag-api",
        "version": SERVICE_VERSION,
        "dependencies": {
            "ollama": {"url": DEFAULT_OLLAMA_URL, "ok": ollama_ok},
            "qdrant": {"url": DEFAULT_QDRANT_URL, "ok": qdrant_ok},
        },
    }
    code = status.HTTP_200_OK if (ollama_ok and qdrant_ok) else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=body, status_code=code)
