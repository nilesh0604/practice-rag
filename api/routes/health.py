"""``GET /api/v1/health`` — liveness probe for the Docker Compose healthcheck.

Built first (per the build order) so the compose healthcheck is honest
before any other endpoint exists. Returns a small JSON document with the
service status and version. This is a *liveness* check (is the process up?)
not a *readiness* check (are Qdrant/Ollama reachable?) — keeping it cheap
and dependency-free so a slow downstream cannot flap the healthcheck and
cause compose restart loops.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

SERVICE_VERSION: str = "0.4.0"
"""Service version surfaced in the health payload (matches the Step 4 build)."""


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — always 200 while the process is serving."""
    return {"status": "ok", "service": "practice-rag-api", "version": SERVICE_VERSION}
