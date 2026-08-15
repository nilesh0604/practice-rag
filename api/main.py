"""FastAPI application factory — app + middleware + router wiring.

Assembles the Step 4 serving layer (extended in Step 7 — Monitoring &
Hardening):

- **CORS** — allows the Vite dev server origin (default ``http://localhost:5173``)
  so the React frontend can call the API during development.
- **structlog JSON logging** — emits structured JSON log lines to stdout;
  falls back to the stdlib logger if structlog is unavailable.
- **Langfuse** — optional. If the ``langfuse`` package is importable and
  ``LANGFUSE_BASE_URL`` is set, the chat flow is traced; otherwise tracing is a
  no-op so the app runs fine with Langfuse down or absent (per the doc's
  "falls back gracefully if Langfuse is down"). Full Langfuse span wiring
  is in ``rag/orchestrator.py`` (Step 7); the enabled-flag seam is here.
- **Ollama warm-up** — on startup a tiny generation call pre-loads the
  model so the first real query is not slow (the doc's "warm-up call on
  startup" mitigation). Degrades to a warning if Ollama is unreachable.
- **Routers** — all endpoint groups mounted under ``/api/v1``, including
  the Step 7 ``/metrics`` and ``/health/ready`` endpoints.

Run with::

    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.observability import warm_up_ollama
from api.routes import chat, feedback, health, history, ingest, metrics

logger = logging.getLogger(__name__)

API_PREFIX: str = "/api/v1"
"""All endpoints live under this prefix (matches the architecture doc)."""

DEFAULT_FRONTEND_ORIGIN: str = "http://localhost:5173"
"""Default Vite dev server origin for CORS. Override with ``FRONTEND_ORIGIN``."""

OLLAMA_WARMUP_MODEL: str = os.getenv("OLLAMA_WARMUP_MODEL", "llama3.2:3b")
"""Model to pre-load on startup (matches the generator's dev model, D2)."""


def _configure_structlog() -> None:
    """Configure structlog for JSON output, falling back to stdlib on error."""
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    except Exception:  # noqa: BLE001 — structlog is optional at runtime
        logging.basicConfig(level=logging.INFO)


def _langfuse_enabled() -> bool:
    """True only if langfuse is importable AND ``LANGFUSE_BASE_URL`` is set."""
    if not os.getenv("LANGFUSE_BASE_URL"):
        return False
    try:
        import langfuse  # noqa: F401

        return True
    except ImportError:
        return False


def create_app() -> FastAPI:
    """Build the FastAPI app with middleware and all routers mounted."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup: warm up Ollama. Shutdown: flush Langfuse."""
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        warm_up_ollama(ollama_url=ollama_url, model=OLLAMA_WARMUP_MODEL)
        yield
        # Shutdown: flush any buffered Langfuse events.
        try:
            from api.deps import get_tracer

            get_tracer().flush()
        except Exception:  # noqa: BLE001
            pass

    _configure_structlog()

    app = FastAPI(
        title="practice-rag API",
        description="RAG knowledge assistant — FastAPI serving layer (Step 7).",
        version="0.7.0",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────
    allowed_origin = os.getenv("FRONTEND_ORIGIN", DEFAULT_FRONTEND_ORIGIN)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[allowed_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── routers ────────────────────────────────────────────────────────
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(metrics.router, prefix=API_PREFIX)
    app.include_router(chat.router, prefix=API_PREFIX)
    app.include_router(history.router, prefix=API_PREFIX)
    app.include_router(feedback.router, prefix=API_PREFIX)
    app.include_router(ingest.router, prefix=API_PREFIX)

    logger.info(
        "practice-rag API ready (langfuse=%s, cors_origin=%s)",
        _langfuse_enabled(),
        allowed_origin,
    )
    return app


app = create_app()
"""Module-level ASGI app for ``uvicorn api.main:app``."""
