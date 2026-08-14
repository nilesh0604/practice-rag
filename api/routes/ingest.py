"""``POST /api/v1/ingest`` — trigger an incremental re-index of the corpus.

Delegates to the Step 2 ``run_ingestion`` orchestrator. By default an
incremental sync is run (only changed files, per the manifest); pass
``full_reindex=true`` to drop and recreate the collection first.

This is a *blocking* endpoint: for a single-user practice project the
ingestion of ~7 docs finishes in well under a minute, so a synchronous
response returning the summary dict is the simplest honest behaviour. A
production system would dispatch this to a background job queue and return
a job id; that gap is documented in the Enterprise Gap Register.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.deps import get_corpus_dir

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestRequest(BaseModel):
    """Optional request body for the ingest endpoint."""

    model_config = ConfigDict(extra="forbid")

    full_reindex: bool = Field(
        default=False,
        description="Drop and recreate the collection; re-index every file.",
    )
    corpus_dir: str | None = Field(
        default=None,
        description="Override the corpus directory (defaults to CORPUS_DIR env).",
    )


class IngestResponse(BaseModel):
    """Summary of the ingestion run."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    summary: dict[str, Any]


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest = IngestRequest(),
    corpus_dir: str = Depends(get_corpus_dir),
) -> IngestResponse:
    """Run the ingestion pipeline (incremental by default)."""
    from ingestion.run import run_ingestion

    target = request.corpus_dir or corpus_dir
    try:
        summary = run_ingestion(target, full_reindex=request.full_reindex)
    except Exception as exc:  # noqa: BLE001 — surface any pipeline failure as 500
        logger.exception("Ingestion failed for %s", target)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    return IngestResponse(status="ok", summary=summary)
