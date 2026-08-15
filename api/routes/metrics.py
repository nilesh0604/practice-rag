"""``GET /api/v1/metrics`` — online serving metrics endpoint (Step 7).

Surfaces the in-memory ``MetricsCollector`` snapshot (request count, error
count, cache hit rate, TTFT mean/p50/p95) plus the LRU cache stats. The
offline metrics (retrieval recall@5, faithfulness, answer relevancy) come
from the Step 6 eval gate; this endpoint covers the *online* half of the
doc's "Metrics to Track" table.

Intended for a single-process deployment. For multi-process, export these
via Prometheus instead.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.cache import LRUCache
from api.deps import get_cache, get_metrics
from api.observability import MetricsCollector

router = APIRouter()


@router.get("/metrics")
def metrics(
    collector: MetricsCollector = Depends(get_metrics),
    cache: LRUCache = Depends(get_cache),
) -> dict:
    """Return online serving metrics + cache stats as a JSON document."""
    snapshot = collector.snapshot()
    snapshot["cache_store"] = cache.stats()
    return snapshot
