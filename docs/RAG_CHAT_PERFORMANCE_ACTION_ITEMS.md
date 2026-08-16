# RAG Chat Assistant — Performance Action Items

**Date:** 2026-08-16  
**Source:** Langfuse MCP `queryMetrics` over the last 30 days (108 production-style `chat` calls)

> **F500 production-grade note:** A p95 end-to-end latency of 7.34 s is well above the typical enterprise chat SLO of p95 < 2–3 s. The items below are prioritized by impact and are safe to defer until the next performance sprint.

## Observed performance

| Component | Calls | Avg | p50 | p95 |
|---|---:|---:|---:|---:|
| `chat` (end-to-end) | 108 | 3.58 s | 2.86 s | 7.34 s |
| `retrieval` | 110 | 68 ms | 50 ms | 138 ms |
| `rerank` | 36 | 218 ms | 222 ms | 259 ms |
| `generation` | 109 | 1.75 s | 1.06 s | 5.45 s |
| `guardrail_input` | 108 | 765 ms | 728 ms | 940 ms |
| `guardrail_output` | 108 | 256 ms | 244 ms | 314 ms |

### Telemetry gaps

- `totalCost` and `totalTokens` are reported as `0` for all observations, so cost/token efficiency cannot be assessed yet.
- Score data (quality, user feedback) could not be retrieved because the Langfuse MCP connection became intermittent after the first query.

## Action items

### 1. Reduce end-to-end `chat` p95 latency (7.34 s)

**Reasoning:** The end-to-end span is what the user actually feels. A p95 above 7 s will hurt engagement and is far outside enterprise interactivity SLOs (p95 < 2–3 s).

**Action:**
1. Set an SLO for the `chat` span: p95 < 3 s for v1.
2. Profile the full trace; the largest known contributors are `generation` (p95 5.45 s) and `guardrail_input` (p95 940 ms).
3. After each fix, re-run the same `queryMetrics` query and compare p95 / p99.

---

### 2. Optimize the `generation` span

**Reasoning:** `generation` is the biggest single time sink (avg 1.75 s, p95 5.45 s). Everything else in the pipeline is sub-1 s, so reducing generation time has the largest impact on `chat` latency.

**Action:**
1. Capture real `usage` (input/output tokens) and `providedModelName` on every `generation` observation in `rag/orchestrator.py` / `api/observability.py`. This will also populate `totalTokens` in Langfuse.
2. Track `timeToFirstToken` separately from total generation time to know whether the delay is model loading, prompt processing, or token streaming.
3. Evaluate a smaller/faster local model, a remote endpoint with SLAs, or enabling streaming (`stream=true`) with a first-token SLO.

---

### 3. Reduce `guardrail_input` latency

**Reasoning:** The input guardrail is a full LLM judge and adds ~765 ms to every single chat. Enterprise deployments typically use a small, purpose-trained classifier or a managed API that returns in < 100 ms.

**Action:**
1. In `api/guardrails.py`, make sure the fast regex/keyword pre-filter runs first and only invokes the LLM judge on ambiguous or high-risk inputs.
2. Consider caching guardrail verdicts for identical or near-identical queries.
3. Evaluate a smaller guardrail model (e.g., `llama3.2:1b`) or a quantized classifier for the input-judge role.

---

### 4. Enable token and cost tracking in Langfuse

**Reasoning:** Without token and cost data, we cannot optimize spend, set rate limits, or compare model efficiency.

**Action:**
1. Update the Langfuse `generation` observation creation to include `usage={"input": n, "output": n, "total": n}`.
2. Add the model cost mapping in the Langfuse project settings so `totalCost` / `inputCost` / `outputCost` are computed automatically.

---

### 5. Add and monitor quality scores

**Reasoning:** Latency is only half of performance. Groundedness, relevance, hallucination, and user feedback are required to detect quality regressions that often trade off with latency.

**Action:**
1. Emit Langfuse scores from the existing `rag/post_processor.py` outputs (`groundedness`, `relevance`, `bias`, `drift_alert`) and from the user-feedback endpoint.
2. Use `queryMetrics` with view `scores-numeric` to trend these weekly and set alert thresholds.

## Validation checklist

- [ ] Re-run `queryMetrics` after each fix and compare p95 / p99 `chat` latency.
- [ ] Confirm `totalTokens` and `totalCost` are non-zero in the Langfuse dashboard.
- [ ] Add a CI or local performance regression test that calls `/api/v1/chat` 50× and asserts p95 < 3 s.
