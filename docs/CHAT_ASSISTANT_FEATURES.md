# Chat Assistant Features — Current Status

> **Scope note:** The feature list below originates from the production AEM/Azure
> target described in [`ai-rag-chat-architecture-2026.md`](../ai-rag-chat-architecture-2026.md).
> This repository is the **local, $0-cost practice implementation** (Ollama +
> Qdrant + SQLite + React). Each feature is mapped to its **actual status in this
> codebase**, not the production target.
>
> **Status legend**
>
> - ✅ **Implemented** — working in this repo (may use a local substitute stack)
> - 🟡 **Partial** — implemented with simplifications/gaps vs. the listed spec
> - ❌ **Not implemented** — target-only; not present in this codebase

**Stack actually in use (vs. the listed Azure/AEM stack):**

| Layer         | Listed (production target)                    | This repo (practice)             |
| ------------- | --------------------------------------------- | -------------------------------- |
| LLM           | Azure OpenAI (GPT-3.5/4)                      | Ollama `llama3.2:3b` (host)      |
| Embeddings    | `text-embedding-ada-002` (1536d)              | Ollama `nomic-embed-text` (768d) |
| Search        | Azure AI Search (BM25+vector+semantic ranker) | Qdrant (dense + sparse + RRF)    |
| Session store | Redis (last 10) + Cosmos DB (90d)             | SQLite (single file)             |
| Cache         | Redis + embedding + semantic cache            | In-memory LRU (exact match)      |
| Portal        | AEM SPA Editor                                | Standalone React + Vite SPA      |

---

## Core Chat Features

### Conversational Q&A — 🟡 Partial

- ✅ Natural-language Q&A over a documentation corpus (FastAPI / Pydantic v2 / SQLModel — _not_ AEM policies/FAQs/org docs).
- ✅ LLM-generated answers with inline source citations.
- ✅ Citations carry `title` + `source_url` + `snippet` (truncated chunk excerpt) + `relevanceScore` (fused RRF score) + `lastModified` (source freshness). See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/schemas/chat.py" /> (`Citation` model), <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/rag/post_processor.py" /> (`extract_citations`, `_make_snippet`), and <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/frontend/src/CitationChip.jsx" />.
- ✅ Multi-turn conversation support via `sessionId` + last-10-turns window passed to the generator. See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/conversation.py" /> (`format_history`, `HISTORY_WINDOW=10`).
- 🟡 Pronoun/coreference resolution (e.g. "that" → "parental leave") is handled by the LLM query rewriter (`LLMQueryRewriter` in <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/rag/query_rewriter.py" />), not a dedicated anaphora resolver. A **latency-skip heuristic** (`_needs_rewrite`) short-circuits the ~1-3s Ollama round-trip for self-contained queries (no anaphoric pronouns, or no conversation history), so only follow-up queries that actually need coreference resolution pay the rewrite cost.
- ❌ **No Redis** (SQLite instead) and **no Cosmos DB** — full history lives in SQLite with no TTL/90-day retention policy. No 30-min sliding window; the window is a fixed turn count.

### Query Classification & Routing — 🟡 Partial

- ✅ Factual Q&A → full RAG pipeline (`documentation` / `compare` / `follow_up` labels). See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/guardrails.py" /> (`QueryClassifier`).
- ✅ Greeting/chitchat → direct canned response, no retrieval (`greeting` label, `handled=True`).
- ✅ Out-of-scope → polite rejection (`off_topic` label, canned answer).
- ❌ **Complex/multi-part → query decomposition into sub-queries** — not implemented. A `compare` label exists but no sub-query splitting.
- ❌ **Ambiguous → clarification prompt** (e.g. "Do you mean parental leave, sick leave, or PTO?") — not implemented.

### Streaming UX — 🟡 Partial

- ✅ SSE streaming via `text/event-stream` with named events. See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/routes/chat.py" /> (`build_event_stream`).
- ✅ SSE frame format uses the listed named events: `event: delta` (one per token), `event: guardrail_replacement` (refusal JSON `{"answer": ...}`, only when the output guardrail blocks, once after the last delta), `event: sources` (citations JSON, once after the last token), `event: metadata` (session id + confidence + trace id, once), `event: done` (terminal `[DONE]` sentinel). The full `ChatResponse` (including the concatenated `answer`) is still persisted to the session store + LRU cache; only the wire format drops the redundant `answer` (the frontend reconstructs it from deltas).
- ✅ Typing/streaming indicator — the frontend renders a streaming state via the `streaming` flag on the assistant message bubble (blinking cursor), not a dedicated `TypingIndicator` component. See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/frontend/src/ChatWidget.jsx" />.
- ✅ Stream-then-verify pattern: input guardrails run pre-stream; output guardrails run on the full buffer post-stream. On output block, the **persisted** answer is replaced with a refusal **and** an `event: guardrail_replacement` SSE event is emitted so the frontend swaps the already-streamed UI message for the refusal (SSE is one-way, so the original tokens cannot be un-sent — they are replaced client-side). See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/rag/orchestrator.py" /> (output guardrail block, sets `PostProcessResult.guardrail_replacement`) and <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/routes/chat.py" /> (`build_event_stream` emits the frame).
- ❌ **TTFT < 800ms target** — not measured/asserted as an SLO; TTFT is collected by `MetricsCollector` but there is no threshold gate.
- ❌ **Sensitive-topic queries: streaming disabled, full response guardrailed before delivery** — not implemented. All queries stream the same way.

---

## React Chat Widget UI — 🟡 Partial

Implemented components (see <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/frontend/src/" />):

| Listed component      | Status | Actual component(s)                                                |
| --------------------- | ------ | ------------------------------------------------------------------ |
| `ChatAssistantWidget` | 🟡     | `ChatWidget` (root, not `position: fixed`)                         |
| `ChatBubble`          | ❌     | No floating trigger button                                         |
| `ChatPanel`           | ❌     | Always-visible panel (no expand/collapse)                          |
| `MessageHistory`      | 🟡     | `MessageList`                                                      |
| `MessageInput`        | 🟡     | `InputBox`                                                         |
| `MessageBubble`       | 🟡     | Rendered inline within `MessageList`                               |
| `SourceCard`          | 🟡     | `CitationChip` (chip with snippet + score + date, not a full card) |
| `FeedbackButtons`     | 🟡     | Thumbs up/down rendered inside `MessageList`                       |
| `TypingIndicator`     | 🟡     | Streaming flag / blinking cursor (no separate comp.)               |
| `ErrorBoundary`       | ✅     | `ErrorBoundary`                                                    |

- 🟡 State is managed via `useState` + `useCallback`, **not `useReducer`** with the listed action types (`TOGGLE_PANEL`, `STREAM_DELTA`, etc.). See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/frontend/src/ChatWidget.jsx" />.
- ❌ **Not embedded into AEM portal** — it is a standalone Vite SPA served via `npm run dev` / `npx serve -s dist`. No SPA Editor integration, no page-template `<script>` tag.

---

## Retrieval Features — 🟡 Partial

- ✅ Hybrid search: **Qdrant** dense (nomic-embed-text) + sparse (BM25/IDF hashing) + **RRF fusion**. See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/rag/retriever.py" />.
- ❌ **Not Azure AI Search** and **no semantic ranker**.
- 🟡 Embeddings are `nomic-embed-text` (768d), **not `text-embedding-ada-002` (1536d)**.
- ✅ Relevance scores normalized to 0–1 (RRF score divided by theoretical max `2/(k+1)`). See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/rag/retriever.py" /> (`_rrf_fuse`).
- ❌ **Configurable `maxSources` per request** — `top_k` is a retriever constructor constant (`DEFAULT_TOP_K=5`), not a per-request option on `ChatRequest`.
- ❌ **Department-scoped retrieval via `department` option** — not implemented. `ChatRequest` has no `department` field; retriever has no filter.

---

## APIs — 🟡 Partial

| Listed endpoint                                                      | Status | Actual                                                                                                                                                                                                                                        |
| -------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /api/v1/chat` (streaming + non-streaming)                      | 🟡     | Streaming SSE only. **No non-streaming JSON mode.** <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/routes/chat.py" />                                                                                                           |
| `POST /api/v1/feedback` (rating, comment, category)                  | 🟡     | `rating` (up/down) + optional `comment` + optional `trace_id`. **No `category` field.** <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/schemas/feedback.py" />                                                                      |
| `GET /api/v1/history?sessionId=...`                                  | 🟡     | Implemented as `GET /api/v1/history/{session_id}` (path param, not query). <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/routes/history.py" />                                                                                 |
| `GET /api/v1/health` (versioned, dependency status, version, uptime) | 🟡     | `GET /api/v1/health` (liveness: status/service/version) + `GET /api/v1/health/ready` (Qdrant+Ollama dependency status, 503 when down). **No uptime field.** <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/routes/health.py" /> |

Additional endpoints not in the listed spec but present: `POST /api/v1/ingest`, `GET /api/v1/metrics`.

---

## Guardrails (User-Facing Safety) — 🟡 Partial

- ✅ **Input guardrails (synchronous, pre-stream):** prompt-injection regex + optional local LLM judge. See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/guardrails.py" /> (`InputGuardrail`, `detect_prompt_injection`).
  - ❌ **PII detection on input** — not implemented (PII scrub is output-only).
  - ❌ **Content safety on input** — the input LLM judge classifies injection, not general content safety.
- ✅ **Output guardrails (post-stream on full buffer):** PII scrub (always) + optional harmful-content LLM judge. See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/guardrails.py" /> (`OutputGuardrail`, `scrub_pii`).
  - 🟡 **Hallucination check (cosine similarity grounding)** — exists as the post-processor's groundedness/confidence score (<ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/rag/post_processor.py" /> `compute_groundedness`), but it produces a **low-confidence UI warning**, not an output block.
- 🟡 **Guardrail replacement:** on output block, the persisted + post-processed answer is replaced with a safe refusal. **No `guardrail_replacement` SSE event** swaps the streamed UI message; the original streamed tokens are not un-sent and are not discarded from the UI (only from persisted history). See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/rag/orchestrator.py" />.
- ❌ **Sensitive-topic queries: streaming disabled, full response guardrailed before delivery (no partial exposure)** — not implemented.

> **F500 enterprise gap:** the guardrail judge uses a 3B local model with no eval gate/SLO, no guardrail metrics, and no red-team suite. Tracked in <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/docs/F500_ENTERPRISE_ACTION_ITEMS.md" />.

---

## Responsible AI Features — mostly ❌

| Feature                                                                      | Status | Notes                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transparency & AI disclosure                                                 | ❌     | No explicit "AI-generated" disclosure in the UI.                                                                                                                                                                                         |
| Bias & fairness monitoring                                                   | ❌     | Not implemented.                                                                                                                                                                                                                         |
| Human-in-the-Loop (HITL) escalation                                          | ❌     | No escalation path to human agents.                                                                                                                                                                                                      |
| AI governance & audit (full decision-chain audit log)                        | 🟡     | Langfuse traces (retrieval/generation/guardrail spans) + feedback scores act as a lightweight audit trail, but there is no formal immutable audit log. <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/observability.py" /> |
| Model drift detection                                                        | ❌     | Not implemented.                                                                                                                                                                                                                         |
| User consent, control & data rights (GDPR Art. 17 right-to-erasure endpoint) | ❌     | No erasure endpoint. SQLite history has no deletion API.                                                                                                                                                                                 |
| Golden dataset evaluation (offline metrics pipeline)                         | ✅     | `eval/golden-dataset.json` (36 Q&A pairs) + `eval/run_eval.py` (Ragas + local LLM judge fallback, threshold gate). <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/eval/run_eval.py" />                                         |
| Responsible AI dashboard (Azure Workbook)                                    | ❌     | Replaced by Langfuse UI + `GET /api/v1/metrics` (TTFT, cache hit rate, request/error counts).                                                                                                                                            |
| User feedback loop → retrieval-quality iteration                             | 🟡     | Feedback is recorded + attached as Langfuse scores. **No automated "weekly prompt + chunking + re-index" workflow.**                                                                                                                     |

---

## Performance & Cost Features — 🟡 Partial

- 🟡 **Multi-tier cache:** only **Tier 1** in-memory LRU exact-match cache exists. **No embedding cache, no AI Search semantic cache.** Hit rate is tracked via `MetricsCollector` but the 35–45% claim is not asserted. See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/cache.py" />.
- ❌ **GPT-3.5 Turbo fallback for simple queries** — Ollama only; no secondary simpler model. The circuit breaker degrades to an error, not a fallback model.
- 🟡 **Sub-100ms responses for cached queries** — cache replay skips retrieval+generation (fast), but the latency target is not measured/asserted.
- ✅ **Circuit breaker for degraded-mode fallback** — `CircuitBreaker` wraps Ollama generation calls (opens after 3 consecutive failures, half-opens after 30s). See <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/api/observability.py" />.

---

## RBAC — ❌ Not implemented

- ❌ **Content filtering by department/role** (e.g. `department: "engineering"`) — not implemented. `ChatRequest` has no department/role field; the retriever applies no payload filter; there is no auth/identity layer.

---

## Summary

| Category             | ✅ Implemented | 🟡 Partial | ❌ Not implemented |
| -------------------- | -------------- | ---------- | ------------------ |
| Core Chat            | 5              | 1          | 3                  |
| React Chat Widget UI | 1              | 8          | 2                  |
| Retrieval            | 2              | 1          | 3                  |
| APIs                 | 0              | 4          | 0 (within scope)   |
| Guardrails           | 2              | 2          | 3                  |
| Responsible AI       | 1              | 2          | 6                  |
| Performance & Cost   | 1              | 2          | 1                  |
| RBAC                 | 0              | 0          | 1                  |

**Bottom line:** The conversational RAG core (Q&A + citations + multi-turn + streaming + hybrid retrieval + basic guardrails + golden-dataset eval + circuit breaker + LRU cache) is **implemented** using a local substitute stack. The **enterprise/production features** (RBAC/department filtering, Azure services, Redis/Cosmos tiering, semantic cache, HITL escalation, GDPR erasure, model drift, AI disclosure, query decomposition, ambiguity clarification, sensitive-topic stream disabling, guardrail-replacement SSE event) remain **target-only** and are documented as future action items in <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/docs/F500_ENTERPRISE_ACTION_ITEMS.md" />.
