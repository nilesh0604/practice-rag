# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Renamed project identifier from `ctc-rag` to `practice-rag` across README repo layout, `docker-compose.yml` container names (qdrant, postgres, clickhouse, redis, minio, langfuse-worker, langfuse-web), and frontend `package.json` / `package-lock.json` package name.

### Added

- `README.md` with project overview, stack, repo layout, quick start, build order, testing strategy, API surface, observability, cost, interview discussion points, and project conventions.
- `.gitignore` covering macOS system files, Python artifacts, Node modules, environment files, and Colima volumes.
- `requirements.txt` with the Python dependency set from the architecture doc (FastAPI, LangChain, Ollama, Qdrant, Langfuse, Ragas, pytest).
- `docker-compose.yml` with Qdrant, Postgres, ClickHouse, Redis, MinIO, Langfuse web + worker — all with healthchecks and minimal resource limits (1 CPU / 1 GB RAM; Langfuse web+worker at 2 GB RAM).
- `.env.example` with all Langfuse secrets documented (CHANGEME placeholders).
- `frontend/` scaffold: `package.json` (Vite + React 18 + Jest + RTL), `vite.config.js`, `index.html`, `src/main.jsx`, `src/App.jsx` stub, `jest.setup.js`, and `__tests__/App.test.jsx`.
- Reference to the full design document `ai-rag-chat-architecture-2026.md`.
- Conda env `rag-chat` (Python 3.12) created via Miniforge; all `requirements.txt` deps installed.
- Colima started with `--cpu 4 --memory 8 --disk 100`; full Docker stack verified healthy.

### Added — Step 1 (Shared contracts + Qdrant collection helper)

- `schemas/` package: the single Pydantic v2 contract seam between the ingestion pipeline and the serving layer.
  - `schemas/documents.py` — `DocumentChunk` (id, content, title, source_url, section, last_modified, chunk_index, parent_doc_id, optional 768-d embedding; naive datetimes normalized to UTC; `to_payload()` for Qdrant payload) and `RetrievedDoc` (retriever output with fused score in [0, 1]).
  - `schemas/chat.py` — `ChatRequest` (message bounded 1–4000 chars, optional session_id), `ChatResponse` (answer + citations + confidence), `Citation` (title + source_url); citations auto-deduplicated by URL.
  - `schemas/feedback.py` — `FeedbackRequest` (session_id, message_index ≥ 0, rating `up`/`down`, optional comment ≤ 2000 chars).
  - `schemas/__init__.py` re-exports all contracts.
- `rag/` package with Qdrant collection helper `rag/qdrant_collection.py`:
  - `CollectionConfig` dataclass encoding the architecture doc's `docs-knowledge` schema (768-d Cosine dense vector named `dense`, sparse vector named `text` with `on_disk: false`, HNSW `m=16` / `ef_construct=128`).
  - `create_collection()` — idempotent creation (skips if exists); `recreate=True` drops first for full re-indexes.
  - `ensure_collection()` — convenience wrapper that builds a client from `QDRANT_URL` (default `http://localhost:6333`) and creates the collection if missing.
  - `HNSW_EF=64` exposed as a constant for the retriever to use at query time (it is a search-time param, not a build-time `HnswConfigDiff` field).
- `pyproject.toml` with pytest config (`pythonpath=["."]`, `testpaths=["tests"]`, `asyncio_mode=auto`, strict markers, warnings-as-errors with qdrant-client deprecations and unhandled-thread warnings ignored).
- `tests/test_schemas.py` — 19 contract tests (validation, bounds, UTC normalization, payload exclusion, citation dedup, rating literal).
- `tests/test_qdrant_collection.py` — 8 tests using a mocked `QdrantClient` (config defaults, create-when-missing, skip-when-exists, recreate drops-then-creates, schema field assertions, `QDRANT_URL` env resolution).
- All 27 tests pass in the `rag-chat` conda env (`python -m pytest -q`).
- Smoke-tested `ensure_collection()` against the live Qdrant container: created `docs-knowledge` with the exact documented schema (768-d Cosine, sparse `text`, HNSW m=16/ef_construct=128), confirmed idempotent re-run skips, then deleted the collection to leave the store pristine for Step 2.

### Notes — Step 1

- **qdrant-client version mismatch:** installed `qdrant-client` 1.19.0 vs server `qdrant/qdrant:v1.11.3` in `docker-compose.yml`. Collection creation works, but the client emits a compatibility warning. Revisit before Step 2 ingestion — either pin `qdrant-client==1.11.*` in `requirements.txt` to match the server, or bump the server image to a 1.18/1.19 tag. Documented here so it is not lost.
- The architecture doc's collection schema JSON lists `ef: 64` alongside `m`/`ef_construct`. In qdrant-client, `ef` is a **search-time** parameter (passed per query), not a field on the build-time `HnswConfigDiff`. The helper keeps `HNSW_EF=64` as a constant for the Step 3 retriever to consume; collection creation sets only `m` and `ef_construct`. This is a clarification of the doc, not a deviation from its intent.

### Added — Step 2 (Ingestion pipeline)

- `ingestion/` package — the offline pipeline that converts a documentation corpus into embedded chunks stored in Qdrant.
  - `ingestion/sync.py` — `discover_files()` finds supported source files (`.md`, `.html`, `.pdf`) under a corpus directory, skipping `.git`/`node_modules`/hidden dirs. `sync_from_git()` helper for future git-clone ingestion.
  - `ingestion/parser.py` — `parse_file()` dispatches to format-specific parsers (Markdown with YAML frontmatter, HTML via BeautifulSoup, PDF via pypdf). Produces `ParsedDocument` with title, source_url, section, last_modified (UTC), and deterministic `parent_doc_id`.
  - `ingestion/chunker.py` — `chunk_document()` two-stage split: `MarkdownHeaderTextSplitter` (H1/H2/H3 → section metadata) then `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (512 tokens, 64 overlap, `cl100k_base`). Chunk IDs are deterministic UUIDs (SHA-256 of `parent_doc_id:chunk_index` → UUID) so re-indexes overwrite rather than duplicate.
  - `ingestion/embedder.py` — `Embedder` class wraps Ollama `nomic-embed-text` for batched 768-d dense embeddings. `sparse_embed_text()` generates BM25-style sparse vectors via a dependency-free hashing trick (`zlib.crc32` → index, `1 + log(tf)` → weight), with Qdrant's `Modifier.IDF` applying IDF weighting at query time. Shared `tokenize()` function for index-time and query-time use.
  - `ingestion/index_writer.py` — `IndexWriter` class wraps QdrantClient for batched upserts (dense + sparse + payload) and stale-chunk deletion by `parent_doc_id` filter (incremental re-index strategy from the architecture doc).
  - `ingestion/manifest.py` — `Manifest` dataclass + `file_hash()` (SHA-256), `save_manifest()`/`load_manifest()` (JSON). Tracks per-file hash, last-indexed timestamp, and chunk count for incremental sync.
  - `ingestion/run.py` — `run_ingestion()` orchestrator: discover → hash-compare against manifest → parse → chunk → embed (dense) → upsert (dense + sparse). Supports `--full-reindex` (drops + recreates collection). Prunes deleted files from manifest. CLI entrypoint via `python -m ingestion.run`.
- `data/corpus/` seed corpus — 7 Markdown files across 3 doc sets: `fastapi/` (query-params, path-params, dependency-injection), `pydantic/` (models, types), `sqlmodel/` (intro, relationships). Each has YAML frontmatter with title + source_url.
- Updated `rag/qdrant_collection.py` — added `Modifier.IDF` to the sparse vector config (`SPARSE_MODIFIER` constant + `CollectionConfig.sparse_modifier` field) so Qdrant applies BM25-style IDF weighting at query time.
- `tests/test_ingestion_manifest.py` — 13 tests (file hashing, manifest CRUD, stale-path detection, JSON roundtrip).
- `tests/test_ingestion_sync.py` — 9 tests (file discovery, nested dirs, skip dirs, sorted output, supported extensions).
- `tests/test_ingestion_parser.py` — 12 tests (frontmatter extraction, H1 fallback, parent_doc_id determinism, UTC normalization, source_url heuristics, HTML parsing, dispatch).
- `tests/test_ingestion_chunker.py` — 12 tests (single/multiple chunks, deterministic UUIDs, sequential indices, section metadata, empty content, overlap constants).
- `tests/test_ingestion_embedder.py` — 22 tests (tokenizer, sparse vector determinism/bounds/term-frequency weighting, dense embedder with mocked Ollama, dimension validation, lazy client).
- `tests/test_ingestion_index_writer.py` — 12 tests (upsert single/batch/empty, payload exclusion, sparse generation from content, missing-embedding error, delete-by-parent, 404 handling).
- `tests/test_ingestion_run.py` — 7 integration tests (full first run, incremental skip, changed-file re-index, full re-index, manifest persistence, empty corpus, deleted-file pruning).
- All 114 tests pass in the `rag-chat` conda env (`python -m pytest -q`).

### Notes — Step 2

- **Sparse vector strategy:** the architecture doc mentions SPLADE for sparse vectors, but running a SPLADE model locally adds complexity and RAM for a practice project. Instead, the embedder generates BM25-style sparse vectors client-side using a dependency-free hashing trick (token → `zlib.crc32` index, `1 + log(tf)` weight), and Qdrant's `Modifier.IDF` applies inverse-document-frequency weighting at query time. This produces proper BM25-like scoring without a separate model, and the same `tokenize()` function is shared between ingestion and the retriever (Step 3). This is a **deviation** from the doc's SPLADE mention — documented here for interview defensibility.
- **Chunk IDs as UUIDs:** Qdrant 1.11.3 requires point IDs to be either unsigned integers or UUIDs (16-char hex strings are rejected with a 400). Chunk IDs are therefore deterministic UUIDs derived from `SHA-256(parent_doc_id:chunk_index)[:32]` → `uuid.UUID()`. This keeps re-indexes idempotent (same source → same UUID → upsert overwrites) while satisfying Qdrant's ID format.
- **qdrant-client version mismatch (updated):** the mismatch (client 1.19.0 vs server 1.11.3) noted in Step 1 did not block ingestion. The `check_compatibility=False` flag is used in smoke tests to suppress the warning. The collection/upsert/search APIs are compatible across these versions. Still recommended to pin or bump before production.
- **Smoke test results:** full re-index of 7 Markdown files → 45 chunks in Qdrant. Dense search for "How do I declare path parameters in FastAPI?" correctly ranks "Path Parameters - FastAPI" first (score 0.88). Sparse (BM25+IDF) search also ranks it first (score 10.30). Incremental re-run skips all 7 unchanged files. Collection deleted after smoke test to leave the store pristine for Step 3.

### Changed

- Ollama runs on the **host** (not in Docker) to avoid port conflicts and leverage host Metal acceleration. The `docker-compose.yml` no longer includes an Ollama service.
- Generation model changed from `llama3.1:8b` to `llama3.2:3b` (already available on the host; lighter and faster for dev). The architecture doc's `llama3.1:8b` remains the documented target — `llama3.2:3b` is the current dev substitute.
- Langfuse web + worker memory limit raised from 1 GB to 2 GB (Next.js + Prisma OOM at 1 GB during migrations).
- README Quick Start updated to reflect host Ollama + `llama3.2:3b`.

### Documented

- Added "Phase 0 Implementation Decisions" section to `ai-rag-chat-architecture-2026.md` documenting four deviations from the original design (host Ollama, llama3.2:3b, Langfuse 2 GB RAM, Miniforge install) with reasoning and revert plans.

### Notes

- Repository is in **Step 4 — FastAPI serving layer** (complete). Steps 0–4 are done. The next step is Step 5 — React frontend (Vite + React 18 scaffold, ChatWidget → MessageList + InputBox + SendButton, SSE consumer, CitationChip, feedback buttons).
- All health endpoints verified: Ollama `localhost:11434`, Qdrant `localhost:6333/healthz`, Langfuse `localhost:3000` (HTTP 200).

### Added — Step 3 (Core RAG orchestrator)

- `rag/` package — the online RAG flow as plain Python (unit-testable, no FastAPI dependency):
  - `rag/retriever.py` — `HybridRetriever` runs two Qdrant `query_points` calls (dense via `nomic-embed-text`, sparse via the shared `sparse_embed_text` hashing trick from Step 2) and fuses them client-side with Reciprocal Rank Fusion (RRF, k=60). Returns top-K=5 `RetrievedDoc` objects with scores normalized to [0, 1]. RRF is done in Python (not Qdrant's native `RrfQuery`) so the fusion math is unit-testable without a live Qdrant. Prefetch limit=20 per branch so RRF has enough overlap signal to reorder.
  - `rag/context_assembler.py` — `ContextAssembler` formats retrieved chunks into the numbered, delimited CONTEXT block for the system prompt (`[n] Title — section` headers, soft 12000-char cap, `(No relevant context found.)` placeholder when empty).
  - `rag/generator.py` — `Generator` wraps the Ollama `chat` API with the architecture doc's generation settings (`temp 0.1`, `num_ctx 8192`, `top_p 0.9`, `num_predict 512`) and the documented system prompt template (`build_system_prompt()` fills `{context}` + `{history}`). Streams tokens via `stream=True`; empty content chunks are skipped. Model defaults to `llama3.2:3b` (Phase 0 dev substitute, D2); `GENERATION_MODEL_TARGET="llama3.1:8b"` documents the production target.
  - `rag/post_processor.py` — `PostProcessor` + pure functions: `extract_citations()` parses `[Source: title]` markers and matches them case-insensitively to retrieved docs (hallucinated titles with no matching doc are dropped; citations deduplicated by URL); `compute_groundedness()` implements the doc's pseudocode — embeds the answer + each chunk, returns max cosine similarity, clamped to [0, 1]. `CONFIDENCE_THRESHOLD=0.65` (per the doc) gates the low-confidence flag. `PostProcessResult` dataclass carries answer + citations + confidence + low_confidence.
  - `rag/query_rewriter.py` — `QueryRewriter` protocol with two implementations: `PassthroughQueryRewriter` (default, zero-latency, returns query unchanged) and `LLMQueryRewriter` (Ollama-backed reformulation with a short prompt, falls back to the original query on error/empty response). The default orchestrator uses passthrough; LLM rewriting is opt-in.
  - `rag/orchestrator.py` — `RAGOrchestrator` wires all five components: `stream_answer()` yields `str` tokens then a final `PostProcessResult`; `answer()` is a non-streaming convenience that collects tokens and returns `(answer, result, docs)`. All collaborators are injected for full mockability.
  - Updated `rag/__init__.py` to re-export all new components.
- `tests/test_rag_retriever.py` — 19 tests (top-K, sorting, RRF normalization, doc-in-both-lists ranks higher, field mapping, empty/dense-only/sparse-only results, embedder called with query, prefetch limit, custom RRF k, dedup by id, defaults).
- `tests/test_rag_context_assembler.py` — 13 tests (empty placeholder, numbering, section in header, max_chunks limit, separator, content stripping, max_chars soft cap, first-chunk-over-cap edge case, defaults).
- `tests/test_rag_generator.py` — 16 tests (token yielding, empty-content skipping, message roles, stream flag, generation settings, model passthrough, lazy client, close, custom settings, concatenated tokens, system prompt template).
- `tests/test_rag_post_processor.py` — 28 tests (citation matching/dedup/case-insensitivity/unmatched-dropped/order, cosine similarity edge cases, groundedness max/perfect/no-match/empty/clamp, PostProcessor result + low-confidence flag, threshold value).
- `tests/test_rag_query_rewriter.py` — 15 tests (passthrough identity/empty/history-ignored, LLM rewrite first-line-only/strip/fallback-on-empty/fallback-on-error/history-in-prompt/stream-false/close/lazy-client, prompt template placeholders).
- `tests/test_rag_orchestrator.py` — 14 tests (tokens-then-result ordering, token count, single result at end, rewriter/retriever/assembler/generator/post-processor call args, default passthrough rewriter, empty token stream, non-streaming `answer()` method).
- All 219 tests pass in the `rag-chat` conda env (`python -m pytest -q`).

### Notes — Step 3

- **RRF in Python vs Qdrant native:** qdrant-client 1.19.0 supports `query_points` with a native `RrfQuery` (server-side fusion), but doing RRF client-side keeps the fusion constant (`k=60`) explicit and the math unit-testable without a live Qdrant. The two-`query_points`-calls approach also lets each branch use its own `SearchParams` (dense gets `hnsw_ef=64`). This is a testability choice, not a performance concern for a single-user practice project.
- **`query_points` vs `search`:** qdrant-client 1.19.0 removed the legacy `search()` method in favor of the unified `query_points()` API. The retriever uses `query_points` with `using=<vector_name>` for both dense and sparse branches.
- **Groundedness uses the same embedder:** `compute_groundedness()` reuses the `nomic-embed-text` `Embedder` (injected) to embed the answer and chunks. This is faithful to the architecture doc's pseudocode. In production this adds one embed call per answer; for a single-user practice project the cost is negligible. The embedder is mocked in tests.
- **Citation matching is title-based, not position-based:** the doc's system prompt asks the LLM to cite `[Source: title]`. The post-processor matches these titles case-insensitively against retrieved doc titles and drops unmatched markers (hallucinated sources). This is stricter than blindly trusting the LLM's citations and prevents the UI from linking to non-existent pages.
- **Query rewriting is passthrough by default:** the architecture doc marks query rewriting as optional (step 3 in the orchestrator pseudocode). For a local 3B model the extra round-trip adds latency and risk of reformulation errors, so the default is `PassthroughQueryRewriter`. `LLMQueryRewriter` is provided and wired but opt-in — pass it to `RAGOrchestrator(query_rewriter=...)` to enable.

### Added — Step 4 (FastAPI serving layer)

- `api/` package — wraps the Step 3 RAG orchestrator in HTTP, exposing all five architecture-doc endpoints under `/api/v1`:
  - `api/main.py` — `create_app()` factory + module-level `app` ASGI instance. Middleware: CORS (default origin `http://localhost:5173`, overridable via `FRONTEND_ORIGIN`), structlog JSON logging (falls back to stdlib on error), optional Langfuse (tracing is a no-op when `langfuse` is unimportable or `LANGFUSE_HOST` is unset, per the doc's "falls back gracefully" requirement). All five routers mounted under `/api/v1`.
  - `api/deps.py` — FastAPI dependency providers wiring the Step 3 components into the HTTP layer. Qdrant client, embedder, conversation store, and cache are `lru_cache`d process-wide singletons. `get_orchestrator()` constructs the real orchestrator from its collaborators; tests override it via `app.dependency_overrides` so the real Ollama/Qdrant stack is never reached.
  - `api/cache.py` — `LRUCache` (Tier 1 exact query match). Thread-safe (`threading.Lock`), bounded by `max_size=256` with LRU eviction via `OrderedDict`. Keys normalized by `normalize_query()` (lowercase, strip, collapse whitespace). Stores `ChatResponse` objects; on replay the session id is rebuilt to the current session so ids never leak. `stats()` exposes hit/miss/size for observability.
  - `api/conversation.py` — `ConversationStore` SQLite session store. Tables: `messages` (session_id, role, content, citations JSON, confidence, created_at) + `feedback` (session_id, message_index, rating, comment, created_at). `format_history()` builds the last-10-turns window as `User: ...\nAssistant: ...` for the generator's system prompt; `get_history()` trims to a turn boundary (drops a leading assistant message so the window starts on a user turn). Thread-safe single connection with `check_same_thread=False`. DB path overridable via `CHAT_DB_PATH` env; tests use `:memory:`.
  - `api/routes/health.py` — `GET /api/v1/health` liveness probe (always 200 while the process is serving; intentionally dependency-free so a slow downstream cannot flap the compose healthcheck).
  - `api/routes/chat.py` — `POST /api/v1/chat` SSE streaming endpoint. `build_event_stream()` yields `data: <token>` frames, then `event: result` with the full `ChatResponse` JSON, then `data: [DONE]`. Records user + assistant messages in the session store; stores the completed `ChatResponse` in the LRU cache. Cache hit → `build_replay_stream()` replays the cached answer without invoking the orchestrator (`X-Cache: HIT` header). Unknown `session_id` → 404. The SSE generators are factored out of the endpoint so they are unit-testable without the HTTP layer.
  - `api/routes/history.py` — `GET /api/v1/history/{session_id}` returns the full chronological message list (role, content, citations, confidence, created_at). 404 for unknown sessions.
  - `api/routes/feedback.py` — `POST /api/v1/feedback` records thumbs up/down + optional comment. 404 for unknown sessions; 422 for invalid rating/comment length.
  - `api/routes/ingest.py` — `POST /api/v1/ingest` delegates to the Step 2 `run_ingestion` orchestrator (incremental by default; `full_reindex=true` drops + recreates the collection). Blocking endpoint — for a single-user project the ~7-doc ingestion finishes in well under a minute. 500 on pipeline failure. Corpus dir overridable via `CORPUS_DIR` env or request body.
- `tests/conftest.py` — shared API fixtures: `TestClient` with mocked deps (orchestrator, in-memory SQLite store, LRU cache), `client_factory` for rebuilding after mutating mocks.
- `tests/test_api_cache.py` — 16 tests (normalization, get/put, LRU eviction, stats, clear, invalid max size).
- `tests/test_api_conversation.py` — 18 tests (session creation/existence, message storage, chronological order, per-session isolation, invalid role, history window truncation/turn-boundary/trailing-unpaired-user, format_history, feedback CRUD/isolation).
- `tests/test_api_health.py` — 3 tests (200, status ok, service + version).
- `tests/test_api_chat.py` — 16 tests (build_event_stream token/result/done frames, result JSON validity, message persistence, cache storage, empty stream, synthesized result, build_replay_stream session id, HTTP SSE streaming, session creation, provided session, 404 unknown session, cache miss/hit headers, cache hit skips orchestrator, validation rejections, history passed to orchestrator).
- `tests/test_api_history.py` — 3 tests (returns messages, 404 unknown, empty session).
- `tests/test_api_feedback.py` — 6 tests (up with comment, down without comment, 404 unknown session, invalid rating, missing fields, comment too long).
- `tests/test_api_ingest.py` — 4 tests (incremental default, full reindex flag, custom corpus dir, pipeline failure 500).
- Updated `pyproject.toml` — added warning filters for starlette's `httpx`/TestClient deprecation warning (the project pins httpx and the TestClient works fine; migration to httpx2 is deferred).
- Updated `README.md` — marked Step 4 ✅ in the build order table, marked all `api/` files ✅ in the repo layout, added new test files to the layout, expanded the "Backend (FastAPI)" running section with health/history/feedback/ingest examples and the SSE format description.
- All 287 tests pass in the `rag-chat` conda env (`python -m pytest -q`).

### Notes — Step 4

- **SSE wire format:** the chat endpoint emits `data: <token>\n\n` per token, then `event: result\ndata: {ChatResponse JSON}\n\n` (so the frontend gets citations + confidence without parsing streamed text), then `data: [DONE]\n\n`. This extends the architecture doc's simpler `data: ... / data: [DONE]` format with the structured `event: result` frame — the doc's React snippet only reads `data:` frames, so the frontend (Step 5) will need to handle the `result` event to render citation chips and the low-confidence warning.
- **Cache is Tier 1 only (exact query match):** the architecture doc describes a three-tier cache (exact match, semantic, response). Only Tier 1 is implemented — an in-memory LRU keyed by normalized query. Semantic caching (Tier 2, embedding-based) and a response cache (Tier 3) are documented as overkill for a single user and are deferred. The cache stores `ChatResponse` objects; on replay the session id is rebuilt to the current session.
- **History window is fixed-size, not summarized:** the build order lists "history summarization" (item 21), but for a local 3B model the extra LLM round-trip to summarize older turns adds latency and reformulation risk. The pragmatic default is a fixed last-10-turns window formatted as `User: ...\nAssistant: ...`. LLM-based summarization is documented as a deferred enhancement.
- **`/api/v1/ingest` is blocking:** for a single-user practice project the ~7-doc ingestion finishes in well under a minute, so a synchronous response returning the summary dict is the simplest honest behaviour. A production system would dispatch to a background job queue and return a job id; this gap is documented in the Enterprise Gap Register.
- **Langfuse is optional and lazy:** `api/main.py` checks for `langfuse` importability + `LANGFUSE_HOST` at startup. If either is missing, tracing is a no-op and the app runs fine — matching the doc's "falls back gracefully if Langfuse is down." Full Langfuse span wiring (retrieval/generation/guardrail spans + feedback scores) is Step 7 (Monitoring & Hardening); the seam is in place so the decorator is available from day one.
- **`get_orchestrator` is parameterless:** FastAPI's dependency system treats annotated function parameters as request fields. The orchestrator provider therefore takes no arguments and constructs its collaborators internally (via the cached `lru_cache` deps). Tests override the provider wholesale via `app.dependency_overrides` rather than injecting mocks through parameters.
- **starlette TestClient deprecation:** starlette warns that `httpx` is deprecated in favor of `httpx2` for the TestClient. The project pins `httpx>=0.27` and the TestClient works correctly; the warning is filtered in `pyproject.toml` until an httpx2 migration is warranted.
