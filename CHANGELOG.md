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

### Changed

- Ollama runs on the **host** (not in Docker) to avoid port conflicts and leverage host Metal acceleration. The `docker-compose.yml` no longer includes an Ollama service.
- Generation model changed from `llama3.1:8b` to `llama3.2:3b` (already available on the host; lighter and faster for dev). The architecture doc's `llama3.1:8b` remains the documented target — `llama3.2:3b` is the current dev substitute.
- Langfuse web + worker memory limit raised from 1 GB to 2 GB (Next.js + Prisma OOM at 1 GB during migrations).
- README Quick Start updated to reflect host Ollama + `llama3.2:3b`.

### Documented

- Added "Phase 0 Implementation Decisions" section to `ai-rag-chat-architecture-2026.md` documenting four deviations from the original design (host Ollama, llama3.2:3b, Langfuse 2 GB RAM, Miniforge install) with reasoning and revert plans.

### Notes

- Repository is in **Step 1 — Shared contracts** (complete). Phase 0 — Foundation is also complete. The next step is Step 2 — ingestion pipeline (doc sync → parse → chunk → embed → Qdrant upsert).
- All health endpoints verified: Ollama `localhost:11434`, Qdrant `localhost:6333/healthz`, Langfuse `localhost:3000` (HTTP 200).
