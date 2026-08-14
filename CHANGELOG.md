# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

### Changed

- Ollama runs on the **host** (not in Docker) to avoid port conflicts and leverage host Metal acceleration. The `docker-compose.yml` no longer includes an Ollama service.
- Generation model changed from `llama3.1:8b` to `llama3.2:3b` (already available on the host; lighter and faster for dev). The architecture doc's `llama3.1:8b` remains the documented target — `llama3.2:3b` is the current dev substitute.
- Langfuse web + worker memory limit raised from 1 GB to 2 GB (Next.js + Prisma OOM at 1 GB during migrations).
- README Quick Start updated to reflect host Ollama + `llama3.2:3b`.

### Documented

- Added "Phase 0 Implementation Decisions" section to `ai-rag-chat-architecture-2026.md` documenting four deviations from the original design (host Ollama, llama3.2:3b, Langfuse 2 GB RAM, Miniforge install) with reasoning and revert plans.

### Notes

- Repository is in **Phase 0 — Foundation** (complete). No application code yet; the next step is Step 1 — `schemas/` Pydantic contracts + Qdrant collection helper.
- All health endpoints verified: Ollama `localhost:11434`, Qdrant `localhost:6333/healthz`, Langfuse `localhost:3000` (HTTP 200).
