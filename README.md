# CTC-RAG — AI-Powered RAG Chat Assistant (2026 Interview Practice Edition)

A **Retrieval-Augmented Generation (RAG) chat assistant** built end-to-end to practice and demonstrate the full RAG lifecycle for technical interviews: ingestion → chunking → embedding → retrieval → generation → streaming → guardrails → evaluation → monitoring → deployment.

> **Status:** Bootstrap phase. The full design lives in [`ai-rag-chat-architecture-2026.md`](./ai-rag-chat-architecture-2026.md) — read it first; this README is the executable companion.

---

## Why this project exists

| Goal                     | How this repo delivers                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **Interview confidence** | Reproduce the _entire_ RAG lifecycle (not just a notebook) so every design decision can be justified with trade-offs |
| **$0 cost**              | Local Ollama LLM + embeddings, self-hosted Qdrant, self-hosted Langfuse — no per-token or managed-service bill       |
| **Apple Silicon first**  | Targets MacBook Pro M1 32 GB RAM, macOS Sequoia, Colima for Docker                                                   |
| **Employable stack**     | FastAPI, Ollama, Qdrant, Langfuse, Ragas, Docker, React — all appear in 2026 AI/LLM job specs                        |
| **Single-language**      | Python for both the offline ingestion pipeline and the online serving layer (one venv, one test suite)               |

The corpus is a small open-source documentation subset (**FastAPI + Pydantic v2 + SQLModel**), so example questions look like:

- _"What is the difference between Pydantic `BaseModel` and `BaseSettings`?"_
- _"How do I declare path parameters in FastAPI?"_
- _"What is the difference between SQLModel and SQLAlchemy?"_

---

## Tech stack at a glance

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
│  React 18 + Vite  │  SSE Streaming  │  Feedback UI          │
├─────────────────────────────────────────────────────────────┤
│                     API LAYER                                │
│  FastAPI  │  Pydantic v2  │  Dependency Injection           │
├─────────────────────────────────────────────────────────────┤
│              RAG ORCHESTRATION LAYER (Python)                │
│  Query rewriting  │  Qdrant hybrid retriever  │  SSE generator│
│  Guardrails  │  In-memory cache  │  Confidence score        │
├─────────────────────────────────────────────────────────────┤
│                     AI SERVICES LAYER                        │
│  Ollama (host) — llama3.2:3b (generation)                   │
│  Ollama (host) — nomic-embed-text (embeddings)              │
│  Qdrant — HNSW dense + sparse hybrid search                │
├─────────────────────────────────────────────────────────────┤
│                     DATA LAYER                               │
│  SQLite (chat history)  │  JSON manifest (ingestion state)  │
├─────────────────────────────────────────────────────────────┤
│           INGESTION LAYER (Python)                           │
│  Unstructured / pypdf / BeautifulSoup                       │
│  LangChain chunkers  │  Ollama embeddings  │  Qdrant client  │
│  Ragas offline eval (golden dataset)                       │
├─────────────────────────────────────────────────────────────┤
│                   OBSERVABILITY                              │
│  Langfuse (self-hosted)  │  structlog JSON                 │
├─────────────────────────────────────────────────────────────┤
│                   INFRASTRUCTURE                             │
│  Docker Compose  │  Colima  │  Local named volumes          │
└─────────────────────────────────────────────────────────────┘
```

Full decision rationale: [`ai-rag-chat-architecture-2026.md` → Technology Stack Summary](./ai-rag-chat-architecture-2026.md#technology-stack-summary).

---

## Repository layout (target)

```
practice-rag/
├── ai-rag-chat-architecture-2026.md   # Full design doc (read first)
├── README.md                          # This file
├── CHANGELOG.md                       # Keep a Changelog — updated before every commit
├── requirements.txt                   # Python deps (installed via pip3 in conda env)
├── pyproject.toml                     # pytest config (pythonpath=. , testpaths=tests)
├── .gitignore                         # macOS, Python, Node, .env, Colima volumes
├── docker-compose.yml                 # Qdrant, Postgres, Langfuse (Phase 1)
│                                      #   Ollama runs on the host, not in Docker
├── schemas/                           # Shared Pydantic contracts (Step 1) ✅
│   ├── __init__.py                    # Re-exports all contracts
│   ├── documents.py                   # DocumentChunk, RetrievedDoc
│   ├── chat.py                        # ChatRequest, ChatResponse, Citation
│   └── feedback.py                    # FeedbackRequest
├── ingestion/                         # Offline pipeline (Step 2)
│   ├── sync.py                        # Doc download / clone
│   ├── parser.py                      # MD / PDF / HTML parsing
│   ├── chunker.py                     # RecursiveCharacterTextSplitter, 512 tok / 64 overlap
│   ├── embedder.py                    # Ollama nomic-embed-text wrapper
│   ├── index_writer.py                # Qdrant upsert + payload
│   └── manifest.json                  # File hashes for incremental sync
├── rag/                               # Core RAG orchestrator (Step 3, plain Python)
│   ├── __init__.py
│   ├── qdrant_collection.py           # docs-knowledge collection helper (Step 1) ✅
│   ├── query_rewriter.py
│   ├── retriever.py                   # Qdrant hybrid (dense + sparse) + RRF fusion
│   ├── context_assembler.py
│   ├── generator.py                   # Ollama llama3.1:8b streaming
│   └── post_processor.py              # Citations + groundedness score
├── api/                               # FastAPI serving layer (Step 4)
│   ├── main.py                        # app + middleware + Langfuse @observe
│   ├── routes/
│   │   ├── chat.py                    # POST /api/v1/chat (SSE)
│   │   ├── feedback.py                # POST /api/v1/feedback
│   │   ├── history.py                 # GET  /api/v1/history/{session_id}
│   │   ├── ingest.py                  # POST /api/v1/ingest
│   │   └── health.py                  # GET  /api/v1/health (compose healthcheck)
│   ├── conversation.py                # SQLite session store, last-10-turns window
│   ├── cache.py                       # In-memory LRU response cache
│   └── guardrails.py                  # Regex + local LLM judge (Step 6)
├── frontend/                          # React + Vite (Step 5)
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── ChatWidget.jsx
│   │   ├── MessageList.jsx
│   │   ├── InputBox.jsx
│   │   ├── CitationChip.jsx
│   │   └── ErrorBoundary.jsx
│   └── __tests__/
├── eval/                              # Offline Ragas eval (Step 6)
│   ├── golden-dataset.json            # 30–50 hand-curated Q&A pairs
│   └── run_eval.py                    # Local llama3.1:8b judge, threshold gate
├── data/
│   └── corpus/                        # Downloaded FastAPI / Pydantic / SQLModel docs
└── tests/                             # pytest backend tests
    ├── test_schemas.py                # DocumentChunk/ChatRequest/Feedback contracts ✅
    ├── test_qdrant_collection.py      # collection helper (mocked client) ✅
    ├── test_guardrails.py
    ├── test_chunker.py
    ├── test_api_chat.py
    ├── test_api_health.py
    └── test_cache.py
```

> Files marked with a phase/step are **targets** — they are created as the build progresses, not all at bootstrap. ✅ = implemented.

---

## Prerequisites

| Tool                               | Why                                                                 | Install                    |
| ---------------------------------- | ------------------------------------------------------------------- | -------------------------- |
| **Colima**                         | Docker runtime on macOS (lighter than Docker Desktop)               | `brew install colima`      |
| **Conda**                          | Isolated Python 3.12 env for host-side dev (ingestion, eval, tests) | Miniconda / Miniforge      |
| **Node.js 20+**                    | React + Vite frontend                                               | `brew install node` or nvm |
| **Ollama** (optional host install) | Pulling models ahead of the Docker stack                            | `brew install ollama`      |

Hardware target: **MacBook Pro M1, 32 GB RAM, macOS Sequoia**. The Colima VM is sized to `--cpu 4 --memory 8 --disk 100`.

---

## Quick start (Phase 1 — Bootstrap)

```bash
# 1. Start Colima with enough headroom for Qdrant + Langfuse
colima start --cpu 4 --memory 8 --disk 100

# 2. Ensure Ollama is running on the HOST (not in Docker)
ollama serve                          # if not already running

# 3. Pull the required models on the host (one-time)
ollama pull llama3.2:3b               # generation (~2 GB)
ollama pull nomic-embed-text          # embeddings 768-d (~0.3 GB)

# 4. Create and activate the dedicated conda env (Python 3.12)
conda create -n rag-chat python=3.12 -y
conda activate rag-chat

# 5. Install Python deps with pip3 (never into system Python)
pip3 install -r requirements.txt

# 6. Bring up the infrastructure stack (Qdrant + Langfuse)
docker compose up -d

# 7. Verify health endpoints
curl http://localhost:11434/api/tags        # Ollama (host)
curl http://localhost:6333/healthz          # Qdrant
curl http://localhost:3000                  # Langfuse UI
```

Once the stack is green, follow the **Dependency-Driven Build Order** below.

---

## Dependency-driven build order

The calendar phases (one weekend) are reordered below into the actual build graph — each step produces something the next step imports. Full table: [`ai-rag-chat-architecture-2026.md` → Implementation Phases](./ai-rag-chat-architecture-2026.md#implementation-phases).

| Step  | What to build                                                | Depends on | Maps to phase                    |
| ----- | ------------------------------------------------------------ | ---------- | -------------------------------- |
| **0** | Colima + `docker-compose.yml` + conda env + repo scaffolding | nothing    | Phase 1 — Bootstrap              |
| **1** | `schemas/` Pydantic contracts + Qdrant collection helper ✅  | Step 0     | seam for Phases 2 & 3            |
| **2** | `ingestion/` pipeline → chunks in Qdrant                     | Step 1     | Phase 2 — Ingestion              |
| **3** | `rag/` orchestrator as plain Python (unit-testable)          | Step 2     | Phase 3 — Core RAG               |
| **4** | `api/` FastAPI layer (`/health` before `/chat`)              | Step 3     | Phase 3 — Core RAG               |
| **5** | `frontend/` React + SSE consumer                             | Step 4     | Phase 4 — Frontend               |
| **6** | `guardrails` + `eval/` golden dataset + Ragas gate           | Step 4     | Phase 5 — Guardrails & Eval      |
| **7** | Langfuse traces + resilience + optional cloud showcase       | Steps 3–6  | Phase 6 — Monitoring & Hardening |

> **Rule of thumb:** never mock something you haven't built yet. Follow the order left-to-right.

---

## Running the app

### Backend (FastAPI)

```bash
conda activate rag-chat
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Smoke-test the SSE stream before touching the frontend:

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I declare optional query parameters in FastAPI?"}'
```

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev          # dev server with HMR
```

Production build + local serve (no global install):

```bash
cd frontend
npm run build
npx serve -s dist
```

---

## Testing strategy

Tests are run **before every commit** (project rule). Three layers:

### 1. Backend — pytest

```bash
conda activate rag-chat
pytest tests/ -v --tb=short
```

Covers guardrail regexes, chunker correctness, FastAPI TestClient with mocked Ollama/Qdrant, async SSE, and the LRU cache.

### 2. Frontend — Jest + React Testing Library

```bash
cd frontend
npm test -- --coverage
```

Covers `ChatWidget` rendering, `CitationChip`, SSE `fetch` mock, and `ErrorBoundary`.

### 3. Golden-dataset eval gate (Ragas)

Acts as a **regression gate** before merge. Fails the commit if quality drops.

```bash
conda activate rag-chat
python eval/run_eval.py --threshold-faithfulness 0.75 --threshold-recall 0.70
```

### Pre-commit checklist

```bash
# 1. Backend tests
pytest tests/ -v

# 2. Frontend tests
cd frontend && npm test && cd ..

# 3. Eval gate (only if ingestion or prompt changed)
python eval/run_eval.py

# 4. Update CHANGELOG.md
# 5. Commit (Conventional Commits)
```

---

## API surface

| Method | Path                           | Purpose                           |
| ------ | ------------------------------ | --------------------------------- |
| `POST` | `/api/v1/chat`                 | Main RAG streaming endpoint (SSE) |
| `POST` | `/api/v1/feedback`             | Thumbs up/down + comment          |
| `GET`  | `/api/v1/history/{session_id}` | Conversation history              |
| `POST` | `/api/v1/ingest`               | Trigger incremental re-index      |
| `GET`  | `/api/v1/health`               | Health check for Docker Compose   |

Request/response examples and the SSE format: [`ai-rag-chat-architecture-2026.md` → API Design](./ai-rag-chat-architecture-2026.md#api-design).

---

## Observability & evaluation

- **Langfuse (self-hosted)** traces every chat request: retrieval span, generation span, guardrail span, feedback event.
- **structlog** emits JSON logs to stdout; falls back gracefully if Langfuse is down.
- **Ragas offline eval** runs against `eval/golden-dataset.json` (30–50 Q&A pairs) with `llama3.1:8b` as the local judge. Optional Groq free tier for a stronger judge.

Metrics tracked: TTFT, retrieval recall@5, faithfulness, answer relevancy, user feedback, cache hit rate.

---

## Cost

**Core workload: $0/month.** Ollama, Qdrant, FastAPI, React dev server, Langfuse, SQLite all run locally in Docker on Colima.

Optional "showcase" cloud deployment for a live resume URL: **$0–$2/month** using Vercel + Render/Railway + Groq free tier + Qdrant Cloud free tier + Turso.

Full breakdown: [`ai-rag-chat-architecture-2026.md` → Cost Estimation](./ai-rag-chat-architecture-2026.md#cost-estimation).

---

## Interview discussion points (quick reference)

The full set lives in [`ai-rag-chat-architecture-2026.md` → Interview Discussion Points](./ai-rag-chat-architecture-2026.md#interview-discussion-points). Be ready to justify:

1. **Why a single Python stack** instead of the 2024 Python + Node.js split.
2. **Why Ollama + local models** instead of OpenAI (cost, latency trade-offs, quantization, context windows).
3. **Why Qdrant** instead of Pinecone/Weaviate (open-source, native hybrid search, HNSW).
4. **Why lightweight guardrails** (regex + local judge) and what the F500 equivalent would be.
5. **How RAG quality is evaluated for free** (golden dataset + Ragas + local judge).
6. **How the architecture scales to production** (vLLM/TGI, Qdrant Cloud, Postgres, API gateway, full guardrail service).
7. **What "production" means for a local stack** (reproducible, containerized, health-checked, observable).

Also review the **Enterprise Gap Register** — it is an explicit, honest list of every deferred control with the production equivalent and the deferral rationale, so trade-offs can be defended rather than discovered under questioning.

---

## Project conventions

- **Python deps:** `pip3 install` inside the `rag-chat` conda env only — never into system Python.
- **Docker:** Colima on macOS; containers capped at minimal specs (1 CPU, 1 GB RAM, 5 GB disk) per project rule.
- **Changelog:** [`CHANGELOG.md`](./CHANGELOG.md) is updated **before every commit** (Keep a Changelog format).
- **Commits:** Conventional Commits — `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`.
- **Branches:** `feature/TICKET-short-desc` | `fix/TICKET-desc` | `hotfix/desc`.
- **Placeholder images:** use `dummyimage.com`.
- **Frontend serve:** `npx serve -s dist` for static HTML/CSS/JS builds.

---

## License

Personal interview-prep project. All referenced libraries are open-source or have strong free tiers as of 2026.
