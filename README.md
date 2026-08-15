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
├── docker-compose.yml                 # Qdrant, Postgres, Langfuse + backend (Phase 1 + Step 7)
│                                      #   Ollama runs on the host, not in Docker
├── Dockerfile                         # Backend image (Step 7) ✅
├── schemas/                           # Shared Pydantic contracts (Step 1) ✅
│   ├── __init__.py                    # Re-exports all contracts
│   ├── documents.py                   # DocumentChunk, RetrievedDoc
│   ├── chat.py                        # ChatRequest, ChatResponse, Citation (+ trace_id Step 7)
│   └── feedback.py                    # FeedbackRequest (+ trace_id Step 7)
├── ingestion/                         # Offline pipeline (Step 2) ✅
│   ├── __init__.py
│   ├── sync.py                        # Doc discovery (local folder / git clone)
│   ├── parser.py                      # MD / HTML / PDF parsing → ParsedDocument
│   ├── chunker.py                     # MarkdownHeader + RecursiveCharacterTextSplitter, 512 tok / 64 overlap
│   ├── embedder.py                    # Ollama nomic-embed-text (dense) + hashing-trick sparse
│   ├── index_writer.py                # Qdrant upsert (dense + sparse + payload)
│   ├── manifest.py                    # File-hash manifest for incremental sync
│   └── run.py                         # Orchestrator: sync→parse→chunk→embed→upsert
├── rag/                               # Core RAG orchestrator (Step 3, plain Python) ✅
│   ├── __init__.py
│   ├── qdrant_collection.py           # docs-knowledge collection helper (Step 1) ✅
│   ├── query_rewriter.py              # Passthrough + optional LLM rewrite ✅
│   ├── retriever.py                   # Qdrant hybrid (dense + sparse) + RRF fusion ✅
│   ├── context_assembler.py           # Format chunks into system-prompt CONTEXT block ✅
│   ├── generator.py                   # Ollama llama3.2:3b streaming (target: llama3.1:8b) ✅
│   ├── post_processor.py              # Citation extraction + groundedness score ✅
│   └── orchestrator.py                # Ties rewrite→retrieve→assemble→generate→post-process + guardrails ✅
├── api/                               # FastAPI serving layer (Step 4) ✅
│   ├── __init__.py
│   ├── main.py                        # app + middleware (CORS, structlog, Langfuse, warm-up lifespan) ✅
│   ├── deps.py                        # FastAPI dependency injection wiring (+ tracer, metrics, breaker) ✅
│   ├── conversation.py                # SQLite session store, last-10-turns window ✅
│   ├── cache.py                       # In-memory LRU response cache ✅
│   ├── observability.py               # CircuitBreaker, MetricsCollector, LangfuseTracer, warm-up (Step 7) ✅
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py                    # POST /api/v1/chat (SSE + TTFT/cache metrics) ✅
│   │   ├── feedback.py                # POST /api/v1/feedback (+ Langfuse score) ✅
│   │   ├── history.py                 # GET  /api/v1/history/{session_id} ✅
│   │   ├── ingest.py                  # POST /api/v1/ingest ✅
│   │   ├── health.py                  # GET /api/v1/health + /health/ready (liveness + readiness) ✅
│   │   └── metrics.py                 # GET /api/v1/metrics (TTFT, cache hit rate, request/error) ✅
│   └── guardrails.py                  # Input/output guardrails + query classifier (Step 6) ✅
├── frontend/                          # React + Vite chat widget (Step 5) ✅
│   ├── package.json                   # Vite + React 18 + Jest + RTL ✅
│   ├── vite.config.js                 # Dev server proxies /api → :8000 ✅
│   ├── babel.config.js                # Babel for Jest (env + react presets) ✅
│   ├── jest.setup.js                  # jsdom polyfills (scrollIntoView, TextEncoder) ✅
│   ├── index.html                     # Vite entry HTML ✅
│   ├── src/
│   │   ├── main.jsx                   # React root + styles.css import ✅
│   │   ├── App.jsx                    # Root → ChatWidget ✅
│   │   ├── ChatWidget.jsx             # State, session, streaming, feedback wiring ✅
│   │   ├── MessageList.jsx            # Messages, citations, confidence warning, feedback ✅
│   │   ├── InputBox.jsx               # Textarea + SendButton, Enter to send ✅
│   │   ├── CitationChip.jsx           # [Source: title] link ✅
│   │   ├── ErrorBoundary.jsx          # Catches render errors, recovery UI ✅
│   │   ├── api.js                     # SSE parser + streamChat + sendFeedback + fetchHistory ✅
│   │   └── styles.css                 # Chat widget styles ✅
│   └── __tests__/
│       ├── App.test.jsx               # Root renders ChatWidget ✅
│       ├── api.test.js                # SSE parser + streamChat + REST helpers (18 tests) ✅
│       ├── CitationChip.test.jsx      # Link rendering + URL fallback (3 tests) ✅
│       ├── MessageList.test.jsx       # Messages, citations, warning, feedback (11 tests) ✅
│       ├── InputBox.test.jsx          # Input, Enter/Shift+Enter, disabled (7 tests) ✅
│       ├── ChatWidget.test.jsx        # Streaming, citations, feedback, session (10 tests) ✅
│       └── ErrorBoundary.test.jsx     # Catch + recovery (3 tests) ✅
├── eval/                              # Offline Ragas eval (Step 6) ✅
│   ├── golden-dataset.json            # 36 hand-curated Q&A pairs (7 corpus docs)
│   └── run_eval.py                    # Ragas + local judge fallback, threshold gate ✅
├── data/
│   └── corpus/                        # Seed corpus: 7 MD files (FastAPI/Pydantic/SQLModel) ✅
└── tests/                             # pytest backend tests
    ├── test_schemas.py                # DocumentChunk/ChatRequest/Feedback contracts ✅
    ├── test_qdrant_collection.py      # collection helper (mocked client) ✅
    ├── test_ingestion_manifest.py     # manifest CRUD + file hashing ✅
    ├── test_ingestion_sync.py         # doc discovery ✅
    ├── test_ingestion_parser.py       # MD/HTML parsing + metadata ✅
    ├── test_ingestion_chunker.py      # two-stage split + deterministic UUIDs ✅
    ├── test_ingestion_embedder.py     # dense (mocked Ollama) + sparse (hashing trick) ✅
    ├── test_ingestion_index_writer.py # Qdrant upsert + delete (mocked client) ✅
    ├── test_ingestion_run.py          # orchestrator integration (mocked) ✅
    ├── test_rag_retriever.py           # hybrid dense+sparse + RRF fusion (mocked) ✅
    ├── test_rag_context_assembler.py   # CONTEXT block formatting ✅
    ├── test_rag_generator.py           # Ollama streaming + system prompt + circuit breaker (mocked) ✅
    ├── test_rag_post_processor.py      # citations + groundedness score (mocked) ✅
    ├── test_rag_query_rewriter.py      # passthrough + LLM rewrite (mocked) ✅
    ├── test_rag_orchestrator.py        # full RAG flow + guardrail + tracer integration (mocked) ✅
    ├── test_guardrails.py              # input/output guardrails + classifier (61 tests) ✅
    ├── test_eval.py                    # golden dataset + threshold gate + CSV (25 tests) ✅
    ├── test_observability.py           # CircuitBreaker, MetricsCollector, LangfuseTracer, warm-up (67 tests) ✅
    ├── test_api_cache.py               # LRU response cache (normalization, eviction, stats) ✅
    ├── test_api_conversation.py        # SQLite session store + history window + feedback ✅
    ├── test_api_health.py              # GET /api/v1/health (liveness) ✅
    ├── test_api_health_ready.py        # GET /api/v1/health/ready (readiness, Qdrant+Ollama) ✅
    ├── test_api_metrics.py             # GET /api/v1/metrics (TTFT, cache hit rate, counts) ✅
    ├── test_api_chat.py                # POST /api/v1/chat SSE (stream, cache, session, 404, metrics) ✅
    ├── test_api_history.py             # GET /api/v1/history/{session_id} ✅
    ├── test_api_feedback.py            # POST /api/v1/feedback (+ trace_id score) ✅
    └── test_api_ingest.py              # POST /api/v1/ingest (mocked pipeline) ✅
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
| **2** | `ingestion/` pipeline → chunks in Qdrant ✅                  | Step 1     | Phase 2 — Ingestion              |
| **3** | `rag/` orchestrator as plain Python (unit-testable) ✅       | Step 2     | Phase 3 — Core RAG               |
| **4** | `api/` FastAPI layer (`/health` before `/chat`) ✅           | Step 3     | Phase 3 — Core RAG               |
| **5** | `frontend/` React + SSE consumer ✅                          | Step 4     | Phase 4 — Frontend               |
| **6** | `guardrails` + `eval/` golden dataset + Ragas gate ✅        | Step 4     | Phase 5 — Guardrails & Eval      |
| **7** | Langfuse traces + resilience + optional cloud showcase ✅    | Steps 3–6  | Phase 6 — Monitoring & Hardening |

> **Rule of thumb:** never mock something you haven't built yet. Follow the order left-to-right.

---

## Running the app

### Ingestion pipeline (Step 2)

Populate Qdrant with the seed corpus (7 Markdown files covering FastAPI, Pydantic v2, and SQLModel):

```bash
conda activate rag-chat

# Full re-index (drops + recreates the collection, indexes every file)
python -m ingestion.run data/corpus --full-reindex

# Incremental sync (skips unchanged files based on manifest.json hashes)
python -m ingestion.run data/corpus

# Verbose logging
python -m ingestion.run data/corpus -v
```

Verify the chunks are in Qdrant:

```bash
curl http://localhost:6333/collections/docs-knowledge | python3 -m json.tool
```

### RAG orchestrator (Step 3)

The RAG flow is plain Python — no HTTP layer yet. After ingesting the corpus,
run an end-to-end query from a Python shell to verify retrieval + generation:

```bash
conda activate rag-chat

python -c "
from ingestion.embedder import Embedder
from rag import (
    HybridRetriever, ContextAssembler, Generator,
    PostProcessor, RAGOrchestrator, get_qdrant_client,
)

client = get_qdrant_client()
embedder = Embedder()
orchestrator = RAGOrchestrator(
    retriever=HybridRetriever(client, embedder),
    context_assembler=ContextAssembler(),
    generator=Generator(),
    post_processor=PostProcessor(embedder),
)

for item in orchestrator.stream_answer('How do I declare path parameters in FastAPI?'):
    if isinstance(item, str):
        print(item, end='', flush=True)
    else:
        print(f'\n\n--- confidence={item.confidence:.3f}, citations={len(item.citations)}')
"
```

> The generator uses `llama3.2:3b` (the dev substitute from Phase 0 deviation
> D2). To use the documented production target, pass `Generator(model="llama3.1:8b")`.

### Backend (FastAPI)

```bash
conda activate rag-chat
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs (OpenAPI/Swagger): `http://localhost:8000/docs`.

Health check (used by the Docker Compose healthcheck):

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","service":"practice-rag-api","version":"0.4.0"}
```

Smoke-test the SSE stream before touching the frontend:

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I declare optional query parameters in FastAPI?"}'
```

The SSE response emits `data: <token>` frames as tokens stream, then an
`event: result` frame with the full `ChatResponse` JSON (citations +
confidence + session id), and finally `data: [DONE]`. Repeated identical
queries hit the in-memory LRU cache (`X-Cache: HIT` header) and replay the
cached answer without invoking the orchestrator.

Other endpoints:

```bash
# Conversation history for a session
curl http://localhost:8000/api/v1/history/<session_id>

# Thumbs up/down feedback
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<id>","message_index":1,"rating":"up","comment":"cited well"}'

# Trigger incremental re-index of the corpus
curl -X POST http://localhost:8000/api/v1/ingest
# Full re-index (drops + recreates the collection)
curl -X POST http://localhost:8000/api/v1/ingest -d '{"full_reindex":true}' \
  -H "Content-Type: application/json"
```

### Frontend (React + Vite)

The chat widget is a React 18 + Vite SPA that consumes the FastAPI SSE
stream. The Vite dev server proxies `/api` to the backend on `:8000`.

```bash
cd frontend
npm install
npm run dev          # dev server with HMR on :5173
```

Production build + local serve (no global install):

```bash
cd frontend
npm run build
npx serve -s dist
```

**Component tree** (per the architecture doc):

```
App → ChatWidget → ErrorBoundary
                 → MessageList → CitationChip
                 → InputBox → SendButton
```

**SSE consumption:** `src/api.js` uses `fetch` + `ReadableStream` reader
(per the doc's snippet) to parse the SSE stream. The parser handles three
frame kinds:

- `data: <token>` — appended to the streaming assistant bubble
- `event: result` — parsed as `ChatResponse` JSON; sets citations,
  confidence, and session id
- `data: [DONE]` — marks streaming complete, enables feedback buttons

**Features:**

- Token-by-token streaming with a blinking cursor
- Citation chips (`[Source: title]` as external links)
- Low-confidence warning when `confidence < 0.65`
- Thumbs up/down feedback → `POST /api/v1/feedback`
- Session id carried across turns (from the `result` frame)
- Error boundary with recovery ("Try again" button)
- "New chat" button to reset the session
- Enter to send, Shift+Enter for newline

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

54 tests across 7 suites covering:

- `api.js` — SSE frame parser (token/result/done), `streamChat` with mocked `ReadableStream`, `sendFeedback`, `fetchHistory`
- `CitationChip` — link rendering, URL fallback, camelCase key
- `MessageList` — messages, citations, low-confidence warning, feedback buttons, error text, active state
- `InputBox` — send on Enter, Shift+Enter newline, disabled state, empty guard
- `ChatWidget` — streaming, citations, confidence warning, feedback wiring, session propagation, error handling, new chat reset
- `ErrorBoundary` — catch + recovery
- `App` — root renders ChatWidget

### 3. Golden-dataset eval gate (Ragas)

Acts as a **regression gate** before merge. Fails the commit if quality drops.

```bash
conda activate rag-chat
# Full eval (36 questions, ~5–10 min with local Ollama)
python eval/run_eval.py --threshold-faithfulness 0.75 --threshold-recall 0.70

# Quick smoke (first 5 questions)
python eval/run_eval.py --limit 5

# Force local LLM judge (skip Ragas)
python eval/run_eval.py --no-ragas
```

> **Ragas fallback:** if the `ragas` package can't import (version mismatch
> with `langchain_community`), the script falls back to a lightweight local
> LLM judge that scores each metric 0–1 with simple Ollama prompts. The
> fallback is less rigorous than Ragas but provides a working regression gate.

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

- **Langfuse (self-hosted)** traces every chat request: retrieval span, generation span, guardrail span, feedback score. The `LangfuseTracer` (`api/observability.py`) degrades to structlog logging when Langfuse is down or unconfigured — serving never breaks.
- **structlog** emits JSON logs to stdout; falls back gracefully if Langfuse is down.
- **Online metrics** (`GET /api/v1/metrics`) — TTFT (mean/p50/p95), request count, error count, cache hit rate. Collected in-process by `MetricsCollector`.
- **Circuit breaker** (`CircuitBreaker`) wraps Ollama generation calls — opens after 3 consecutive failures, half-opens after 30 s, never stalls serving on a flaky model.
- **Ollama warm-up** — on startup a tiny generation call pre-loads the model so the first real query is not slow.
- **Readiness probe** (`GET /api/v1/health/ready`) — pings Qdrant + Ollama, returns 503 if either is down (liveness `/health` stays 200).
- **Ragas offline eval** runs against `eval/golden-dataset.json` (36 Q&A pairs) with a local Ollama judge (Ragas fallback when the library can't import). Threshold gate: `faithfulness >= 0.75`, `context_recall >= 0.70`, `answer_relevancy >= 0.70`.

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
