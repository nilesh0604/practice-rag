# Ollama Model Recommendations for practice-rag

> Companion to `docs/NVIDIA_NIM_INTEGRATION_PLAN.md` (NIM integration) and
> `docs/F500_ENTERPRISE_ACTION_ITEMS.md` (production-grade gaps).
>
> **Date:** 2026-08-15
> **Hardware:** MacBook Pro M1, 32 GB RAM, macOS Sequoia
> **Ollama:** host-installed (not containerized — architecture doc deviation D1)

---

## Table of Contents

1. [Current State — What's Pulled vs. What's Used](#1-current-state--whats-pulled-vs-whats-used)
2. [Model Usage Points in the Codebase](#2-model-usage-points-in-the-codebase)
3. [Recommended Model Assignments](#3-recommended-model-assignments)
4. [Models to Pull](#4-models-to-pull)
5. [Memory Budget Analysis](#5-memory-budget-analysis)
6. [Configuration Changes Required](#6-configuration-changes-required)
7. [Embedding Model — Keep vs. Upgrade](#7-embedding-model--keep-vs-upgrade)
8. [Pull Commands Summary](#8-pull-commands-summary)

---

## 1. Current State — What's Pulled vs. What's Used

### Currently pulled in Ollama

| Model | Size | Modified | Used in code? |
|---|---|---|---|
| `llama3.2:3b` | 2.0 GB | 17 hours ago | Yes — generator, guardrails, rewriter, warm-up, eval judge |
| `qwen2.5:14b` | 9.0 GB | 2 months ago | **No — pulled but not referenced anywhere in the codebase** |
| `nomic-embed-text:latest` | 274 MB | 11 months ago | Yes — dense embeddings (768-d) |

### Key finding

**`qwen2.5:14b` is already pulled (9.0 GB) but is not used by any component.**
This is a 14.8B parameter model (Q4_K_M quantization, 32K context, tool-use
capable) that is significantly stronger than the current `llama3.2:3b` used
everywhere. It should be assigned to the generator and eval judge — the two
roles that benefit most from stronger reasoning.

---

## 2. Model Usage Points in the Codebase

Every location where an Ollama model ID is hardcoded:

| File | Constant | Current value | Role |
|---|---|---|---|
| `rag/generator.py` | `GENERATION_MODEL` | `llama3.2:3b` | Answer generation (streaming) |
| `rag/generator.py` | `GENERATION_MODEL_TARGET` | `llama3.1:8b` | Doc's production target (not yet active) |
| `api/guardrails.py` | `GUARD_MODEL` | `llama3.2:3b` | Input injection judge, output harm judge, query classifier |
| `rag/query_rewriter.py` | `REWRITE_MODEL` | `llama3.2:3b` | LLM query rewriting (coreference resolution) |
| `ingestion/embedder.py` | `EMBEDDING_MODEL` | `nomic-embed-text` | Dense embeddings (768-d) |
| `api/main.py` | `OLLAMA_WARMUP_MODEL` | `llama3.2:3b` (env: `OLLAMA_WARMUP_MODEL`) | Startup warm-up call |
| `eval/run_eval.py` | `DEFAULT_JUDGE_MODEL` | `llama3.2:3b` (env: `EVAL_JUDGE_MODEL`) | Offline Ragas eval judge |

---

## 3. Recommended Model Assignments

### Design principle

Different roles have different requirements. The current approach of using
`llama3.2:3b` for everything is the Phase 0 dev shortcut (architecture doc
deviation D2). The recommended assignments below split models by role to
optimize the quality/latency/memory trade-off for each component.

### Assignment table

| Role | File | Current | Recommended | Status | Why |
|---|---|---|---|---|---|
| **Answer generation** | `rag/generator.py` | `llama3.2:3b` | **`qwen2.5:14b`** | Already pulled | 14.8B params, 32K context, strong instruction-following + reasoning. The generator is the most quality-sensitive component — it produces the user-facing answer. 32K context easily fits the 5-chunk × 512-token context window + system prompt + history (~4-5K tokens). Q4_K_M quantization (9 GB) fits comfortably in 32 GB M1. Tool-use capability aligns with the project's agentic RAG direction. |
| **Eval judge** | `eval/run_eval.py` | `llama3.2:3b` | **`qwen2.5:14b`** | Already pulled | The eval judge scores answer quality (faithfulness, relevancy, context recall). A 3B model is too weak to reliably judge quality — it will both miss real issues and hallucinate problems. `qwen2.5:14b` has the reasoning depth to make reliable quality judgments. Same model as the generator is acceptable for the eval judge (the judge sees generated answers, not the generator's internal state). |
| **Content safety judge** | `api/guardrails.py` (`OutputGuardrail`) | `llama3.2:3b` | **`llama-guard3:8b`** | **Pull needed** (4.9 GB) | Purpose-built by Meta for content safety classification against the MLCommons 14-hazard taxonomy (violence, hate, self-harm, sexual content, privacy, etc.). Directly addresses F500 action item #1: "a 3B general-purpose chat model is not a security boundary an enterprise compliance review would accept." Fine-tuned from Llama 3.1 8B — same parameter class but specialized for safety. 128K context (overkill for safety checks, but no downside). |
| **Injection judge** | `api/guardrails.py` (`InputGuardrail`) | `llama3.2:3b` | **`llama3.1:8b`** | **Pull needed** (4.9 GB) | Prompt-injection detection requires understanding subtle linguistic manipulation (role-reset framing, indirect injection via history, multi-turn jailbreaks). A 3B model misses subtler attacks. `llama3.1:8b` is the architecture doc's documented production target (D2) and has enough depth to catch patterns a 3B model misses. Note: `llama-guard3:8b` is specialized for content safety, NOT prompt injection — so a general-purpose 8B model is the right choice for the injection judge. |
| **Query classifier** | `api/guardrails.py` (`QueryClassifier`) | `llama3.2:3b` | **`llama3.2:3b`** (keep) | Already pulled | The classifier outputs a single label (`documentation` / `greeting` / `off_topic` / `compare` / `follow_up`) with a 10-token prompt. This is a fast, shallow task — a 3B model is sufficient and keeps latency low (the classifier runs on every request). The keyword fallback (`classify_keywords`) is the safety net. |
| **Query rewriter** | `rag/query_rewriter.py` | `llama3.2:3b` | **`llama3.2:3b`** (keep) | Already pulled | The rewriter outputs a short reformulated query (~10-20 tokens). It runs before retrieval on every request, so latency matters. A 3B model is sufficient for coreference resolution ("summarize the above" → "summarize the FastAPI path parameter docs"). If rewrite quality is poor in eval, upgrade to `llama3.1:8b` — but try 3B first. |
| **Embeddings** | `ingestion/embedder.py` | `nomic-embed-text` | **`nomic-embed-text`** (keep) | Already pulled | 768-d, multilingual, well-tested in the project. The Qdrant collection is sized for 768-d. Changing the embedder requires re-indexing the entire corpus + creating a new collection — defer to a separate phase (see Section 7). |
| **Warm-up** | `api/main.py` | `llama3.2:3b` | **`qwen2.5:14b`** | Already pulled | The warm-up call pre-loads the generator's model into memory so the first real query is not slow. It should match the generator model. Set `OLLAMA_WARMUP_MODEL=qwen2.5:14b` in `.env`. |

### Summary: which model for which role

```
┌─────────────────────────────────────────────────────────────────────┐
│  qwen2.5:14b (9.0 GB)     →  generator + eval judge + warm-up      │
│  llama-guard3:8b (4.9 GB) →  content safety judge (OutputGuardrail) │
│  llama3.1:8b (4.9 GB)     →  injection judge (InputGuardrail)       │
│  llama3.2:3b (2.0 GB)     →  query classifier + query rewriter      │
│  nomic-embed-text (274 MB)→  dense embeddings (768-d)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Models to Pull

### Required pulls (2 models)

#### 1. `llama3.1:8b` — injection judge + doc's production target

```bash
ollama pull llama3.1:8b
```

| Attribute | Value |
|---|---|
| Size | 4.9 GB (Q4_K_M) |
| Parameters | 8B |
| Context | 128K tokens |
| Architecture | Llama 3.1 |
| Why | Architecture doc's documented production target (deviation D2). Used for the injection judge — needs enough depth to catch subtle prompt-injection patterns that a 3B model misses. Also serves as a comparison baseline against `qwen2.5:14b` for the generator role. |

#### 2. `llama-guard3:8b` — purpose-built content safety classifier

```bash
ollama pull llama-guard3:8b
```

| Attribute | Value |
|---|---|
| Size | 4.9 GB (Q4_K_M) |
| Parameters | 8B (fine-tuned from Llama 3.1 8B) |
| Context | 128K tokens |
| Architecture | Llama Guard 3 (Meta) |
| Why | Purpose-built for content safety classification against the MLCommons 14-hazard taxonomy. Directly addresses F500 action item #1's future action item #1: "Swap `GUARD_MODEL` to Llama Guard 3 behind a config flag, pinned to a specific tag." Replaces the repurposed `llama3.2:3b` chat model as the output harm judge. |

### Already available (no pull needed)

| Model | Size | Role |
|---|---|---|
| `qwen2.5:14b` | 9.0 GB | Generator + eval judge + warm-up (**already pulled, not used — start using it**) |
| `llama3.2:3b` | 2.0 GB | Query classifier + query rewriter + fallback for all LLM roles |
| `nomic-embed-text` | 274 MB | Dense embeddings (768-d) |

### Optional future pulls (embeddings upgrade — requires re-indexing)

See [Section 7](#7-embedding-model--keep-vs-upgrade) for the full trade-off analysis.

| Model | Size | Dim | Context | When to consider |
|---|---|---|---|---|
| `bge-m3` | 1.2 GB | 1024 | 8K | If you want dense + sparse in one model (replaces both `nomic-embed-text` and the hashing-trick sparse). Requires re-indexing + new Qdrant collection. |
| `mxbai-embed-large` | 670 MB | 1024 | 512 | If you want stronger English retrieval quality. Requires re-indexing + new collection. |
| `snowflake-arctic-embed` | 669 MB | 1024 | 512 | Alternative to mxbai — similar quality, same trade-offs. |

---

## 5. Memory Budget Analysis

### M1 32 GB RAM allocation

| Component | Estimated RAM |
|---|---|
| macOS + system | ~8 GB |
| Qdrant (Docker/Colima) | ~0.5 GB |
| FastAPI backend | ~0.3 GB |
| React dev server (Vite) | ~0.3 GB |
| Colima VM overhead | ~1 GB |
| **Available for Ollama models** | **~22 GB** |

### Model loading scenarios

| Scenario | Models loaded | Total model RAM | Fits? |
|---|---|---|---|
| **Hot path (default)** | `qwen2.5:14b` + `llama3.2:3b` + `nomic-embed-text` | 11.3 GB | Yes (comfortable) |
| **Full guardrails** | + `llama-guard3:8b` + `llama3.1:8b` | 21.1 GB | Yes (tight — may swap if memory pressure) |
| **Comparison testing** | `qwen2.5:14b` + `llama3.1:8b` + `llama3.2:3b` + `nomic-embed-text` | 16.2 GB | Yes |

### Ollama multi-model loading

Ollama keeps models loaded based on `OLLAMA_MAX_LOADED_MODELS` (default: 2 on
most setups, but auto-tuned based on available memory). On 32 GB M1, Ollama
will typically keep 2-3 models resident. When a model is evicted, the next
call to it re-loads it (~5-10s for a 9 GB model on M1).

**Recommendation:** set `OLLAMA_MAX_LOADED_MODELS=4` in the Ollama launch
config so all hot-path models stay resident:

```bash
# ~/.ollama/config.json or launchctl env
OLLAMA_MAX_LOADED_MODELS=4
```

This ensures the generator (`qwen2.5:14b`), guardrails (`llama3.2:3b` +
`llama-guard3:8b` / `llama3.1:8b`), and embedder (`nomic-embed-text`) are all
resident — no cold-load latency on any request.

---

## 6. Configuration Changes Required

### `.env.example` additions

```bash
# ── Ollama models (practice-rag) ──
# Generator model (answer generation). qwen2.5:14b = strong reasoning, 32K context.
OLLAMA_GENERATION_MODEL=qwen2.5:14b
# Guardrail model for query classifier + query rewriter (fast, short-output tasks).
OLLAMA_GUARD_MODEL=llama3.2:3b
# Content safety judge (purpose-built). llama-guard3:8b = Meta Llama Guard 3.
OLLAMA_SAFETY_MODEL=llama-guard3:8b
# Injection judge (general-purpose, needs depth). llama3.1:8b = doc's production target.
OLLAMA_INJECTION_MODEL=llama3.1:8b
# Embedding model (768-d). Changing requires re-indexing the corpus.
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
# Warm-up model (should match the generator).
OLLAMA_WARMUP_MODEL=qwen2.5:14b
# Eval judge model (should be the strongest available model).
EVAL_JUDGE_MODEL=qwen2.5:14b
```

### Code changes (constants → env-driven)

Each model constant should read from env with the current value as fallback:

| File | Constant | Change to |
|---|---|---|
| `rag/generator.py` | `GENERATION_MODEL` | `os.getenv("OLLAMA_GENERATION_MODEL", "qwen2.5:14b")` |
| `api/guardrails.py` | `GUARD_MODEL` | `os.getenv("OLLAMA_GUARD_MODEL", "llama3.2:3b")` |
| `api/guardrails.py` | (new) `SAFETY_MODEL` | `os.getenv("OLLAMA_SAFETY_MODEL", "llama-guard3:8b")` |
| `api/guardrails.py` | (new) `INJECTION_MODEL` | `os.getenv("OLLAMA_INJECTION_MODEL", "llama3.1:8b")` |
| `rag/query_rewriter.py` | `REWRITE_MODEL` | `os.getenv("OLLAMA_REWRITE_MODEL", "llama3.2:3b")` |
| `api/main.py` | `OLLAMA_WARMUP_MODEL` | Already env-driven — update `.env` default |
| `eval/run_eval.py` | `DEFAULT_JUDGE_MODEL` | Already env-driven — update `.env` default |

> **Note:** The guardrails split (`GUARD_MODEL` → `SAFETY_MODEL` +
> `INJECTION_MODEL`) requires a code change in `api/guardrails.py` to pass
> different models to `InputGuardrail` vs `OutputGuardrail`. Currently both
> use the same `GUARD_MODEL`. This is a Phase 1 code change, not just a
> config change.

---

## 7. Embedding Model — Keep vs. Upgrade

### Recommendation: keep `nomic-embed-text` for now

**Why keep:**
- 768-d, multilingual, well-tested in the project.
- The Qdrant collection (`rag/qdrant_collection.py`) is sized for 768-d.
  Changing the embedder requires creating a new collection + re-indexing the
  entire corpus — a batch job, not a runtime swap.
- `nomic-embed-text` is the de-facto Ollama default for embeddings and is
  well-regarded for English documentation retrieval.
- The project's hybrid retrieval (dense + hashing-trick sparse + RRF fusion)
  already compensates for dense-embedding weaknesses.

**When to upgrade (defer to a separate phase):**

| Model | Dim | Size | Context | Why upgrade |
|---|---|---|---|---|
| `bge-m3` | 1024 | 1.2 GB | 8K | Dense + multi-vector + sparse in one model — could replace both `nomic-embed-text` (dense) and the hashing-trick sparse embedder with a single, stronger model. Multilingual. This is the most interesting upgrade but requires re-architecting the retrieval pipeline. |
| `mxbai-embed-large` | 1024 | 670 MB | 512 | Stronger English retrieval than `nomic-embed-text` (competitive with cloud embedding models). Simple like-for-like dense swap. 512 context is tight for long chunks (current chunk size is 512 tokens — at the limit). |
| `snowflake-arctic-embed` | 1024 | 669 MB | 512 | Similar to `mxbai-embed-large` — strong English, 1024-d. Same 512-context limitation. |

**Upgrade path (if pursued later):**
1. Pull the new embedder (`ollama pull bge-m3`).
2. Create a new Qdrant collection sized for the new dimensionality (1024-d).
3. Re-run the ingestion pipeline against the new collection.
4. Run a retrieval recall@5 comparison: old collection vs. new collection.
5. If the new embedder wins, swap the default collection + update
   `EMBEDDING_MODEL` + `EMBEDDING_DIM` + `rag/qdrant_collection.py`.
6. Keep the old collection as a fallback during the transition.

---

## 8. Pull Commands Summary

### Required (2 pulls, ~9.8 GB total)

```bash
# Injection judge + doc's production target (4.9 GB)
ollama pull llama3.1:8b

# Content safety judge — purpose-built (4.9 GB)
ollama pull llama-guard3:8b
```

### Already available (no pull needed)

```bash
# Already pulled — just start using it for generator + eval judge
# qwen2.5:14b (9.0 GB) — currently idle

# Already pulled — keep for classifier + rewriter + fallback
# llama3.2:3b (2.0 GB)

# Already pulled — keep for embeddings
# nomic-embed-text (274 MB)
```

### Optional (embeddings upgrade — defer to a separate phase)

```bash
# Only if pursuing the embedding upgrade (requires re-indexing)
# bge-m3 (1.2 GB, 1024-d, dense+sparse)
ollama pull bge-m3

# mxbai-embed-large (670 MB, 1024-d, strong English)
ollama pull mxbai-embed-large
```

### Verify all models are available

```bash
ollama list
```

Expected output after pulls:

```
NAME                       ID              SIZE
llama3.1:8b                <id>            4.9 GB
llama-guard3:8b            <id>            4.9 GB
llama3.2:3b                a80c4f17acd5    2.0 GB
qwen2.5:14b                7cdf5a0187d5    9.0 GB
nomic-embed-text:latest    0a109f422b47    274 MB
```

---

## F500 Enterprise Note

These Ollama model recommendations improve the project's quality bar but do
**not** close the F500 enterprise gaps. The local models are:

- **Not version-pinned** to a specific tag (Ollama `latest` tags drift).
- **Not eval-gated** — swapping models without measuring FPR/FNR/faithfulness
  delta is a practice-stage shortcut (F500 item #2).
- **Not audited** — model versions are not recorded in trace metadata for
  incident correlation (F500 item #8, action item #8).

For production, each model should be pinned to a specific digest
(`ollama pull llama3.1:8b@sha256:...`) and the version recorded in the
Langfuse trace. See `docs/F500_ENTERPRISE_ACTION_ITEMS.md` for the full gap
analysis.

---

*This document is a planning artifact. Code changes to wire these models are
tracked in `CHANGELOG.md` as they land.*
