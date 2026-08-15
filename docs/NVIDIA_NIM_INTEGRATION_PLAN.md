# NVIDIA NIM Integration Plan — Model Selections, Fallback Architecture & F500 Gaps

> Companion to `docs/NVIDIA_NIM_FREE_MODELS.md` (catalog) and
> `docs/F500_ENTERPRISE_ACTION_ITEMS.md` (item #8 — production-grade gaps).
>
> **Status:** Planning doc — no code integration promoted to production path yet.
> **Date:** 2026-08-15

---

## Table of Contents

1. [Design Principle — Configurable Primary with Graceful Degradation](#1-design-principle--configurable-primary-with-graceful-degradation)
2. [Model Selections with Reasoning](#2-model-selections-with-reasoning)
   - [2.1 Answer Generation (`rag/generator.py`)](#21-answer-generation-raggeneratorpy)
   - [2.2 Content Safety Guard (`api/guardrails.py` — input + output judge)](#22-content-safety-guard-apiguardrailspy--input--output-judge)
   - [2.3 Topic Control Guard (`api/guardrails.py`)](#23-topic-control-guard-apiguardrailspy)
   - [2.4 PII Scrubbing (`api/guardrails.py`)](#24-pii-scrubbing-apiguardrailspy)
   - [2.5 Embedding / RAG Retrieval (`ingestion/embedder.py`)](#25-embedding--rag-retrieval-ingestionembedderpy)
   - [2.6 Reranking (not yet implemented)](#26-reranking-not-yet-implemented)
3. [Fallback Architecture](#3-fallback-architecture)
4. [What Does NOT Change](#4-what-does-not-change)
5. [F500 Enterprise Gaps (Summary)](#5-f500-enterprise-gaps-summary)
6. [Integration Phases](#6-integration-phases)

---

## 1. Design Principle — Configurable Primary with Graceful Degradation

The project is a **local, $0-cost, single-user practice project** (per README +
architecture doc deviations D1/D2). NVIDIA NIM is a **free tier with no SLA**,
a **shared ~40 RPM global rate limit**, a **rotating catalog** (models get
retired → 404), and **data egress to NVIDIA infrastructure with no DPA**.

Making NIM the unconditional primary would couple the project's default
behavior to an unreliable free tier. The correct enterprise pattern — and the
one the codebase already follows (LLM judge → regex fallback) — is
**configurable primary with graceful degradation**:

```
┌─────────────────────────────────────────────────────────────┐
│  DEFAULT (no config flag):                                  │
│    Ollama (local)  →  regex/keyword fallback                │
│                                                             │
│  OPT-IN (NIM_ENABLED=true):                                 │
│    NIM (hosted)  →  Ollama (local)  →  regex/keyword        │
│                                                             │
│  PRODUCTION (future, after F500 gaps closed):               │
│    Paid-tier hosted LLM (SLA)  →  Ollama  →  regex/keyword  │
└─────────────────────────────────────────────────────────────┘
```

**Key correction on the fallback direction:**

- Ollama is **not** demoted to fallback by default. It remains the **default
  primary** when NIM is not opted into.
- When NIM **is** explicitly enabled via config, NIM becomes primary and
  Ollama becomes the **automatic fallback** on NIM failure (rate limit, 404,
  network error, circuit-breaker open).
- This preserves the local-first invariant while enabling comparison testing
  against stronger hosted models.

This matches the existing `CircuitBreaker` pattern in `api/observability.py`
(3 consecutive failures → open for 30s → half-open probe) — the same breaker
can be extended to wrap NIM calls.

---

## 2. Model Selections with Reasoning

Each selection below records: the NIM model, why it was chosen over
alternatives, the current Ollama baseline it would augment, and the trade-offs.

### 2.1 Answer Generation (`rag/generator.py`)

| Field                | Value                                                              |
| -------------------- | ------------------------------------------------------------------ |
| **Current (Ollama)** | `llama3.2:3b` (default), `llama3.1:8b` (doc's production target)   |
| **NIM primary**      | `moonshotai/kimi-k2.6`                                             |
| **NIM alternatives** | `z-ai/glm-5.2`, `deepseek-ai/deepseek-v4-flash`                    |
| **Fallback**         | Ollama `llama3.2:3b` → refusal ("I don't have enough information") |

**Why `moonshotai/kimi-k2.6` as the NIM primary:**

- **262K context** — comfortably fits the generator's 5-chunk × 512-token
  context window + system prompt + history, with headroom for longer
  conversations. The current Ollama `num_ctx=8192` is tight for multi-turn.
- **Agentic + tool use** — successor to K2.5; the doc (Section 10) flags it
  as the model that prompted the research. Aligns with the project's
  direction toward agentic RAG (query rewriter, classifier, orchestrator).
- **Confirmed online** (free tier, 2026-08-06 probe) — not in the
  "check provider" list (Section 8), so it has a lower retirement risk.
- **Chat-tuned** — drop-in for the existing Chat Completions call shape.

**Why not `z-ai/glm-5.2` as primary (kept as alternative):**

- 1.0M context is overkill for a 5-chunk RAG context window; the extra
  context headroom is unused cost/latency for this use case.
- Flagship models on free tiers carry higher cold-start latency (Section 12
  caveat: "Large models cold-start — 253B/397B models may time out").
- Better reserved for a **second comparison run** to measure whether the
  1M context + stronger reasoning justifies the latency trade-off.

**Why not `deepseek-ai/deepseek-v4-flash` as primary (kept as alternative):**

- 284B MoE, optimized for coding + agents — strong but coding-focused;
  the corpus is FastAPI/Pydantic/SQLModel **documentation**, not code
  generation. Kimi K2.6's general chat + agentic profile is a closer fit.
- Listed in Section 8 ("check provider") for the `deepseek-v4-pro` variant;
  the `flash` variant is confirmed online but the family has availability
  uncertainty.

**Trade-offs accepted:**

- NIM's 40 RPM shared limit means burst traffic (e.g. a multi-question eval
  run) will hit 429s → fallback to Ollama kicks in. Acceptable for
  comparison testing; would need a multi-key/multi-provider router for
  production volume.
- Data egress: prompts (query + retrieved context + history) leave the
  machine to NVIDIA. The corpus is public docs with no real PII, so this
  is acceptable for the practice stage. **Becomes a blocker if the corpus
  ever contains PII/PHI** — see F500 item #8, action item #9.

---

### 2.2 Content Safety Guard (`api/guardrails.py` — input + output judge)

| Field                | Value                                                         |
| -------------------- | ------------------------------------------------------------- |
| **Current (Ollama)** | `llama3.2:3b` LLM judge, 8 output tokens, `temperature=0.0`   |
| **NIM primary**      | `nvidia/llama-3.1-nemoguard-8b-content-safety`                |
| **NIM alternative**  | `meta/llama-guard-4-12b`                                      |
| **Fallback**         | Ollama `llama3.2:3b` judge → regex-only (already implemented) |

**Why `nvidia/llama-3.1-nemoguard-8b-content-safety` as the NIM primary:**

- **Purpose-built** for content safety classification — not a general chat
  model repurposed as a judge. This directly addresses F500 action item #1
  ("a 3B general-purpose chat model is not a security boundary an enterprise
  compliance review would accept").
- **8B parameter class** — comparable latency to the current `llama3.2:3b`
  judge, so swapping it in does not blow the guardrail latency budget.
- **NVIDIA-hosted** — same provider as the generator, so one API key + one
  rate-limit budget covers both (simpler ops than mixing providers).
- **Confirmed online** (Section 9.5).

**Why not `meta/llama-guard-4-12b` as primary (kept as alternative):**

- 12B is heavier than the 8B nemoguard; higher latency per guardrail call,
  and guardrails run on **every** input + output (2 calls per request),
  so the latency multiplier matters.
- 1.0M context is unnecessary for a safety judge that looks at one message
  or one answer at a time.
- Better reserved as a **second comparison run** to measure whether the
  larger model reduces FPR/FNR enough to justify the latency cost — but
  that requires the eval gate from F500 item #2, which does not yet exist.

**Trade-offs accepted:**

- Sending the user message (input judge) and the generated answer (output
  judge) to a hosted model = **data egress of user content**. For the
  practice corpus (public docs, single user) this is acceptable. **Becomes
  a blocker the moment the corpus or user input contains PII** — F500 item
  #5 already flags this: "If moving to a hosted guardrail model, history
  scrubbing becomes mandatory."
- No eval gate yet (F500 item #2) — swapping the judge without measuring
  FPR/FNR delta is a practice-stage shortcut, not a production practice.

---

### 2.3 Topic Control Guard (`api/guardrails.py`)

| Field                | Value                                                               |
| -------------------- | ------------------------------------------------------------------- |
| **Current (Ollama)** | regex fallback (no LLM topic-control model)                         |
| **NIM primary**      | `nvidia/llama-3.1-nemoguard-8b-topic-control`                       |
| **Fallback**         | regex/keyword fallback (already implemented in `classify_keywords`) |

**Why `nvidia/llama-3.1-nemoguard-8b-topic-control`:**

- **Purpose-built** topic-control guard — directly addresses the gap noted
  in `NVIDIA_NIM_FREE_MODELS.md` Section 13: the current project uses a
  "regex fallback" for topic control, which is brittle for nuanced
  off-topic detection.
- Same 8B parameter class as the content-safety guard → consistent latency
  profile and same provider/key.
- Confirmed online (Section 9.5).

**Trade-offs accepted:**

- Same data-egress caveat as 2.2 — the user query leaves the machine to
  NVIDIA for topic classification.
- The current `QueryClassifier` already routes `off_topic` queries to a
  canned refusal, so a NIM topic-control model would **augment** the
  classifier, not replace it. The fallback to `classify_keywords` remains.

---

### 2.4 PII Scrubbing (`api/guardrails.py`)

| Field                | Value                                        |
| -------------------- | -------------------------------------------- |
| **Current (Ollama)** | regex (`scrub_pii` — email + phone patterns) |
| **NIM primary**      | `nvidia/gliner-pii`                          |
| **Fallback**         | regex `scrub_pii` (already implemented)      |

**Why `nvidia/gliner-pii`:**

- **ML-based PII detection** — catches PII patterns regex misses
  (names, addresses, SSNs, account numbers, contextual PII). Regex is
  fundamentally limited to pattern-shaped PII (emails, phones).
- Directly addresses F500 item #5's future action item #1: "PII-scrub
  history before it enters judge/classifier/rewriter prompts."

**Trade-offs accepted:**

- Listed in Section 8 ("check provider") — availability is uncertain;
  must probe before relying on it. The regex fallback is the safety net.
- **PII detection runs on text that may itself contain PII** — sending
  raw text to a hosted PII detector to find PII is a chicken-and-egg
  data-egress problem. For the practice corpus (no real PII) this is
  theoretical. **In production, PII scrubbing must happen locally first**
  (regex), and the hosted ML model should only see already-regex-scrubbed
  text as a second-pass refinement. This is a non-trivial architectural
  point — documented as F500 item #8, action item #9.

---

### 2.5 Embedding / RAG Retrieval (`ingestion/embedder.py`)

| Field                | Value                                                               |
| -------------------- | ------------------------------------------------------------------- |
| **Current (Ollama)** | `nomic-embed-text` (768-d dense) + hashing-trick sparse (BM25-like) |
| **NIM primary**      | `nvidia/llama-nemotron-embed-1b-v2`                                 |
| **NIM alternative**  | `baai/bge-m3`                                                       |
| **Fallback**         | Ollama `nomic-embed-text` (already implemented)                     |

**Why `nvidia/llama-nemotron-embed-1b-v2` as the NIM primary:**

- **Multilingual (26 languages)** + **long-doc QA retrieval** tuning — the
  doc (Section 9.4) specifically tags it for "long-doc QA retrieval,"
  which is the exact use case (FastAPI/Pydantic/SQLModel documentation).
- 1B parameter class — comparable inference cost to `nomic-embed-text`,
  so swapping does not dramatically change per-query latency.
- NVIDIA-hosted → same key/rate-limit budget as the generator + guards.

**Why not `baai/bge-m3` as primary (kept as alternative):**

- `bge-m3` supports **dense + multi-vector + sparse** retrieval in one
  model — architecturally attractive because it could replace both the
  Ollama dense embedder **and** the hashing-trick sparse embedder with a
  single model. That is a bigger refactor than a like-for-like dense
  embedding swap.
- Better reserved for a **second phase** that re-architects the retrieval
  pipeline to use a unified dense+sparse model, rather than bolting it
  onto the current hybrid (Ollama dense + hashing sparse) design.

**Critical trade-off — embedding swap requires re-indexing:**

- Embeddings are **not interoperable**: vectors from `nomic-embed-text`
  (768-d) cannot be mixed with vectors from `nemotron-embed-1b-v2`
  (different dimensionality/space). Swapping the embedder requires
  **re-ingesting the entire corpus** with the new model.
- The Qdrant collection (`rag/qdrant_collection.py`) is sized for 768-d.
  A NIM embedder with a different dimensionality requires a new collection
  - re-ingestion. This is a **batch job**, not a runtime swap.
- **Decision:** the NIM embedder is **not** a runtime fallback. It is a
  **separate collection** (e.g. `ctc_rag_nim`) populated by a separate
  ingestion run, used for **comparison retrieval quality** (recall@5)
  against the Ollama-indexed collection. The Ollama collection remains
  the default for the live app.

---

### 2.6 Reranking (not yet implemented)

| Field           | Value                                                    |
| --------------- | -------------------------------------------------------- |
| **Current**     | Not implemented (no reranker in the retrieval pipeline)  |
| **NIM primary** | `nvidia/llama-nemotron-rerank-1b-v2`                     |
| **Fallback**    | No reranking (current behavior — retrieved order stands) |

**Why `nvidia/llama-nemotron-rerank-1b-v2`:**

- The doc (Section 13) flags reranking as a **net-new capability** ("not
  implemented" → "GPU-accelerated reranking"). This is not a swap; it is
  an addition.
- Reranking sits between retrieval and generation: retrieve top-K (e.g. 20) → rerank to top-5 → feed to generator. This is a standard RAG
  quality lever the project does not yet have.

**Trade-offs accepted:**

- Adds one extra hosted call per query (retrieve → rerank → generate),
  consuming rate-limit budget. With a 40 RPM shared cap, this tightens
  the budget further.
- **No fallback** if the reranker fails — the retrieved order stands.
  This is acceptable: reranking is a quality enhancement, not a
  correctness requirement. A failed rerank degrades to the current
  behavior (no reranking), which is the status quo.

---

## 3. Fallback Architecture

The fallback chain is **per-component** because different components have
different failure semantics.

### 3.1 Generator (answer generation)

```
NIM_ENABLED=true:
  NIM kimi-k2.6  ──(429/404/timeout/circuit-open)──▶  Ollama llama3.2:3b
                                                          │
                                                          └─(failure)─▶  refusal

NIM_ENABLED=false (default):
  Ollama llama3.2:3b  ──(failure)──▶  refusal
```

- The existing `CircuitBreaker` (3 failures → 30s open → half-open probe)
  wraps the NIM call. On `CircuitOpenError`, fall back to Ollama
  immediately without a network round-trip.
- NIM-specific errors (429 rate limit, 404 model retired) are mapped to
  fallback, not retried — retrying a 429 against a shared global limit
  would waste the budget.

### 3.2 Guardrails (input judge, output judge, topic control, PII)

```
NIM_ENABLED=true:
  NIM nemoguard/gliner  ──(failure)──▶  Ollama llama3.2:3b judge
                                              │
                                              └─(failure)──▶  regex/keyword

NIM_ENABLED=false (default):
  Ollama llama3.2:3b judge  ──(failure)──▶  regex/keyword
```

- This **extends** the existing 2-tier (LLM → regex) to 3-tier
  (NIM LLM → Ollama LLM → regex) when NIM is enabled.
- Guardrail failures **never block** legitimate traffic — the existing
  graceful-degradation contract is preserved.

### 3.3 Embeddings (retrieval)

```
NIM_ENABLED=true:
  NIM nemotron-embed-1b-v2  (separate Qdrant collection, comparison only)
  Ollama nomic-embed-text   (default collection, live app)

NIM_ENABLED=false (default):
  Ollama nomic-embed-text  (default collection, live app)
```

- **No runtime fallback** for embeddings — vectors are not interoperable.
- The NIM embedder populates a **separate collection** for retrieval-quality
  comparison (recall@5). The live app continues to query the Ollama
  collection.
- A production-grade unified embedder swap is deferred to a later phase
  that includes re-ingestion + eval gate (F500 item #8, action item #4).

### 3.4 Reranker (net-new)

```
NIM_ENABLED=true:
  NIM nemotron-rerank-1b-v2  ──(failure)──▶  no reranking (retrieved order stands)

NIM_ENABLED=false (default):
  no reranking
```

- Reranking is a quality enhancement; failure degrades to the current
  behavior (no reranking). No fallback model — there is no local reranker.

---

## 4. What Does NOT Change

- **Default behavior with no config flag** — Ollama remains primary for
  all components. The project stays local-first, $0-cost, no external
  dependency, matching README + architecture doc deviations D1/D2.
- **Regex/keyword fallbacks** in `api/guardrails.py` — these remain the
  final safety net for every guardrail, regardless of which LLM tier is
  enabled.
- **System prompt template** in `rag/generator.py` — the NIM generator
  uses the same `SYSTEM_PROMPT_TEMPLATE` (NIM is OpenAI-compatible; the
  prompt is model-agnostic).
- **Circuit breaker contract** — the existing `CircuitBreaker` semantics
  (3 failures → 30s open → half-open probe) are extended to NIM, not
  replaced.
- **Lazy-client pattern** — NIM clients are lazily constructed (like the
  Ollama clients) so unit tests inject mocks without touching the network.
- **Qdrant collection schema** — the default 768-d collection is unchanged.
  A NIM embedder uses a separate collection.

---

## 5. F500 Enterprise Gaps (Summary)

This integration is **not** F500 enterprise production-grade. The full gap
analysis lives in `docs/F500_ENTERPRISE_ACTION_ITEMS.md` item #8. Summary:

| #   | Gap                             | Practice-stage stance                           | Production blocker?            |
| --- | ------------------------------- | ----------------------------------------------- | ------------------------------ |
| 1   | No SLA/SLO                      | Free tier, no SLA — fine for comparison testing | Yes                            |
| 2   | Shared 40 RPM rate limit        | Acceptable for single-user eval runs            | Yes                            |
| 3   | Data egress, no DPA             | Public docs, no PII — acceptable                | Yes (if PII/PHI)               |
| 4   | Long-lived API key in client    | Practice project — acceptable                   | Yes (secrets manager)          |
| 5   | No model version pinning        | Probe script (Section 11) for monitoring        | Yes (pin in IaC)               |
| 6   | No eval gate on swap            | Comparison testing is the eval — no SLO yet     | Yes (blocking CI gate)         |
| 7   | No multi-provider failover      | Ollama fallback is the current failover         | Yes (router + circuit breaker) |
| 8   | No audit trail for hosted calls | Langfuse spans capture metadata                 | Yes (formalize)                |
| 9   | PII egress to hosted guardrail  | No PII in corpus — acceptable                   | Yes (scrub upstream)           |
| 10  | No retry/backoff/timeout policy | Circuit breaker is the current policy           | Yes (backoff + jitter)         |

**Bottom line:** NIM integration as described here is appropriate for
**comparison testing in the practice stage**. It must not be promoted to
the production generation/guardrail path until all 10 gaps are closed.

---

## 6. Integration Phases

### Phase 0 — Documentation (this doc + F500 item #8) ✅

- Catalog research (`NVIDIA_NIM_FREE_MODELS.md`).
- Model selections + fallback architecture (this doc).
- F500 gap analysis recorded (`F500_ENTERPRISE_ACTION_ITEMS.md` item #8).

### Phase 1 — Generator comparison test (smallest blast radius) ✅ (code landed)

- ✅ Add `NIM_ENABLED` + `NVIDIA_API_KEY` env vars to `.env.example`.
- ✅ Add a NIM-backed `Generator` variant behind the config flag
  (`rag/nim_generator.py` — `NIMGenerator` + `FallbackGenerator` +
  `build_generator()` factory).
- ✅ Wire fallback: NIM → Ollama → refusal, via the existing
  `CircuitBreaker` (owned by `FallbackGenerator`, checked before the
  primary attempt; failures recorded on pre-first-token failure).
- ✅ Unit tests: `tests/test_rag_nim_generator.py` (29 tests, all mocked).
- ✅ `api/deps.py` wired through `build_generator()` — default path
  unchanged (Ollama primary when `NIM_ENABLED` is unset/false).
- ⬜ Run a comparison eval: same queries, Ollama vs. NIM answers,
  side-by-side review (no automated SLO yet — that is F500 item #2).
  This is a **manual step** requiring a live `NVIDIA_API_KEY` + Ollama
  running; not automatable in CI at the practice stage.

### Phase 2 — Guardrail comparison test ✅ (code landed)

- ✅ Add NIM-backed `InputGuardrail` / `OutputGuardrail` / `QueryClassifier`
  variants behind the config flag (`api/nim_guardrails.py` —
  `NIMGuardrailClient` + `NIMInputGuardrail` + `NIMOutputGuardrail` +
  `NIMQueryClassifier` + `build_guardrail_suite()` factory).
- ✅ Wire 3-tier fallback: NIM LLM → Ollama LLM → regex/keyword. The three
  subclasses override only the LLM-judge / LLM-classify method of the
  existing guardrail classes, so the regex/keyword tiers and the
  `GuardrailDecision` / `QueryClassification` contracts are inherited
  unchanged.
  - `NIMInputGuardrail` — regex injection scan (tier 1, inherited) → NIM
    content-safety judge (tier 2) → Ollama injection judge (tier 3,
    inherited) → regex-only on all-LLM failure.
  - `NIMOutputGuardrail` — PII regex scrub (tier 1, inherited) → NIM
    content-safety judge (tier 2) → Ollama harmful-content judge (tier 3,
    inherited) → scrub-only on all-LLM failure.
  - `NIMQueryClassifier` — NIM topic-control guard (binary on-topic/off-topic,
    tier 1) → Ollama 5-way classifier (tier 2, inherited) → keyword fallback
    (tier 3, inherited). NIM "off-topic" routes directly to `off_topic`;
    NIM "on-topic" defers to the Ollama 5-way for fine-grained routing.
- ✅ The NIM guardrail client gets its **own** `CircuitBreaker` (separate
  from the generator's NIM breaker and the Ollama breaker). All three NIM
  guardrails share one `NIMGuardrailClient` (one API key + one rate-limit
  budget + one breaker).
- ✅ Unit tests: `tests/test_api_nim_guardrails.py` (50 tests, all mocked).
- ✅ `api/deps.py` wired through `build_guardrail_suite()` — default path
  unchanged (plain Ollama `GuardrailSuite` when `NIM_ENABLED` is unset/false).
- ⬜ The hosted PII model (`nvidia/gliner-pii`, §2.4) is **not** wired here.
  It is in the "check provider" list (uncertain availability), carries a
  chicken-and-egg data-egress problem, and F500 item #5 flags PII scrubbing
  upstream of hosted models as a blocker. The regex `scrub_pii` (inherited,
  always-on) remains the PII tier. The hosted PII refinement is deferred to
  a sub-phase after availability is probed.
- ⬜ Run a comparison eval: same queries, Ollama-judge vs. NIM-judge
  verdicts, side-by-side FPR/FNR review (no automated SLO yet — that is
  F500 item #2). This is a **manual step** requiring a live
  `NVIDIA_API_KEY` + Ollama running; not automatable in CI at the practice
  stage.
- **Do not** enable NIM guardrails on any corpus with real PII until
  F500 item #5 (PII scrubbing upstream) is closed.

### Phase 3 — Embedding comparison test (separate collection)

- Add a NIM embedder ingestion path that populates a separate Qdrant
  collection (`ctc_rag_nim`).
- Run retrieval recall@5 comparison: Ollama collection vs. NIM collection.
- **Do not** swap the live app's default collection until a re-ingestion
  - eval gate (F500 item #8, action item #4) is in place.

### Phase 4 — Reranker (net-new capability)

- Add a reranker stage between retrieval and generation, behind the config
  flag.
- Measure answer-quality uplift (faithfulness, relevance) with vs. without
  reranking.

### Phase 5 — Production promotion (BLOCKED on F500 item #8)

- Do not promote NIM (or any hosted provider) to the default production
  path until all 10 F500 gaps are closed.
- At that point, the architecture flips: paid-tier hosted LLM (with SLA)
  becomes primary, Ollama becomes the automatic fallback, regex/keyword
  remains the final safety net.

---

_This document is a planning artifact. Code integration is tracked in
`CHANGELOG.md` as it lands. F500 enterprise gaps are tracked in
`docs/F500_ENTERPRISE_ACTION_ITEMS.md` item #8._
