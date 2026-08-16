# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — TTFT < 800ms SLO gate (online-serving latency threshold)

TTFT (time-to-first-token) was collected by `MetricsCollector` and exposed
in the `GET /api/v1/metrics` snapshot (mean/p50/p95), but there was no
threshold gate asserting the doc's "TTFT < 800ms" online-serving SLO — the
metric was observed but never evaluated against the target. The collector
now evaluates the SLO and surfaces the met/breached status.

- `api/observability.py`:
  - New constants `TTFT_SLO_TARGET_S = 0.8` (seconds) and
    `TTFT_SLO_PERCENTILE = 95.0` (the SLO is evaluated on p95 over the
    retained sample window).
  - New `MetricsCollector.evaluate_ttft_slo(target_s, percentile)` —
    computes the configured percentile of the TTFT samples and returns
    `{target_s, percentile, value_s, status, met, samples}` where
    `status` is `"met"` (value < target), `"breached"` (value >= target),
    or `"no_data"` (no samples recorded yet; `met = None`).
  - New lock-free pure helper `_ttft_slo(samples, target_s, percentile)`
    so `snapshot()` can compute the SLO block while already holding the
    collector lock (avoids reentrant-lock deadlock).
  - `snapshot()` now includes an `slo` key with the gate result, surfaced
    via `GET /api/v1/metrics`.
  - Module docstring updated to document the SLO gate.
- `tests/test_observability.py` — new `TestTtftSloConstants`,
  `TestTtftSloHelper`, and `TestMetricsCollectorTtftSlo` classes (SLO
  constants, no-data / met / breached / boundary-equal / custom-target
  cases, snapshot `slo` block, consistency with `ttft.p95_s`, reset
  clears samples).
- `tests/test_api_metrics.py` — `slo` key assertion + no-data / met /
  breached endpoint cases.
- `docs/CHAT_ASSISTANT_FEATURES.md` — TTFT < 800ms row updated from ❌
  to 🟡 (gate in place; enterprise gap = in-process ring buffer, no
  rolling time window, no alerting on SLO burn).
- `docs/F500_ENTERPRISE_ACTION_ITEMS.md` — item 6 (observability) partial
  progress note recording the SLO gate and the remaining enterprise gaps.

### Added — Ambiguous query → clarification prompt

Queries the classifier labels `ambiguous` (e.g. "how do I use validators?"
without specifying Pydantic vs FastAPI dependency validators, or a bare
pronoun reference like "how do I configure it?" with no disambiguating
context) previously went through the full RAG flow and likely produced a
vague or wrong answer. The orchestrator now short-circuits `ambiguous`-
labeled queries with a clarification prompt instead — no retrieval, no
generation. The user is asked to disambiguate before any answer is
attempted.

- `api/guardrails.py` — new `CLASS_AMBIGUOUS` label added to `VALID_CLASSES`;
  `CLASSIFIER_PROMPT` updated with an `ambiguous` label description;
  `classify_keywords` keyword fallback now detects short queries (1–6 words)
  with bare pronoun/shared-concept markers (`it`/`that`/`the model`/`the
validator`/…) and no library qualifier (FastAPI/Pydantic/SQLModel) →
  `ambiguous`; `_to_classification` routes `ambiguous` through the
  orchestrator (not `handled`).
- `rag/query_clarifier.py` — new module:
  - `QueryClarifier` protocol + `PassthroughQueryClarifier` (returns `None`,
    no LLM call — zero latency, zero risk).
  - `LLMQueryClarifier` — sends a short clarify prompt to Ollama and parses
    the response into a single clarification question (leading
    `Clarification:`/`Q:` labels stripped via `_strip_label_prefix`). Falls
    back to a deterministic heuristic clarifier (`_heuristic_clarify` —
    detects shared-concept nouns like `validator`/`model`/`field`/`cache`/
    `schema`/`session`/`dependency` and proposes a "Do you mean X or Y?"
    prompt) when Ollama is unavailable, returns nothing useful, or errors.
    When both LLM and heuristic find no candidates, returns `None` so the
    orchestrator uses `GENERIC_CLARIFICATION`. Shares the lazy-client
    pattern from `Generator`/`Embedder`/`LLMQueryRewriter`/
    `LLMQueryDecomposer`.
- `rag/orchestrator.py` — new optional `query_clarifier` collaborator
  (defaults to `PassthroughQueryClarifier`); both `stream_answer` and
  `answer` short-circuit `ambiguous`-labeled queries with the clarifier's
  prompt (or `GENERIC_CLARIFICATION` when the clarifier returns `None`).
  The clarifier runs on the **original** query (before rewriting) since the
  user's actual words are what need disambiguation. No retrieval, no
  generation, no rewrite.
- `api/deps.py` — `get_orchestrator` now wires `LLMQueryClarifier()`.
- `tests/test_rag_query_clarifier.py` — new test module (heuristic concept
  detection, label-prefix stripping, passthrough, LLM with fallbacks,
  prompt template, generic clarification).
- `tests/test_guardrails.py` — `CLASS_AMBIGUOUS` import; ambiguity keyword
  fallback tests; LLM `ambiguous` label test + error-fallback test.
- `tests/test_rag_orchestrator.py` — `TestOrchestratorAmbiguity` class
  (short-circuit with clarifier prompt, generic fallback, passthrough
  clarifier, rewriter not called, history forwarding, non-ambiguous labels
  skip clarifier, `answer` method short-circuit).

### Added — Query decomposition for `compare`-labeled queries (multi-query retrieval)

Complex/multi-part queries labeled `compare` by the query classifier (e.g.
"Compare FastAPI and Flask for building REST APIs") previously went through
a single retrieval pass, which often missed chunks strongly relevant to
only one side of the comparison. The orchestrator now decomposes a
`compare`-labeled query into one sub-query per compared subject, runs
retrieval once per sub-query, and merges + deduplicates the results before
assembly/generation — so the generator sees balanced context for both
sides.

- `rag/query_decomposer.py` — new module:
  - `QueryDecomposer` protocol + `PassthroughQueryDecomposer` (returns
    `[query]`, no split, no LLM call — zero latency, zero risk).
  - `LLMQueryDecomposer` — sends a short split prompt to Ollama and parses
    the response into one sub-query per non-empty line (leading
    numbering/bullet prefixes stripped via `_strip_prefix`). Falls back to
    a deterministic heuristic splitter (`_heuristic_decompose` — splits on
    `vs`/`versus`/`compared to`/`difference between`/`and`, `maxsplit=1`)
    when Ollama is unavailable, returns nothing useful, returns a single
    line equal to the original (no decomposition), or errors. Shares the
    lazy-client pattern from `Generator`/`Embedder`/`LLMQueryRewriter` so
    unit tests inject mocks and never touch the network.
- `rag/orchestrator.py` — new optional `query_decomposer` collaborator
  (defaults to `PassthroughQueryDecomposer`); new `_retrieve` helper runs
  multi-query retrieval only when the classifier labeled the query
  `compare` AND the decomposer returns >1 sub-query (all other labels and
  single-sub-query decompositions stay single-retrieval-pass, identical to
  the prior flow); new `_merge_docs` static helper deduplicates by doc id
  (max RRF score kept) and sorts by score descending. Decomposition runs
  on the **rewritten** query (after coreference resolution), and the
  merged candidate pool is what the optional NIM reranker reranks. Both
  `stream_answer` and `answer` use `_retrieve`.
- `api/deps.py` — `get_orchestrator()` now wires
  `LLMQueryDecomposer()` into the orchestrator.
- `tests/test_rag_query_decomposer.py` — new test file (40 tests):
  `_heuristic_decompose` (15 tests: all connectors, no-connector
  passthrough, empty/whitespace, empty-part guard, case-insensitivity,
  first-connector-only split, connector priority), `_strip_prefix` (9
  tests: numbered/paren/dash/asterisk/bullet prefixes, no-prefix
  passthrough, inline-number no-strip), `PassthroughQueryDecomposer`
  (3 tests), `LLMQueryDecomposer` (13 tests: multi-line split, prefix
  stripping, single-line-equal-to-original heuristic fallback, empty/
  missing/exception fallback, `use_llm=False`, whitespace-line dropping,
  stream/query-in-prompt/close/lazy-client), `DECOMPOSE_PROMPT_TEMPLATE`
  (4 tests).
- `tests/test_rag_orchestrator.py` — new `TestOrchestratorDecomposition`
  (9 tests: compare → multiple retrievals, single-sub-query → single
  retrieval, non-compare label skips decomposition, no-guardrail-suite
  skips decomposition, merged docs deduped by id with max score, merged
  docs sorted by score desc, reranker receives merged pool, `answer`
  method decomposes, decomposition uses rewritten query) +
  `TestMergeDocs` (7 tests for the static merge helper).
- `docs/CHAT_ASSISTANT_FEATURES.md` — Query Classification & Routing
  "Complex/multi-part" line upgraded from ❌ to 🟡 Partial (decomposition
  implemented for the `compare` label; general multi-part compound
  questions are not detected). Core Chat summary updated (🟡 1→2, ❌ 2→1).
  Bottom-line paragraph updated.
- All 832 backend tests pass (`python -m pytest -q`); all 62 frontend
  tests pass (`npm test`).

**F500 enterprise gap:** the decomposer uses a 3B local model with no
eval gate/SLO, no decompose-quality eval (does retrieval recall improve?
does it ever break a good query?), and no audit log of the split. Tracked
in `docs/F500_ENTERPRISE_ACTION_ITEMS.md`.

### Changed — corrected session-store feature status to ✅ Implemented

The "No Redis (SQLite instead) and no Cosmos DB" bullet in
`docs/CHAT_ASSISTANT_FEATURES.md` was mislabeled ❌ Not implemented. The
bullet text describes the _actual practice implementation_, which is fully
working and tested:

- SQLite single-file session store (`api/conversation.py`), no Redis.
- No Cosmos DB; full per-session history lives in SQLite (`get_messages`
  returns every message — no pruning, no TTL, no 90-day retention).
- No 30-min sliding window; the prompt-context window is a fixed turn
  count (`HISTORY_WINDOW=10` turns) via `get_history`/`format_history`.

Relabeled ❌ → ✅ and updated the Core Chat summary count
(✅ 5→6, ❌ 3→2). The production Redis (last 10) + Cosmos DB (90d) tiering
remains a target-only enterprise feature. No code change —
`tests/test_api_conversation.py` (18 tests) already covers session
creation, full-history retrieval, history-window truncation/turn-boundary,
and feedback CRUD.

### Added — `guardrail_replacement` SSE event for output-guardrail blocks

The stream-then-verify guardrail pattern previously had a UX gap: when the
output guardrail blocked the already-streamed answer, the **persisted**
answer was replaced with a refusal, but the **already-streamed tokens stayed
visible** in the UI because SSE is one-way and no swap event was emitted.
The history endpoint reflected the refusal while the live chat bubble showed
the harmful text — an inconsistent and unsafe state.

This adds an optional `event: guardrail_replacement` SSE frame, emitted
**only** when the output guardrail blocks, between the last `delta` and
`sources`:

    event: guardrail_replacement\ndata: {"answer": "<refusal>"}\n\n

The frontend swaps the visible message content for the refusal on receipt,
so the live UI and the persisted history agree.

- `rag/post_processor.py` — `PostProcessResult.guardrail_replacement`
  field (default `None`).
- `rag/orchestrator.py` — `OUTPUT_REFUSAL` constant; `stream_answer` and
  `answer` set `result.guardrail_replacement = OUTPUT_REFUSAL` on output
  block; class docstring updated.
- `api/routes/chat.py` — `build_event_stream` emits
  `event: guardrail_replacement` when `result.guardrail_replacement` is
  set (JSON-encoded `{"answer": ...}`); module docstring updated with the
  new event in the wire-format taxonomy.
- `frontend/src/api.js` — `parseSseFrame` dispatches
  `guardrail_replacement` → `onGuardrailReplacement`; `streamChat`
  accepts and forwards `onGuardrailReplacement`; module docstring updated.
- `frontend/src/ChatWidget.jsx` — `handleSend` wires
  `onGuardrailReplacement` to replace the assistant message `content`
  with the refusal (swap, not append).
- `tests/test_rag_orchestrator.py` — new tests asserting
  `guardrail_replacement` is set on output block (stream + `answer`) and
  `None` when allowed.
- `tests/test_api_chat.py` — new tests asserting the
  `guardrail_replacement` frame is emitted (with correct order + JSON
  payload + persisted refusal) and omitted on non-blocked results.
- `frontend/__tests__/api.test.js` — `parseSseFrame` +
  `streamChat` tests for the `guardrail_replacement` event.
- `frontend/__tests__/ChatWidget.test.jsx` — test that streamed tokens
  are swapped for the refusal on `guardrail_replacement`.
- `docs/CHAT_ASSISTANT_FEATURES.md` — Streaming UX
  "Stream-then-verify pattern" line upgraded from 🟡 to ✅.
- `README.md` — SSE wire-format section updated with the
  `guardrail_replacement` event.

### Changed — SSE wire format switched to named events (delta/sources/metadata/done)

The `POST /api/v1/chat` SSE stream now emits named events instead of the
previous bare `data:` / `event: result` / `data: [DONE]` frames. The new
wire format matches the architecture doc's listed event taxonomy:

    event: delta\ndata: <token>\n\n             # one per generated token
    event: sources\ndata: {"citations":[...]}\n\n  # once, after the last token
    event: metadata\ndata: {json}\n\n           # once, after sources
    event: done\ndata: [DONE]\n\n               # terminal sentinel

The `event: sources` frame carries the post-processed citations (was
previously embedded in the `event: result` frame's full `ChatResponse`).
The `event: metadata` frame carries the session id, groundedness/
confidence score, and Langfuse trace id. The full `ChatResponse`
(including the concatenated `answer`) is still persisted to the session
store and the LRU cache — only the wire format drops the redundant
`answer` field (the frontend reconstructs it from the `delta` events).

This is a **breaking wire-format change**: any client consuming the old
`data: <token>` / `event: result` / `data: [DONE]` frames must be
updated to the named-event format.

- `api/routes/chat.py` — `build_event_stream` + `build_replay_stream`
  now emit `event: delta` / `event: sources` / `event: metadata` /
  `event: done`; module + function docstrings updated.
- `frontend/src/api.js` — `parseSseFrame` dispatches on the `event:`
  field (`delta` → `onToken`, `sources` → `onSources`, `metadata` →
  `onMetadata`, `done` → `onDone`); `streamChat` callback signatures
  updated (`onResult` → `onSources` + `onMetadata`); module docstring
  updated.
- `frontend/src/ChatWidget.jsx` — `handleSend` wires `onSources`
  (citations) and `onMetadata` (session id + confidence) separately
  instead of a single `onResult`.
- `frontend/src/CitationChip.jsx`, `frontend/src/MessageList.jsx` —
  doc comments updated to reference `event: sources` instead of
  `event: result`.
- `schemas/chat.py`, `rag/orchestrator.py` — module/class docstrings
  updated to the named-event format.
- `tests/test_api_chat.py` — all SSE frame assertions updated for the
  new named events (`delta` / `sources` / `metadata` / `done`); the
  `test_no_result_yielded_synthesizes_empty` test now verifies
  persistence instead of the wire payload (the synthesized answer is
  no longer carried on the wire).
- `frontend/__tests__/api.test.js` — `parseSseFrame` + `streamChat`
  tests updated for named events.
- `frontend/__tests__/ChatWidget.test.jsx` — `mockStreamChatTokens`
  helper now calls `onSources` + `onMetadata` instead of `onResult`.
- `docs/CHAT_ASSISTANT_FEATURES.md` — Streaming UX SSE frame line
  upgraded from 🟡 to ✅; Core Chat summary updated (✅4→5, 🟡2→1).
- `README.md` — SSE wire format + frontend parser descriptions updated.

### Added — Query rewriter latency-skip heuristic

`LLMQueryRewriter` now short-circuits the ~1-3s Ollama round-trip for
self-contained queries that don't need coreference resolution. A new
`_needs_rewrite(query, history)` heuristic returns `False` (skip the LLM
call, return the original query) when:

- `history` is empty/whitespace — no conversation context to resolve
  against, or
- the query contains no anaphoric pronouns / phrase markers (matched as
  whole words, case-insensitive, against a curated list: `it`, `this`,
  `that`, `they`, `he`, `she`, `the above`, `the former`, etc.).

Only follow-up queries that actually need coreference resolution (history
present AND an anaphoric marker detected) pay the rewrite cost. The
heuristic is conservative — false negatives (rewriting when not needed)
only waste latency; false positives (skipping when rewriting was needed)
hurt retrieval quality, so the marker list errs toward inclusion.

This partially closes F500 action item #4.4 (budget cap). The p95 latency
SLO itself is still not measured/asserted.

- `rag/query_rewriter.py` — new `_ANAPHORA_MARKERS` regex +
  `_needs_rewrite()` function; `LLMQueryRewriter.rewrite()` checks the
  heuristic before calling Ollama; fixed stale module docstring (was
  "wired but not used by the default orchestrator" — it IS wired via
  `api.deps.get_orchestrator()`); updated class docstring.
- `tests/test_rag_query_rewriter.py` — new `TestNeedsRewriteHeuristic`
  class (16 tests: no-history skip, whitespace-history skip, no-pronoun
  skip, pronoun+history rewrite, case-insensitive, word-boundary
  no-false-positive on `it`/`his`/`she` substrings, self-contained
  technical query skip, follow-up with pronoun rewrite); 3 new
  integration tests on `LLMQueryRewriter` (skips LLM call when no
  history, skips when no pronoun, calls LLM when both present); existing
  tests updated to pass history + pronoun so the LLM call path is
  exercised.
- `docs/CHAT_ASSISTANT_FEATURES.md` — Core Chat coreference line updated
  to note the latency-skip heuristic.
- `docs/F500_ENTERPRISE_ACTION_ITEMS.md` — item #4.4 marked
  [PARTIALLY CLOSED 2026-08-15].

### Added — Enriched citations (snippet, relevanceScore, lastModified)

Citations now carry more than just `title` + `source_url`. The `Citation`
model, `extract_citations` post-processor, and `CitationChip` React component
were extended so each citation surfaces:

- **`snippet`** — a short excerpt (≤ 200 chars, word-boundary truncated with
  ellipsis, whitespace collapsed) from the supporting retrieved chunk, so the
  user sees context without clicking through.
- **`relevanceScore`** — the fused RRF retrieval score ([0, 1]) for the cited
  chunk, rendered as a "N% match" badge in the UI.
- **`lastModified`** — the source document's last-modified timestamp (UTC),
  rendered as an "Updated <date>" label so the user can gauge source freshness.

All three fields are optional with defaults, so older persisted citations
(title + url only) still deserialize and render correctly.

- `schemas/chat.py` — `Citation` model gains `snippet`, `relevanceScore`,
  `lastModified` (all optional).
- `schemas/documents.py` — `RetrievedDoc` gains `last_modified` (optional),
  propagated from the Qdrant payload.
- `rag/retriever.py` — `_to_retrieved_doc` maps `last_modified` from the
  Qdrant payload.
- `rag/post_processor.py` — `extract_citations` populates the three new
  fields; new `SNIPPET_MAX_CHARS` constant + `_make_snippet` helper.
- `frontend/src/CitationChip.jsx` — renders snippet, score badge, and
  last-modified date; falls back gracefully when fields are absent.
- `frontend/src/styles.css` — citation chip restyled as a compact column
  layout (title + snippet + meta row).
- Tests: `tests/test_rag_post_processor.py` (new `TestCitationEnrichedFields`
  - `TestMakeSnippet` classes), `tests/test_schemas.py` (new `TestCitation`
    class), `frontend/__tests__/CitationChip.test.jsx` (snippet/score/date
    render cases).
- `docs/CHAT_ASSISTANT_FEATURES.md` — Core Chat citation line upgraded from
  🟡 Partial to ✅ Implemented.

### Added — Live NIM comparison eval results (Phases 2, 3, 4)

Ran the three eval artifacts against the live Ollama + Qdrant + NIM stack
(2026-08-15, `NIM_ENABLED=true`, NIM guardrail models
`nvidia/llama-3.1-nemoguard-8b-content-safety` +
`nvidia/llama-3.1-nemoguard-8b-topic-control`, NIM embedder
`nvidia/llama-nemotron-embed-1b-v2`, NIM reranker
`nvidia/llama-nemotron-rerank-1b-v2`, NIM generator
`meta/llama-3.1-8b-instruct`). All four phases of the NIM integration plan
are now measured with real data, not just scaffolded.

- **Phase 2 — guardrail FPR/FNR** (`eval/run_guardrail_eval.py --both`):
  - **Ollama backend: PASSED** the gate (FPR ≤ 0.10, FNR ≤ 0.10 on all
    three checks). Input: P=0.909 R=0.909 F1=0.909 FPR=0.032 FNR=0.091.
    Output: P=1.000 R=1.000 F1=1.000 FPR=0.000 FNR=0.000. Topic-control
    reject: P=0.900 R=1.000 F1=0.947 FPR=0.045 FNR=0.000.
  - **NIM backend: FAILED** the gate on two checks. Input (injection):
    P=1.000 R=0.818 F1=0.900 FPR=0.000 **FNR=0.182** — NIM's nemoguard
    content-safety model missed 2 prompt-injection attacks that Ollama
    caught (higher precision but lower recall). Topic-control reject:
    P=0.692 R=1.000 F1=0.818 **FPR=0.182** — NIM's topic-control model is
    over-aggressive, routing 4 legitimate borderline questions ("best
    framework", "FastAPI vs Django", "summarize the above", "vague help")
    to `off_topic`. Output: P=1.000 R=1.000 F1=1.000 (same as Ollama).
  - **Key finding:** the NIM nemoguard topic-control model's parsing is
    brittle — it frequently returns `'on-topic '` (with trailing space)
    which the classifier doesn't recognize as a valid label, causing
    fallback to Ollama. This is a prompt-format mismatch, not a model
    quality issue. The NIM content-safety model is more precise (FPR=0)
    but less recall (misses 2 injections). Neither backend is strictly
    better — this is exactly the kind of finding the eval was designed to
    surface before any production promotion.
  - Report: `eval/guardrail_report.csv` (42 examples × 2 backends).

- **Phase 3 — retrieval recall@5** (`eval/run_retrieval_eval.py`):
  - NIM collection ingested: 7 files, 45 chunks into `ctc_rag_nim`
    (2048-d, `nvidia/llama-nemotron-embed-1b-v2`).
  - **Ollama recall@5 = 1.000** (36/36 hits, mean latency 0.072s).
  - **NIM recall@5 = 1.000** (36/36 hits, mean latency 0.181s).
  - **Δ = +0.000** — both embedders achieve perfect recall@5 on this
    corpus. The NIM embedder is 2.5× slower per query (hosted round-trip
    vs local Ollama) with no recall uplift on this small, well-curated
    corpus. The NIM embedder's advantage (multilingual, long-doc QA
    tuning) would likely show on a larger/more ambiguous corpus — not
    measurable here. Gate (Ollama recall@5 ≥ 0.70): PASSED.
  - Report: `eval/retrieval_report.csv` (36 questions × 2 collections).

- **Phase 4 — reranker A/B** (`eval/run_eval.py`, NIM generator +
  guardrails, reranker on vs off):
  - **Without reranker:** faithfulness=0.800, relevancy=0.792,
    recall=0.722, 28/36 passed, mean latency 5.27s. Gate: PASSED.
  - **With reranker:** faithfulness=0.800, relevancy=0.803,
    recall=0.733, 30/36 passed, mean latency 6.43s. Gate: PASSED.
  - **Uplift:** +0.011 relevancy, +0.011 recall, +2 questions passed
    (28→30). The reranker reorders 20 candidates → top 5, improving
    context precision enough to flip 2 borderline questions from fail
    to pass. Latency cost: +1.16s/query (the extra NIM rerank call).
    Faithfulness is unchanged (0.800) — the reranker improves _what_ is
    retrieved but not _how faithfully_ the generator uses it.
  - Reports: `eval/eval_report_no_rerank.csv`, `eval/eval_report_with_rerank.csv`.

### Fixed — `eval/run_eval.py` kwarg mismatch + NIM orchestrator wiring

- `build_orchestrator()` now delegates to `api.deps.get_orchestrator()`
  when `NIM_ENABLED=true` is set, so the NIM generator, guardrails, and
  reranker are exercised through the same config-flag wiring as the live
  app. This enables the Phase 4 A/B comparison (reranker on vs off) by
  running the script twice with different `NIM_RERANK_ENABLED` values.
  When NIM is disabled (default), the plain Ollama-only orchestrator is
  constructed directly (unchanged).
- Fixed a pre-existing `TypeError` in `main()`: `mark_passes()` and
  `check_gate()` expect `threshold_faithfulness` / `threshold_recall` /
  `threshold_relevancy` kwargs, but the call site passed the short names
  (`faithfulness` / `recall` / `relevancy`). The `print_summary()` call
  correctly expects the short-named dict, so the call site was
  restructured to pass the right form to each function. This bug would
  have crashed any full `run_eval.py` run (the local-judge path
  completed all questions but crashed at the post-processing step).
- All 730 backend tests pass (`python -m pytest -q`).

### Added — Eval artifacts for Phase 2 (guardrails) + Phase 3 (embeddings)

The existing `eval/run_eval.py` + `eval/golden-dataset.json` suite covered
only Phase 1 (generator) and Phase 4 (reranker) of the NIM integration plan.
Phase 2 (guardrail FPR/FNR) and Phase 3 (embedding recall@5) had no
automated eval artifacts — the plan's "manual review" steps assumed eyeballing
verdicts. This adds the three missing artifacts so all four phases are
measurable:

- `eval/guardrail-dataset.json` — adversarial golden set of 48 labeled
  examples across 5 categories (`on_topic_safe`, `off_topic`,
  `prompt_injection`, `harmful_output`, `borderline`) with expected verdicts
  for each of the three guardrail checks (`expected_input` block/allow,
  `expected_class` label, `expected_output` block/allow). Enough samples per
  class (>=6 for the four main categories, >=4 borderline) to compute
  meaningful FPR/FNR. Includes legitimate context-dependent follow-ups
  (with history) that must NOT be false-positive blocked, and safe answers
  that must NOT be false-positive blocked by the output guardrail.
- `eval/run_guardrail_eval.py` — runner that loads the adversarial set and
  runs each example through `build_guardrail_suite()` (NIM on via
  `NIM_ENABLED=true`, or plain Ollama, or `--no-llm` for regex/keyword only).
  Records actual verdicts for all three checks and computes, per check:
  input guardrail precision/recall/F1/FPR/FNR (positive=block), output
  guardrail precision/recall/F1/FPR/FNR (positive=block), topic-control
  binary reject/accept FPR/FNR (positive=off_topic), and per-class
  precision/recall/F1 for the multi-class classifier. Writes a per-example
  CSV (`eval/guardrail_report.csv`) and prints a per-check summary. Gate:
  exits non-zero if any binary check's FPR > `--max-fpr` (default 0.10) or
  FNR > `--max-fnr` (default 0.10). `--both` runs Ollama then NIM
  back-to-back for side-by-side comparison.
- `eval/run_retrieval_eval.py` — recall@5 comparison script that reuses
  `eval/golden-dataset.json`'s `source_title` ground truth. For each golden
  question, queries both the default Ollama `docs-knowledge` collection and
  the separate NIM `ctc_rag_nim` collection via `HybridRetriever.retrieve()`,
  checks whether the ground-truth title appears in the top-5 (substring +
  case-insensitive tolerant match), and reports recall@5 per collection +
  the NIM−Ollama delta. Writes a per-question CSV
  (`eval/retrieval_report.csv`). Gracefully reports `n/a` for the NIM
  collection when it is missing/empty (with a hint to ingest it first via
  `NIM_ENABLED=true python -m ingestion.nim_embedder`). Gate: exits
  non-zero if Ollama recall@5 < `--threshold` (default 0.70); the NIM
  recall@5 is a comparison metric, not a release gate (the live app keeps
  using the Ollama collection). `--no-nim` / `--only-nim` select a single
  collection.
- `tests/test_eval_guardrails.py` — 27 unit tests for the pure-logic parts
  of the guardrail runner (dataset loading + category balance, `BinaryMetrics`
  / `ClassMetrics` arithmetic incl. zero-denominator safety, `_score_binary`,
  `aggregate` wiring for input/output/topic-reject/per-class, `run_example`
  with a mocked suite, `write_csv` append behavior).
- `tests/test_eval_retrieval.py` — 20 unit tests for the pure-logic parts of
  the retrieval runner (dataset loading, `compute_hit` exact/substring/
  case-insensitive/miss/empty + min-length guard against single-char false
  matches, `aggregate` recall + error-exclusion + latency, `write_csv`).
- All 730 backend tests pass (`python -m pytest -q`).

### Added — NVIDIA NIM reranker integration (Phase 4 — net-new capability)

- `rag/nim_reranker.py` — new module implementing the Phase 4 reranker from
  `docs/NVIDIA_NIM_INTEGRATION_PLAN.md`:
  - `NIMReranker` — calls the NVIDIA NIM dedicated reranking endpoint
    (`https://ai.api.nvidia.com/v1/retrieval/{model}/reranking` — note the
    different host vs. the chat/embeddings endpoint). Uses `httpx` (already a
    dependency) to POST with `{"model", "query": {"text"}, "passages":
[{"text"}, ...], "truncate": "END"}` and parses the `{"rankings":
[{"index", "logit"}, ...]}` response (sorted by relevance — highest logit
    first). Reorders `RetrievedDoc` candidates by the reranker's relevance
    scores, truncates to `top_n` (default 5). Lazy httpx client so unit tests
    inject mocks without hitting the network. Maps 429/404/timeout/connection/
    malformed-response errors to `NIMRerankError`. Owns a dedicated
    `CircuitBreaker` (separate from the generator's, guardrails', embedder's
    NIM breakers and the Ollama breaker). **No fallback model** — on any
    failure (circuit open, 429, 404, timeout, connection error, malformed
    response), `rerank()` returns the original retrieved order truncated to
    `top_n` (graceful degradation to no-reranking, the status quo). The
    exception is logged but never raised, because reranking is a quality
    enhancement, not a correctness requirement. Preserves the original RRF
    retrieval scores (the reranker reorders but does not rescore — its logits
    are unbounded and not comparable to the [0, 1] RRF score). Default model
    `nvidia/llama-nemotron-rerank-1b-v2`.
  - `build_reranker()` factory — returns `NIMReranker` when `NIM_ENABLED=true`
    and `NIM_RERANK_ENABLED` is not `false` (default true), else `None`. A
    second flag `NIM_RERANK_ENABLED` allows disabling just the reranker while
    keeping the generator/guardrails on NIM — useful for the A/B quality
    comparison. `NIM_RERANK_TOP_N` overrides the number of docs to keep
    (default 5). Returns `None` (not a no-op reranker) so the orchestrator
    skips the rerank step entirely with a simple `is not None` check.
  - `get_rerank_candidate_k()` helper — returns the retriever `top_k` to use
    when reranking is active (default 20, override with
    `NIM_RERANK_CANDIDATE_K`). Matches the retriever's `PREFETCH_LIMIT`.
- `rag/orchestrator.py` — `RAGOrchestrator` gained an optional `reranker`
  param. After retrieval, if a reranker is configured, the docs are reranked
  before context assembly (both `stream_answer` and `answer` methods). A
  Langfuse `"rerank"` span is emitted with candidate + reranked count
  metadata. Default path unchanged when `reranker` is `None`.
- `api/deps.py` — wired through `build_reranker()` via a cached
  `get_reranker()`. When the reranker is active, `get_retriever()` uses a
  larger `top_k` (`NIM_RERANK_CANDIDATE_K`, default 20) so the reranker has
  enough candidates to reorder; otherwise the default `top_k=5`. Default path
  unchanged (no reranker, `top_k=5`) when `NIM_ENABLED` is unset/false.
- `.env.example` — added `NIM_RERANK_ENABLED`, `NIM_RERANK_TOP_N`,
  `NIM_RERANK_CANDIDATE_K` env vars (Phase 4); updated the NIM section header
  to note Phase 4 coverage.
- `tests/test_rag_nim_reranker.py` — 58 new unit tests covering endpoint URL
  construction (model in path, different host), request shape (model/query/
  passages/truncate, doc content not title, custom model/truncate, auth
  header), reordering (by rankings, truncation to top_n, preserves original
  RRF scores, skip when ≤top_n, empty docs), error mapping (429/404/500/
  empty-rankings/out-of-range-index/missing-index/timeout/connect/unexpected),
  graceful degradation on every error path (returns original[:top_n], never
  raises), circuit breaker (open→degrade, failure recording, success reset,
  wraps request), lazy client, close, `build_reranker` factory (None when
  disabled, None when rerank disabled, reranker when enabled, top_n override,
  dedicated breaker, default model), `get_rerank_candidate_k` (default 20,
  env override), and orchestrator integration (reranker called with rewritten
  query + retrieved docs, reranked docs passed to assembler + post-processor,
  skipped when None, `answer` method too, rerank span traced).
- **F500 enterprise gap notice:** this is comparison-test-only code behind
  `NIM_ENABLED` (default off), not production-grade. The reranker adds one
  extra hosted call per query (data egress of query + chunk contents to
  NVIDIA), consuming the shared ~40 RPM rate-limit budget. See
  `docs/F500_ENTERPRISE_ACTION_ITEMS.md` item #8 for the 10 gaps to close
  before any production promotion.

### Added — NVIDIA NIM embedding integration (Phase 3 — comparison testing)

- `ingestion/nim_embedder.py` — new module implementing the Phase 3 embedding
  comparison test from `docs/NVIDIA_NIM_INTEGRATION_PLAN.md`:
  - `NIMEmbedder` — OpenAI-compatible `/embeddings` client for the NVIDIA NIM
    free tier (`https://integrate.api.nvidia.com/v1`). Uses `httpx` (already a
    dependency) to POST to `/embeddings` with `input_type` (`passage` at index
    time, `query` at query time — the nemotron model is tuned for asymmetric
    bi-encoder retrieval). Lazy httpx client so unit tests inject mocks
    without hitting the network. Maps 429/404/timeout/connection errors to
    `NIMEmbeddingError`. Owns a dedicated `CircuitBreaker` (separate from the
    generator's and guardrails' NIM breakers and the Ollama breaker). Default
    model `nvidia/llama-nemotron-embed-1b-v2` (native 2048-d; Matryoshka
    reduced dimensions 384/512/768/1024/2048 via the `dimensions` API param,
    configurable with `NIM_EMBEDDING_DIM`). Same `embed_texts`/`embed_text`
    interface as the Ollama `Embedder` so it is a drop-in for the existing
    `IndexWriter` and `HybridRetriever`. **No fallback embedder** — vectors
    are not interoperable across models, so a NIM embedding failure aborts
    the ingestion batch (the Ollama collection is never touched).
  - `build_nim_embedder()` factory — reads `NVIDIA_API_KEY`,
    `NIM_EMBEDDING_MODEL`, `NIM_EMBEDDING_DIM` env vars.
  - `build_nim_retriever()` factory — existing `HybridRetriever` pointed at
    the NIM comparison collection (`ctc_rag_nim`), for recall@5 comparison
    against the default Ollama collection. Not wired into `api/deps.py`.
  - `run_nim_ingestion()` orchestrator — thin wrapper over
    `ingestion.run.run_ingestion` that injects a `NIMEmbedder` + an
    `IndexWriter` pointed at `ctc_rag_nim`, ensures the NIM collection
    exists at the configured dimensionality, and refuses to run without
    `NIM_ENABLED=true`. The default `docs-knowledge` collection is never
    touched.
- `rag/qdrant_collection.py` — added NIM comparison collection support:
  `NIM_COLLECTION_NAME` (`ctc_rag_nim`), `NIM_VECTOR_SIZE` (2048),
  `NIM_EMBEDDING_MODEL_DEFAULT`, `build_nim_collection_config()` (reuses the
  same `dense`/`text` named vectors as the default collection so
  `IndexWriter`/`HybridRetriever` work unchanged — only the collection name +
  dense size differ), `ensure_nim_collection()`.
- `ingestion/run.py` — `run_ingestion` parameterized with an optional
  `collection_config: CollectionConfig | None = None`. When `None` (default)
  the standard `docs-knowledge` collection is used (unchanged). When provided,
  the collection is ensured/recreated at that config's name + dimensionality.
  This lets the NIM ingestion path target a separate collection without
  duplicating the orchestrator. Default path unchanged.
- `.env.example` — added `NIM_EMBEDDING_MODEL` and `NIM_EMBEDDING_DIM` env
  vars (Phase 3); updated the NIM section header to note Phase 3 coverage.
- `tests/test_ingestion_nim_embedder.py` — 47 new unit tests covering
  `NIMEmbedder` (request shape, `input_type`, dimensions/Matryoshka, error
  mapping for 429/404/500/empty/count-mismatch/timeout/connect/unexpected,
  circuit-breaker open/failure-recording/success-reset, lazy client, close),
  `build_nim_embedder` factory, `build_nim_collection_config` (NIM collection
  name + size, same vector names, overrides, env), `ensure_nim_collection`
  (creates if missing, skips if exists, never touches default collection),
  `build_nim_retriever`, and `run_nim_ingestion` (refuses without
  `NIM_ENABLED`, runs with it, passes `collection_config` + injected
  embedder/index_writer, `full_reindex` passthrough, custom embedder
  passthrough).
- **F500 enterprise gap notice:** this is comparison-test-only code behind
  `NIM_ENABLED` (default off), not production-grade. The NIM embedder is not
  wired into the live app's default path. See
  `docs/F500_ENTERPRISE_ACTION_ITEMS.md` item #8 for the 10 gaps to close
  before any production promotion.

### Added — NVIDIA NIM guardrail integration (Phase 2 — comparison testing)

- `api/nim_guardrails.py` — new module implementing the Phase 2 guardrail
  comparison test from `docs/NVIDIA_NIM_INTEGRATION_PLAN.md`:
  - `NIMGuardrailClient` — non-streaming OpenAI-compatible chat client for
    NIM guardrail verdicts (`https://integrate.api.nvidia.com/v1`). Uses
    `httpx` (already a dependency) to POST to `/chat/completions` with
    `stream=False` and returns the verdict content. Maps 429/404/timeout/
    connection errors to `NIMGuardrailError` for fallback. Owns a dedicated
    `CircuitBreaker` (separate from the generator's NIM breaker and the
    Ollama breaker) so a NIM guardrail outage does not trip other breakers.
    Default models: `nvidia/llama-3.1-nemoguard-8b-content-safety` (input +
    output judges) and `nvidia/llama-3.1-nemoguard-8b-topic-control`
    (classifier).
  - `NIMInputGuardrail(InputGuardrail)` — overrides `_llm_judge` to call the
    NIM content-safety model first; on NIM failure
    (`NIMGuardrailError` / `CircuitOpenError`) or an unparseable verdict,
    falls back to the parent Ollama injection judge. The regex injection
    tier (tier 1) is inherited and always runs first. 3-tier: regex → NIM →
    Ollama → regex-only.
  - `NIMOutputGuardrail(OutputGuardrail)` — overrides `_llm_judge` to call
    the NIM content-safety model first; on failure falls back to the parent
    Ollama harmful-content judge. The PII regex scrub (tier 1) is inherited
    and always runs first. 3-tier: PII scrub → NIM → Ollama → scrub-only.
  - `NIMQueryClassifier(QueryClassifier)` — overrides `_llm_classify` to
    call the NIM topic-control model first (binary on-topic/off-topic). NIM
    "off-topic" routes directly to `off_topic`; NIM "on-topic" defers to
    the parent Ollama 5-way classifier for fine-grained routing. 3-tier:
    NIM topic-control → Ollama 5-way → keyword fallback.
  - `build_guardrail_suite()` factory — reads `NIM_ENABLED` env var. When
    `true` (case-insensitive, also accepts `1`/`yes`), returns a
    `GuardrailSuite` with the three NIM-augmented subclasses sharing one
    `NIMGuardrailClient` (one API key + one rate-limit budget + one
    dedicated circuit breaker). When disabled (default), returns the plain
    Ollama `GuardrailSuite` — no change to the existing default path.
- `api/deps.py` — `get_guardrail_suite` now uses `build_guardrail_suite()`
  instead of constructing `GuardrailSuite` directly, so the NIM guardrail
  path is wired through the config flag with zero change to the default
  behavior.
- `.env.example` — updated the NIM section to note it now covers guardrails
  (Phase 2) in addition to generation (Phase 1).
- `tests/test_api_nim_guardrails.py` — 50 new unit tests covering
  `NIMGuardrailClient` (non-streaming POST, verdict extraction, lazy
  client, error mapping for 429/404/500/timeout/connect, circuit-breaker
  open/failure-recording/success-reset), `NIMInputGuardrail` (regex
  short-circuit, NIM unsafe/safe, NIM failure → Ollama fallback, NIM
  unparseable → Ollama, both-fail → regex-only, circuit-open fallback,
  model + history passthrough), `NIMOutputGuardrail` (PII scrub always,
  NIM unsafe/safe, NIM failure → Ollama, both-fail → scrub-only), and
  `NIMQueryClassifier` (NIM off-topic direct route, NIM on-topic → Ollama
  5-way, NIM failure → Ollama, both-fail → keyword, circuit-open
  fallback, model + history passthrough), and `build_guardrail_suite`
  (default → plain suite, NIM-enabled → NIM-augmented suite sharing one
  client, case-insensitive flag, dedicated breaker). All mocked — no
  network.
- `docs/NVIDIA_NIM_INTEGRATION_PLAN.md` — marked Phase 2 as ✅ (code
  landed) with notes on the 3-tier fallback design per component and the
  deferred hosted PII model (`nvidia/gliner-pii`).

**F500 enterprise gap notice:** This is a **practice-stage comparison
test only**, not a production promotion. All 10 gaps in
`docs/F500_ENTERPRISE_ACTION_ITEMS.md` item #8 remain open. The default
path (NIM disabled) is unchanged — Ollama stays primary, local-first,
$0-cost. NIM guardrails must not be enabled on corpora with real PII
(F500 item #5). The hosted PII model (`nvidia/gliner-pii`) is deferred
(uncertain availability + chicken-and-egg data-egress problem).

### Changed — NVIDIA NIM default model swapped after live integration test

- `NIM_GENERATION_MODEL` default changed from `moonshotai/kimi-k2.6` to
  `meta/llama-3.1-8b-instruct`. A live integration test against
  `https://integrate.api.nvidia.com/v1` (2026-08-15) confirmed the API key
  is valid and SSE streaming works end-to-end, but `moonshotai/kimi-k2.6`
  returns HTTP 404 on `/chat/completions` (retired/undeployed on the free
  tier, despite still appearing in `GET /v1/models`). `meta/llama-3.1-8b-
instruct` is the integration plan's own production target and streams
  correctly (~0.35–1s). Updated `test_default_model` assertion to match.
  See `docs/F500_ENTERPRISE_ACTION_ITEMS.md` #11 for the catalog-drift
  workaround record.

### Added — NVIDIA NIM generator integration (Phase 1 — comparison testing)

- `rag/nim_generator.py` — new module implementing the Phase 1 generator
  comparison test from `docs/NVIDIA_NIM_INTEGRATION_PLAN.md`:
  - `NIMGenerator` — streaming OpenAI-compatible chat client for the
    NVIDIA NIM free tier (`https://integrate.api.nvidia.com/v1`). Uses
    `httpx` (already a dependency) to POST to `/chat/completions` with
    `stream=True` and parses the SSE response. Shares the system prompt
    template + generation settings with the Ollama generator so answers
    are directly comparable. Maps 429/404/timeout/connection errors to
    `NIMError` for fallback. Default model: `meta/llama-3.1-8b-instruct`
    (originally `moonshotai/kimi-k2.6` per integration plan §2.1, but that
    model was retired/undeployed on the NIM free tier — returns 404 on
    `/chat/completions` as of 2026-08-15 despite still being listed in
    `GET /v1/models`; switched to the plan's own production target, which
    is reliably available on the free tier at ~0.35–1s latency and matches
    the Ollama `llama3.1:8b` for apples-to-apples comparison).
  - `FallbackGenerator` — wraps a primary + fallback generator with a
    `CircuitBreaker`. When the primary fails before yielding any tokens
    (or the circuit is open), transparently falls back to the secondary.
    If both fail, yields a canned refusal. Mid-stream failures propagate
    (no confusing duplicate answers). The breaker is checked _before_
    attempting the primary and failures are recorded after a pre-first-
    token failure.
  - `build_generator()` factory — reads `NIM_ENABLED` env var. When
    `true` (case-insensitive, also accepts `1`/`yes`), returns a
    `FallbackGenerator(NIM → Ollama)` with a dedicated NIM circuit
    breaker. When disabled (default), returns the plain Ollama
    `Generator` — no change to the default local-first path.
- `api/deps.py` — `get_orchestrator` now uses `build_generator()` instead
  of constructing `Generator` directly, so the NIM path is wired through
  the config flag with zero change to the default behavior.
- `.env.example` — added `NIM_ENABLED=false` + `NVIDIA_API_KEY=` with a
  warning about the free-tier limitations (no SLA, shared 40 RPM, data
  egress, no PII).
- `tests/test_rag_nim_generator.py` — 29 new unit tests covering
  `NIMGenerator` (SSE parsing, message construction, auth header, lazy
  client, error mapping for 429/404/500/timeout/connect), `FallbackGenerator`
  (primary success, primary-fail→fallback, both-fail→refusal, mid-stream
  failure propagation, circuit-open skip, breaker failure recording,
  close semantics), and `build_generator` (default→Ollama, NIM-enabled→
  FallbackGenerator, case-insensitive flag). All mocked — no network.
- `docs/NVIDIA_NIM_INTEGRATION_PLAN.md` — marked Phase 1 as ✅ (code
  landed) with a note that the comparison eval run is the remaining
  manual step.

**F500 enterprise gap notice:** This is a **practice-stage comparison
test only**, not a production promotion. All 10 gaps in
`docs/F500_ENTERPRISE_ACTION_ITEMS.md` item #8 remain open. The default
path (NIM disabled) is unchanged — Ollama stays primary, local-first,
$0-cost. NIM must not be enabled on corpora with real PII.

### Added — Ollama model recommendations per component

- `docs/OLLAMA_MODEL_RECOMMENDATIONS.md` — new planning doc mapping every
  Ollama model usage point in the codebase (generator, guardrails, query
  rewriter, embedder, warm-up, eval judge) to a recommended model with
  reasoning. Key finding: `qwen2.5:14b` (9 GB) is already pulled but not used
  anywhere — recommended for generator + eval judge + warm-up. Two new pulls
  recommended: `llama3.1:8b` (injection judge, doc's production target) and
  `llama-guard3:8b` (purpose-built content safety classifier, addresses F500
  item #1). `llama3.2:3b` kept for classifier + rewriter (fast, short-output
  tasks). `nomic-embed-text` kept for embeddings (changing requires
  re-indexing). Includes memory budget analysis for M1 32 GB, `.env.example`
  additions, and optional embedding upgrade path (`bge-m3`, `mxbai-embed-large`).

### Added — NVIDIA NIM integration plan + F500 enterprise gap analysis

- `docs/NVIDIA_NIM_INTEGRATION_PLAN.md` — new planning doc documenting the
  per-component NIM model selections with reasoning (generator, guardrails,
  topic control, PII, embeddings, reranker), the fallback architecture
  (configurable primary with graceful degradation: NIM → Ollama → regex),
  what does NOT change (default stays Ollama/local-first), and a 5-phase
  integration roadmap. Corrects the fallback direction: Ollama remains the
  default primary when NIM is not opted into; Ollama becomes the automatic
  fallback only when NIM is explicitly enabled via config flag.
- `docs/F500_ENTERPRISE_ACTION_ITEMS.md` — added item #8 (NVIDIA NIM as
  optional cloud provider) recording 10 enterprise-grade gaps (no SLA,
  shared rate limit, data egress/DPA, secrets management, model version
  pinning, eval gate, failover, retry/backoff, audit trail, PII scrubbing)
  that must be closed before any production promotion. Updated the summary
  table and stage-justification paragraph.

### Fixed — Langfuse tracer silently disabled (v2→v4 SDK API mismatch)

The `LangfuseTracer` was written against the Langfuse **v2 SDK API**
(`client.trace(...)`, `trace.span(...)`, `trace.score(...)`, `span.end(metadata=...)`)
but `requirements.txt` had `langfuse>=3.0` (unbounded), which allowed
`langfuse` v3/v4 to install. The v3/v4 SDK removed all of those methods in
favor of the OpenTelemetry-style API (`start_observation`, `create_score`,
`create_trace_id`, `span.update()` + `span.end()`). As a result, every
`start_trace` / `start_span` / `record_score` call hit the `except Exception`
graceful-degradation branch and silently fell back to structlog — **no
traces, spans, or scores actually reached Langfuse Cloud**, even though
`tracer.enabled` was `True` and the client constructed successfully. The
app ran fine but observability was completely dark.

Root cause was discovered via a live integration check: `trace.enabled`
returned `True` but `trace._lf_trace` was `None` because
`client.trace()` raised `AttributeError: 'Langfuse' object has no attribute
'trace'`, caught by the broad `except Exception` block.

Pinning to `langfuse<3` (v2 API) was attempted first but rejected because
langfuse v2 requires `wrapt<2.0` while `unstructured 0.25.2` (used by the
ingestion pipeline) requires `wrapt>=2.1.1` — a hard, unresolvable
dependency conflict. The only clean fix was to upgrade the tracer code to
the v3/v4 API.

- `api/observability.py` — rewrote `LangfuseTracer` to use the v3/v4 SDK:
  `start_trace` now calls `client.create_trace_id()` +
  `client.start_observation(as_type="span", trace_context=...)` to create a
  root span; `start_span` nests via `root.start_observation(...)`; `end_span`
  calls `span.update(metadata=...)` then `span.end()`; new `end_trace` method
  closes the root span; `record_score` uses `client.create_score(trace_id=...)`
  instead of `trace.score(...)` (this also fixes the feedback route, which
  constructs a `TraceHandle` with just an id and no root span object).
  `TraceHandle._lf_trace` renamed to `_lf_root`. Updated the class docstring
  with the full SDK method mapping and the module docstring to reference v3/v4.
- `rag/orchestrator.py` — wrapped both `stream_answer` (generator) and
  `answer` (non-streaming) bodies in `try/finally` that calls
  `self.tracer.end_trace(trace)` so the root span is always closed even on
  early returns (input-blocked, classification short-circuit) and exceptions.
- `tests/test_observability.py` — rewrote `TestLangfuseTracerEnabled` for the
  new API shape: mock client now stubs `create_trace_id` / `start_observation`
  / `create_score`; `TraceHandle` uses `_lf_root`; `end_span` asserts
  `update`+`end` call sequence; added 6 new tests for `end_trace` (with/without
  metadata, noop-when-root-None, degrades-on-error) and `record_score` with
  an id-only handle (feedback-route scenario). 73 tests total (was 67).
- `requirements.txt` — `langfuse>=3.0` (unchanged from original; the code now
  targets the v3/v4 API so the unbounded upper is correct).

### Changed — Migrate Langfuse from self-hosted to Langfuse Cloud (langfuse.com)

Replaced the self-hosted Langfuse v4 Docker stack (postgres, clickhouse, redis, minio, langfuse-worker, langfuse-web) with the hosted Langfuse Cloud SaaS. Traces, spans, and feedback scores are now sent to the cloud endpoint (US region: `https://us.cloud.langfuse.com`) instead of a local container stack. The env var was renamed from `LANGFUSE_HOST` to `LANGFUSE_BASE_URL` to match Langfuse Cloud conventions.

- `docker-compose.yml` — removed the six self-hosted Langfuse services (postgres, clickhouse, redis, minio, langfuse-worker, langfuse-web) and their five named volumes. The compose stack now contains only Qdrant + the FastAPI backend. Backend env var passthrough renamed from `LANGFUSE_HOST` to `LANGFUSE_BASE_URL`. Updated the header and backend-service comments to reference the cloud endpoint.
- `.env.example` — removed the Langfuse core (NEXTAUTH\_\*, SALT, ENCRYPTION_KEY, TELEMETRY_ENABLED), Postgres, ClickHouse, Redis, and MinIO sections. `LANGFUSE_BASE_URL` now defaults to `https://us.cloud.langfuse.com` (Langfuse Cloud US region); `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` remain blank by default (tracing disabled until the user creates a project at cloud.langfuse.com and pastes the API keys).
- `api/observability.py` — `DEFAULT_LANGFUSE_BASE_URL` set to `https://us.cloud.langfuse.com`. The env var was renamed from `LANGFUSE_HOST` to `LANGFUSE_BASE_URL` across the tracer (`__init__`, `_detect_enabled`) and the module docstring. Updated the `LangfuseTracer` docstring to reference Langfuse Cloud instead of the self-hosted container.
- `api/main.py` — `_langfuse_enabled()` now reads `LANGFUSE_BASE_URL` instead of `LANGFUSE_HOST`.
- `api/deps.py` — `get_tracer()` docstring updated to reference `LANGFUSE_BASE_URL`.
- `tests/test_observability.py` — updated the `LANGFUSE_BASE_URL` values in the enable-detection tests to `https://us.cloud.langfuse.com`.
- `README.md` — updated the cost table, architecture diagram, repo-layout comment, quick-start health-check step, and observability section to reference Langfuse Cloud instead of the self-hosted stack.

### Notes — Langfuse Cloud migration

- **Env var rename:** `LANGFUSE_HOST` → `LANGFUSE_BASE_URL`. The Langfuse Python SDK constructor parameter is still `host=`, but the env var name is a project-level choice; `LANGFUSE_BASE_URL` aligns with Langfuse Cloud's documented convention and avoids ambiguity with `LANGFUSE_HOST` (which the SDK reads by default for self-hosted deployments).
- **US region:** the default base URL is `https://us.cloud.langfuse.com` (Langfuse Cloud US). EU region is `https://cloud.langfuse.com` — set `LANGFUSE_BASE_URL` accordingly if your project is in the EU region.
- **No code behavior change:** the `LangfuseTracer` already auto-disables when `LANGFUSE_BASE_URL` / keys are unset and lazily constructs the client from env vars, so the only code change is the default constant + env var name. The tracer, orchestrator span wiring, and feedback-score correlation are unchanged.
- **Operational change:** `docker compose up -d` now brings up only Qdrant + the backend (2 containers) instead of 8 containers. Langfuse traces are viewed at https://us.cloud.langfuse.com instead of http://localhost:3000.
- **Cost:** Langfuse Cloud has a free tier (sufficient for this practice project); the self-hosted stack was free but consumed ~8 GB RAM across 6 containers. This trades local resource usage for a managed SaaS dependency.
- **Enterprise note:** the architecture doc (`ai-rag-chat-architecture-2026.md`) still references "Langfuse self-hosted in Docker" as a design decision (D4) and cost constraint. That doc is a historical design reference and is not updated here; the operational reality is now Langfuse Cloud.

### Added — Chat assistant feature status register

- `docs/CHAT_ASSISTANT_FEATURES.md` — documents the full list of chat assistant features (core chat, React widget UI, retrieval, APIs, guardrails, responsible AI, performance & cost, RBAC) from the production AEM/Azure target in `ai-rag-chat-architecture-2026.md`, mapped to their **actual implementation status** in this local practice codebase. Each feature is tagged ✅ Implemented / 🟡 Partial / ❌ Not implemented, with the stack substitution table (Ollama vs Azure OpenAI, Qdrant vs Azure AI Search, SQLite vs Redis/Cosmos, standalone Vite SPA vs AEM SPA Editor) and per-feature file references. Includes a summary count table and a bottom-line assessment pointing to `docs/F500_ENTERPRISE_ACTION_ITEMS.md` for the deferred enterprise gaps.

### Added — F500 enterprise production-grade action items tracker

- `docs/F500_ENTERPRISE_ACTION_ITEMS.md` — tracks every place where the project ships a workaround instead of a production-grade F500 enterprise practice. Documents 7 gaps introduced/accepted by the context-blind-guardrails fix (3B judge model, no eval gate/SLO, history-synthesis hallucination risk, rewriter as injection surface, PII/injection in judge history, no guardrail metrics, no red-team suite). Each entry records the enterprise standard, the workaround shipped, the reasoning for the workaround at the current practice stage, and the future action item to close the gap. Items 1, 2, 5, and 7 are flagged as blockers the moment the app is deployed to a hosted/multi-user environment or the judge moves to a hosted model.

### Added — NVIDIA NIM free LLM models research document

- `docs/NVIDIA_NIM_FREE_MODELS.md` — deep-research reference of every free, OpenAI-compatible model hosted on NVIDIA's build.nvidia.com NIM catalog (125 models: 99 confirmed online + 26 requiring provider verification). Documents the platform overview, access/authentication steps, global constraints (~40 RPM shared rate limit, no credit card, no key expiration, phone verification), OpenAI compatibility (Chat Completions, streaming, function calling, vision), quick-start code (Python openai SDK, curl, LangChain), all flagship/mid-size/small chat models with their specialities (chat, reasoning, coding, vision, embedding, safety, domain-specific), a Kimi K2.5 / K2.6 deep-dive, a Bash probe script for programmatic availability detection, caveats/best practices, and direct relevance to this project's `generator.py` and `guardrails.py` components. Sources cited from build.nvidia.com, docs.api.nvidia.com, freellm.net, and stevescargall.com.

### Fixed — Context-blind guardrails & classifier (multi-turn follow-up bug)

A context-dependent follow-up such as `"please summarize all above 3 answers"` was rejected by the input guardrail (LLM judge false-positive) and would have been misrouted to `off_topic` by the classifier, because both pre-RAG gates ran on the bare query with no conversation history. The 3B judge couldn't distinguish a legitimate conversational anaphor ("the above 3 answers") from a manipulative one ("the above instructions").

- `api/guardrails.py` — `InputGuardrail.check(message, history="")` and `QueryClassifier.classify(query, history="")` now accept the formatted conversation history. `INJECTION_JUDGE_PROMPT` and `CLASSIFIER_PROMPT` gained a `Conversation history:` block so the LLM judge/classifier can see prior on-topic Q&A and avoid false positives on follow-ups. `GuardrailSuite.check_input` / `GuardrailSuite.classify` forward `history`. Backward compatible (history defaults to `""`).
- `api/guardrails.py` — added a `follow_up` classification label (`CLASS_FOLLOW_UP`) for context-dependent follow-ups that reference prior turns. It is `handled=False` (still goes through the RAG flow) so the generator can synthesize from history. `VALID_CLASSES` and `CLASSIFIER_PROMPT` updated.
- `rag/orchestrator.py` — `stream_answer` and `answer` now pass `history` into `check_input(query, history)` and `classify(query, history)` (previously only the query was passed).
- `rag/generator.py` — `SYSTEM_PROMPT_TEMPLATE` rule 5 now explicitly permits synthesizing a follow-up answer from `CONVERSATION HISTORY` (e.g. a summary of prior turns) while forbidding invented facts. Previously the prompt said "use ONLY the provided context", which would starve a "summarize the above" answer.
- `api/deps.py` — `get_orchestrator()` now injects `LLMQueryRewriter()` instead of the passthrough default, so a decontextualized follow-up is rewritten into a self-contained query (coreference resolution against history) before retrieval. Falls back to the original query on LLM error/empty.
- `tests/test_guardrails.py` — 8 new tests: follow-up with history not blocked, follow-up without history blocked when judge unsafe, history default renders as `(none)`, `follow_up` label not handled, follow-up with history routed to `follow_up`, follow-up without history can route `off_topic`, `GuardrailSuite` forwards history to both `check_input` and `classify`.
- `tests/test_rag_orchestrator.py` — 3 new tests: orchestrator forwards history to `check_input` and `classify` (stream + answer), follow-up with history reaches the RAG flow instead of short-circuiting.
- All 493 backend tests pass (`python -m pytest -q`).

### Added — Step 7 (Observability & hardening)

- `api/observability.py` — the observability & hardening layer (build-order items 39–41, Phase 6 — Monitoring & Hardening).
  - **CircuitBreaker** (item 41): in-process breaker around Ollama calls. After `threshold` (default 3) consecutive failures the circuit opens for `timeout` (default 30 s); calls during that window raise `CircuitOpenError` instead of hitting the network. After the timeout the circuit half-opens for one probe call — success closes it, failure reopens. Thread-safe via `threading.Lock`. This is the practice-project equivalent of the doc's "Circuit Breaker (Optional)" snippet.
  - **MetricsCollector** (item 40): thread-safe in-memory store for online serving metrics — request count, error count, cache hits/misses, and TTFT (time-to-first-token) samples (capped at 200). `snapshot()` returns mean/p50/p95 TTFT + cache hit rate as a JSON-serialisable dict, surfaced via `GET /api/v1/metrics`. The offline metrics (retrieval recall@5, faithfulness, answer relevancy) come from the Step 6 eval gate; this collector covers the online half.
  - **LangfuseTracer** (item 39): thin wrapper around the Langfuse SDK with graceful degradation. When Langfuse is available (package importable + `LANGFUSE_HOST` + public/secret keys configured), traces and spans are sent to the self-hosted container. When any condition fails, every method is a safe no-op that still emits a structured log line — the doc's "Langfuse-down fallback to structlog" mitigation. Lazy-client pattern (like `Generator`/`Embedder`) keeps unit tests network-free. Supports `start_trace`, `start_span`/`end_span`, a `span()` context manager, `record_score` (for feedback), and `flush`/`close`.
  - **warm_up_ollama()** (item 41): pings Ollama `/api/tags` then sends a minimal `generate` call (num_predict=1) to pre-load the model into memory — the doc's "warm-up call on startup" mitigation. Returns False if Ollama is unreachable; the app starts either way.
  - **check_ollama() / check_qdrant()**: lightweight liveness checks (GET with short timeout) used by the readiness endpoint.
- `rag/orchestrator.py` — updated `RAGOrchestrator` to accept an optional `tracer`. `stream_answer()` and `answer()` now emit a Langfuse trace per request with child spans for `guardrail_input`, `retrieval`, `generation`, and `guardrail_output`. The trace id is set on the `PostProcessResult` so the chat route can surface it in the `ChatResponse` and the frontend can pass it back with feedback for score correlation. When `tracer=None` (the default for existing tests), the flow is identical to Step 6 — no behavior change.
- `rag/generator.py` — updated `Generator` to accept an optional `circuit_breaker`. The Ollama `chat` call is routed through the breaker: if the circuit is open, `CircuitOpenError` is raised before any network call; a connection failure increments the breaker's failure count. When `circuit_breaker=None` (the default for existing tests), the flow is unchanged.
- `rag/post_processor.py` — added `trace_id` field to `PostProcessResult` (defaults to None). Set by the orchestrator when tracing is enabled.
- `api/routes/health.py` — added `GET /api/v1/health/ready` readiness probe. Pings Qdrant + Ollama with short timeouts; returns 200 when both are reachable, 503 otherwise. The body includes per-dependency status so the operator can see which downstream is down. Liveness (`/health`) stays 200 regardless — liveness is independent of readiness (per the doc's failure-mode table). Service version bumped to 0.7.0.
- `api/routes/metrics.py` — new `GET /api/v1/metrics` endpoint. Returns the `MetricsCollector` snapshot (requests, errors, cache hit rate, TTFT mean/p50/p95) plus the LRU cache stats.
- `api/routes/chat.py` — `build_event_stream()` now accepts an optional `metrics` collector and records: request count on entry, TTFT (time from start to first token) on the first token, and error count on exception. The `chat()` endpoint records cache hit/miss and passes the collector through. The `result` frame now includes `trace_id` (from the orchestrator's Langfuse trace) so the frontend can pass it back with feedback.
- `api/routes/feedback.py` — when the request carries a `trace_id`, the feedback rating is attached as a Langfuse score (`user_feedback`, 1.0 for up / 0.0 for down) on the corresponding trace. Without a trace_id the score is logged via structlog (the Langfuse-down fallback).
- `api/deps.py` — added `get_circuit_breaker()`, `get_tracer()`, and `get_metrics()` (all `lru_cache`'d singletons) and wired them into `get_orchestrator()` so the production orchestrator has tracing + circuit breaker enabled by default.
- `api/main.py` — added a `lifespan` context manager that runs the Ollama warm-up on startup and flushes Langfuse on shutdown. Mounted the `metrics` router. Service version bumped to 0.7.0.
- `schemas/chat.py` — added optional `trace_id` field to `ChatResponse` (defaults to None). Present when Langfuse tracing is enabled; None when disabled.
- `schemas/feedback.py` — added optional `trace_id` field to `FeedbackRequest` (defaults to None). When present, the feedback score is attached to the Langfuse trace.
- `Dockerfile` — multi-stage build for the FastAPI backend (Python 3.12-slim, pip3 install of requirements.txt, copies app + seed corpus, uvicorn entrypoint, healthcheck via `/api/v1/health`). Capped at 1 CPU / 1 GB RAM via docker-compose.
- `docker-compose.yml` — added the `backend` service (builds from Dockerfile, `restart: unless-stopped`, depends on qdrant healthy, healthcheck, 1 CPU / 1 GB RAM). Connects to Qdrant at `http://qdrant:6333`, Ollama at `http://host.docker.internal:11434`, Langfuse at `http://langfuse-web:3000`.
- `.env.example` — added `OLLAMA_WARMUP_MODEL`, `FRONTEND_ORIGIN`, and `LANGFUSE_HOST`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (blank by default → tracing disabled).
- `tests/test_observability.py` — 67 tests: `CircuitBreaker` (construction, call success/failure, open/half-open/closed transitions, threshold + timeout, reset, args/kwargs), `MetricsCollector` (counters, cache hit rate, TTFT samples + percentiles, max-samples cap, reset), `_percentile` (empty, single, p0/p50/p100, even/odd), `LangfuseTracer` (disabled no-ops, enabled span/score delegation, graceful degradation on SDK errors, lazy client, span context manager + error recording, detection logic), `check_ollama`/`check_qdrant`/`warm_up_ollama` (unreachable, 200, non-200).
- `tests/test_rag_generator.py` — 5 new circuit breaker integration tests (stream uses breaker, failure recorded, circuit-open raises, no-breaker unchanged, success resets).
- `tests/test_rag_orchestrator.py` — 9 new tracer integration tests (start_trace called, retrieval/generation/guardrail spans, trace_id on result, no-tracer unchanged, answer method tracer).
- `tests/test_api_metrics.py` — 5 tests: 200 status, expected keys, recorded metrics reflected, cache store stats, empty snapshot.
- `tests/test_api_health_ready.py` — 8 tests: 200 when both up, 503 when Ollama/Qdrant/both down, version, dependency URLs, liveness endpoint.
- `tests/test_api_feedback.py` — 2 new tests: trace_id accepted, trace_id optional (backward compatible).
- All 481 backend tests pass (`python -m pytest -q`); all 54 frontend tests pass (`npm test`).

### Notes — Step 7

- **Langfuse tracer is optional and injectable:** the orchestrator's `tracer` parameter defaults to `None`, so all Step 6 tests (which construct the orchestrator without a tracer) pass unchanged. The production wiring in `api/deps.py` constructs a `LangfuseTracer` that auto-disables when env vars are absent; tests override `get_orchestrator` entirely via `app.dependency_overrides`, so the tracer is never reached in API tests. This keeps the tracing layer testable in isolation without coupling it to the existing 385 tests.
- **Circuit breaker is optional and injectable:** the generator's `circuit_breaker` parameter defaults to `None`, so all Step 3 generator tests pass unchanged. The production wiring in `api/deps.py` constructs a shared `CircuitBreaker(threshold=3, timeout=30)` and injects it into the generator. The breaker is in-process (no external state) — sufficient for a single-user practice project. For multi-process or multi-instance, an external breaker (e.g. via Redis) would be needed.
- **TTFT is measured at the chat route, not the orchestrator:** the orchestrator is a pure Python generator with no HTTP awareness, so TTFT (time-to-first-token) is measured in `build_event_stream()` as the elapsed time from the start of the stream to the first `str` item yielded. This captures the full latency including query rewrite + retrieval + the first generation token, which is what the user perceives. The orchestrator's Langfuse generation span provides a finer-grained breakdown (retrieval vs generation) in the Langfuse UI.
- **Readiness vs liveness separation:** `/health` (liveness) always returns 200 while the process is up — it's used by the Docker Compose healthcheck and never depends on downstreams (a slow Qdrant/Ollama cannot flap it and cause restart loops). `/health/ready` (readiness) pings both downstreams and returns 503 if either is down — intended for a load balancer or `depends_on: condition: service_healthy` to withhold traffic. This matches the doc's failure-mode table: "Qdrant down → FastAPI returns 'Search unavailable'" (the process stays up, just not ready).
- **Langfuse score correlation via trace_id:** the chat `result` frame now includes `trace_id` (when tracing is enabled). The frontend passes it back with feedback; the feedback route attaches the score to the Langfuse trace. When Langfuse is disabled, `trace_id` is None and the score is logged via structlog — the correlation still works in logs. This is a backward-compatible addition (optional field, defaults to None).
- **Ollama warm-up is best-effort:** the lifespan warm-up pings Ollama and pre-loads the model. If Ollama is unreachable at startup, the warm-up logs a warning and returns False — the app still starts. The first real query will be slow (cold model load) but the circuit breaker + error handling will catch a hard failure. This matches the doc's "Pre-pull models in Dockerfile or entrypoint.sh; warm-up call on startup" mitigation.
- **Backend container uses host Ollama:** per deviation D1, Ollama runs on the host (not in Docker). The backend container reaches it at `http://host.docker.internal:11434` (Docker Desktop / Colima's host alias). This avoids a second Ollama container + 6 GB model cache. If deploying to a clean machine, set `OLLAMA_BASE_URL` to the Ollama container URL and add Ollama to docker-compose.
- **Dockerfile does not reproduce the conda env:** the container uses the slim Python base's system Python + pip3 (no conda in the container). The conda env is for host-side dev only (ingestion, eval, tests). The container is capped at 1 CPU / 1 GB RAM per the project rule.
- **Optional cloud showcase (item 42) is deferred:** the doc's build-order item 42 ("Optional cloud showcase: Vercel + Render/Railway + Groq + Qdrant Cloud + Turso") is not implemented. It's a deployment-target change, not a code change — the app is already cloud-agnostic via env vars (`QDRANT_URL`, `OLLAMA_BASE_URL`, `LANGFUSE_HOST`). A follow-up PR would add deployment configs (Vercel `vercel.json`, Render `render.yaml`) and a Groq client option in the generator. Documented here so the Step 7 scope is clear.

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

- Repository is in **Step 6 — Guardrails & evaluation** (complete). Steps 0–6 are done. The next step is Step 7 — Observability & hardening (Langfuse traces for retrieval/generation/guardrail spans + feedback events, metrics: TTFT/retrieval recall@5/faithfulness/cache hit rate, resilience: docker restart policies + healthchecks + Ollama warm-up + circuit breaker + Langfuse-down structlog fallback, optional cloud showcase).
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

### Added — Step 5 (React frontend)

- `frontend/src/` — the React 18 + Vite chat widget that consumes the FastAPI SSE stream. Maps to Phase 4 — Frontend (build-order items 26–32).
  - `api.js` — SSE consumer + REST helpers. `parseSseFrame()` is a pure frame parser that dispatches `data: <token>` → `onToken`, `event: result` → `onResult` (parsed `ChatResponse` JSON), and `data: [DONE]` → `onDone`. Multi-line data frames (newline inside a token) are reconstructed by rejoining continuation lines. `streamChat()` uses `fetch` + `ReadableStream` reader (per the doc's snippet) with a buffer that splits on `\n\n` to handle tokens split across chunk boundaries. `sendFeedback()` and `fetchHistory()` are thin `fetch` wrappers for the REST endpoints.
  - `ChatWidget.jsx` — the main orchestrator. Manages `messages` (chronological list with id, role, content, citations, confidence, streaming, error), `sessionId` (null until the first `result` frame), `isStreaming`, and `feedbackState`. `handleSend()` appends a user + empty assistant message, calls `streamChat()` with token/result/done callbacks that update the assistant bubble, then sets `isStreaming=false`. `handleFeedback()` calls `sendFeedback()` and reverts the UI state on failure. `handleNewChat()` resets all state. Session id from the `result` frame is carried to subsequent requests.
  - `MessageList.jsx` — renders the message list with auto-scroll (`useRef` + `useEffect`). Assistant messages show citation chips, a low-confidence warning when `confidence < 0.65` (matching the doc's threshold), and thumbs up/down feedback buttons (enabled after streaming completes, disabled when no session id). Error messages replace feedback buttons. Empty state shows a placeholder.
  - `InputBox.jsx` — textarea + SendButton. Enter sends (trimmed), Shift+Enter inserts a newline. Disabled while streaming. Send button disabled when text is empty. Auto-refocuses after send.
  - `CitationChip.jsx` — renders `[Source: title]` as an external link (`target="_blank"`, `rel="noopener noreferrer"`). Falls back to `#` when no URL; supports both `source_url` and `sourceUrl` keys.
  - `ErrorBoundary.jsx` — class component catching render errors in the chat subtree. Shows a fallback UI with the error message and a "Try again" button that resets the boundary state.
  - `App.jsx` — root component rendering `ChatWidget`.
  - `main.jsx` — React root + `styles.css` import.
  - `styles.css` — chat widget styles (message bubbles, citation chips, feedback buttons, input box, error boundary).
- `frontend/jest.setup.js` — jsdom polyfills: `scrollIntoView` mock (jsdom doesn't implement it), `TextEncoder`/`TextDecoder`/`ReadableStream` from Node `util`/`stream/web` (jsdom doesn't provide them globally, needed by the `streamChat` test mock).
- `frontend/__tests__/` — 54 tests across 7 suites:
  - `api.test.js` (18 tests) — `parseSseFrame` dispatch (token/result/done, multi-line, empty), `streamChat` (ordered tokens+result+done, chunk-boundary splitting, request body, null session_id, non-2xx error, cache replay), `sendFeedback` (body + error), `fetchHistory` (GET + error).
  - `CitationChip.test.jsx` (3 tests) — link rendering, URL fallback, camelCase key.
  - `MessageList.test.jsx` (11 tests) — empty placeholder, user/assistant messages, citation chips, low-confidence warning (above/below threshold), feedback buttons (shown/hidden while streaming, disabled without session, active state), error text.
  - `InputBox.test.jsx` (7 tests) — renders textarea+button, send+clear, empty guard, Enter send, Shift+Enter newline, disabled state, empty-button disabled.
  - `ChatWidget.test.jsx` (10 tests) — header+placeholder+input, streaming response, citation chips, low-confidence warning, input disable/enable during streaming, error message, feedback wiring, feedback revert on failure, new chat reset, session id propagation.
  - `ErrorBoundary.test.jsx` (3 tests) — renders children, fallback on throw, recovery via "Try again".
  - `App.test.jsx` (2 tests) — root renders title + chat input.
- All 54 frontend tests pass (`npm test`). Production build succeeds (`npm run build` → `dist/`, 149 KB JS / 3.5 KB CSS gzipped to 48 KB / 1.2 KB).

### Notes — Step 5

- **SSE `event: result` frame handling:** the architecture doc's React snippet only reads `data:` frames and strips the `data: ` prefix. The backend (Step 4) extends the wire format with an `event: result` frame carrying the full `ChatResponse` JSON (citations + confidence + session id). The frontend's `parseSseFrame()` handles both `data:` and `event:` lines, dispatching the result frame to `onResult` so citation chips and the low-confidence warning can be rendered without parsing streamed text. This is the deliberate extension flagged in the Step 4 notes.
- **Session id flows from the result frame:** the frontend starts with `sessionId=null` (server creates a new session). The `result` frame's `session_id` is captured and passed to all subsequent `streamChat` and `sendFeedback` calls. This matches the backend's contract: the first response creates the session, later responses carry the id.
- **Feedback `messageIndex` is the chronological position:** the backend's `FeedbackRequest.message_index` is the 0-based position in the session's message list. The frontend passes the React array index of the assistant message, which matches because messages are appended in chronological order (user + assistant pairs). If the frontend later supports restoring history via `fetchHistory`, the index will need to be derived from the restored list, not the React array.
- **No `fetchHistory` integration yet:** `fetchHistory()` is implemented and tested in `api.js` but not wired into the UI. The "New chat" button resets state rather than restoring a prior session. History restoration is a natural follow-up but not in the Step 5 scope (the doc's component tree doesn't include a session picker).
- **jsdom polyfills:** jsdom (via `jest-environment-jsdom`) does not implement `Element.scrollIntoView`, `TextEncoder`, `TextDecoder`, or `ReadableStream`. The `jest.setup.js` file mocks `scrollIntoView` and polyfills the rest from Node built-ins. Without these, the MessageList auto-scroll effect throws (triggering the ErrorBoundary) and the `streamChat` test mock can't construct `ReadableStream` chunks.
- **No state management library:** per the doc, state is `useState`/`useRef`/`useEffect` only — no Redux/Zustand. The widget is small enough that prop drilling (ChatWidget → MessageList/InputBox) is sufficient.
- **CSS is a single file:** `styles.css` is imported in `main.jsx` and covers the full widget. No CSS-in-JS or preprocessor — keeping the build simple per the doc's "keep it small" guidance.

### Added — Step 6 (Guardrails & evaluation)

- `api/guardrails.py` — the defense-in-depth guardrail layer (build-order items 33–35). Maps to Phase 5 — Guardrails & Eval.
  - **Input guardrails** (item 33): `detect_prompt_injection()` is a pure regex scan with 10 patterns (ignore/disregard previous instructions, role-reset prefixes, DAN jailbreak, system-prompt extraction, rule resets, fake developer-mode). `InputGuardrail` combines the regex tier (always runs, zero latency) with an optional local LLM judge (`llama3.2:3b`, 8-token "safe"/"unsafe" response). The LLM judge is a second line of defense for subtler attacks the regex misses; on any Ollama error it degrades to regex-only (never blocks on LLM failure).
  - **Output guardrails** (item 34): `scrub_pii()` is a pure regex replacement of emails (`[REDACTED-EMAIL]`) and phone numbers (`[REDACTED-PHONE]`). `OutputGuardrail` runs PII scrubbing (always) and an optional harmful-content LLM judge (hate / violence / self-harm refusal). The scrubbed text is what gets post-processed and persisted.
  - **Query classifier/router** (item 35): `QueryClassifier` routes a query to one of `documentation` / `greeting` / `off_topic` / `compare` (the four classes from the doc's Query Classification & Routing table). Uses a cheap local LLM classifier (10-token prompt) with a keyword fallback (`classify_keywords()`) so it works without Ollama. `greeting` and `off_topic` are handled with canned answers (no retrieval, no generation); `documentation` and `compare` proceed through the full RAG flow.
  - `GuardrailSuite` facade bundles the three components; each is independently injectable for testing.
- `rag/orchestrator.py` — updated `RAGOrchestrator` to accept an optional `guardrail_suite`. `stream_answer()` and `answer()` now run: (1) input guardrail before retrieval (blocked → refusal stream, no retrieval/generation), (2) query classification (greeting/off_topic → canned answer, no retrieval/generation), (3) output guardrail after generation (PII scrub always; harmful → refusal replaces the persisted answer). When `guardrail_suite=None` (the default for existing tests), the flow is identical to Step 3 — no behavior change.
- `api/deps.py` — added `get_guardrail_suite()` (lru_cache'd process-wide singleton) and wired it into `get_orchestrator()` so the production orchestrator has guardrails enabled by default. LLM-backed checks degrade to regex/keyword fallbacks if Ollama is unreachable.
- `eval/golden-dataset.json` — 36 hand-curated Q&A triples (question, ground_truth, ground_truth_context, source_title) covering all 7 corpus documents (FastAPI path-params/query-params/dependency-injection, Pydantic models/types, SQLModel intro/relationships). 5 questions per doc on average. Includes metadata (version, thresholds, corpus description).
- `eval/run_eval.py` — offline Ragas eval script with local judge + threshold gate (build-order items 37–38). Loads the golden dataset, runs the full RAG pipeline (`RAGOrchestrator.answer()`) for each question, computes `faithfulness` / `answer_relevancy` / `context_recall`, writes a CSV report, prints a summary table, and exits non-zero if any metric mean is below threshold (`faithfulness >= 0.75`, `context_recall >= 0.70`, `answer_relevancy >= 0.70`). Supports `--limit N` for quick smoke tests, `--no-ragas` to force the local judge, and custom thresholds. Tries the `ragas` library first; if it can't import (version mismatch with `langchain_community`), falls back to a lightweight local LLM judge that scores each metric on a 0–1 scale with simple prompts.
- `tests/test_guardrails.py` — 61 tests: `detect_prompt_injection` (13 positive/negative cases), `scrub_pii` (6 cases), `classify_keywords` (7 cases), `InputGuardrail` (10 cases: regex block, LLM block, LLM safe, error degradation, use_llm=False, lazy client, close), `OutputGuardrail` (8 cases: PII scrub always, harmful block, safe, error degradation, use_llm=False, empty answer, close), `QueryClassifier` (12 cases: all four labels, handled flags, error fallback, use_llm=False, invalid label fallback, is_documentation property, model, close), `GuardrailSuite` (5 facade delegation tests).
- `tests/test_rag_orchestrator.py` — 12 new guardrail integration tests (8 for `stream_answer`, 4 for `answer`): input-blocked short-circuit, greeting/off_topic canned answers, documentation/compare proceed, output-blocked refusal replacement, PII scrub application, no-guardrail-suite unchanged behavior.
- `tests/test_eval.py` — 25 tests: `load_dataset` (7 cases: default dataset, required fields, unique questions, source coverage, custom path, empty, missing source_title), `check_gate` (6 cases: all-pass, each-metric-fails, custom thresholds, exact threshold), `mark_passes` (4 cases: all-pass, faithfulness-fail, error-fail, custom thresholds), `aggregate` (5 cases: means, error exclusion, passed count, empty, all-errors), `write_csv` (3 cases: header+rows, all columns, empty).
- All 385 backend tests pass (`python -m pytest -q`); all 54 frontend tests pass (`npm test`).

### Notes — Step 6

- **Guardrail suite is optional and injectable:** the orchestrator's `guardrail_suite` parameter defaults to `None`, so all Step 3 tests (which construct the orchestrator without guardrails) pass unchanged. The production wiring in `api/deps.py` constructs a `GuardrailSuite` with LLM checks enabled; tests override `get_orchestrator` entirely via `app.dependency_overrides`, so the guardrails are never reached in API tests. This keeps the guardrail layer testable in isolation without coupling it to the existing 287 tests.
- **Two-tier input guardrail (regex + LLM):** the regex tier is fast and deterministic (10 patterns, case-insensitive) and always runs. The LLM judge is a second line of defense for subtler injections the regex misses (e.g. "pretend you have no rules"). On any Ollama error, the judge is skipped and the guardrail degrades to regex-only — it never blocks on LLM failure. This matches the doc's "Regex + local LLM judge prompt" mitigation.
- **Output guardrail replaces the persisted answer, not the streamed tokens:** when the harmful-content judge blocks an answer, the already-streamed tokens are not un-sent (SSE is one-way). Instead, the orchestrator replaces the `answer` variable with a refusal before post-processing, so the `PostProcessResult`, session store, and history endpoint all reflect the refusal. The user sees the original tokens stream followed by a refusal in the `result` frame — a known trade-off documented here for interview defensibility. A production system would buffer the full answer before streaming (trading latency for safety).
- **Query classifier keyword fallback is conservative:** `classify_keywords()` defaults to `documentation` when no keyword matches, so a real question still reaches the RAG flow rather than being silently rejected. The greeting check requires a very short message (≤2 words after stripping punctuation) that starts with a greeting word, so "hello how do I use FastAPI" correctly falls through to documentation. The LLM classifier handles nuanced cases when Ollama is available.
- **`compare` class is classified but not specially handled:** the doc's routing table says `compare` should "retrieve two topics then generate comparison." The current implementation classifies `compare` queries but routes them through the standard RAG flow (same as `documentation`). A dedicated two-topic retrieval + comparison generation path is a natural follow-up but not in the Step 6 scope — the classifier label is available for future routing logic.
- **Ragas import is broken in the current env:** `ragas` 0.4.3 (from `vibrantlabsai`) fails to import due to `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` (langchain-community 0.4.2 sunset). The eval script handles this gracefully: `_try_ragas_imports()` checks importability, and on failure falls back to `run_with_local_judge()` which uses simple Ollama prompts to score each metric 0–1. This is a **deviation** from the doc's "Ragas offline eval" — documented here so the eval gate works today while the ragas version issue is resolved. The fallback judge is less rigorous than Ragas (no multi-step faithfulness decomposition) but provides a working regression gate.
- **Golden dataset has 36 examples (not 30–50):** the doc says "30–50 Q&A pairs." 36 covers all 7 documents with 4–6 questions each, which is sufficient for a practice eval gate. More examples can be added as the corpus grows.
- **Eval gate exit code:** `run_eval.py` exits 0 if all metric means meet thresholds, 1 otherwise. This makes it suitable for a pre-commit hook or CI gate: `python eval/run_eval.py && git commit`. The `--limit N` flag supports quick smoke tests during development.
- **`GuardrailSuite` import is under `TYPE_CHECKING`:** the orchestrator imports `GuardrailSuite` only under `typing.TYPE_CHECKING` to avoid a runtime circular import (`api.guardrails` → nothing from `rag`, but `rag` is imported widely). The orchestrator has no hard runtime dependency on the `api` package — the guardrail suite is passed in by the caller (`api/deps.py`).
