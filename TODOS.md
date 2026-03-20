# openHas — TODOS

Deferred work from CEO review (2026-03-18 and 2026-03-19, /plan-ceo-review, SELECTIVE EXPANSION mode).

> **2026-03-19 update:** TODO-1 completed. Eval now uses production `sqlite_vec_store` adapters,
> expansion cache eliminates cross-run variance, and query rewriting bridges the vocabulary gap.
> See `docs/designs/recall-pipeline.md` for the final design.

---

## P1 — Must do soon

### ~~[TODO-1] Eval-production adapter convergence~~ ✅

**What:** Update `evals/recall_eval.py` to use the same `sqlite_vec_store` adapter as the
production path, replacing the custom in-memory numpy index.

**Why:** The eval currently validates a code path that doesn't exist in production. Eval
scores are meaningless until they reflect the same stack the user actually runs.

**Pros:** Eval quality = production quality. One source of truth for the search implementation.

**Cons:** Eval becomes slower (SQLite I/O vs in-memory) and requires llama-server running.
Integration test rather than pure benchmark.

**Context:** The eval's `_make_search_fn` builds a numpy normed matrix from scratch each run.
The production `SearchFn` will use sqlite-vec's ANN search. These must converge.
Start by adding an adapter flag (`--use-prod-adapter`) before removing the numpy path entirely.

**Effort:** S (human: ~4h / CC: ~10 min)
**Priority:** P1
**Depends on:** sqlite-vec adapter (`src/adapters/sqlite_vec_store.py`) complete

---

### [TODO-2] Multi-user schema isolation (one-way door)

**What:** Add `user_id TEXT NOT NULL DEFAULT 'default'` to the `nodes` and `edges` tables.
Use `Claims.sub` from the auth module as the user identity.

**Why:** The auth module already supports multi-user JWT. The schema doesn't. Adding `user_id`
after rows are populated requires a migration touching every row. Cost today: near-zero.
Cost after 1k+ memories exist: painful.

**Pros:** Future-proof from day one. Per-user memory isolation comes for free.
Enables multi-user deployment without schema migration.

**Cons:** Tiny increase in schema complexity. Single-user use is unaffected (default='default').

**Context:** See `src/domain_types/auth.py` — `Claims.sub` is the natural user_id.
Add `user_id` to every `INSERT` and every `WHERE` clause in the storage adapter.
MCP server: extract `Claims.sub` from JWT and pass to all store/recall calls.

**Effort:** S (human: ~2h / CC: ~5 min)
**Priority:** P1 (one-way door — do in the same commit as schema creation)
**Depends on:** SQLite schema design (do this before any rows are written)

---

## P2 — Important but not blocking

### [TODO-3] FastAPI REST entrypoint

**What:** Add `src/entrypoints/api.py` with FastAPI. Endpoints:
`POST /ingest`, `GET /recall?q=...&top_k=5`, `DELETE /memory/{node_id}`.
JWT auth on all endpoints using the existing `core/auth.authenticate`.

**Why:** MCP server covers LLM integration. REST API covers all other clients: web apps,
mobile, Go/JS services, curl-based automation.

**Pros:** Makes openHas a self-hosted memory microservice. Non-Python clients can integrate.

**Cons:** Adds a FastAPI dependency. A second entrypoint to maintain.

**Context:** Auth, core functions, and domain types are all ready. This is pure wiring.
Follow the same factory-function pattern as the CLI: `_make_app(db_path, secret)`.

**Effort:** M (human: ~1 day / CC: ~15 min)
**Priority:** P2
**Depends on:** P0 wiring + MCP server complete

---

### [TODO-4] Auto-ingest adversarial robustness

**What:** Harden the fact extraction pipeline against adversarial inputs:
(a) Store `source_turn_id` provenance on each extracted fact.
(b) Add a confidence threshold: LLM must express confidence before fact is ingested.
(c) Validate extracted facts against existing graph before inserting contradictions.

**Why:** If the user summarizes a malicious document in conversation, the auto-ingest
pipeline may extract and store adversarial facts. This is the highest-severity security
concern for auto-ingest mode.

**Pros:** Safe to use in any conversation, including with untrusted input.

**Cons:** Adds complexity to the extraction pipeline. Confidence scoring requires LLM
prompt tuning and calibration work.

**Context:** See Section 3 of the CEO review (2026-03-18). The ingestion gate's value
threshold doesn't protect against adversarial facts — it only filters low-value content.
A high-confidence malicious fact passes the gate unchanged.

**Effort:** L (human: ~1 week / CC: ~30 min)
**Priority:** P2 (before any multi-user or public deployment)
**Depends on:** Auto-ingest baseline working and validated on honest inputs first

---

### [TODO-5] Consolidation scheduling strategy

**What:** Define when and how often `consolidate()` is triggered. Three options to evaluate:
(a) On every ingest — correct, O(n) per write, may be acceptable at small scale.
(b) When `len(graph.nodes)` crosses a threshold — amortized.
(c) Background cron / timer-based.

**Why:** Currently `consolidate()` is never called in production. Superseded facts
(e.g. `sched-sync`) will surface in recall indefinitely until consolidation runs.

**Pros:** Correct behavior: stale facts decay, contradictions resolve, graph stays clean.

**Cons:** Consolidation is O(n) over the full graph. At 100k nodes, running on every ingest
is expensive. Need to measure before choosing a strategy.

**Context:** The eval wires consolidation into the search index at build time (`_build_index`).
Production needs an equivalent. Start with on-ingest triggering, add a threshold guard
(`if len(nodes) > _CONSOLIDATION_MIN_SIZE`), then move to background if perf degrades.

**Effort:** M (human: ~1 day / CC: ~15 min)
**Priority:** P2
**Depends on:** P0 consolidation executor working

---

### [TODO-6] SQL-native consolidation (avoid full MemoryGraph load)

**What:** Rewrite the consolidation executor to work directly against the DB using SQL queries
(`SELECT * FROM edges WHERE kind = 'contradicts'`, etc.) instead of loading all nodes into
a `MemoryGraph` first.

**Why:** The current design loads the full graph into memory (O(n)) to feed `consolidate()`.
For personal memory (≤10k nodes) this is ~50ms and fine. At higher scale it wastes RAM.

**Pros:** Eliminates the full-graph RAM load. Consolidation becomes O(edges) + O(low-value nodes).
Enables running consolidation more frequently without cost concerns.

**Cons:** Requires redesigning the consolidation interface from `consolidate(graph)` to
SQL-native passes. Some of the beautiful pure-function design is lost to the DB layer.

**Context:** `core/consolidation.py` is pure and elegant. The executor loads a MemoryGraph
before calling it. The three passes (contradiction, supersession, low-value) could each be
expressed as SQL queries that the executor runs directly, bypassing the in-memory graph.
Decision: only do this if consolidation becomes a measurable bottleneck first.

**Effort:** M (human: ~2 days / CC: ~20 min)
**Priority:** P3
**Depends on:** P0 consolidation executor working and proven stable

---

### ~~[TODO-7] kind/label filter support in SearchFn~~ ✅

**What:** Extend `SearchFn` protocol to accept optional `kinds: tuple[NodeKind, ...]` and
`labels: tuple[str, ...]` parameters. Add `WHERE kind IN (...)` and label JSON filtering to
`sqlite_vec_store.make_search_fn`. Wire through `make_recall`.

**Why:** `MemoryQuery.kinds` and `MemoryQuery.labels` are silently dropped today — callers who
pass them get unfiltered results with no error. This is a broken API contract.

**Pros:** Enables structured recall (e.g. "only preferences", "only work-labeled facts").
Reduces reranker input noise by pre-filtering irrelevant node kinds.

**Cons:** `SearchFn` protocol signature change is a breaking change for all adapters and tests.
Must update `tests/adapters/test_sqlite_vec_store.py` and any stub implementations.

**Context:** `SearchFn` currently has signature `(embedding, top_k) -> list[(id, score)]`.
New signature: `(embedding, top_k, kinds=(), labels=()) -> list[(id, score)]`.
The sqlite-vec KNN query can't filter inline — fetch `top_k*2`, then filter by kind/label in
Python after resolving node_ids via vec_meta → nodes join.

**Effort:** S (human: ~4h / CC: ~10 min)
**Priority:** P2
**Depends on:** Core recall pipeline wiring complete

---

### ~~[TODO-8] Adaptive blend weight tuning~~ ✅

**Done:** Grid search over `_W_HYDE ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]` on 142-case
eval benchmark. Result: `_W_HYDE=0.25` (R@1=0.606) beats previous default 0.15 (R@1=0.570).

`_W_EXP / _W_DOC` removed — reverted to MAX scoring per node. Only `_W_HYDE` remains as a
tunable blend weight. `RecallDeps.hyde_weight` allows per-instance override. Eval `--tune`
flag runs grid search with index-once optimization.

---

### ~~[TODO-9] Streaming recall (return base hits immediately, upgrade when reranker finishes)~~ ✅

**What:** Return `RecallResult` from base KNN immediately (< 100ms), then emit an updated
result when the reranker pass completes (~1-2s later). Requires an async streaming protocol.

**Why:** With all stages enabled, recall latency is ~2s. For interactive use (chat assistant,
autocomplete), 2s is too slow. Streaming returns useful results in 60ms and silently upgrades.

**Pros:** Perceived latency drops from ~2s to ~60ms for the first result. Real-time UX.
No accuracy tradeoff — the final result is still fully reranked.

**Cons:** Requires a streaming `RecallResult` type (e.g. `AsyncGenerator`). Caller must handle
two emissions. Complicates CLI output (would need to overwrite the printed list).

**Context:** This becomes relevant when openHas is used as a service (TODO-3: FastAPI entrypoint)
rather than a CLI tool. The streaming protocol should be modeled as
`AsyncGenerator[RecallResult, None]` — first yield is base KNN, second yield is reranked.
The existing `make_*` factory pattern can be extended: `make_streaming_recall(...)`.

**Effort:** L (human: ~2 days / CC: ~20 min)
**Priority:** P2 (when FastAPI entrypoint exists)
**Depends on:** TODO-3 (FastAPI entrypoint)
