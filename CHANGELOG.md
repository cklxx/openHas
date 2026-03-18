# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-03-18

### Changed

- **`evals/recall_eval.py`** — wire `consolidate()` into search index; superseded nodes
  (`sched-sync`, `work-role-old`) are decayed by 0.1× so they don't surface in results.
- **`evals/recall_eval.py`** — add AR@5 anti-recall metric; `--no-predict` flag and
  prediction-sample step removed.
- **`evals/hard_cases.jsonl`** — add 18 `ku_filter` cases (9 schedule + 9 role
  supersession) with `forbidden_ids` assertions.

## [0.1.0] - 2026-03-18

### Added

- **Domain types** — `MemoryNode`, `MemoryGraph`, `Edge`, `MemoryQuery`, `RecallResult`,
  `QueryDistribution`, `WriteDecision`, `ConsolidationAction` as frozen dataclasses with
  dual timestamps (`event_time` / `record_time`) and permanence classification.
- **`core/scoring`** — `compute_predictive_value`: decay-weighted scoring that combines
  recency, access frequency, and query-distribution match.
- **`core/consolidation`** — `consolidate`: three-pass background consolidation
  (contradiction resolution, supersession propagation, low-value pruning).
- **`core/memory`** — `make_recall`: hybrid retrieval combining doc embeddings and
  context-expanded query embeddings.
- **`adapters/llama_embed`** — doc and query embedding via llama-server HTTP.
- **`adapters/llama_expand`** — hypothetical-query context expansion via llama-server.
- **`adapters/llama_predict`** — next-query prediction via llama-server.
- **`evals/recall_eval.py`** — end-to-end recall evaluation with R@1/R@3/R@5, MRR,
  and AR@5 (anti-recall) metrics; consolidation wired into the search index so
  superseded nodes are decayed.
- **`evals/hard_cases.jsonl`** — 155-case benchmark covering multi-hop, cue-trigger,
  KU (knowledge update), temporal, referential, implicit, adversarial, and
  `ku_filter` categories (18 cases that assert superseded facts do not surface).
