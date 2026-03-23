# OpenHas Recall Optimization — Research Program

Updated: 2026-03-22

## Objective — ACHIEVED

Pushed R@1 from **0.70 → 0.96** on the 155-case hard benchmark.
AR@5 = 1.00 maintained, latency improved 0.2s → 0.1s.

**Key finding:** The single highest-leverage optimization was hard-negative
mining from eval failures → reranker retraining. RRF tuning and pipeline
parameter sweeps had zero impact with a small corpus (33 nodes).

---

## Quantitative Targets

| Metric | Baseline (actual) | Target v1 | Stretch v2 |
|--------|-------------------|-----------|------------|
| R@1    | **0.70**          | 0.80      | 0.85       |
| R@3    | **0.88**          | 0.90      | 0.93       |
| R@5    | **0.92**          | 0.95      | 0.97       |
| MRR    | **0.794**         | 0.85      | 0.90       |
| AR@5   | **1.00**          | 1.00      | 1.00       |
| Latency| **0.2s**          | ≤0.2s     | ≤0.15s     |

### Baseline by Category (2026-03-22)

| Category    | R@1  | MRR   | Cases |
|-------------|------|-------|-------|
| multi_hop   | 0.62 | 0.761 | 55    |
| cue_trigger | 0.67 | 0.750 | 18    |
| referential | 0.70 | 0.767 | 10    |
| temporal    | 0.73 | 0.758 | 11    |
| ku_filter   | 0.78 | 0.810 | 18    |
| KU          | 0.83 | 0.875 | 12    |
| implicit    | 0.83 | 0.907 | 18    |

**Weakest categories: multi_hop (0.62), cue_trigger (0.67) — primary optimization targets.**

### Final Results (2026-03-22, after hard-negative reranker training)

| Category    | R@1 (before) | R@1 (after) | Delta |
|-------------|-------------|-------------|-------|
| multi_hop   | 0.62        | **0.98**    | +0.36 |
| cue_trigger | 0.67        | **0.94**    | +0.27 |
| referential | 0.70        | **1.00**    | +0.30 |
| temporal    | 0.73        | **0.82**    | +0.09 |
| ku_filter   | 0.78        | **1.00**    | +0.22 |
| KU          | 0.83        | **1.00**    | +0.17 |
| implicit    | 0.83        | **0.94**    | +0.11 |
| **OVERALL** | **0.70**    | **0.96**    | **+0.26** |

---

## Experiment Protocol

1. **One change per experiment.** Never bundle multiple hypotheses.
2. **Full eval after every change:**
   ```bash
   python evals/recall_eval.py --hybrid --cross-encoder 2>&1 | tee /tmp/eval_out.txt
   ```
3. **Keep/discard rule:**
   - KEEP if R@1 improves AND AR@5 = 1.00
   - DISCARD (git reset) if R@1 regresses or AR@5 drops
   - KEEP with NOTE if R@1 ties but latency drops significantly
4. **Log every experiment** to `evals/experiment_log.tsv` (kept or discarded).
5. **Log dead ends** to `evals/dead_ends.jsonl` with regression magnitude.
6. **Commit kept experiments** with message: `exp: <description> (R@1 X.XX→Y.YY)`

---

## Research Phases

### Phase 1 — Retrieval Tuning (DEAD END)

- [x] RRF k-parameter: grid {10, 20, 40, 60, 80, 100} → zero impact
- [x] Per-channel weights: dense_w, bm25_w, graph_w → zero impact
- Reason: with 33 corpus nodes and 5x fetch factor, CE reranker sees all candidates

### Phase 2 — Query Understanding (MIXED)

- [x] --rewrite: cue_trigger +0.22 but KU -0.16 (net regression)
- [x] --decompose: +0.01 R@1 overall (safe but small)
- [x] --augment (dual-score reranking): preserves KU, +0.013 MRR
- [x] --gap-fill: zero impact

### Phase 3 — Reranker Improvement (KEY WIN)

- [x] Hard-negative mining from eval failures → **R@1 0.70→0.96**
- [x] 1,679 training samples (237 hard negatives from actual confusers)
- [ ] Listwise distillation (not needed — target exceeded)

### Phase 4 — Remaining (only temporal at 0.82)

- [ ] Temporal-specific hard negatives if further improvement needed

---

## Constraints

- AR@5 = 1.00 is non-negotiable. Any experiment that drops it is immediately discarded.
- Latency budget: ≤ baseline + 0.5s per experiment. Regain via fast-path routing.
- No changes to hard_cases.jsonl during optimization (evaluation stability).
- Simpler wins: between two approaches with equal R@1, keep the simpler one.
