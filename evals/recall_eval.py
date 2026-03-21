#!/usr/bin/env python3
"""Recall quality evaluation — llama-server HTTP backend.

Fully automatic: downloads models from HuggingFace, starts llama-server
processes, runs the eval, then shuts everything down.

Usage:
    pip install "openhas[eval]"
    python evals/recall_eval.py
    python evals/recall_eval.py --cases evals/hard_cases.jsonl
    python evals/recall_eval.py --no-predict

Requires llama-server in PATH:
    brew install llama.cpp
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from huggingface_hub import hf_hub_download
from src.adapters.cross_encoder_rerank import make_cross_encoder_rerank_fn
from src.adapters.llama_decompose import make_llama_decompose_fn
from src.adapters.llama_embed import make_doc_embed_fn, make_query_embed_fn
from src.adapters.llama_expand import make_llama_expand_fn
from src.adapters.llama_gap_check import make_llama_gap_check_fn
from src.adapters.llama_hyde import make_llama_hyde_fn
from src.adapters.llama_rerank import make_llama_rerank_fn
from src.adapters.llama_rewrite import make_llama_rewrite_fn
from src.adapters.sqlite_vec_store import (
    make_bm25_search_fn,
    make_decay_map_fn,
    make_graph_search_fn,
    make_hydrate_fn,
    make_primary_search_fn,
    make_search_fn,
    make_store_expansion_fn,
    make_store_fn,
    make_update_access_fn,
    open_db,
)
from src.core.consolidation import consolidate
from src.core.memory import (
    HybridDeps,
    RecallDeps,
    HybridConfig,
    make_decomposed_recall,
    make_gap_filled_recall,
    make_hybrid_recall,
    make_hyde_recall,
    make_iterative_recall,
    make_recall,
    make_reranked_recall,
    make_rewritten_recall,
)
from src.domain_types.memory import (
    Edge,
    MemoryGraph,
    MemoryNode,
    MemoryQuery,
    QueryDistribution,
)

# ── Model registry ─────────────────────────────────────────────────────────

_EMBED_REPO = "Qwen/Qwen3-Embedding-0.6B-GGUF"
_EMBED_FILE = "Qwen3-Embedding-0.6B-Q8_0.gguf"
_PREDICT_REPO = "unsloth/Qwen3.5-4B-GGUF"
_PREDICT_FILE = "Qwen3.5-4B-Q4_K_M.gguf"

_EMBED_PORT = 18080
_PREDICT_PORT = 18081
_HTTP_OK = 200

_DEFAULT_CASES = Path(__file__).parent / "hard_cases.jsonl"

# ── Corpus ──────────────────────────────────────────────────────────────────

_CORPUS: list[tuple[str, str]] = [
    # diet / health
    ("diet-veg",        "User follows a strict vegetarian diet, no meat, poultry or seafood"),
    ("health-lactose",  "User has lactose intolerance, causes digestive discomfort"),
    ("health-back",     "User has a lower back condition, max 2 hours sitting without a break"),
    ("health-migraine", "User suffers from migraines triggered by bright screens and loud noise"),
    ("health-meds",
     "User takes a daily medication that must not be combined with alcohol or grapefruit juice"),
    ("partner-allergy", "User's partner is allergic to shellfish and peanuts"),
    ("food-spicy",      "User loves Sichuan cuisine and spicy food, regularly orders 麻辣 dishes"),
    # schedule / work — sched-sync-new supersedes sched-sync
    ("sched-sync",      "Weekly product sync every Tuesday 2pm with the engineering team"),
    ("sched-sync-new",  "Product sync rescheduled: moved from Tuesday 2pm to Thursday 4pm"),
    ("sched-1on1",      "Biweekly 1:1 with manager every other Monday at 10am"),
    ("work-deadline",   "Current sprint deadline is Friday end-of-day, high priority"),
    ("work-remote",     "User works from home on Mondays and Fridays, in-office Tuesday–Thursday"),
    ("focus-block",     "User blocks 2–5pm daily for deep work, disables all notifications"),
    # role — work-role-new supersedes work-role-old
    ("work-role-old",   "User was a senior software engineer"),
    ("work-role-new",   "User was promoted to engineering manager 3 months ago"),
    # life / routine
    ("home-city",       "User currently lives in Shanghai, Jing'an district"),
    ("home-prev",       "User previously lived in Beijing for 5 years before moving to Shanghai"),
    ("routine-run",     "User runs 5km every morning before 8am, skips on rainy days"),
    ("drink-coffee",    "User drinks black coffee every morning, no milk, no sugar"),
    ("sleep-habit",     "User goes to bed at 11pm and wakes at 6:30am every day"),
    ("gym-habit",       "User goes to the gym on Tuesday and Thursday evenings after work"),
    # family / relationships
    ("family-kids",     "User has two young children aged 5 and 8, school pickup at 4pm"),
    ("partner-diet",    "User's partner is pescatarian — eats fish and seafood but no meat"),
    ("parent-care",
     "User's elderly father lives with the family and needs assistance in the evenings"),
    # skills / personality
    ("hobby-jazz",      "User plays jazz piano as a hobby, practices on weekends"),
    ("lang-native",     "User's first language is Mandarin, second language is English"),
    ("lang-third",      "User is learning Spanish, currently at intermediate (B1) level"),
    ("tech-stack",
     "User primarily codes in Python and Go, 8 years of engineering experience"),
    ("comm-style",
     "User strongly prefers async communication, dislikes unexpected phone calls"),
    # travel / comfort
    ("travel-pref",     "User always requests window seat, avoids middle seats on any flight"),
    ("budget-travel",
     "User prefers economy for flights under 3 hours, business class for 5+ hours"),
    # values
    ("value-privacy",
     "User values data privacy strongly, avoids apps that monetise user data"),
    ("value-worklife",
     "User enforces work-life balance, does not check work messages after 9pm"),
]

_CORPUS_PERMANENCE: dict[str, str] = {
    "sched-sync":     "transient",   # superseded
    "sched-sync-new": "permanent",
    "work-role-old":  "transient",   # superseded
    "work-role-new":  "permanent",
}

_CORPUS_EDGES: list[tuple[str, str]] = [
    ("sched-sync-new", "sched-sync"),
    ("work-role-new",  "work-role-old"),
]


def _build_memory_graph() -> MemoryGraph:
    now = time.time()
    nodes = tuple(
        MemoryNode(
            id=nid, kind='fact', content=text,
            event_time=now, record_time=now, last_accessed=now,
            permanence=_CORPUS_PERMANENCE.get(nid, 'unknown'),  # type: ignore[arg-type]
        )
        for nid, text in _CORPUS
    )
    edges = tuple(
        Edge(source_id=src, target_id=tgt, kind='supersedes')
        for src, tgt in _CORPUS_EDGES
    )
    return MemoryGraph(nodes=nodes, edges=edges)


# ── Case loading ────────────────────────────────────────────────────────────

def _load_cases(path: str) -> list[dict]:  # type: ignore[type-arg]
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases

# ── Server lifecycle ────────────────────────────────────────────────────────

def _start_server(
    model_path: str, port: int, extra: list[str]
) -> subprocess.Popen:  # type: ignore[type-arg]
    cmd = ["llama-server", "--model", model_path, "--port", str(port),
           "--log-disable", *extra]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def _wait_ready(url: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            try:
                if (await c.get(f"{url}/health")).status_code == _HTTP_OK:
                    return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"{url} not ready after {timeout:.0f}s")

# ── Metrics ─────────────────────────────────────────────────────────────────

def _recall_at(returned: list[str], expected: set[str], k: int) -> float:
    return float(bool(set(returned[:k]) & expected))


def _mrr(returned: list[str], expected: set[str]) -> float:
    for rank, nid in enumerate(returned, 1):
        if nid in expected:
            return 1.0 / rank
    return 0.0


def _anti_recall_at(returned: list[str], forbidden: set[str], k: int) -> float:
    """1.0 when no forbidden ID appears in top-k; 0.0 when any forbidden ID does."""
    return 0.0 if set(returned[:k]) & forbidden else 1.0

# ── Eval loops ──────────────────────────────────────────────────────────────

async def _run_recall(
    recall_fn, cases: list[dict]  # type: ignore[type-arg]
) -> list[dict]:  # type: ignore[type-arg]
    rows = []
    for case in cases:
        if case.get("unanswerable"):
            continue
        t0 = time.monotonic()
        result = await recall_fn(MemoryQuery(text=case["query"], top_k=5))
        elapsed = time.monotonic() - t0
        if result[0] == "err":
            print(f"  ERR {case['query']!r}: {result[1]}", file=sys.stderr)
            continue
        rows.append({
            "query": case["query"],
            "category": case["category"],
            "expected": set(case["expected_ids"]),
            "returned": [n.id for n in result[1].nodes],
            "forbidden": set(case.get("forbidden_ids", [])),
            "elapsed": elapsed,
        })
    return rows


# ── Report ───────────────────────────────────────────────────────────────────

def _dw(s: str) -> int:
    """Terminal display width: CJK wide chars count as 2 columns."""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + ' ' * max(width - _dw(s), 0)


_EXPANSION_CACHE = Path(__file__).parent / "expansion_cache.json"


_DEFAULT_CE_MODEL = Path(__file__).parent / "reranker_model"


@dataclass(frozen=True, slots=True)
class _EvalFlags:
    use_hyde: bool
    use_iterative: bool
    use_rerank: bool
    use_rewrite: bool
    use_decompose: bool
    use_gap_fill: bool
    category: str | None = None
    cross_encoder_path: str | None = None
    hybrid: bool = False
    use_bm25: bool = True
    use_graph: bool = True
    use_temporal: bool = True


def _print_row(row: dict) -> tuple[float, ...]:  # type: ignore[type-arg]
    q, ret, exp = row["query"], row["returned"], row["expected"]
    forb, t = row["forbidden"], row["elapsed"]
    r1, r3, r5 = (_recall_at(ret, exp, k) for k in (1, 3, 5))
    rr = _mrr(ret, exp)
    mark = "✓" if r1 else "✗"
    print(f"  {mark} {_pad(q, 48)}  {r1:.0f} {r3:.0f} {r5:.0f}  {rr:.3f}  {t:.1f}s")
    if not r1:
        first_hit = next((r for r in ret if r in exp), "—")
        print(f"      got: {ret[:3]}  first hit: {first_hit}")
    ar5 = _anti_recall_at(ret, forb, 5) if forb else -1.0
    if forb and not ar5:
        print(f"      anti-R@5 FAIL  forbidden: {sorted(set(ret[:5]) & forb)}")
    return r1, r3, r5, rr, ar5


def _print_category_report(rows: list[dict]) -> None:  # type: ignore[type-arg]
    cats: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        r1 = _recall_at(row["returned"], row["expected"], 1)
        rr = _mrr(row["returned"], row["expected"])
        ar5 = _anti_recall_at(row["returned"], row["forbidden"], 5) if row["forbidden"] else -1.0
        cats.setdefault(row["category"], []).append((r1, rr, ar5))
    print("\n── By category " + "─" * 47)
    for cat, metrics in sorted(cats.items()):
        n = len(metrics)
        mr1 = sum(m[0] for m in metrics) / n
        mrr = sum(m[1] for m in metrics) / n
        ar_vals = [m[2] for m in metrics if m[2] >= 0.0]
        ar5_str = f"  AR@5={sum(ar_vals) / len(ar_vals):.2f}" if ar_vals else ""
        print(f"  {cat:<18}  R@1={mr1:.2f}  MRR={mrr:.3f}{ar5_str}  ({n} cases)")
    print()


def _accumulate_row(row: dict, totals: list[float]) -> int:  # type: ignore[type-arg]
    """Print one row and accumulate its metrics; returns 1 if AR@5 was tracked."""
    r1, r3, r5, rr, ar5 = _print_row(row)
    for i, v in enumerate((r1, r3, r5, rr)):
        totals[i] += v
    if ar5 >= 0.0:
        totals[4] += ar5
        return 1
    return 0


def _print_recall_report(rows: list[dict]) -> None:  # type: ignore[type-arg]
    header = f"  {'Query':<50} R@1  R@3  R@5    MRR    time"
    sep = "─" * len(header)
    print(f"\n{header}\n{sep}")
    totals = [0.0] * 5
    n_ar = sum(_accumulate_row(row, totals) for row in rows)
    n = len(rows)
    t1, t3, t5, tmrr = (totals[i] / n for i in range(4))
    ar5_str = f"  AR@5={totals[4] / n_ar:.2f}" if n_ar else ""
    avg_time = sum(r["elapsed"] for r in rows) / n
    mean = f"  {'MEAN':<50}  {t1:.2f}  {t3:.2f}  {t5:.2f}  {tmrr:.3f}{ar5_str}  {avg_time:.1f}s"
    print(f"{sep}\n{mean}")
    _print_category_report(rows)

# ── Entry point ─────────────────────────────────────────────────────────────

def _print_categories(cases_path: str) -> None:
    cases = _load_cases(cases_path)
    counts: dict[str, int] = {}
    for c in cases:
        cat = c.get("category", "")
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
    print("Available categories:")
    for cat in sorted(counts, key=counts.__getitem__, reverse=True):
        print(f"  {cat:<20} {counts[cat]} cases")


def _add_hybrid_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hybrid", action="store_true",
                   help="Use hybrid multi-channel recall")
    p.add_argument("--no-bm25", action="store_true",
                   help="Disable BM25 channel in hybrid mode")
    p.add_argument("--no-graph", action="store_true",
                   help="Disable graph-edge channel in hybrid mode")
    p.add_argument("--no-temporal", action="store_true",
                   help="Disable temporal re-scoring in hybrid mode")


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-hyde", action="store_true",
                   help="Disable HyDE query-time expansion")
    p.add_argument("--no-iterative", action="store_true",
                   help="Disable multi-hop iterative retrieval")
    p.add_argument("--no-rerank", action="store_true",
                   help="Disable LLM reranker")
    p.add_argument("--rewrite", action="store_true",
                   help="Enable query rewriting")
    p.add_argument("--decompose", action="store_true",
                   help="Enable query decomposition")
    p.add_argument("--gap-fill", action="store_true",
                   help="Enable evidence-gap detection and filling")
    p.add_argument("--cross-encoder", metavar="PATH", nargs="?",
                   const=str(_DEFAULT_CE_MODEL),
                   help="Use cross-encoder reranker instead of LLM")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recall quality evaluation")
    p.add_argument("--embed-model", metavar="PATH",
                   help="Embedding GGUF (auto-downloaded if omitted)")
    p.add_argument("--predict-model", metavar="PATH",
                   help="Predict GGUF (auto-downloaded if omitted)")
    p.add_argument("--cases", metavar="PATH", default=str(_DEFAULT_CASES),
                   help="JSONL file with eval cases")
    p.add_argument("--rebuild-expansion", action="store_true",
                   help="Regenerate expansion cache")
    p.add_argument("--tune", action="store_true",
                   help="Grid search _W_HYDE")
    p.add_argument("--category", metavar="CAT",
                   help="Run only cases matching this category")
    p.add_argument("--list-categories", action="store_true",
                   help="Print available categories, then exit")
    _add_pipeline_args(p)
    _add_hybrid_args(p)
    return p.parse_args()


def _compute_decayed_ids(graph: MemoryGraph) -> frozenset[str]:
    consolidation = consolidate(graph, time.time(), QueryDistribution())
    if consolidation[0] != 'ok':
        return frozenset()
    return frozenset(
        nid
        for action in consolidation[1].actions
        if action.action in ('decay', 'supersede', 'remove')
        for nid in action.node_ids
    )


def _load_expansion_cache() -> dict[str, list[str]] | None:
    if _EXPANSION_CACHE.exists():
        return json.loads(_EXPANSION_CACHE.read_text())  # type: ignore[no-any-return]
    return None


def _save_expansion_cache(cache: dict[str, list[str]]) -> None:
    _EXPANSION_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


async def _expand_node(
    nid: str, text: str, expand_fn: object, cache: dict[str, list[str]]
) -> list[str]:
    if nid in cache:
        return cache[nid]
    queries = await expand_fn(text)
    cache[nid] = queries
    return queries


async def _store_corpus(conn: object, embed_url: str, predict_url: str) -> None:
    """Embed all corpus nodes + store expansion embeddings into DB."""
    doc_embed, query_embed = make_doc_embed_fn(embed_url), make_query_embed_fn(embed_url)
    expand_fn = make_llama_expand_fn(predict_url)
    node_map = {n.id: n for n in _build_memory_graph().nodes}
    store_fn = make_store_fn(conn, 'eval')  # type: ignore[arg-type]
    store_expansion = make_store_expansion_fn(conn, 'eval')  # type: ignore[arg-type]
    exp_cache: dict[str, list[str]] = _load_expansion_cache() or {}
    print(f"Embedding + expanding {len(_CORPUS)} corpus nodes …")
    for nid, text in _CORPUS:
        n = node_map[nid]
        emb = await doc_embed(text)
        await store_fn(MemoryNode(
            id=n.id, kind=n.kind, content=n.content,
            event_time=n.event_time, record_time=n.record_time,
            last_accessed=n.last_accessed, permanence=n.permanence, embedding=emb,
        ))
        queries = await _expand_node(nid, text, expand_fn, exp_cache)
        await store_expansion(nid, list(await asyncio.gather(*[query_embed(q) for q in queries])))
    if not _EXPANSION_CACHE.exists():
        _save_expansion_cache(exp_cache)


def _store_edges(conn: object) -> None:
    """Insert corpus edges into the edges table for graph search."""
    now = time.time()
    for src, tgt in _CORPUS_EDGES:
        conn.execute(  # type: ignore[union-attr]
            "INSERT OR IGNORE INTO edges "
            "(source_id, target_id, user_id, kind, weight, created_at) "
            "VALUES (?, ?, 'eval', 'supersedes', 1.0, ?)",
            (src, tgt, now),
        )
    conn.commit()  # type: ignore[union-attr]


def _make_all_fns(conn: object, embed_url: str) -> tuple[object, ...]:
    """Create all adapter fns from a populated DB."""
    return (
        make_query_embed_fn(embed_url),
        make_doc_embed_fn(embed_url),
        make_search_fn(conn, 'eval'),  # type: ignore[arg-type]
        make_primary_search_fn(conn, 'eval'),  # type: ignore[arg-type]
        make_hydrate_fn(conn, 'eval'),  # type: ignore[arg-type]
        make_update_access_fn(conn, 'eval'),  # type: ignore[arg-type]
        _compute_decayed_ids(_build_memory_graph()),
        make_bm25_search_fn(conn, 'eval'),  # type: ignore[arg-type]
        make_graph_search_fn(conn, 'eval'),  # type: ignore[arg-type]
        make_decay_map_fn(conn, 'eval'),  # type: ignore[arg-type]
    )


async def _setup_db(embed_url: str, predict_url: str) -> tuple[object, ...]:
    """Store corpus into in-memory sqlite-vec DB; return factories + decayed_ids."""
    conn = open_db(':memory:')
    await _store_corpus(conn, embed_url, predict_url)
    _store_edges(conn)
    return _make_all_fns(conn, embed_url)


@dataclass(frozen=True, slots=True)
class _BaseCtx:
    deps: RecallDeps
    doc_embed: object
    predict_url: str
    update_access: object


def _make_base(ctx: _BaseCtx, use_hyde: bool):  # type: ignore[return]
    if use_hyde:
        hyde_fn = make_llama_hyde_fn(ctx.predict_url)
        return make_hyde_recall(ctx.deps, ctx.doc_embed, hyde_fn, ctx.update_access)
    return make_recall(
        ctx.deps.search, ctx.deps.query_embed, ctx.deps.hydrate, ctx.update_access
    )


def _apply_postprocessing(
    recall: object, db_tuple: tuple[object, ...],
    predict_url: str, flags: _EvalFlags,
):  # type: ignore[return]
    """Apply decompose, gap-fill, rerank, rewrite wrappers."""
    (query_embed, _doc, search, _ps, hydrate,
     _ua, decayed_ids, _bm25, _graph, _decay) = db_tuple
    if flags.use_decompose:
        recall = make_decomposed_recall(
            recall, make_llama_decompose_fn(predict_url), hydrate,
        )
    if flags.use_gap_fill:
        recall = make_gap_filled_recall(
            recall, make_llama_gap_check_fn(predict_url),
            query_embed, search, hydrate,
        )
    if flags.use_rerank:
        rerank_fn = (
            make_cross_encoder_rerank_fn(flags.cross_encoder_path)
            if flags.cross_encoder_path
            else make_llama_rerank_fn(predict_url)
        )
        recall = make_reranked_recall(recall, rerank_fn, hydrate, decayed_ids)
    if flags.use_rewrite:
        recall = make_rewritten_recall(recall, make_llama_rewrite_fn(predict_url))
    return recall


def _build_from_db(
    db_tuple: tuple[object, ...], predict_url: str,
    flags: _EvalFlags, hyde_weight: float = 0.15,
):  # type: ignore[return]
    """Build recall stack from pre-indexed DB — no re-indexing."""
    if flags.hybrid:
        return _build_hybrid(db_tuple, predict_url, flags)
    (query_embed, doc_embed, search, primary_search, hydrate,
     update_access, *_rest) = db_tuple
    deps = RecallDeps(  # type: ignore[arg-type]
        search=search, query_embed=query_embed, hydrate=hydrate,
        primary_search=primary_search, hyde_weight=hyde_weight,
    )
    ctx = _BaseCtx(
        deps=deps, doc_embed=doc_embed,
        predict_url=predict_url, update_access=update_access,
    )
    recall = _make_base(ctx, flags.use_hyde)
    if flags.use_iterative:
        recall = make_iterative_recall(recall, query_embed, search, hydrate)
    return _apply_postprocessing(recall, db_tuple, predict_url, flags)


def _build_hybrid(
    db_tuple: tuple[object, ...], predict_url: str,
    flags: _EvalFlags,
):  # type: ignore[return]
    """Build hybrid multi-channel recall pipeline."""
    (query_embed, _doc, search, _ps, hydrate,
     _ua, _dec, bm25_search, graph_search, decay_map_fn) = db_tuple
    hybrid_deps = HybridDeps(  # type: ignore[arg-type]
        dense_search=search,
        bm25_search=bm25_search,
        graph_search=graph_search,
        embed=query_embed,
        hydrate=hydrate,
        decay_map_fn=decay_map_fn,
    )
    hybrid_cfg = HybridConfig(
        use_bm25=flags.use_bm25,
        use_graph=flags.use_graph,
        use_temporal=flags.use_temporal,
        temporal_decay=0.0001,
    )
    recall = make_hybrid_recall(hybrid_deps, now_fn=time.time, cfg=hybrid_cfg)
    return _apply_postprocessing(recall, db_tuple, predict_url, flags)


async def _build_recall(
    embed_url: str, predict_url: str, flags: _EvalFlags, hyde_weight: float = 0.15,
):  # type: ignore[return]
    db_tuple = await _setup_db(embed_url, predict_url)
    return _build_from_db(db_tuple, predict_url, flags, hyde_weight)


_TUNE_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def _mean_r1(rows: list[dict]) -> float:  # type: ignore[type-arg]
    if not rows:
        return 0.0
    return sum(_recall_at(r["returned"], r["expected"], 1) for r in rows) / len(rows)


def _load_scored_cases(cases_path: str, category: str | None) -> list[dict]:  # type: ignore[type-arg]
    cases = _load_cases(cases_path)
    if category:
        cases = [c for c in cases if c.get("category") == category]
    return [c for c in cases if not c.get("unanswerable")]


async def _run_tune(embed_url: str, predict_url: str, cases_path: str, flags: _EvalFlags) -> None:
    """Grid search _W_HYDE — index once, vary only the blend weight."""
    cases = _load_scored_cases(cases_path, flags.category)
    await asyncio.gather(_wait_ready(embed_url), _wait_ready(predict_url))
    db_tuple = await _setup_db(embed_url, predict_url)
    print(f"Tuning _W_HYDE over {_TUNE_GRID} ({len(cases)} cases)\n")
    best_w, best_r1 = 0.15, 0.0
    for w in _TUNE_GRID:
        recall = _build_from_db(db_tuple, predict_url, flags, w)
        rows = await _run_recall(recall, cases)
        r1 = _mean_r1(rows)
        marker = " ← best" if r1 > best_r1 else ""
        print(f"  w={w:.2f}  R@1={r1:.3f}{marker}")
        if r1 > best_r1:
            best_w, best_r1 = w, r1
    print(f"\nBest: _W_HYDE={best_w:.2f} (R@1={best_r1:.3f})")


async def _run_eval(embed_url: str, predict_url: str, cases_path: str, flags: _EvalFlags) -> None:
    cases = _load_cases(cases_path)
    if flags.category:
        cases = [c for c in cases if c.get("category") == flags.category]
        print(f"Category filter: {flags.category!r} → {len(cases)} cases")
    non_adversarial = sum(1 for c in cases if not c.get("unanswerable"))
    ku_filter = sum(1 for c in cases if c.get("forbidden_ids"))
    print(f"Loaded {len(cases)} cases ({non_adversarial} scored, "
          f"{len(cases) - non_adversarial} adversarial, {ku_filter} ku_filter)")
    await asyncio.gather(_wait_ready(embed_url), _wait_ready(predict_url))
    recall = await _build_recall(embed_url, predict_url, flags)
    rows = await _run_recall(recall, cases)
    _print_recall_report(rows)


def _flags_from_args(args: argparse.Namespace) -> _EvalFlags:
    return _EvalFlags(
        use_hyde=not args.no_hyde,
        use_iterative=not args.no_iterative,
        use_rerank=not args.no_rerank,
        use_rewrite=args.rewrite,
        use_decompose=args.decompose,
        use_gap_fill=getattr(args, 'gap_fill', False),
        category=args.category,
        cross_encoder_path=getattr(args, 'cross_encoder', None),
        hybrid=getattr(args, 'hybrid', False),
        use_bm25=not getattr(args, 'no_bm25', False),
        use_graph=not getattr(args, 'no_graph', False),
        use_temporal=not getattr(args, 'no_temporal', False),
    )


def _maybe_clear_cache(args: argparse.Namespace) -> None:
    if args.rebuild_expansion and _EXPANSION_CACHE.exists():
        _EXPANSION_CACHE.unlink()
        print("Expansion cache cleared — will regenerate")


async def _dispatch(args: argparse.Namespace, embed_url: str, predict_url: str) -> None:
    flags = _flags_from_args(args)
    if args.tune:
        await _run_tune(embed_url, predict_url, args.cases, flags)
    else:
        await _run_eval(embed_url, predict_url, args.cases, flags)


async def _run_with_servers(args: argparse.Namespace) -> None:
    _maybe_clear_cache(args)
    embed_path = args.embed_model or hf_hub_download(_EMBED_REPO, _EMBED_FILE)
    predict_path = args.predict_model or hf_hub_download(_PREDICT_REPO, _PREDICT_FILE)
    embed_url, predict_url = f"http://localhost:{_EMBED_PORT}", f"http://localhost:{_PREDICT_PORT}"
    print("Starting embed + predict servers …")
    embed_proc = _start_server(embed_path, _EMBED_PORT, ["--embedding", "--pooling", "last"])
    predict_proc = _start_server(predict_path, _PREDICT_PORT, [])
    try:
        await _dispatch(args, embed_url, predict_url)
    finally:
        embed_proc.terminate()
        predict_proc.terminate()


async def main() -> None:
    args = _parse_args()
    if args.list_categories:
        _print_categories(args.cases)
        return
    await _run_with_servers(args)


if __name__ == "__main__":
    asyncio.run(main())
