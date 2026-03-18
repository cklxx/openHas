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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import numpy as np
from huggingface_hub import hf_hub_download
from src.adapters.llama_embed import make_doc_embed_fn, make_query_embed_fn
from src.adapters.llama_expand import make_llama_expand_fn
from src.adapters.llama_hyde import make_llama_hyde_fn
from src.adapters.llama_rerank import make_llama_rerank_fn
from src.core.consolidation import consolidate
from src.core.memory import RecallError, make_recall
from src.domain_types.memory import (
    Edge,
    MemoryGraph,
    MemoryNode,
    MemoryQuery,
    QueryDistribution,
    RecallResult,
)
from src.domain_types.ports import EmbedFn, ExpandContextFn, HydrateFn, SearchFn

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
    nodes = tuple(
        MemoryNode(
            id=nid, kind='fact', content=text,
            event_time=float(i), record_time=float(i), last_accessed=float(i),
            permanence=_CORPUS_PERMANENCE.get(nid, 'unknown'),  # type: ignore[arg-type]
        )
        for i, (nid, text) in enumerate(_CORPUS, start=1)
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

# ── Numpy cosine search ─────────────────────────────────────────────────────

def _build_normed(embs: list[tuple[float, ...]]) -> "np.ndarray":  # type: ignore[type-arg]
    m = np.array(embs, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)  # type: ignore[return-value]


_DocIndex = tuple[list[str], "np.ndarray"]  # type: ignore[type-arg]

# Blend weights: doc (query-emb ↔ doc-normed) > exp (context expansions) > hyde (hypothetical)
# w_doc + w_exp must sum to 1.0 for the base search; _W_HYDE is applied as a post-blend.
_W_DOC, _W_EXP = 0.70, 0.30
_W_HYDE = 0.15   # HyDE contribution: final = base * (1 - _W_HYDE) + hyde * _W_HYDE
_DECAY_FACTOR = 0.1


def _make_search_fn(
    doc_index: _DocIndex, exp_index: _DocIndex, decayed_ids: frozenset[str]
) -> SearchFn:
    doc_ids, doc_normed = doc_index
    exp_ids, exp_normed = exp_index

    async def search(embedding: tuple[float, ...], top_k: int) -> list[tuple[str, float]]:
        q = np.array(embedding, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-8
        doc_scores = {doc_ids[i]: float(s) for i, s in enumerate(doc_normed @ q)}
        exp_raw = exp_normed @ q
        exp_best: dict[str, float] = {}
        for i, nid in enumerate(exp_ids):
            s = float(exp_raw[int(i)])
            if s > exp_best.get(nid, -1.0):
                exp_best[nid] = s
        raw = {
            nid: doc_scores.get(nid, 0.0) * _W_DOC + exp_best.get(nid, 0.0) * _W_EXP
            for nid in set(doc_scores) | set(exp_best)
        }
        blended = {
            nid: (score * _DECAY_FACTOR if nid in decayed_ids else score)
            for nid, score in raw.items()
        }
        return sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return search


# ── HyDE recall ──────────────────────────────────────────────────────────────
#
# HyDE (Hypothetical Document Embedding):
#   query text ──▶ LLM ──▶ hypothetical memory snippets
#                            ──▶ doc-side embed ──▶ cosine sim vs doc_normed
#                                                    ──▶ hyde_scores
#   final = base_score * (1 - W_HYDE) + hyde_score * W_HYDE
#
async def _compute_hyde_scores(
    query_text: str,
    hyde_fn: ExpandContextFn,
    doc_embed: EmbedFn,
    doc_ids: list[str],
    doc_normed: "np.ndarray",  # type: ignore[type-arg]
    decayed_ids: frozenset[str],
) -> dict[str, float]:
    """Compute per-node HyDE scores for a query. Returns {} on any failure."""
    try:
        snippets = await hyde_fn(query_text)
    except Exception:
        return {}
    result: dict[str, float] = {}
    for snippet in snippets:
        try:
            s_emb = await doc_embed(snippet)
            q_vec = np.array(s_emb, dtype=np.float32)
            q_vec /= np.linalg.norm(q_vec) + 1e-8
            scores = doc_normed @ q_vec
            for i, nid in enumerate(doc_ids):
                score = float(scores[i])
                if nid in decayed_ids:
                    score *= _DECAY_FACTOR
                if score > result.get(nid, -1.0):
                    result[nid] = score
        except Exception:
            continue
    return result


def _make_hyde_recall(
    base_search: SearchFn,
    query_embed: EmbedFn,
    doc_embed: EmbedFn,
    hydrate: HydrateFn,
    hyde_fn: ExpandContextFn,
    doc_ids: list[str],
    doc_normed: "np.ndarray",  # type: ignore[type-arg]
    decayed_ids: frozenset[str],
):  # type: ignore[return]
    """Wrap base search with HyDE: blend base + hypothetical-snippet scores."""
    async def recall(query: MemoryQuery):  # type: ignore[return]
        if not query.text.strip():
            return ('err', RecallError(code='EMPTY_QUERY'))
        try:
            q_emb = await query_embed(query.text)
            base_hits = await base_search(q_emb, query.top_k)
        except Exception as e:
            return ('err', RecallError(code='SEARCH_FAILED', detail=str(e)))
        base_map: dict[str, float] = dict(base_hits)
        hyde_map = await _compute_hyde_scores(
            query.text, hyde_fn, doc_embed, doc_ids, doc_normed, decayed_ids
        )
        all_nids = set(base_map) | set(hyde_map)
        blended = {
            nid: base_map.get(nid, 0.0) * (1.0 - _W_HYDE)
                 + hyde_map.get(nid, 0.0) * _W_HYDE
            for nid in all_nids
        }
        top_pairs = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:query.top_k]
        hydrated = await hydrate(tuple(nid for nid, _ in top_pairs))
        pairs = [(hydrated[nid], blended[nid]) for nid, _ in top_pairs if nid in hydrated]
        return ('ok', RecallResult(
            nodes=tuple(n for n, _ in pairs),
            scores=tuple(s for _, s in pairs),
        ))
    return recall


# ── Multi-hop iterative recall ────────────────────────────────────────────────
#
# Round 1: search with original query embedding
# Round 2: re-embed (query_text + top-3 node content), search again
# Merge:   max score per node across both rounds
#
def _make_iterative_recall(base_recall, query_embed: EmbedFn, search: SearchFn, hydrate: HydrateFn):  # type: ignore[return]
    """Wrap a recall fn with one additional search round using context from round-1 hits."""
    async def iterative_recall(query: MemoryQuery):  # type: ignore[return]
        r1 = await base_recall(query)
        if r1[0] == 'err' or not r1[1].nodes:
            return r1
        ctx = ' '.join(n.content for n in r1[1].nodes[:3])
        try:
            r2_emb = await query_embed(f"{query.text} {ctx}")
            r2_hits = await search(r2_emb, query.top_k)
        except Exception:
            return r1
        r1_map: dict[str, float] = dict(zip((n.id for n in r1[1].nodes), r1[1].scores))
        r2_map: dict[str, float] = dict(r2_hits)
        merged = {
            nid: max(r1_map.get(nid, 0.0), r2_map.get(nid, 0.0))
            for nid in set(r1_map) | set(r2_map)
        }
        top_pairs = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:query.top_k]
        hydrated = await hydrate(tuple(nid for nid, _ in top_pairs))
        pairs = [(hydrated[nid], merged[nid]) for nid, _ in top_pairs if nid in hydrated]
        return ('ok', RecallResult(
            nodes=tuple(n for n, _ in pairs),
            scores=tuple(s for _, s in pairs),
        ))
    return iterative_recall


# ── LLM reranker ─────────────────────────────────────────────────────────────
#
# Fetches top-k * _RERANK_FETCH_FACTOR candidates, then asks the LLM to rank
# them by true relevance — handles cue_trigger, referential, multi_hop.
#
_RERANK_FETCH_FACTOR = 3  # fetch 15 when top_k=5


def _make_reranked_recall(base_recall, rerank_fn, hydrate: HydrateFn):  # type: ignore[return]
    """Wrap a recall fn with an LLM reranking pass over a wider candidate pool."""
    async def reranked_recall(query: MemoryQuery):  # type: ignore[return]
        wide_query = MemoryQuery(text=query.text, top_k=query.top_k * _RERANK_FETCH_FACTOR)
        r = await base_recall(wide_query)
        if r[0] == 'err' or not r[1].nodes:
            return r
        try:
            ranked_ids = await rerank_fn(query.text, list(r[1].nodes))
        except Exception:
            return r
        hydrated = await hydrate(tuple(ranked_ids[:query.top_k]))
        top_ids = [nid for nid in ranked_ids[:query.top_k] if nid in hydrated]
        score_map = dict(zip((n.id for n in r[1].nodes), r[1].scores))
        return ('ok', RecallResult(
            nodes=tuple(hydrated[nid] for nid in top_ids),
            scores=tuple(score_map.get(nid, 0.0) for nid in top_ids),
        ))
    return reranked_recall


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
        result = await recall_fn(MemoryQuery(text=case["query"], top_k=5))
        if result[0] == "err":
            print(f"  ERR {case['query']!r}: {result[1]}", file=sys.stderr)
            continue
        rows.append({
            "query": case["query"],
            "category": case["category"],
            "expected": set(case["expected_ids"]),
            "returned": [n.id for n in result[1].nodes],
            "forbidden": set(case.get("forbidden_ids", [])),
        })
    return rows


# ── Report ───────────────────────────────────────────────────────────────────

def _dw(s: str) -> int:
    """Terminal display width: CJK wide chars count as 2 columns."""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + ' ' * max(width - _dw(s), 0)


def _print_row(
    query_text: str, expected: set[str], returned: list[str], forbidden: set[str]
) -> tuple[float, ...]:
    r1, r3, r5 = (_recall_at(returned, expected, k) for k in (1, 3, 5))
    rr = _mrr(returned, expected)
    mark = "✓" if r1 else "✗"
    print(f"  {mark} {_pad(query_text, 48)}  {r1:.0f}    {r3:.0f}    {r5:.0f}  {rr:.3f}")
    if not r1:
        first_hit = next((r for r in returned if r in expected), "—")
        print(f"      got: {returned[:3]}  first hit: {first_hit}")
    ar5 = _anti_recall_at(returned, forbidden, 5) if forbidden else -1.0
    if forbidden and not ar5:
        print(f"      anti-R@5 FAIL  forbidden appeared: {sorted(set(returned[:5]) & forbidden)}")
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
    r1, r3, r5, rr, ar5 = _print_row(
        row["query"], row["expected"], row["returned"], row["forbidden"]
    )
    for i, v in enumerate((r1, r3, r5, rr)):
        totals[i] += v
    if ar5 >= 0.0:
        totals[4] += ar5
        return 1
    return 0


def _print_recall_report(rows: list[dict]) -> None:  # type: ignore[type-arg]
    header = f"  {'Query':<50} R@1  R@3  R@5    MRR"
    sep = "─" * len(header)
    print(f"\n{header}\n{sep}")
    totals = [0.0] * 5
    n_ar = sum(_accumulate_row(row, totals) for row in rows)
    n = len(rows)
    t1, t3, t5, tmrr = (totals[i] / n for i in range(4))
    ar5_str = f"  AR@5={totals[4] / n_ar:.2f}" if n_ar else ""
    print(f"{sep}\n  {'MEAN':<50}  {t1:.2f}  {t3:.2f}  {t5:.2f}  {tmrr:.3f}{ar5_str}")
    _print_category_report(rows)

# ── Entry point ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recall quality evaluation")
    p.add_argument("--embed-model", metavar="PATH",
                   help="Embedding GGUF (auto-downloaded if omitted)")
    p.add_argument("--predict-model", metavar="PATH",
                   help="Predict GGUF (auto-downloaded if omitted)")
    p.add_argument("--cases", metavar="PATH", default=str(_DEFAULT_CASES),
                   help="JSONL file with eval cases")
    p.add_argument("--no-hyde", action="store_true",
                   help="Disable HyDE query-time expansion (faster, lower recall)")
    p.add_argument("--no-iterative", action="store_true",
                   help="Disable multi-hop iterative retrieval (faster, lower recall)")
    p.add_argument("--no-rerank", action="store_true",
                   help="Disable LLM reranker (faster, lower recall on cue/referential/multi-hop)")
    return p.parse_args()


async def _embed_corpus(
    doc_embed, query_embed, expand  # type: ignore[type-arg]
) -> tuple[list[str], list[tuple[float, ...]], list[str], list[tuple[float, ...]]]:
    doc_ids: list[str] = []
    doc_embs: list[tuple[float, ...]] = []
    exp_ids: list[str] = []
    exp_embs: list[tuple[float, ...]] = []
    print(f"Expanding contexts for {len(_CORPUS)} memories …")
    for nid, text in _CORPUS:
        doc_ids.append(nid)
        doc_embs.append(await doc_embed(text))
        for ctx in await expand(text):
            exp_ids.append(nid)
            exp_embs.append(await query_embed(ctx))
    return doc_ids, doc_embs, exp_ids, exp_embs


async def _build_index(  # type: ignore[return]
    embed_url: str, predict_url: str, use_hyde: bool, use_iterative: bool, use_rerank: bool
):
    doc_embed = make_doc_embed_fn(embed_url)
    query_embed = make_query_embed_fn(embed_url)
    expand = make_llama_expand_fn(predict_url)
    doc_ids, doc_embs, exp_ids, exp_embs = await _embed_corpus(doc_embed, query_embed, expand)
    graph = _build_memory_graph()
    consolidation = consolidate(graph, time.time(), QueryDistribution())
    decayed_ids: frozenset[str] = frozenset()
    if consolidation[0] == 'ok':
        decayed_ids = frozenset(
            nid
            for action in consolidation[1].actions
            if action.action in ('decay', 'supersede', 'remove')
            for nid in action.node_ids
        )
    doc_normed = _build_normed(doc_embs)
    search = _make_search_fn(
        (doc_ids, doc_normed),
        (exp_ids, _build_normed(exp_embs)),
        decayed_ids,
    )
    node_lookup = {n.id: n for n in graph.nodes}

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: node_lookup[i] for i in ids if i in node_lookup}

    if use_hyde:
        print("HyDE enabled — generating hypothetical snippets at query time")
        recall = _make_hyde_recall(
            search, query_embed, doc_embed, hydrate,
            make_llama_hyde_fn(predict_url),
            doc_ids, doc_normed, decayed_ids,
        )
    else:
        recall = make_recall(search, query_embed, hydrate)  # type: ignore[arg-type]

    if use_iterative:
        print("Iterative retrieval enabled — two-round multi-hop search")
        recall = _make_iterative_recall(recall, query_embed, search, hydrate)

    if use_rerank:
        print("LLM reranker enabled — reasoning pass over wider candidate pool")
        recall = _make_reranked_recall(recall, make_llama_rerank_fn(predict_url), hydrate)

    return recall


async def _run_eval(
    embed_url: str, predict_url: str, cases_path: str,
    use_hyde: bool, use_iterative: bool, use_rerank: bool
) -> None:
    cases = _load_cases(cases_path)
    non_adversarial = sum(1 for c in cases if not c.get("unanswerable"))
    ku_filter = sum(1 for c in cases if c.get("forbidden_ids"))
    print(f"Loaded {len(cases)} cases ({non_adversarial} scored, "
          f"{len(cases) - non_adversarial} adversarial, {ku_filter} ku_filter)")
    await asyncio.gather(_wait_ready(embed_url), _wait_ready(predict_url))
    recall = await _build_index(embed_url, predict_url, use_hyde, use_iterative, use_rerank)
    rows = await _run_recall(recall, cases)
    _print_recall_report(rows)


async def main() -> None:
    args = _parse_args()
    embed_path = args.embed_model or hf_hub_download(_EMBED_REPO, _EMBED_FILE)
    predict_path = args.predict_model or hf_hub_download(_PREDICT_REPO, _PREDICT_FILE)
    embed_url = f"http://localhost:{_EMBED_PORT}"
    predict_url = f"http://localhost:{_PREDICT_PORT}"
    print("Starting embed + predict servers …")
    embed_proc = _start_server(embed_path, _EMBED_PORT, ["--embedding", "--pooling", "last"])
    predict_proc = _start_server(predict_path, _PREDICT_PORT, [])
    try:
        await _run_eval(embed_url, predict_url, args.cases,
                        use_hyde=not args.no_hyde, use_iterative=not args.no_iterative,
                        use_rerank=not args.no_rerank)
    finally:
        embed_proc.terminate()
        predict_proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
