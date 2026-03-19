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
from src.adapters.llama_embed import make_doc_embed_fn, make_query_embed_fn
from src.adapters.llama_expand import make_llama_expand_fn
from src.adapters.llama_hyde import make_llama_hyde_fn
from src.adapters.llama_rerank import make_llama_rerank_fn
from src.adapters.sqlite_vec_store import (
    make_hydrate_fn,
    make_search_fn,
    make_store_expansion_fn,
    make_store_fn,
    make_update_access_fn,
    open_db,
)
from src.core.consolidation import consolidate
from src.core.memory import (
    RecallDeps,
    make_hyde_recall,
    make_iterative_recall,
    make_recall,
    make_reranked_recall,
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


@dataclass(frozen=True, slots=True)
class _EvalFlags:
    use_hyde: bool
    use_iterative: bool
    use_rerank: bool


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


async def _store_corpus(conn: object, embed_url: str, predict_url: str) -> None:
    """Embed all corpus nodes + store expansion embeddings into DB."""
    doc_embed = make_doc_embed_fn(embed_url)
    query_embed = make_query_embed_fn(embed_url)
    expand_fn = make_llama_expand_fn(predict_url)
    store_fn = make_store_fn(conn, 'eval')  # type: ignore[arg-type]
    store_expansion = make_store_expansion_fn(conn, 'eval')  # type: ignore[arg-type]
    node_map = {n.id: n for n in _build_memory_graph().nodes}
    print(f"Embedding + expanding {len(_CORPUS)} corpus nodes …")
    for nid, text in _CORPUS:
        n = node_map[nid]
        emb = await doc_embed(text)
        await store_fn(MemoryNode(
            id=n.id, kind=n.kind, content=n.content,
            event_time=n.event_time, record_time=n.record_time,
            last_accessed=n.last_accessed, permanence=n.permanence, embedding=emb,
        ))
        exp_embs = await asyncio.gather(*[query_embed(q) for q in await expand_fn(text)])
        await store_expansion(nid, list(exp_embs))


async def _setup_db(embed_url: str, predict_url: str) -> tuple[object, ...]:
    """Store corpus into in-memory sqlite-vec DB; return factories + decayed_ids."""
    conn = open_db(':memory:')
    await _store_corpus(conn, embed_url, predict_url)
    graph = _build_memory_graph()
    decayed_ids = _compute_decayed_ids(graph)
    search = make_search_fn(conn, 'eval')
    hydrate = make_hydrate_fn(conn, 'eval')
    update_access = make_update_access_fn(conn, 'eval')
    query_embed = make_query_embed_fn(embed_url)
    doc_embed = make_doc_embed_fn(embed_url)
    return query_embed, doc_embed, search, hydrate, update_access, decayed_ids


async def _build_recall(embed_url: str, predict_url: str, flags: _EvalFlags):  # type: ignore[return]
    query_embed, doc_embed, search, hydrate, update_access, decayed_ids = (
        await _setup_db(embed_url, predict_url)
    )
    deps = RecallDeps(search=search, query_embed=query_embed, hydrate=hydrate)  # type: ignore[arg-type]
    if flags.use_hyde:
        print("HyDE enabled — generating hypothetical snippets at query time")
        recall = make_hyde_recall(
            deps, doc_embed, make_llama_hyde_fn(predict_url), update_access  # type: ignore[arg-type]
        )
    else:
        recall = make_recall(search, query_embed, hydrate, update_access)  # type: ignore[arg-type]
    if flags.use_iterative:
        print("Iterative retrieval enabled — two-round multi-hop search")
        recall = make_iterative_recall(recall, query_embed, search, hydrate)  # type: ignore[arg-type]
    if flags.use_rerank:
        print("LLM reranker enabled — reasoning pass over wider candidate pool")
        recall = make_reranked_recall(
            recall, make_llama_rerank_fn(predict_url), hydrate, decayed_ids  # type: ignore[arg-type]
        )
    return recall


async def _run_eval(embed_url: str, predict_url: str, cases_path: str, flags: _EvalFlags) -> None:
    cases = _load_cases(cases_path)
    non_adversarial = sum(1 for c in cases if not c.get("unanswerable"))
    ku_filter = sum(1 for c in cases if c.get("forbidden_ids"))
    print(f"Loaded {len(cases)} cases ({non_adversarial} scored, "
          f"{len(cases) - non_adversarial} adversarial, {ku_filter} ku_filter)")
    await asyncio.gather(_wait_ready(embed_url), _wait_ready(predict_url))
    recall = await _build_recall(embed_url, predict_url, flags)
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
    flags = _EvalFlags(
        use_hyde=not args.no_hyde,
        use_iterative=not args.no_iterative,
        use_rerank=not args.no_rerank,
    )
    try:
        await _run_eval(embed_url, predict_url, args.cases, flags)
    finally:
        embed_proc.terminate()
        predict_proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
