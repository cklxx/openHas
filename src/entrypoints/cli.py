"""Memory system CLI — wires real adapters into core."""

import argparse
import asyncio
import sqlite3
import time
from typing import Any

from src.adapters.llama_embed import make_doc_embed_fn, make_query_embed_fn
from src.adapters.llama_expand import make_llama_expand_fn
from src.adapters.llama_hyde import make_llama_hyde_fn
from src.adapters.llama_rerank import make_llama_rerank_fn
from src.adapters.sqlite_vec_store import (
    load_graph,
    make_hydrate_fn,
    make_list_pending_fn,
    make_primary_search_fn,
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
    make_store_memory,
)
from src.domain_types.memory import MemoryQuery, QueryDistribution

_DEFAULT_DB = "memory.db"
_DEFAULT_EMBED_URL = "http://localhost:18080"
_DEFAULT_PREDICT_URL = "http://localhost:18081"
_DEFAULT_USER = "default"


# ── Parser helpers ────────────────────────────────────────────────────────────

def _add_store_cmd(sub: Any) -> None:
    p: argparse.ArgumentParser = sub.add_parser('store', help='Store a memory node')
    p.add_argument('content', help='Memory content text')
    p.add_argument('--id', required=True, help='Node ID')
    p.add_argument('--kind', default='fact', help='Node kind')
    p.add_argument('--labels', nargs='*', default=[], help='Labels')


def _add_recall_cmd(sub: Any) -> None:
    p: argparse.ArgumentParser = sub.add_parser('recall', help='Recall memories')
    p.add_argument('query', help='Query text')
    p.add_argument('--top-k', type=int, default=10, help='Number of results')
    p.add_argument('--hyde', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--iterative', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--rerank', action=argparse.BooleanOptionalAction, default=True)


def _add_expand_cmd(sub: Any) -> None:
    p: argparse.ArgumentParser = sub.add_parser('expand', help='Expand pending nodes')
    p.add_argument('--concurrency', type=int, default=4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='openhas', description='Memory system CLI')
    parser.add_argument('--db', default=_DEFAULT_DB)
    parser.add_argument('--embed-url', default=_DEFAULT_EMBED_URL)
    parser.add_argument('--predict-url', default=_DEFAULT_PREDICT_URL)
    parser.add_argument('--user', default=_DEFAULT_USER)
    sub = parser.add_subparsers(dest='command')
    _add_store_cmd(sub)
    _add_recall_cmd(sub)
    _add_expand_cmd(sub)
    return parser


# ── Recall stack builder ──────────────────────────────────────────────────────

def _compute_decayed(conn: sqlite3.Connection, user_id: str) -> frozenset[str]:
    graph = load_graph(conn, user_id)
    result = consolidate(graph, time.time(), QueryDistribution())
    if result[0] == 'err':
        return frozenset()
    return frozenset(
        nid
        for action in result[1].actions
        if action.action in ('decay', 'supersede', 'remove')
        for nid in action.node_ids
    )


def _build_recall_stack(
    args: argparse.Namespace, conn: sqlite3.Connection
):  # type: ignore[return]
    deps = RecallDeps(
        search=make_search_fn(conn, args.user),
        query_embed=make_query_embed_fn(args.embed_url),
        hydrate=make_hydrate_fn(conn, args.user),
        primary_search=make_primary_search_fn(conn, args.user),
    )
    update_access = make_update_access_fn(conn, args.user)
    if args.hyde:
        recall = make_hyde_recall(
            deps, make_doc_embed_fn(args.embed_url),
            make_llama_hyde_fn(args.predict_url), update_access,
        )
    else:
        recall = make_recall(deps.search, deps.query_embed, deps.hydrate, update_access)
    if args.iterative:
        recall = make_iterative_recall(recall, deps.query_embed, deps.search, deps.hydrate)
    if args.rerank:
        decayed_ids = _compute_decayed(conn, args.user)
        recall = make_reranked_recall(
            recall, make_llama_rerank_fn(args.predict_url), deps.hydrate, decayed_ids
        )
    return recall


# ── Command handlers ──────────────────────────────────────────────────────────

def _print_recall_result(result: object) -> None:
    if result[0] == 'ok':  # type: ignore[index]
        for node in result[1].nodes:  # type: ignore[index]
            print(f"[{node.kind}] {node.id}: {node.content}")  # type: ignore[reportUnknownMemberType]
        print(f"\n{len(result[1].nodes)} result(s)")  # type: ignore[index]
    else:
        print(f"Error: {result[1].code}")  # type: ignore[index]


async def _handle_store(args: argparse.Namespace) -> None:
    conn = open_db(args.db)
    store = make_store_memory(make_doc_embed_fn(args.embed_url), time.time,
                              make_store_fn(conn, args.user))
    result = await store(id=args.id, kind=args.kind, content=args.content,
                         labels=tuple(args.labels))
    if result[0] == 'ok':
        print(f"Stored: {result[1].id} ({result[1].kind})")
    else:
        print(f"Error: {result[1].code} — {result[1].detail}")


async def _handle_recall(args: argparse.Namespace) -> None:
    conn = open_db(args.db)
    recall = _build_recall_stack(args, conn)
    result = await recall(MemoryQuery(text=args.query, top_k=args.top_k))
    _print_recall_result(result)


async def _handle_expand(args: argparse.Namespace) -> None:
    conn = open_db(args.db)
    query_embed = make_query_embed_fn(args.embed_url)
    expand_fn = make_llama_expand_fn(args.predict_url)
    store_expansion = make_store_expansion_fn(conn, args.user)
    pending = await make_list_pending_fn(conn, args.user)()
    sem = asyncio.Semaphore(args.concurrency)

    async def expand_one(node: Any) -> None:
        async with sem:
            queries = await expand_fn(node.content)
            embs = await asyncio.gather(*[query_embed(q) for q in queries])
            await store_expansion(node.id, list(embs))

    await asyncio.gather(*[expand_one(n) for n in pending])
    print(f"Expanded {len(pending)} nodes")


_HANDLERS = {
    'store': _handle_store,
    'recall': _handle_recall,
    'expand': _handle_expand,
}


async def _run(args: argparse.Namespace) -> None:
    handler = _HANDLERS.get(args.command)
    if handler is None:
        print("Use 'store', 'recall', or 'expand'. Run with --help for usage.")
        return
    await handler(args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == '__main__':
    main()
