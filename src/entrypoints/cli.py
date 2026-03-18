"""Memory system CLI — wires real adapters into core."""

import argparse
import asyncio
import time

from src.adapters.llama_embed import make_doc_embed_fn, make_query_embed_fn
from src.adapters.sqlite_vec_store import (
    make_hydrate_fn,
    make_search_fn,
    make_store_fn,
    open_db,
)
from src.core.memory import make_recall, make_store_memory
from src.domain_types.memory import MemoryQuery

_DEFAULT_DB = "memory.db"
_DEFAULT_EMBED_URL = "http://localhost:18080"
_DEFAULT_USER = "default"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='openhas', description='Memory system CLI')
    parser.add_argument('--db', default=_DEFAULT_DB, help='SQLite database path')
    parser.add_argument('--embed-url', default=_DEFAULT_EMBED_URL, help='llama-server embed URL')
    parser.add_argument('--user', default=_DEFAULT_USER, help='User ID for memory isolation')
    sub = parser.add_subparsers(dest='command')

    store_p = sub.add_parser('store', help='Store a memory node')
    store_p.add_argument('content', help='Memory content text')
    store_p.add_argument('--id', required=True, help='Node ID')
    store_p.add_argument('--kind', default='fact', help='Node kind')
    store_p.add_argument('--labels', nargs='*', default=[], help='Labels')

    recall_p = sub.add_parser('recall', help='Recall memories')
    recall_p.add_argument('query', help='Query text')
    recall_p.add_argument('--top-k', type=int, default=10, help='Number of results')

    return parser


async def _handle_store(args: argparse.Namespace) -> None:
    conn = open_db(args.db)
    embed = make_doc_embed_fn(args.embed_url)
    store_node = make_store_fn(conn, args.user)
    store = make_store_memory(embed, time.time, store_node)
    result = await store(
        id=args.id,
        kind=args.kind,
        content=args.content,
        labels=tuple(args.labels),
    )
    if result[0] == 'ok':
        print(f"Stored: {result[1].id} ({result[1].kind})")
    else:
        print(f"Error: {result[1].code} — {result[1].detail}")


async def _handle_recall(args: argparse.Namespace) -> None:
    conn = open_db(args.db)
    query_embed = make_query_embed_fn(args.embed_url)
    search = make_search_fn(conn, args.user)
    hydrate = make_hydrate_fn(conn, args.user)
    recall = make_recall(search, query_embed, hydrate)
    query = MemoryQuery(text=args.query, top_k=args.top_k)
    result = await recall(query)
    if result[0] == 'ok':
        for node in result[1].nodes:
            print(f"[{node.kind}] {node.id}: {node.content}")
        print(f"\n{len(result[1].nodes)} result(s)")
    else:
        print(f"Error: {result[1].code}")


_HANDLERS = {
    'store': _handle_store,
    'recall': _handle_recall,
}


async def _run(args: argparse.Namespace) -> None:
    handler = _HANDLERS.get(args.command)
    if handler is None:
        print("Use 'store' or 'recall'. Run with --help for usage.")
        return
    await handler(args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == '__main__':
    main()
