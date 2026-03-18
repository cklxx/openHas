"""Memory system CLI — entrypoint that wires adapters into core."""

import argparse
import asyncio
import time

from src.adapters.embeddings import make_embed_fn
from src.core.memory import make_recall, make_store_memory
from src.domain_types.memory import MemoryQuery


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='openhas', description='Memory system CLI')
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
    embed = make_embed_fn()
    store = make_store_memory(embed, time.time)
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
    embed = make_embed_fn()
    search_hits: list[tuple[str, float]] = []

    async def stub_search(
        embedding: tuple[float, ...], top_k: int
    ) -> list[tuple[str, float]]:
        return search_hits[:top_k]

    recall = make_recall(stub_search, embed)  # type: ignore[arg-type]
    query = MemoryQuery(text=args.query, top_k=args.top_k)
    result = await recall(query)
    if result[0] == 'ok':
        print(f"Recalled {len(result[1].nodes)} nodes")
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
