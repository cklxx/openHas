# openHas

Self-hosted personal memory layer for LLMs. Any MCP-compatible client (Claude, Cursor, Zed) gains persistent context — no cloud, no API key, no data leaving your machine.

## What it does

- **Ingest** — extract memorable facts from conversation text via a local LLM
- **Recall** — hybrid retrieval: HyDE + multi-hop query expansion + vector search
- **Consolidate** — background pass resolves contradictions, propagates supersessions, prunes low-value nodes

## Architecture

```
domain_types/   pure dataclass types, zero external imports
core/           pure functions → Result[T, E], no IO
adapters/       SQLite + sqlite-vec + llama-server HTTP
entrypoints/    CLI  (MCP server: planned)
```

## Quick start

```bash
pip install -e ".[dev]"
# start llama-server on localhost:8080 (any embedding model)
python -m src.entrypoints.cli recall "what should I order for dinner?"
```

## Eval

```bash
python evals/recall_eval.py
# R@1 / R@3 / R@5 · MRR · AR@5 (anti-recall) against 155-case hard benchmark
```

## Status

Core algorithms complete. SQLite persistence and MCP server wiring in progress — see `docs/designs/p0-completion-and-mcp.md`.
