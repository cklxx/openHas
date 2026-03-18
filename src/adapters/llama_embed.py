"""Embedding adapter — llama-server HTTP backend (OpenAI-compatible /v1/embeddings).

Supports asymmetric retrieval: query embeddings prepend an instruction prefix
(required for Qwen3-Embedding and similar instruction-tuned models).
"""

import httpx

from src.domain_types.ports import EmbedFn

_RETRIEVAL_TASK = "Retrieve relevant personal memories for this user query."


def make_doc_embed_fn(base_url: str) -> EmbedFn:
    """Embed document text (no prefix)."""
    client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def embed(text: str) -> tuple[float, ...]:
        r = await client.post("/v1/embeddings", json={"input": text})
        return tuple(r.json()["data"][0]["embedding"])

    return embed  # type: ignore[return-value]


def make_query_embed_fn(base_url: str, task: str = _RETRIEVAL_TASK) -> EmbedFn:
    """Embed query text with instruction prefix for asymmetric retrieval."""
    prefix = f"Instruct: {task}\nQuery: "
    client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def embed(text: str) -> tuple[float, ...]:
        r = await client.post("/v1/embeddings", json={"input": f"{prefix}{text}"})
        return tuple(r.json()["data"][0]["embedding"])

    return embed  # type: ignore[return-value]
