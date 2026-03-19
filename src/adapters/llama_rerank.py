"""LLM reranker adapter.

Post-retrieval reasoning pass: given a query and candidate MemoryNodes,
the LLM ranks them by true relevance — handling indirect cues, vague
references, and multi-hop reasoning that vector search cannot.
"""

import httpx

from src.domain_types.memory import MemoryNode

_SYSTEM = (
    "You are a memory relevance ranker. "
    "Given a question about a person and a list of their stored memory facts, "
    "rank the facts from most to least useful for answering the question. "
    "Consider INDIRECT relevance: a fact about work-from-home schedule is relevant "
    "to questions about in-person availability; a morning run habit is relevant to "
    "questions about early-morning activities. "
    "Output ONLY the node IDs in ranked order, one per line, most relevant first. "
    "Include all IDs. No explanation."
)


def make_llama_rerank_fn(base_url: str):  # type: ignore[return]
    client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def rerank(query: str, nodes: list[MemoryNode]) -> list[str]:
        if not nodes:
            return []
        node_block = "\n".join(f"[{n.id}] {n.content[:150]}" for n in nodes)
        prompt = f"Question: {query}\n\nMemory facts:\n{node_block}"
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        })
        lines = r.json()["choices"][0]["message"]["content"].splitlines()
        valid_ids = {n.id for n in nodes}
        ranked = [ln.strip() for ln in lines if ln.strip() in valid_ids]
        # append any IDs the LLM dropped (preserve coverage)
        seen = set(ranked)
        ranked += [n.id for n in nodes if n.id not in seen]
        return ranked

    return rerank
