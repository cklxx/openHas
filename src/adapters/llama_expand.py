"""Context expansion adapter — generates hypothetical retrieval queries for a memory.

At index-build time, the LLM produces N query-like strings per memory.
These are embedded on the query side (instruction prefix), so they sit close
to real user queries in embedding space and bridge the vocabulary gap.
"""

import httpx

from src.domain_types.ports import ExpandContextFn

_SYSTEM = (
    "Given a personal memory fact, output 5 queries where knowing THIS specific fact "
    "is essential to answer correctly. "
    "Include both direct questions and indirect scenario queries — real situations "
    "where this fact determines the correct answer, even if the query does not "
    "mention the fact directly (e.g. a food restriction applies to specific "
    "ingredients; a schedule determines when someone is available). "
    "Each query must be unanswerable without this fact. "
    "Avoid generic questions about the person's life. "
    "Mix Chinese and English. One query per line. No numbering, no explanation."
)


def make_llama_expand_fn(base_url: str) -> ExpandContextFn:
    client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def expand(memory_text: str) -> list[str]:
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": memory_text},
            ],
            "max_tokens": 256,
            "temperature": 0.3,
        })
        text: str = r.json()["choices"][0]["message"]["content"]
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    return expand  # type: ignore[return-value]
