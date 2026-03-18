"""HyDE (Hypothetical Document Embedding) adapter.

At query time, the LLM generates short memory snippets that would answer the query.
These are embedded doc-side and blended with the standard KNN score.

Inverse of llama_expand.py:
  llama_expand: memory → hypothetical queries  (index-time, doc-side embed)
  llama_hyde:   query  → hypothetical memories (query-time, doc-side embed)
"""

import httpx

from src.domain_types.ports import ExpandContextFn

_SYSTEM = (
    "Given a question about a person, output 3 short factual memory snippets "
    "that would fully answer the question if they existed in the person's memory store. "
    "Each snippet: one sentence, written as a stored personal fact. "
    "No questions. No explanations. One snippet per line."
)

N_SNIPPETS = 3  # public — used in tests and eval


def make_llama_hyde_fn(base_url: str) -> ExpandContextFn:
    """Return an ExpandContextFn that generates hypothetical memory snippets for a query."""
    client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def hyde(query_text: str) -> list[str]:
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query_text},
            ],
            "max_tokens": 128,
            "temperature": 0.3,
        })
        text: str = r.json()["choices"][0]["message"]["content"]
        return [ln.strip() for ln in text.splitlines() if ln.strip()][:N_SNIPPETS]

    return hyde  # type: ignore[return-value]
