"""Query decomposition adapter — splits multi-constraint queries into sub-queries.

For multi-hop questions like "Can the user attend a Friday evening party and
drink wine?", the LLM decomposes into independent constraint checks:
  1. "Does the user have evening time restrictions?"
  2. "Can the user drink alcohol?"

Each sub-query retrieves independently, then results are merged — widening
the candidate pool so the reranker sees all relevant constraints.
"""

import httpx

_SYSTEM = (
    "Given a question about a person, break it into 2-3 independent "
    "sub-questions that each check ONE specific constraint or fact needed "
    "to answer the original question.\n\n"
    "Rules:\n"
    "- Each sub-question should target a different type of personal fact "
    "(schedule, diet, health, family, preferences, etc.)\n"
    "- Keep sub-questions short and specific\n"
    "- If the question is already simple (one constraint), output just "
    "that question unchanged\n"
    "- One sub-question per line. No numbering, no explanation."
)


def make_llama_decompose_fn(base_url: str):  # type: ignore[return]
    client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def decompose(query: str) -> list[str]:
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        })
        text: str = r.json()["choices"][0]["message"]["content"]
        subs = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return subs[:3] if subs else [query]

    return decompose
