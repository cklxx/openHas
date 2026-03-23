"""Query rewriting adapter — makes implicit domain knowledge explicit.

At query time, the LLM rewrites the user query to surface hidden
connections (e.g. "whey protein" → "whey protein (dairy, contains lactose)"),
bridging the vocabulary gap between query and stored memory facts.
"""

import httpx

_SYSTEM = (
    "You help retrieve personal memories. Rewrite the question by adding "
    "implicit real-world knowledge in parentheses.\n\n"
    "Rules:\n"
    "- Food: list key ingredients and allergens. "
    "E.g. 'tiramisu (mascarpone=dairy, Marsala wine=alcohol, coffee)'\n"
    "- Health: spell out medical implications. "
    "E.g. 'grapefruit (interacts with many medications)'\n"
    "- Schedule: expand time references. "
    "E.g. 'this evening (after work, ~6pm-11pm)'\n"
    "- Travel: note duration, logistics, absence from home.\n"
    "- Keep the original meaning. Do NOT answer the question.\n"
    "- Output ONLY the rewritten question."
)


def make_llama_rewrite_fn(base_url: str):  # type: ignore[return]
    client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def rewrite(query: str) -> str:
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        })
        text: str = r.json()["choices"][0]["message"]["content"]
        return text.strip()

    return rewrite
