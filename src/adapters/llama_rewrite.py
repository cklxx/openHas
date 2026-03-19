"""Query rewriting adapter — makes implicit domain knowledge explicit.

At query time, the LLM rewrites the user query to surface hidden
connections (e.g. "whey protein" → "whey protein (dairy, contains lactose)"),
bridging the vocabulary gap between query and stored memory facts.
"""

import httpx

_SYSTEM = (
    "Rewrite the following question to make implicit real-world knowledge "
    "explicit. Add specific ingredients, substances, medical implications, "
    "or scheduling consequences in parentheses where they help connect the "
    "question to relevant personal facts. Keep the original meaning and tone. "
    "Output ONLY the rewritten question, nothing else."
)


def make_llama_rewrite_fn(base_url: str):  # type: ignore[return]
    client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def rewrite(query: str) -> str:
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        })
        text: str = r.json()["choices"][0]["message"]["content"]
        return text.strip()

    return rewrite
