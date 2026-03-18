"""Permanence classifier — llama-server HTTP backend (OpenAI-compatible).

Returns 'permanent', 'transient', or 'unknown'. Never raises on bad LLM
output — falls back to 'unknown'. Re-raises httpx transport errors so
the ingestion gate can convert them to IngestError.
"""

import httpx

from src.domain_types.ports import ClassifyFn

_VALID = frozenset(('permanent', 'transient', 'unknown'))

_PROMPT = (
    "Classify whether this memory describes a permanent or transient fact.\n"
    "Permanent: long-lasting (diet, personality, skills, relationships).\n"
    "Transient: time-bound (schedules, deadlines, current tasks).\n"
    "Reply with exactly one word: permanent, transient, or unknown.\n"
    "Memory: {content}"
)


def make_llama_classify_fn(base_url: str, timeout: float = 30.0) -> ClassifyFn:
    """Return a ClassifyFn backed by a llama-server chat endpoint."""
    client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def classify(content: str) -> str:
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": _PROMPT.format(content=content)}],
                "max_tokens": 5,
                "temperature": 0,
            },
        )
        r.raise_for_status()
        word = r.json()["choices"][0]["message"]["content"].strip().lower()
        return word if word in _VALID else 'unknown'

    return classify  # type: ignore[return-value]
