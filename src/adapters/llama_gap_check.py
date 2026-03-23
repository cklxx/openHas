"""Evidence-gap detector adapter — identifies uncovered constraints.

After initial recall, the LLM checks whether the retrieved facts cover
all constraints implied by the query. If gaps exist, it generates
targeted sub-queries to fill them.

Inspired by MemR³ (arxiv:2512.20237) evidence-gap tracking.
"""

import httpx

_SYSTEM = (
    "Given a question about a person and a list of facts already retrieved, "
    "determine if any important constraints in the question are NOT covered "
    "by the retrieved facts.\n\n"
    "If all constraints are covered, output: COMPLETE\n"
    "If there are gaps, output 1-2 short queries that would retrieve the "
    "missing facts. One query per line.\n\n"
    "Examples of constraints: dietary restrictions, schedule conflicts, "
    "health conditions, family obligations, communication preferences.\n"
    "Only output gap queries for constraints clearly implied by the question. "
    "Do not speculate. No explanation."
)


def make_llama_gap_check_fn(base_url: str):  # type: ignore[return]
    client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def gap_check(query: str, retrieved_facts: list[str]) -> list[str]:
        facts_block = "\n".join(f"- {f}" for f in retrieved_facts)
        prompt = f"Question: {query}\n\nRetrieved facts:\n{facts_block}"
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        })
        text: str = r.json()["choices"][0]["message"]["content"]
        if "COMPLETE" in text.upper():
            return []
        return [ln.strip() for ln in text.splitlines() if ln.strip()][:2]

    return gap_check
