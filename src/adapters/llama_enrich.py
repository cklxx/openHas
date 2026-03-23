"""Document enrichment adapter — adds implicit knowledge to memory facts.

At index time, the LLM expands a memory fact with its real-world
implications: food ingredients, medical interactions, scheduling
consequences, etc. The enriched text is stored alongside the original
so that even a generic cross-encoder can match indirect queries.
"""

import httpx

_SYSTEM = (
    "Given a personal memory fact, list its real-world implications "
    "that someone might search for indirectly. Be specific and concrete.\n\n"
    "Categories to cover:\n"
    "- If FOOD/DIET: list 10+ specific foods, dishes, and ingredients "
    "that are affected (e.g. for lactose intolerance: tiramisu, caesar "
    "salad, latte, whey protein, cheese pizza, butter, cream sauce, "
    "ice cream, yogurt, milk chocolate)\n"
    "- If MEDICAL: list drug interactions, activity restrictions, "
    "symptoms, and common triggers\n"
    "- If SCHEDULE: list affected time slots, conflicting activities, "
    "and availability windows\n"
    "- If RELATIONSHIP: list the person's role, needs, and how they "
    "affect the user's decisions\n"
    "- If LOCATION: list commute implications, nearby landmarks, "
    "and accessibility\n"
    "- If SKILL/ROLE: list responsibilities, expectations, and "
    "related competencies\n\n"
    "Output a single paragraph of comma-separated implications. "
    "No headers, no bullet points, no explanation. "
    "Start directly with the implications."
)


def make_llama_enrich_fn(base_url: str):  # type: ignore[return]
    client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def enrich(memory_text: str) -> str:
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": memory_text},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        })
        text: str = r.json()["choices"][0]["message"]["content"]
        return text.strip()

    return enrich
