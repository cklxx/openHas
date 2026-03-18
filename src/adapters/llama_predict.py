"""Query prediction adapter — llama-server HTTP backend (OpenAI-compatible API)."""

import httpx

from src.domain_types.ports import PredictQueryFn

_SYSTEM = (
    "Predict the user's 3 most likely follow-up questions based on the context. "
    "Output exactly 3 questions, one per line. Nothing else."
)
_PROBS = (0.9, 0.7, 0.5)


def _parse_predictions(text: str) -> list[tuple[str, float]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [(q, p) for q, p in zip(lines, _PROBS, strict=False)]


def make_llama_predict_fn(base_url: str) -> PredictQueryFn:
    client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def predict(recent_context: str) -> list[tuple[str, float]]:
        r = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": recent_context},
            ],
            "max_tokens": 128,
            "temperature": 0.3,
        })
        text: str = r.json()["choices"][0]["message"]["content"]
        return _parse_predictions(text)

    return predict  # type: ignore[return-value]
