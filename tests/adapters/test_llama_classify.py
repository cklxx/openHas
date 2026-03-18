"""Tests for llama_classify adapter — parsing logic and error propagation."""

import httpx
import pytest
from src.adapters.llama_classify import _PROMPT, _VALID


def _make_classify(response_text: str):  # type: ignore[return]
    """Return a classify fn backed by a mock HTTP transport."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = f'{{"choices": [{{"message": {{"content": "{response_text}"}}}}]}}'
        return httpx.Response(200, text=body)

    client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )

    async def classify(content: str) -> str:
        r = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": _PROMPT.format(content=content)}],
                  "max_tokens": 5, "temperature": 0},
        )
        r.raise_for_status()
        word = r.json()["choices"][0]["message"]["content"].strip().lower()
        return word if word in _VALID else 'unknown'

    return classify


@pytest.mark.asyncio
async def test_classify_permanent() -> None:
    result = await _make_classify("permanent")("user is vegetarian")
    assert result == 'permanent'


@pytest.mark.asyncio
async def test_classify_transient() -> None:
    result = await _make_classify("transient")("meeting at 2pm today")
    assert result == 'transient'


@pytest.mark.asyncio
async def test_classify_unknown_fallback() -> None:
    """Garbage LLM output falls back to 'unknown'."""
    result = await _make_classify("I cannot determine this")("some content")
    assert result == 'unknown'


@pytest.mark.asyncio
async def test_classify_timeout_returns_classify_failed() -> None:
    """TimeoutException propagates — ingestion gate converts it to CLASSIFY_FAILED."""
    from src.core.ingestion import make_ingestion_gate
    from src.domain_types.memory import QueryDistribution

    async def timeout_classify(content: str) -> str:
        raise httpx.TimeoutException("timeout", request=None)  # type: ignore[arg-type]

    async def embed(t: str) -> tuple[float, ...]:
        return (0.1,)

    async def search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return []

    gate = make_ingestion_gate(timeout_classify, embed, lambda: 1.0, search)  # type: ignore[arg-type]
    result = await gate('content', 'fact', QueryDistribution())
    assert result[0] == 'err' and result[1].code == 'CLASSIFY_FAILED'


@pytest.mark.asyncio
async def test_classify_5xx_returns_classify_failed() -> None:
    """HTTPStatusError propagates — ingestion gate converts it to CLASSIFY_FAILED."""
    from src.core.ingestion import make_ingestion_gate
    from src.domain_types.memory import QueryDistribution

    async def http_error_classify(content: str) -> str:
        request = httpx.Request("POST", "http://test/v1/chat/completions")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("500", request=request, response=response)

    async def embed(t: str) -> tuple[float, ...]:
        return (0.1,)

    async def search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return []

    gate = make_ingestion_gate(http_error_classify, embed, lambda: 1.0, search)  # type: ignore[arg-type]
    result = await gate('content', 'fact', QueryDistribution())
    assert result[0] == 'err' and result[1].code == 'CLASSIFY_FAILED'
