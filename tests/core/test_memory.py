"""Tests for memory store & recall — using Protocol stubs, no mock.patch."""

import pytest
from src.core.memory import make_recall, make_store_memory
from src.domain_types.memory import MemoryQuery

_EXPECTED_HIT_COUNT = 2


def _noop_embed(text: str) -> None:
    return None


@pytest.mark.asyncio
async def test_store_rejects_empty_content() -> None:
    result = await make_store_memory(_noop_embed, lambda: 1.0)('id1', 'fact', '   ')  # type: ignore[arg-type]
    assert result[0] == 'err' and result[1].code == 'EMPTY_CONTENT'


@pytest.mark.asyncio
async def test_store_success() -> None:
    async def embed(text: str) -> tuple[float, ...]:
        return (0.1, 0.2)

    store = make_store_memory(embed, lambda: 42.0)  # type: ignore[arg-type]
    result = await store('n1', 'fact', 'hello world')
    assert result[0] == 'ok' and result[1].id == 'n1'


@pytest.mark.asyncio
async def test_recall_rejects_empty_query() -> None:
    async def embed(text: str) -> tuple[float, ...]:
        return (0.0,)

    async def search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return []

    recall = make_recall(search, embed)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='  '))
    assert result[0] == 'err' and result[1].code == 'EMPTY_QUERY'


@pytest.mark.asyncio
async def test_recall_returns_hits() -> None:
    async def embed(text: str) -> tuple[float, ...]:
        return (0.5,)

    async def search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('node1', 0.9), ('node2', 0.7)]

    recall = make_recall(search, embed)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='test query'))
    assert result[0] == 'ok' and len(result[1].nodes) == _EXPECTED_HIT_COUNT
