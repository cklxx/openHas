"""Tests for memory store & recall — using Protocol stubs, no mock.patch."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
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


@pytest.mark.asyncio
async def test_store_embed_fails() -> None:
    async def bad_embed(text: str) -> tuple[float, ...]:
        raise RuntimeError("down")
    store = make_store_memory(bad_embed, lambda: 1.0)  # type: ignore[arg-type]
    result = await store('id', 'fact', 'content')
    assert result[0] == 'err' and result[1].code == 'EMBED_FAILED'


@pytest.mark.asyncio
async def test_recall_search_fails() -> None:
    async def embed(t: str) -> tuple[float, ...]:
        return (0.1,)
    async def bad_search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        raise RuntimeError("down")
    result = await make_recall(bad_search, embed)(MemoryQuery(text='query'))  # type: ignore[arg-type]
    assert result[0] == 'err' and result[1].code == 'SEARCH_FAILED'


@pytest.mark.asyncio
async def test_recall_passes_top_k_to_search() -> None:
    seen: list[int] = []
    async def embed(t: str) -> tuple[float, ...]:
        return (0.1,)
    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        seen.append(k)
        return []
    await make_recall(search, embed)(MemoryQuery(text='q', top_k=7))  # type: ignore[arg-type]
    assert seen == [7]


async def _stub_embed(t: str) -> tuple[float, ...]:
    return (0.1,)


@given(n=st.integers(min_value=0, max_value=15))
@settings(max_examples=20)
async def test_scores_length_always_matches_nodes(n: int) -> None:
    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [(f"n{i}", 0.5) for i in range(n)]
    result = await make_recall(search, _stub_embed)(MemoryQuery(text='q'))  # type: ignore[arg-type]
    assert result[0] == 'ok' and len(result[1].nodes) == len(result[1].scores)
