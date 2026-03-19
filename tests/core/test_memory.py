"""Tests for memory store & recall — using Protocol stubs, no mock.patch."""

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from src.core.memory import (
    RecallDeps,
    make_hyde_recall,
    make_iterative_recall,
    make_recall,
    make_reranked_recall,
    make_rewritten_recall,
    make_store_memory,
)
from src.domain_types.memory import MemoryNode, MemoryQuery

_EXPECTED_HIT_COUNT = 2
_NODES_AFTER_MISSING = 2


async def _noop_store(node: MemoryNode):  # type: ignore[return]
    return ('ok', None)


async def _noop_hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
    return {}


def _make_node(id: str) -> MemoryNode:
    return MemoryNode(id=id, kind='fact', content=id, event_time=0.0,
                      record_time=0.0, last_accessed=0.0)


@pytest.mark.asyncio
async def test_store_rejects_empty_content() -> None:
    async def embed(t: str) -> tuple[float, ...]:
        return (0.1,)
    result = await make_store_memory(embed, lambda: 1.0, _noop_store)('id1', 'fact', '   ')  # type: ignore[arg-type]
    assert result[0] == 'err' and result[1].code == 'EMPTY_CONTENT'


@pytest.mark.asyncio
async def test_store_success() -> None:
    async def embed(text: str) -> tuple[float, ...]:
        return (0.1, 0.2)

    store = make_store_memory(embed, lambda: 42.0, _noop_store)  # type: ignore[arg-type]
    result = await store('n1', 'fact', 'hello world')
    assert result[0] == 'ok' and result[1].id == 'n1'


@pytest.mark.asyncio
async def test_store_write_failure_returns_err() -> None:
    async def embed(t: str) -> tuple[float, ...]:
        return (0.1,)

    from src.domain_types.memory import NodeWriteError
    async def bad_store(node: MemoryNode):  # type: ignore[return]
        return ('err', NodeWriteError(code='WRITE_FAILED', detail='disk full'))

    store = make_store_memory(embed, lambda: 1.0, bad_store)  # type: ignore[arg-type]
    result = await store('n1', 'fact', 'content')
    assert result[0] == 'err' and result[1].code == 'WRITE_FAILED'


@pytest.mark.asyncio
async def test_recall_rejects_empty_query() -> None:
    async def embed(text: str) -> tuple[float, ...]:
        return (0.0,)

    async def search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return []

    recall = make_recall(search, embed, _noop_hydrate)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='  '))
    assert result[0] == 'err' and result[1].code == 'EMPTY_QUERY'


@pytest.mark.asyncio
async def test_recall_returns_hits() -> None:
    async def embed(text: str) -> tuple[float, ...]:
        return (0.5,)

    async def search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('node1', 0.9), ('node2', 0.7)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: _make_node(i) for i in ids}

    recall = make_recall(search, embed, hydrate)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='test query'))
    assert result[0] == 'ok' and len(result[1].nodes) == _EXPECTED_HIT_COUNT


@pytest.mark.asyncio
async def test_recall_calls_hydrate() -> None:
    seen: list[tuple[str, ...]] = []

    async def embed(t: str) -> tuple[float, ...]:
        return (0.5,)

    async def search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('node1', 0.9)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        seen.append(ids)
        return {i: _make_node(i) for i in ids}

    await make_recall(search, embed, hydrate)(MemoryQuery(text='q'))  # type: ignore[arg-type]
    assert seen == [('node1',)]


@pytest.mark.asyncio
async def test_recall_hydrate_skips_missing() -> None:
    """If hydrate returns fewer nodes, result is still aligned."""
    async def embed(t: str) -> tuple[float, ...]:
        return (0.5,)

    async def search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('n1', 0.9), ('n2', 0.7), ('n3', 0.5)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {'n1': _make_node('n1'), 'n3': _make_node('n3')}  # n2 missing

    recall = make_recall(search, embed, hydrate)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='q'))
    assert result[0] == 'ok'
    assert len(result[1].nodes) == _NODES_AFTER_MISSING
    assert result[1].nodes[0].id == 'n1'
    assert result[1].nodes[1].id == 'n3'
    assert len(result[1].scores) == _NODES_AFTER_MISSING


@pytest.mark.asyncio
async def test_store_embed_fails() -> None:
    async def bad_embed(text: str) -> tuple[float, ...]:
        raise RuntimeError("down")
    store = make_store_memory(bad_embed, lambda: 1.0, _noop_store)  # type: ignore[arg-type]
    result = await store('id', 'fact', 'content')
    assert result[0] == 'err' and result[1].code == 'EMBED_FAILED'


@pytest.mark.asyncio
async def test_recall_search_fails() -> None:
    async def embed(t: str) -> tuple[float, ...]:
        return (0.1,)

    async def bad_search(emb: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        raise RuntimeError("down")

    result = await make_recall(bad_search, embed, _noop_hydrate)(  # type: ignore[arg-type]
        MemoryQuery(text='query')
    )
    assert result[0] == 'err' and result[1].code == 'SEARCH_FAILED'


@pytest.mark.asyncio
async def test_recall_passes_top_k_to_search() -> None:
    seen: list[int] = []

    async def embed(t: str) -> tuple[float, ...]:
        return (0.1,)

    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        seen.append(k)
        return []

    await make_recall(search, embed, _noop_hydrate)(MemoryQuery(text='q', top_k=7))  # type: ignore[arg-type]
    assert seen == [7]


async def _stub_embed(t: str) -> tuple[float, ...]:
    return (0.1,)


@given(n=st.integers(min_value=0, max_value=15))
@settings(max_examples=20)
async def test_scores_length_always_matches_nodes(n: int) -> None:
    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [(f"n{i}", 0.5) for i in range(n)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: _make_node(i) for i in ids}

    result = await make_recall(search, _stub_embed, hydrate)(  # type: ignore[arg-type]
        MemoryQuery(text='q')
    )
    assert result[0] == 'ok' and len(result[1].nodes) == len(result[1].scores)


@pytest.mark.asyncio
async def test_hyde_recall_timeout_falls_back_to_base() -> None:
    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('n1', 0.9)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: _make_node(i) for i in ids}

    async def slow_hyde(text: str) -> list[str]:
        await asyncio.sleep(100)
        return []

    deps = RecallDeps(  # type: ignore[arg-type]
        search=search, query_embed=_stub_embed, hydrate=hydrate, primary_search=search,
    )
    recall = make_hyde_recall(deps, _stub_embed, slow_hyde)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='q', top_k=1))
    assert result[0] == 'ok' and len(result[1].nodes) == 1


@pytest.mark.asyncio
async def test_hyde_recall_empty_snippets_returns_base() -> None:
    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('n1', 0.9)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: _make_node(i) for i in ids}

    async def empty_hyde(text: str) -> list[str]:
        return []

    deps = RecallDeps(  # type: ignore[arg-type]
        search=search, query_embed=_stub_embed, hydrate=hydrate, primary_search=search,
    )
    recall = make_hyde_recall(deps, _stub_embed, empty_hyde)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='q', top_k=1))
    assert result[0] == 'ok' and result[1].nodes[0].id == 'n1'


@pytest.mark.asyncio
async def test_reranked_recall_all_decayed_returns_base() -> None:
    async def base_recall(query: MemoryQuery):  # type: ignore[return]
        return ('ok', type('R', (), {
            'nodes': (_make_node('n1'),),
            'scores': (0.9,),
        })())

    async def rerank(query: str, nodes: list) -> list[str]:
        return [n.id for n in nodes]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: _make_node(i) for i in ids}

    recall = make_reranked_recall(base_recall, rerank, hydrate, frozenset({'n1'}))  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='q', top_k=1))
    assert result[0] == 'ok'
    assert result[1].nodes[0].id == 'n1'


@pytest.mark.asyncio
async def test_iterative_recall_r1_empty_returns_r1() -> None:
    async def base_recall(query: MemoryQuery):  # type: ignore[return]
        return ('ok', type('R', (), {'nodes': (), 'scores': ()})())

    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('n1', 0.9)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: _make_node(i) for i in ids}

    recall = make_iterative_recall(base_recall, _stub_embed, search, hydrate)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='q'))
    assert result[0] == 'ok' and result[1].nodes == ()


@pytest.mark.asyncio
async def test_recall_updates_access_count() -> None:
    updated: list[tuple[str, ...]] = []

    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('n1', 0.9)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: _make_node(i) for i in ids}

    async def update_access(ids: tuple[str, ...]):  # type: ignore[return]
        updated.append(ids)
        return ('ok', None)

    recall = make_recall(search, _stub_embed, hydrate, update_access)  # type: ignore[arg-type]
    await recall(MemoryQuery(text='q'))
    assert updated == [('n1',)]


@pytest.mark.asyncio
async def test_recall_access_update_failure_does_not_fail_recall() -> None:
    async def search(e: tuple[float, ...], k: int) -> list[tuple[str, float]]:
        return [('n1', 0.9)]

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        return {i: _make_node(i) for i in ids}

    async def bad_update(ids: tuple[str, ...]):  # type: ignore[return]
        raise RuntimeError("db down")

    recall = make_recall(search, _stub_embed, hydrate, bad_update)  # type: ignore[arg-type]
    result = await recall(MemoryQuery(text='q'))
    assert result[0] == 'ok'


@pytest.mark.asyncio
async def test_rewritten_recall_rewrites_query() -> None:
    queries_seen: list[str] = []

    async def base_recall(query: MemoryQuery):  # type: ignore[return]
        queries_seen.append(query.text)
        return ('ok', type('R', (), {'nodes': (), 'scores': ()})())

    async def rewrite(q: str) -> str:
        return f"REWRITTEN: {q}"

    recall = make_rewritten_recall(base_recall, rewrite)  # type: ignore[arg-type]
    await recall(MemoryQuery(text='original'))
    assert queries_seen == ['REWRITTEN: original']


@pytest.mark.asyncio
async def test_rewritten_recall_timeout_falls_back() -> None:
    queries_seen: list[str] = []

    async def base_recall(query: MemoryQuery):  # type: ignore[return]
        queries_seen.append(query.text)
        return ('ok', type('R', (), {'nodes': (), 'scores': ()})())

    async def slow_rewrite(q: str) -> str:
        await asyncio.sleep(60)
        return "never"

    recall = make_rewritten_recall(base_recall, slow_rewrite)  # type: ignore[arg-type]
    await recall(MemoryQuery(text='original'))
    assert queries_seen == ['original']
