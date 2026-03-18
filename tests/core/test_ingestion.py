"""Tests for write strategy / ingestion gate (P0.3)."""

import pytest
from src.core.ingestion import make_ingestion_gate
from src.domain_types.memory import MemoryGraph, MemoryNode, QueryDistribution

_DIST = QueryDistribution()
_EMPTY_GRAPH = MemoryGraph(nodes=(), edges=())


async def _classify_permanent(content: str) -> str:
    return 'permanent'


async def _classify_transient(content: str) -> str:
    return 'transient'


async def _embed(text: str) -> tuple[float, ...]:
    return (0.5, 0.5, 0.5)


@pytest.mark.asyncio
async def test_rejects_empty_content() -> None:
    gate = make_ingestion_gate(_classify_permanent, _embed, lambda: 1.0)  # type: ignore[arg-type]
    result = await gate('  ', 'fact', _EMPTY_GRAPH, _DIST)
    assert result[0] == 'err' and result[1].code == 'EMPTY_CONTENT'


@pytest.mark.asyncio
async def test_stores_valuable_content() -> None:
    gate = make_ingestion_gate(_classify_permanent, _embed, lambda: 1.0)  # type: ignore[arg-type]
    result = await gate('user is vegetarian', 'preference', _EMPTY_GRAPH, _DIST)
    assert result[0] == 'ok' and result[1].action == 'store'


@pytest.mark.asyncio
async def test_detects_near_duplicate() -> None:
    existing = MemoryNode(
        id='n1', kind='fact', content='vegetarian',
        event_time=1.0, record_time=1.0, last_accessed=1.0,
        embedding=(0.5, 0.5, 0.5),
    )
    graph = MemoryGraph(nodes=(existing,), edges=())
    gate = make_ingestion_gate(_classify_permanent, _embed, lambda: 2.0)  # type: ignore[arg-type]
    result = await gate('vegetarian diet', 'fact', graph, _DIST)
    assert result[0] == 'ok' and result[1].action == 'merge'
