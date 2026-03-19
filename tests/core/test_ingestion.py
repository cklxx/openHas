"""Tests for write strategy / ingestion gate (P0.3)."""

import pytest
from src.core.ingestion import make_ingestion_gate
from src.domain_types.memory import QueryDistribution

_DIST = QueryDistribution()


async def _classify_permanent(content: str) -> str:
    return 'permanent'


async def _embed(text: str) -> tuple[float, ...]:
    return (0.5, 0.5, 0.5)


async def _search_empty(emb: tuple[float, ...], k: int, *_: object) -> list[tuple[str, float]]:
    return []


@pytest.mark.asyncio
async def test_rejects_empty_content() -> None:
    gate = make_ingestion_gate(_classify_permanent, _embed, lambda: 1.0, _search_empty)  # type: ignore[arg-type]
    result = await gate('  ', 'fact', _DIST)
    assert result[0] == 'err' and result[1].code == 'EMPTY_CONTENT'


@pytest.mark.asyncio
async def test_stores_valuable_content() -> None:
    gate = make_ingestion_gate(_classify_permanent, _embed, lambda: 1.0, _search_empty)  # type: ignore[arg-type]
    result = await gate('user is vegetarian', 'preference', _DIST)
    assert result[0] == 'ok' and result[1].action == 'store'


@pytest.mark.asyncio
async def test_stores_valuable_content_includes_node() -> None:
    gate = make_ingestion_gate(_classify_permanent, _embed, lambda: 1.0, _search_empty)  # type: ignore[arg-type]
    result = await gate('user is vegetarian', 'preference', _DIST)
    assert result[0] == 'ok' and result[1].node is not None
    assert result[1].node.content == 'user is vegetarian'


@pytest.mark.asyncio
async def test_detects_near_duplicate() -> None:
    async def search_hits(emb: tuple[float, ...], k: int, *_: object) -> list[tuple[str, float]]:
        return [('n1', 0.95)]

    gate = make_ingestion_gate(_classify_permanent, _embed, lambda: 2.0, search_hits)  # type: ignore[arg-type]
    result = await gate('vegetarian diet', 'fact', _DIST)
    assert result[0] == 'ok' and result[1].action == 'merge'
    assert result[1].merge_target_id == 'n1'


@pytest.mark.asyncio
async def test_classify_failure_returns_err() -> None:
    async def bad_classify(content: str) -> str:
        raise RuntimeError("timeout")

    gate = make_ingestion_gate(bad_classify, _embed, lambda: 1.0, _search_empty)  # type: ignore[arg-type]
    result = await gate('some content', 'fact', _DIST)
    assert result[0] == 'err' and result[1].code == 'CLASSIFY_FAILED'


@pytest.mark.asyncio
async def test_embed_failure_returns_err() -> None:
    async def bad_embed(text: str) -> tuple[float, ...]:
        raise RuntimeError("timeout")

    gate = make_ingestion_gate(_classify_permanent, bad_embed, lambda: 1.0, _search_empty)  # type: ignore[arg-type]
    result = await gate('some content', 'fact', _DIST)
    assert result[0] == 'err' and result[1].code == 'EMBED_FAILED'
