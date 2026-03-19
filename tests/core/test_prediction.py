"""Tests for predictive buffer (P1.1)."""

import pytest
from src.core.prediction import make_fill_predictive_buffer

_HIT_COUNT = 2


async def _predict(recent_context: str) -> list[tuple[str, float]]:
    return [('what to eat', 0.6), ('weather', 0.3), ('noise', 0.05)]


async def _embed(text: str) -> tuple[float, ...]:
    return (0.1, 0.2)


async def _search(emb: tuple[float, ...], k: int, *_: object) -> list[tuple[str, float]]:
    return [('vegetarian-node', 0.9)]


@pytest.mark.asyncio
async def test_fills_buffer_for_high_prob_predictions() -> None:
    fill = make_fill_predictive_buffer(_predict, _embed, _search)  # type: ignore[arg-type]
    result = await fill('user asked about lunch')
    assert result[0] == 'ok'
    assert len(result[1].predicted_queries) == _HIT_COUNT


@pytest.mark.asyncio
async def test_empty_predictions_return_empty_buffer() -> None:
    async def no_predictions(ctx: str) -> list[tuple[str, float]]:
        return []

    fill = make_fill_predictive_buffer(no_predictions, _embed, _search)  # type: ignore[arg-type]
    result = await fill('context')
    assert result[0] == 'ok' and len(result[1].nodes) == 0


@pytest.mark.asyncio
async def test_scores_are_probability_weighted() -> None:
    fill = make_fill_predictive_buffer(_predict, _embed, _search)  # type: ignore[arg-type]
    result = await fill('context')
    assert result[0] == 'ok'
    assert all(s > 0 for s in result[1].scores)
