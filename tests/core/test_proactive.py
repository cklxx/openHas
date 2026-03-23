"""Tests for proactive memory surfacing engine."""

import pytest
from src.core.proactive import (
    _CODING_THRESHOLD,
    _SURFACE_THRESHOLD,
    ProactiveDeps,
    _build_recommendations,
    _classify_urgency,
    _effective_threshold,
    _is_coding_context,
    _score_candidates,
    make_proactive_surface,
)
from src.domain_types.memory import MemoryNode


def _node(nid: str, content: str = "test", labels: tuple[str, ...] = ()) -> MemoryNode:
    return MemoryNode(
        id=nid, kind='fact', content=content,
        event_time=0.0, record_time=0.0, last_accessed=0.0,
        permanence='permanent', labels=labels, embedding=(),
    )


def test_is_coding_context_true() -> None:
    ctx = "I'm writing a Python function with import and async def"
    assert _is_coding_context(ctx)


def test_is_coding_context_false() -> None:
    ctx = "booking a dinner reservation at a seafood restaurant"
    assert not _is_coding_context(ctx)


def test_is_coding_context_boundary_two_keywords() -> None:
    ctx = "using import and class"
    assert not _is_coding_context(ctx)


def test_effective_threshold_coding() -> None:
    assert _effective_threshold("def class import async code") == _CODING_THRESHOLD


def test_effective_threshold_normal() -> None:
    assert _effective_threshold("dinner at restaurant") == _SURFACE_THRESHOLD


def test_classify_urgency_immediate() -> None:
    node = _node("n1", labels=("health", "allergy"))
    assert _classify_urgency(0.8, node) == 'immediate'


def test_classify_urgency_deferred() -> None:
    node = _node("n1", labels=("hobby",))
    assert _classify_urgency(0.6, node) == 'deferred'


def test_classify_urgency_never() -> None:
    node = _node("n1")
    assert _classify_urgency(0.3, node) == 'never'


def test_classify_urgency_high_but_no_urgent_label() -> None:
    node = _node("n1", labels=("hobby",))
    assert _classify_urgency(0.8, node) == 'deferred'


_HIGH_SCORE = 0.9
_LOW_SCORE = 0.1


def test_score_candidates() -> None:
    nodes = {"a": _node("a", "alpha"), "b": _node("b", "beta")}
    def score_fn(ctx: str, doc: str) -> float:
        return _HIGH_SCORE if "alpha" in doc else _LOW_SCORE
    scores = _score_candidates(score_fn, "test", nodes)
    assert scores["a"] == _HIGH_SCORE
    assert scores["b"] == _LOW_SCORE


def test_build_recommendations_filters_below_threshold() -> None:
    nodes = {"a": _node("a", labels=("health",)), "b": _node("b")}
    scores = {"a": 0.8, "b": 0.3}
    recs = _build_recommendations(scores, nodes, _SURFACE_THRESHOLD, 3)
    assert len(recs) == 1
    assert recs[0].node_id == "a"


_TOP_K_LIMIT = 2


def test_build_recommendations_respects_top_k() -> None:
    nodes = {f"n{i}": _node(f"n{i}", labels=("health",)) for i in range(5)}
    scores = {f"n{i}": 0.9 - i * 0.05 for i in range(5)}
    recs = _build_recommendations(scores, nodes, _SURFACE_THRESHOLD, _TOP_K_LIMIT)
    assert len(recs) == _TOP_K_LIMIT


def test_build_recommendations_skips_never_urgency() -> None:
    nodes = {"a": _node("a")}
    scores = {"a": 0.55}
    recs = _build_recommendations(scores, nodes, _SURFACE_THRESHOLD, 3)
    assert len(recs) == 1
    assert recs[0].urgency == 'deferred'


def _stub_deps(node: MemoryNode) -> ProactiveDeps:
    """Build deps for a single-node test scenario."""
    async def embed(text: str) -> tuple[float, ...]:
        return (0.1, 0.2)

    async def search(emb: object, k: int, kinds: object, labels: object) -> list:
        return [(node.id, _HIGH_SCORE)]

    async def hydrate(ids: tuple[str, ...]) -> dict:
        return {node.id: node}

    return ProactiveDeps(
        embed=embed, search=search,  # type: ignore[arg-type]
        score_fn=lambda ctx, doc: 0.85,
        neighbor_fn=lambda ids: [],
        hydrate=hydrate,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_make_proactive_surface_empty_context() -> None:
    node = _node("n1")
    fn = make_proactive_surface(_stub_deps(node))
    result = await fn("", 3)
    assert result[0] == 'err'
    assert result[1].code == 'EMPTY_CONTEXT'


@pytest.mark.asyncio
async def test_make_proactive_surface_returns_results() -> None:
    node = _node("diet-veg", "strict vegetarian", labels=("health",))
    fn = make_proactive_surface(_stub_deps(node))
    result = await fn("booking seafood dinner", 3)
    assert result[0] == 'ok'
    assert len(result[1].recommendations) == 1
    assert result[1].recommendations[0].node_id == "diet-veg"
