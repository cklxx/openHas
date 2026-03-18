"""Property-based tests for predictive value scoring (P0.2)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.core.scoring import compute_predictive_value
from src.domain_types.memory import MemoryNode, QueryDistribution

_DIST = QueryDistribution()


def _make_node(
    permanence: str = 'unknown', access_count: int = 0
) -> MemoryNode:
    return MemoryNode(
        id='n1', kind='fact', content='x',
        event_time=1.0, record_time=1.0, last_accessed=1.0,
        permanence=permanence, access_count=access_count,  # type: ignore[arg-type]
    )


_node_strategy = st.builds(
    MemoryNode,
    id=st.text(min_size=1, max_size=10),
    kind=st.sampled_from(['fact', 'episode', 'preference', 'entity', 'procedure']),
    content=st.text(max_size=100),
    event_time=st.floats(min_value=0, max_value=1e12),
    record_time=st.floats(min_value=0, max_value=1e12),
    last_accessed=st.floats(min_value=0, max_value=1e12),
    permanence=st.sampled_from(['permanent', 'transient', 'unknown']),
    access_count=st.integers(min_value=0, max_value=10000),
)


@given(_node_strategy, st.floats(min_value=0, max_value=1e12))
def test_score_is_non_negative(node: MemoryNode, now: float) -> None:
    score = compute_predictive_value(node, now, _DIST)
    assert score >= 0.0


@given(_node_strategy)
def test_permanent_beats_transient(node: MemoryNode) -> None:
    now = node.record_time + 100_000.0
    perm = compute_predictive_value(_make_node('permanent'), now, _DIST)
    trans = compute_predictive_value(_make_node('transient'), now, _DIST)
    assert perm >= trans


def test_high_dependency_reduces_uniqueness() -> None:
    node = _make_node()
    unique = compute_predictive_value(node, 2.0, _DIST, dependency_count=0)
    redundant = compute_predictive_value(node, 2.0, _DIST, dependency_count=10)
    assert unique > redundant


def test_kind_prior_affects_score() -> None:
    node = _make_node()
    fact_heavy = QueryDistribution(
        fact=0.8, episode=0.05, preference=0.05, entity=0.05, procedure=0.05,
    )
    episode_heavy = QueryDistribution(
        fact=0.05, episode=0.8, preference=0.05, entity=0.05, procedure=0.05,
    )
    v_fact = compute_predictive_value(node, 2.0, fact_heavy)
    v_ep = compute_predictive_value(node, 2.0, episode_heavy)
    assert v_fact > v_ep


def test_transient_decays_monotonically_with_age() -> None:
    node = _make_node('transient', access_count=5)
    ages = [0.0, 1_000.0, 10_000.0, 100_000.0]
    scores = [compute_predictive_value(node, node.record_time + a, _DIST) for a in ages]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_permanent_does_not_decay_with_age() -> None:
    node = _make_node('permanent', access_count=5)
    v_young = compute_predictive_value(node, node.record_time + 1.0, _DIST)
    v_old = compute_predictive_value(node, node.record_time + 1_000_000.0, _DIST)
    assert v_young == v_old


def test_access_count_zero_uses_neutral_prior() -> None:
    # access_count=0 → access_utility=0.5; access_count=10 → 1.0
    node_zero = _make_node(access_count=0)
    node_used = _make_node(access_count=10)
    v_zero = compute_predictive_value(node_zero, 2.0, _DIST)
    v_used = compute_predictive_value(node_used, 2.0, _DIST)
    assert v_used == pytest.approx(v_zero * 2.0)


def test_access_count_caps_at_ten() -> None:
    node_10 = _make_node(access_count=10)
    node_100 = _make_node(access_count=100)
    assert compute_predictive_value(node_10, 2.0, _DIST) == \
        compute_predictive_value(node_100, 2.0, _DIST)


@given(st.integers(min_value=0, max_value=1000))
def test_uniqueness_strictly_decreases_with_dependency_count(dep: int) -> None:
    node = _make_node(access_count=1)
    v0 = compute_predictive_value(node, 2.0, _DIST, dependency_count=0)
    vn = compute_predictive_value(node, 2.0, _DIST, dependency_count=dep)
    assert v0 >= vn
