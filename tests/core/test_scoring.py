"""Property-based tests for predictive value scoring (P0.2)."""

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
