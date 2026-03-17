"""Property-based tests for predictive value scoring."""

from hypothesis import given
from hypothesis import strategies as st
from src.core.scoring import compute_predictive_value
from src.domain_types.memory import MemoryNode

_node_strategy = st.builds(
    MemoryNode,
    id=st.text(min_size=1, max_size=10),
    kind=st.sampled_from(['fact', 'episode', 'preference', 'entity', 'procedure']),
    content=st.text(max_size=100),
    created_at=st.floats(min_value=0, max_value=1e12),
    last_accessed=st.floats(min_value=0, max_value=1e12),
    access_count=st.integers(min_value=0, max_value=10000),
)


@given(_node_strategy, st.floats(min_value=0, max_value=1e12))
def test_score_is_bounded(node: MemoryNode, now: float) -> None:
    score = compute_predictive_value(node, now)
    assert score >= 0.0


@given(_node_strategy)
def test_recent_access_scores_higher(node: MemoryNode) -> None:
    recent = compute_predictive_value(node, node.last_accessed + 1.0)
    stale = compute_predictive_value(node, node.last_accessed + 100_000.0)
    assert recent >= stale
