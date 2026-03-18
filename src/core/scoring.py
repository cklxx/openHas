"""P0.2: Predictive value objective function.

Replace recency × frequency (the Zep path) with:
  V(m) = E_Q [ information_gain(m) ]
       ≈ permanence_weight × kind_prior × uniqueness × access_signal

Key insight: a memory's value is the conditional probability that removing it
degrades the next response. This depends on:
  1. Permanence — permanent facts ("vegetarian") don't decay with time
  2. Kind prior — weighted by user's query distribution
  3. Uniqueness — can this be inferred from other memories? (non-redundancy)
  4. Access signal — was it actually used when available? (empirical utility)
"""

from src.domain_types.memory import MemoryNode, QueryDistribution

_PERMANENCE_WEIGHTS = {'permanent': 1.0, 'transient': 0.3, 'unknown': 0.6}
_KIND_FIELDS = ('fact', 'episode', 'preference', 'entity', 'procedure')


def _permanence_factor(node: MemoryNode, age: float) -> float:
    """Permanent facts don't decay. Transient facts decay fast."""
    base = _PERMANENCE_WEIGHTS.get(node.permanence, 0.6)
    if node.permanence == 'permanent':
        return base
    return base / (1.0 + 0.001 * age)


def _kind_prior(node: MemoryNode, dist: QueryDistribution) -> float:
    """How likely the user's next query needs this kind of memory."""
    return getattr(dist, node.kind, 0.2)


def _uniqueness_score(dependency_count: int) -> float:
    """Memories with fewer dependents are harder to reconstruct.

    dependency_count = number of other nodes that could substitute.
    0 dependents → uniqueness = 1.0 (irreplaceable).
    """
    return 1.0 / (1.0 + dependency_count)


def _access_utility(node: MemoryNode) -> float:
    """Empirical signal: was this memory useful when retrieved?"""
    if node.access_count == 0:
        return 0.5  # no data, neutral prior
    return min(node.access_count / 10.0, 1.0)


def compute_predictive_value(
    node: MemoryNode,
    now: float,
    dist: QueryDistribution,
    dependency_count: int = 0,
) -> float:
    """V(m) ≈ permanence × kind_prior × uniqueness × access_utility.

    This is the objective function for all downstream decisions:
    - Write strategy uses it to gate ingestion
    - Proactive consolidation uses it to decide merge/prune
    - Context budget allocation uses it to rank memories
    """
    age = max(now - node.record_time, 0.0)
    p = _permanence_factor(node, age)
    k = _kind_prior(node, dist)
    u = _uniqueness_score(dependency_count)
    a = _access_utility(node)
    return p * k * u * a
