"""Predictive value scoring — pure functions."""

from src.domain_types.memory import MemoryNode


def compute_predictive_value(
    node: MemoryNode, now: float, decay_rate: float = 0.01
) -> float:
    """Score = recency-weighted access frequency, biased by decay."""
    age = max(now - node.last_accessed, 1.0)
    recency = 1.0 / (1.0 + decay_rate * age)
    frequency = node.access_count / (1.0 + age)
    return recency * 0.6 + frequency * 0.4
