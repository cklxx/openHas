"""P0.4: Proactive background consolidation.

Runs in query-interval idle time. Optimization target: predictive value.

Three operations:
  1. Contradiction detection — find 'contradicts' edges, keep the newer belief
  2. Supersession propagation — when A supersedes B, decay B and B's dependents
  3. Low-value pruning — remove nodes below threshold
"""

from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import (
    ConsolidationAction,
    MemoryGraph,
    MemoryNode,
    QueryDistribution,
)
from src.domain_types.result import Result

from .scoring import compute_predictive_value

_PRUNE_THRESHOLD = 0.01


@dataclass(frozen=True, slots=True)
class ConsolidationError:
    code: Literal['EMPTY_GRAPH']
    detail: str = ''


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    actions: tuple[ConsolidationAction, ...]


def _resolve_contradiction(
    index: dict[str, MemoryNode], src_id: str, tgt_id: str
) -> ConsolidationAction | None:
    src = index.get(src_id)
    tgt = index.get(tgt_id)
    if src is None or tgt is None:
        return None
    older = src if src.record_time <= tgt.record_time else tgt
    return ConsolidationAction(
        action='supersede',
        node_ids=(older.id,),
        reason='contradicted by newer belief',
    )


def _find_contradictions(graph: MemoryGraph) -> tuple[ConsolidationAction, ...]:
    index = {n.id: n for n in graph.nodes}
    actions = [
        a for edge in graph.edges
        if edge.kind == 'contradicts'
        if (a := _resolve_contradiction(index, edge.source_id, edge.target_id)) is not None
    ]
    return tuple(actions)


def _collect_superseded_ids(graph: MemoryGraph) -> set[str]:
    return {e.target_id for e in graph.edges if e.kind == 'supersedes'}


def _collect_dependent_ids(graph: MemoryGraph, source_ids: set[str]) -> set[str]:
    return {
        e.target_id for e in graph.edges
        if e.source_id in source_ids and e.kind in ('causes', 'part_of')
    }


def _find_supersession_targets(graph: MemoryGraph) -> tuple[ConsolidationAction, ...]:
    superseded = _collect_superseded_ids(graph)
    dependents = _collect_dependent_ids(graph, superseded)
    all_decay = superseded | dependents
    if not all_decay:
        return ()
    return (ConsolidationAction(
        action='decay',
        node_ids=tuple(sorted(all_decay)),
        reason='superseded or dependent on superseded node',
    ),)


def _find_low_value_nodes(
    graph: MemoryGraph, now: float, dist: QueryDistribution
) -> tuple[ConsolidationAction, ...]:
    removable = [
        n.id for n in graph.nodes
        if compute_predictive_value(n, now, dist) < _PRUNE_THRESHOLD
    ]
    if not removable:
        return ()
    return (ConsolidationAction(
        action='remove',
        node_ids=tuple(removable),
        reason=f'predictive value below {_PRUNE_THRESHOLD}',
    ),)


def consolidate(
    graph: MemoryGraph, now: float, dist: QueryDistribution
) -> Result[ConsolidationResult, ConsolidationError]:
    """Run all consolidation passes. Pure — does not mutate graph."""
    if not graph.nodes:
        return ('err', ConsolidationError(code='EMPTY_GRAPH'))
    contradictions = _find_contradictions(graph)
    supersessions = _find_supersession_targets(graph)
    pruned = _find_low_value_nodes(graph, now, dist)
    all_actions = contradictions + supersessions + pruned
    return ('ok', ConsolidationResult(actions=all_actions))
