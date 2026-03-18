"""Tests for proactive background consolidation (P0.4)."""

from src.core.consolidation import consolidate
from src.domain_types.memory import (
    Edge,
    MemoryGraph,
    MemoryNode,
    QueryDistribution,
)

_DIST = QueryDistribution()


def _node(nid: str, record_time: float = 1.0, perm: str = 'unknown') -> MemoryNode:
    return MemoryNode(
        id=nid, kind='fact', content=f'content-{nid}',
        event_time=record_time, record_time=record_time,
        last_accessed=record_time, permanence=perm,  # type: ignore[arg-type]
    )


def test_empty_graph_returns_error() -> None:
    graph = MemoryGraph(nodes=(), edges=())
    result = consolidate(graph, 100.0, _DIST)
    assert result[0] == 'err' and result[1].code == 'EMPTY_GRAPH'


def test_finds_contradictions() -> None:
    graph = MemoryGraph(
        nodes=(_node('old', 1.0), _node('new', 10.0)),
        edges=(Edge(source_id='old', target_id='new', kind='contradicts'),),
    )
    result = consolidate(graph, 100.0, _DIST)
    assert result[0] == 'ok'
    supersede_actions = [a for a in result[1].actions if a.action == 'supersede']
    assert len(supersede_actions) >= 1
    assert 'old' in supersede_actions[0].node_ids


def test_propagates_supersession() -> None:
    graph = MemoryGraph(
        nodes=(_node('a'), _node('b'), _node('c')),
        edges=(
            Edge(source_id='a', target_id='b', kind='supersedes'),
            Edge(source_id='b', target_id='c', kind='causes'),
        ),
    )
    result = consolidate(graph, 100.0, _DIST)
    assert result[0] == 'ok'
    decay_actions = [a for a in result[1].actions if a.action == 'decay']
    assert len(decay_actions) >= 1
    decayed = decay_actions[0].node_ids
    assert 'b' in decayed and 'c' in decayed
