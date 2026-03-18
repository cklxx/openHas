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


def test_contradiction_keeps_newer_node() -> None:
    # newer node (record_time=10) survives; older (record_time=1) gets superseded
    graph = MemoryGraph(
        nodes=(_node('old', record_time=1.0), _node('new', record_time=10.0)),
        edges=(Edge(source_id='new', target_id='old', kind='contradicts'),),
    )
    result = consolidate(graph, 100.0, _DIST)
    assert result[0] == 'ok'
    supersede_actions = [a for a in result[1].actions if a.action == 'supersede']
    assert 'old' in supersede_actions[0].node_ids
    assert 'new' not in supersede_actions[0].node_ids


def test_supersession_no_dependents_decays_only_target() -> None:
    # 'new' supersedes 'old', no further edges — only 'old' decays
    graph = MemoryGraph(
        nodes=(_node('old'), _node('new')),
        edges=(Edge(source_id='new', target_id='old', kind='supersedes'),),
    )
    result = consolidate(graph, 100.0, _DIST)
    assert result[0] == 'ok'
    decay_actions = [a for a in result[1].actions if a.action == 'decay']
    assert len(decay_actions) == 1
    assert decay_actions[0].node_ids == ('old',)


def test_prunes_low_value_transient_node() -> None:
    # transient node with very high age and zero access → below _PRUNE_THRESHOLD
    old_transient = _node('stale', record_time=0.0, perm='transient')
    graph = MemoryGraph(nodes=(old_transient,), edges=())
    result = consolidate(graph, 1_000_000.0, _DIST)
    assert result[0] == 'ok'
    remove_actions = [a for a in result[1].actions if a.action == 'remove']
    assert len(remove_actions) == 1
    assert 'stale' in remove_actions[0].node_ids


def test_no_actions_on_healthy_graph() -> None:
    # permanent node, high access, no edges — nothing to prune or resolve
    healthy = MemoryNode(
        id='h', kind='fact', content='x',
        event_time=1.0, record_time=1.0, last_accessed=1.0,
        permanence='permanent', access_count=10,
    )
    graph = MemoryGraph(nodes=(healthy,), edges=())
    result = consolidate(graph, 2.0, _DIST)
    assert result[0] == 'ok'
    assert result[1].actions == ()


def test_multiple_contradictions_resolved_independently() -> None:
    # two independent contradiction pairs, both older nodes superseded
    graph = MemoryGraph(
        nodes=(
            _node('a_old', record_time=1.0), _node('a_new', record_time=5.0),
            _node('b_old', record_time=2.0), _node('b_new', record_time=8.0),
        ),
        edges=(
            Edge(source_id='a_old', target_id='a_new', kind='contradicts'),
            Edge(source_id='b_old', target_id='b_new', kind='contradicts'),
        ),
    )
    result = consolidate(graph, 100.0, _DIST)
    assert result[0] == 'ok'
    superseded_ids = {nid for a in result[1].actions if a.action == 'supersede'
                      for nid in a.node_ids}
    assert 'a_old' in superseded_ids
    assert 'b_old' in superseded_ids
