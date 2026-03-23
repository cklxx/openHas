"""Proactive memory surfacing — surface relevant memories without being asked."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import MemoryNode
from src.domain_types.ports import (
    EmbedFn,
    GraphNeighborFn,
    HydrateFn,
    ScoreContextFn,
    SearchFn,
)
from src.domain_types.proactive import (
    ProactiveError,
    ProactiveOutcome,
    ProactiveResult,
    SurfaceRecommendation,
)

logger = logging.getLogger(__name__)

URGENT_LABELS: frozenset[str] = frozenset({
    'health', 'allergy', 'medication',
    'deadline', 'scheduling', 'safety',
})

CODING_KEYWORDS: frozenset[str] = frozenset({
    'function', 'class', 'import', 'def', 'async', 'await',
    'return', 'const', 'let', 'var', 'struct', 'impl',
    'fn', 'pub', 'int', 'str', 'list', 'dict', 'for', 'while',
})

_PRE_FILTER_K = 20
_SURFACE_THRESHOLD = 0.5
_CODING_THRESHOLD = 0.8
_GRAPH_BOOST = 0.2
_GRAPH_BOOST_MIN_RELEVANCE = 0.3
_BOOST_EDGE_KINDS = frozenset({'contradicts', 'causes'})
_IMMEDIATE_THRESHOLD = 0.7
_CODING_KEYWORD_MIN = 3


ProactiveFn = Callable[[str, int], Awaitable[ProactiveOutcome]]


@dataclass(frozen=True, slots=True)
class ProactiveDeps:
    """Bundled IO dependencies for proactive surfacing."""
    embed: EmbedFn
    search: SearchFn
    score_fn: ScoreContextFn
    neighbor_fn: GraphNeighborFn
    hydrate: HydrateFn


def _is_coding_context(context: str) -> bool:
    words = set(context.lower().split())
    return len(words & CODING_KEYWORDS) >= _CODING_KEYWORD_MIN


def _effective_threshold(context: str) -> float:
    if _is_coding_context(context):
        return _CODING_THRESHOLD
    return _SURFACE_THRESHOLD


def _classify_urgency(
    relevance: float, node: MemoryNode,
) -> Literal['immediate', 'deferred', 'never']:
    """Classify urgency based on relevance and node labels."""
    has_urgent = any(label in URGENT_LABELS for label in node.labels)
    if relevance >= _IMMEDIATE_THRESHOLD and has_urgent:
        return 'immediate'
    if relevance >= _SURFACE_THRESHOLD:
        return 'deferred'
    return 'never'


def _apply_graph_boost(
    scores: dict[str, float],
    candidates: dict[str, MemoryNode],
    neighbor_fn: GraphNeighborFn,
    threshold: float,
) -> None:
    """Boost neighbors of high-scoring candidates via graph edges."""
    seeds = tuple(
        nid for nid, score in scores.items()
        if score > _GRAPH_BOOST_MIN_RELEVANCE
    )
    if not seeds:
        return
    for nid, kind in neighbor_fn(seeds):
        if kind not in _BOOST_EDGE_KINDS or nid not in candidates:
            continue
        current = scores.get(nid, 0.0)
        if current < threshold:
            scores[nid] = min(current + _GRAPH_BOOST, 1.0)


def _build_recommendations(
    scores: dict[str, float],
    candidates: dict[str, MemoryNode],
    threshold: float,
    top_k: int,
) -> tuple[SurfaceRecommendation, ...]:
    """Build sorted recommendations from scored candidates."""
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    recs: list[SurfaceRecommendation] = []
    for nid in ranked:
        if scores[nid] < threshold or nid not in candidates:
            continue
        urgency = _classify_urgency(scores[nid], candidates[nid])
        if urgency == 'never':
            continue
        recs.append(SurfaceRecommendation(
            node_id=nid, relevance=scores[nid],
            urgency=urgency, reason=candidates[nid].content[:80],
        ))
        if len(recs) >= top_k:
            break
    return tuple(recs)


def make_proactive_surface(deps: ProactiveDeps) -> ProactiveFn:
    """Factory: proactive surfacing engine.

    Embeds context → dense pre-filter → CE score → graph boost
    → urgency classify → anti-noise filter → recommendations.
    """

    async def surface(context: str, top_k: int) -> ProactiveOutcome:
        if not context.strip():
            return ('err', ProactiveError(code='EMPTY_CONTEXT'))
        threshold = _effective_threshold(context)
        try:
            emb = await deps.embed(context)
            hits = await deps.search(emb, _PRE_FILTER_K, (), ())
        except Exception as e:
            return ('err', ProactiveError(
                code='SCORING_FAILED', detail=str(e),
            ))
        node_ids = tuple(nid for nid, _ in hits)
        candidates = await deps.hydrate(node_ids)
        scores = _score_candidates(deps.score_fn, context, candidates)
        _apply_graph_boost(scores, candidates, deps.neighbor_fn, threshold)
        recs = _build_recommendations(scores, candidates, threshold, top_k)
        return ('ok', ProactiveResult(recommendations=recs))

    return surface


def _score_candidates(
    score_fn: ScoreContextFn,
    context: str,
    candidates: dict[str, MemoryNode],
) -> dict[str, float]:
    """Score each candidate against the context using the CE."""
    return {
        nid: score_fn(context, node.content)
        for nid, node in candidates.items()
    }
