"""P1.1: Predictive buffer — fills the orthogonal retrieval blind spot.

Problem: "今天吃什么" and "素食者" are semantically orthogonal.
No vector/BM25/label pipeline bridges this. The blind spot is structural.

Solution: predict what user will ask next, pre-fetch memories for those
predicted queries. Two-phase retrieval:
  1. Active buffer — direct retrieval for actual query (standard)
  2. Predictive buffer — retrieval for predicted future queries (proactive)
"""

from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import MemoryNode
from src.domain_types.ports import EmbedFn, PredictQueryFn, SearchFn
from src.domain_types.result import Result

_MIN_PREDICTION_PROB = 0.1


@dataclass(frozen=True, slots=True)
class BufferError:
    code: Literal['PREDICT_FAILED', 'SEARCH_FAILED']
    detail: str = ''


@dataclass(frozen=True, slots=True)
class PredictiveBuffer:
    """Pre-fetched memories for predicted future queries."""
    nodes: tuple[MemoryNode, ...]
    scores: tuple[float, ...]
    predicted_queries: tuple[str, ...]


def _stub_node(node_id: str) -> MemoryNode:
    """Placeholder — adapter fills real data."""
    return MemoryNode(
        id=node_id, kind='fact', content='',
        event_time=0.0, record_time=0.0, last_accessed=0.0,
    )


def _filter_predictions(
    predictions: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    return [(q, p) for q, p in predictions if p >= _MIN_PREDICTION_PROB]


async def _search_one(
    query_text: str, prob: float, embed: EmbedFn, search: SearchFn,
) -> tuple[list[MemoryNode], list[float]]:
    """Search for one predicted query, return weighted nodes + scores."""
    q_emb = await embed(query_text)
    hits = await search(q_emb, 10)
    nodes = [_stub_node(hit_id) for hit_id, _ in hits]
    scores = [score * prob for _, score in hits]
    return nodes, scores


def make_fill_predictive_buffer(
    predict: PredictQueryFn, embed: EmbedFn, search: SearchFn
):
    """Factory: fills the predictive buffer asynchronously."""

    async def fill_buffer(
        recent_context: str,
    ) -> Result[PredictiveBuffer, BufferError]:
        try:
            raw = await predict(recent_context)
        except Exception as exc:
            return ('err', BufferError(code='PREDICT_FAILED', detail=str(exc)))
        relevant = _filter_predictions(raw)
        if not relevant:
            return ('ok', PredictiveBuffer(nodes=(), scores=(), predicted_queries=()))
        return await _collect_hits(relevant, embed, search)

    return fill_buffer


async def _collect_hits(
    predictions: list[tuple[str, float]],
    embed: EmbedFn, search: SearchFn,
) -> Result[PredictiveBuffer, BufferError]:
    all_nodes: list[MemoryNode] = []
    all_scores: list[float] = []
    queries: list[str] = []
    for q_text, prob in predictions:
        try:
            nodes, scores = await _search_one(q_text, prob, embed, search)
        except Exception as exc:
            return ('err', BufferError(code='SEARCH_FAILED', detail=str(exc)))
        all_nodes.extend(nodes)
        all_scores.extend(scores)
        queries.append(q_text)
    return ('ok', PredictiveBuffer(
        nodes=tuple(all_nodes), scores=tuple(all_scores),
        predicted_queries=tuple(queries),
    ))
