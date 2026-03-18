"""P0.3: Write strategy — ingestion gate.

Every incoming piece of information goes through value estimation
before entering the graph. This is the first filter against noise.

Decision: store, skip, or merge with existing node.
ClassifyFn is LLM now, replaceable by RL policy later.
"""

from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import (
    MemoryGraph,
    MemoryNode,
    QueryDistribution,
    WriteDecision,
)
from src.domain_types.ports import ClassifyFn, EmbedFn, NowFn
from src.domain_types.result import Result

from .scoring import compute_predictive_value

_VALUE_THRESHOLD = 0.05
_SIMILARITY_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class IngestError:
    code: Literal['EMPTY_CONTENT', 'CLASSIFY_FAILED', 'EMBED_FAILED']
    detail: str = ''


@dataclass(frozen=True, slots=True)
class PreparedInput:
    """Intermediate result: classified + embedded content."""
    permanence: Literal['permanent', 'transient', 'unknown']
    embedding: tuple[float, ...]
    timestamp: float


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _find_near_duplicate(
    embedding: tuple[float, ...], graph: MemoryGraph
) -> str:
    """Return node ID of closest match above threshold, or ''."""
    best_id = ''
    best_sim = _SIMILARITY_THRESHOLD
    for node in graph.nodes:
        if not node.embedding:
            continue
        sim = _cosine_similarity(embedding, node.embedding)
        if sim > best_sim:
            best_sim = sim
            best_id = node.id
    return best_id


async def _classify_and_embed(
    content: str, classify: ClassifyFn, embed: EmbedFn, now: NowFn
) -> Result[PreparedInput, IngestError]:
    """Classify permanence and embed content."""
    try:
        permanence = await classify(content)
    except Exception as exc:
        return ('err', IngestError(code='CLASSIFY_FAILED', detail=str(exc)))
    try:
        embedding = await embed(content)
    except Exception as exc:
        return ('err', IngestError(code='EMBED_FAILED', detail=str(exc)))
    return ('ok', PreparedInput(permanence=permanence, embedding=embedding, timestamp=now()))


def _to_provisional_node(content: str, kind: str, prep: PreparedInput) -> MemoryNode:
    return MemoryNode(
        id='_provisional', kind=kind,  # type: ignore[arg-type]
        content=content, event_time=prep.timestamp, record_time=prep.timestamp,
        last_accessed=prep.timestamp, permanence=prep.permanence, embedding=prep.embedding,
    )


def _decide(
    node: MemoryNode, now: float, graph: MemoryGraph, dist: QueryDistribution
) -> WriteDecision:
    """Pure decision: store, skip, or merge."""
    value = compute_predictive_value(node, now, dist)
    if value < _VALUE_THRESHOLD:
        return WriteDecision(
            action='skip', estimated_value=value, reason='below value threshold',
        )
    dup_id = _find_near_duplicate(node.embedding, graph)
    if dup_id:
        return WriteDecision(
            action='merge', estimated_value=value,
            merge_target_id=dup_id, reason='near-duplicate',
        )
    return WriteDecision(action='store', estimated_value=value)


def make_ingestion_gate(classify: ClassifyFn, embed: EmbedFn, now: NowFn):
    """Factory: returns an ingestion gate function."""

    async def ingest(
        content: str, kind: str, graph: MemoryGraph, dist: QueryDistribution,
    ) -> Result[WriteDecision, IngestError]:
        if not content.strip():
            return ('err', IngestError(code='EMPTY_CONTENT'))
        prep = await _classify_and_embed(content, classify, embed, now)
        if prep[0] == 'err':
            return prep  # type: ignore[return-value]
        node = _to_provisional_node(content, kind, prep[1])
        return ('ok', _decide(node, now(), graph, dist))

    return ingest
