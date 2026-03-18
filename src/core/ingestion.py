"""P0.3: Write strategy — ingestion gate.

Every incoming piece of information goes through value estimation
before entering the graph. This is the first filter against noise.

Decision: store, skip, or merge with existing node.
ClassifyFn is LLM now, replaceable by RL policy later.
"""

import uuid
from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import MemoryNode, QueryDistribution, WriteDecision
from src.domain_types.ports import ClassifyFn, EmbedFn, NowFn, SearchFn
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
    permanence: str
    embedding: tuple[float, ...]
    timestamp: float


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


def _to_node(content: str, kind: str, prep: PreparedInput) -> MemoryNode:
    return MemoryNode(
        id=str(uuid.uuid4()),
        kind=kind,  # type: ignore[arg-type]
        content=content,
        event_time=prep.timestamp,
        record_time=prep.timestamp,
        last_accessed=prep.timestamp,
        permanence=prep.permanence,  # type: ignore[arg-type]
        embedding=prep.embedding,
    )


async def _find_near_dup(embedding: tuple[float, ...], search: SearchFn) -> str:
    """Return node ID of closest match above threshold, or ''."""
    if not embedding:
        return ''
    hits = await search(embedding, 1)
    if hits and hits[0][1] >= _SIMILARITY_THRESHOLD:
        return hits[0][0]
    return ''


async def _decide(
    node: MemoryNode, now: float, dist: QueryDistribution, search: SearchFn
) -> WriteDecision:
    """Decide: store, skip, or merge."""
    value = compute_predictive_value(node, now, dist)
    if value < _VALUE_THRESHOLD:
        return WriteDecision(action='skip', estimated_value=value, reason='below value threshold')
    dup_id = await _find_near_dup(node.embedding, search)
    if dup_id:
        return WriteDecision(
            action='merge', estimated_value=value,
            merge_target_id=dup_id, reason='near-duplicate', node=node,
        )
    return WriteDecision(action='store', estimated_value=value, node=node)


def make_ingestion_gate(classify: ClassifyFn, embed: EmbedFn, now: NowFn, search: SearchFn):
    """Factory: returns an ingestion gate function."""

    async def ingest(
        content: str, kind: str, dist: QueryDistribution,
    ) -> Result[WriteDecision, IngestError]:
        if not content.strip():
            return ('err', IngestError(code='EMPTY_CONTENT'))
        prep = await _classify_and_embed(content, classify, embed, now)
        if prep[0] == 'err':
            return ('err', prep[1])
        node = _to_node(content, kind, prep[1])
        return ('ok', await _decide(node, now(), dist, search))

    return ingest
