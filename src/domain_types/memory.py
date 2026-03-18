"""Domain types for the memory graph — P0 infrastructure.

Dual timestamps: event_time (when fact occurred) vs record_time (when we learned it).
Permanence classification: transient facts decay, permanent facts don't.
Edge semantics carry belief-revision meaning (supersedes, contradicts).
"""

from dataclasses import dataclass
from typing import Literal

NodeKind = Literal['fact', 'episode', 'preference', 'entity', 'procedure']
EdgeKind = Literal['related', 'causes', 'part_of', 'contradicts', 'supersedes']
Permanence = Literal['permanent', 'transient', 'unknown']


@dataclass(frozen=True, slots=True)
class MemoryNode:
    id: str
    kind: NodeKind
    content: str
    # dual timestamps — the core P0 infrastructure
    event_time: float       # when the fact occurred (0.0 = unknown)
    record_time: float      # when we learned / stored it
    last_accessed: float
    permanence: Permanence = 'unknown'
    access_count: int = 0
    predictive_value: float = 0.0
    labels: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Edge:
    source_id: str
    target_id: str
    kind: EdgeKind
    weight: float = 1.0
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class MemoryGraph:
    """Immutable snapshot of the knowledge graph."""
    nodes: tuple[MemoryNode, ...]
    edges: tuple[Edge, ...]


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    text: str
    top_k: int = 10
    kinds: tuple[NodeKind, ...] = ()
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallResult:
    nodes: tuple[MemoryNode, ...]
    scores: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryDistribution:
    """User's query-type distribution — the prior for predictive value.

    Weights sum to 1.0. Maps NodeKind to probability that the user's
    next query will need memories of that kind.
    """
    fact: float = 0.2
    episode: float = 0.2
    preference: float = 0.2
    entity: float = 0.2
    procedure: float = 0.2


@dataclass(frozen=True, slots=True)
class NodeWriteError:
    """Storage-layer write failure."""
    code: Literal['WRITE_FAILED', 'DUPLICATE_ID']
    detail: str = ''


@dataclass(frozen=True, slots=True)
class ExecutorError:
    """Consolidation executor failure."""
    code: Literal['WRITE_FAILED', 'UNKNOWN_NODE']
    detail: str = ''


@dataclass(frozen=True, slots=True)
class WriteDecision:
    """Output of the ingestion gate — store, skip, or merge."""
    action: Literal['store', 'skip', 'merge']
    estimated_value: float
    merge_target_id: str = ''
    reason: str = ''
    # Prepared node (embedding already computed); present when action != 'skip'.
    node: 'MemoryNode | None' = None


@dataclass(frozen=True, slots=True)
class ConsolidationAction:
    """A single action proposed by proactive background consolidation."""
    action: Literal['merge', 'supersede', 'decay', 'remove']
    node_ids: tuple[str, ...]
    reason: str = ''
    new_content: str = ''
