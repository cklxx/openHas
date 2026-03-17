from dataclasses import dataclass
from typing import Literal

NodeKind = Literal['fact', 'episode', 'preference', 'entity', 'procedure']
EdgeKind = Literal['related', 'causes', 'part_of', 'contradicts', 'supersedes']


@dataclass(frozen=True, slots=True)
class MemoryNode:
    id: str
    kind: NodeKind
    content: str
    created_at: float
    last_accessed: float
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
class MemoryQuery:
    text: str
    top_k: int = 10
    kinds: tuple[NodeKind, ...] = ()
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallResult:
    nodes: tuple[MemoryNode, ...]
    edges: tuple[Edge, ...] = ()
    scores: tuple[float, ...] = ()
