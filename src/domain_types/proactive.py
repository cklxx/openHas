"""Domain types for proactive memory surfacing."""

from dataclasses import dataclass
from typing import Literal

from .result import Result


@dataclass(frozen=True, slots=True)
class SurfaceRecommendation:
    """A memory node recommended for proactive surfacing."""
    node_id: str
    relevance: float
    urgency: Literal['immediate', 'deferred']
    reason: str


@dataclass(frozen=True, slots=True)
class ProactiveResult:
    """Result of a proactive surfacing pass."""
    recommendations: tuple[SurfaceRecommendation, ...]


@dataclass(frozen=True, slots=True)
class ProactiveError:
    code: Literal['EMPTY_CONTEXT', 'SCORING_FAILED']
    detail: str = ''


ProactiveOutcome = Result[ProactiveResult, ProactiveError]
