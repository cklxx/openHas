from typing import Protocol

from .memory import ConsolidationAction, ExecutorError, MemoryNode, NodeWriteError
from .result import Result


class QueryFn(Protocol):
    async def __call__(self, sql: str, params: tuple[object, ...]) -> list[dict[str, object]]: ...


class EmitFn(Protocol):
    async def __call__(self, event: str, payload: dict[str, object]) -> None: ...


class EmbedFn(Protocol):
    async def __call__(self, text: str) -> tuple[float, ...]: ...


class SearchFn(Protocol):
    async def __call__(
        self, embedding: tuple[float, ...], top_k: int,
        kinds: tuple[str, ...] = (), labels: tuple[str, ...] = (),
    ) -> list[tuple[str, float]]: ...


class StoreNodeFn(Protocol):
    """Persist a MemoryNode to the storage backend."""
    async def __call__(self, node: MemoryNode) -> Result[None, NodeWriteError]: ...


class HydrateFn(Protocol):
    """Fetch full MemoryNodes by ID. Returns only IDs that exist."""
    async def __call__(self, ids: tuple[str, ...]) -> dict[str, MemoryNode]: ...


class ConsolidationExecutorFn(Protocol):
    """Apply consolidation actions against the storage backend."""
    async def __call__(
        self, actions: tuple[ConsolidationAction, ...]
    ) -> Result[int, ExecutorError]: ...


class NowFn(Protocol):
    def __call__(self) -> float: ...


class ClassifyFn(Protocol):
    """Classify content permanence: permanent, transient, or unknown."""
    async def __call__(self, content: str) -> str: ...


class PredictQueryFn(Protocol):
    """Predict what kinds of queries the user will ask next.

    Returns list of (predicted_query_text, probability) pairs.
    """
    async def __call__(self, recent_context: str) -> list[tuple[str, float]]: ...


class ExpandContextFn(Protocol):
    """Generate hypothetical queries that would retrieve a given memory.

    Used at index-build time: each generated query is embedded (query-side)
    and added to the search index alongside the raw doc embedding.
    """
    async def __call__(self, memory_text: str) -> list[str]: ...


class RerankFn(Protocol):
    """Re-rank candidate MemoryNodes by relevance to a query text.

    Returns node IDs in ranked order, most relevant first.
    """
    async def __call__(self, query: str, nodes: list[MemoryNode]) -> list[str]: ...


class RewriteQueryFn(Protocol):
    """Rewrite a query to make implicit domain knowledge explicit."""
    async def __call__(self, query: str) -> str: ...


class UpdateNodeFn(Protocol):
    """Increment access_count + update last_accessed for recalled node IDs."""
    async def __call__(self, ids: tuple[str, ...]) -> Result[None, NodeWriteError]: ...
