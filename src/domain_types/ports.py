from typing import Literal, Protocol


class QueryFn(Protocol):
    async def __call__(self, sql: str, params: tuple[object, ...]) -> list[dict[str, object]]: ...


class EmitFn(Protocol):
    async def __call__(self, event: str, payload: dict[str, object]) -> None: ...


class EmbedFn(Protocol):
    async def __call__(self, text: str) -> tuple[float, ...]: ...


class StoreFn(Protocol):
    async def __call__(self, key: str, value: bytes) -> None: ...


class SearchFn(Protocol):
    async def __call__(
        self, embedding: tuple[float, ...], top_k: int
    ) -> list[tuple[str, float]]: ...


class NowFn(Protocol):
    def __call__(self) -> float: ...


class ClassifyFn(Protocol):
    """Classify content permanence: permanent, transient, or unknown."""
    async def __call__(self, content: str) -> Literal['permanent', 'transient', 'unknown']: ...


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
