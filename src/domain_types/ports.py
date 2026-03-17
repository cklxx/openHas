from typing import Protocol


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
