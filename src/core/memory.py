"""Memory store & recall — pure functions with injected IO."""

from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import MemoryNode, MemoryQuery, RecallResult
from src.domain_types.ports import EmbedFn, HydrateFn, NowFn, SearchFn, StoreNodeFn
from src.domain_types.result import Result


@dataclass(frozen=True, slots=True)
class StoreError:
    code: Literal['EMPTY_CONTENT', 'EMBED_FAILED', 'WRITE_FAILED']
    detail: str = ''


@dataclass(frozen=True, slots=True)
class RecallError:
    code: Literal['EMPTY_QUERY', 'SEARCH_FAILED']
    detail: str = ''


async def _validate_embed(
    embed: EmbedFn, content: str
) -> Result[tuple[float, ...], StoreError]:
    if not content.strip():
        return ('err', StoreError(code='EMPTY_CONTENT'))
    try:
        return ('ok', await embed(content))
    except Exception as e:
        return ('err', StoreError(code='EMBED_FAILED', detail=str(e)))


def make_store_memory(embed: EmbedFn, now: NowFn, store: StoreNodeFn):
    async def store_memory(
        id: str, kind: str, content: str, labels: tuple[str, ...] = ()
    ) -> Result[MemoryNode, StoreError]:
        emb = await _validate_embed(embed, content)
        if emb[0] == 'err':
            return ('err', emb[1])
        ts = now()
        node = MemoryNode(
            id=id,
            kind=kind,  # type: ignore[arg-type]
            content=content,
            event_time=ts,
            record_time=ts,
            last_accessed=ts,
            labels=labels,
            embedding=emb[1],
        )
        res = await store(node)
        if res[0] == 'err':
            return ('err', StoreError(code='WRITE_FAILED', detail=res[1].detail))
        return ('ok', node)

    return store_memory


def make_recall(search: SearchFn, embed: EmbedFn, hydrate: HydrateFn):
    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        if not query.text.strip():
            return ('err', RecallError(code='EMPTY_QUERY'))
        try:
            q_emb = await embed(query.text)
            hits = await search(q_emb, query.top_k)
        except Exception as e:
            return ('err', RecallError(code='SEARCH_FAILED', detail=str(e)))
        hydrated = await hydrate(tuple(i for i, _ in hits)) if hits else {}
        pairs = [(hydrated[i], s) for i, s in hits if i in hydrated]
        return ('ok', RecallResult(
            nodes=tuple(n for n, _ in pairs),
            scores=tuple(s for _, s in pairs),
        ))

    return recall
