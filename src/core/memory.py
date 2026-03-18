"""Memory store & recall — pure functions with injected IO."""

from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import MemoryNode, MemoryQuery, RecallResult
from src.domain_types.ports import EmbedFn, NowFn, SearchFn
from src.domain_types.result import Result


@dataclass(frozen=True, slots=True)
class StoreError:
    code: Literal['EMPTY_CONTENT', 'EMBED_FAILED']
    detail: str = ''


@dataclass(frozen=True, slots=True)
class RecallError:
    code: Literal['EMPTY_QUERY', 'SEARCH_FAILED']
    detail: str = ''


def make_store_memory(embed: EmbedFn, now: NowFn):
    async def store_memory(
        id: str, kind: str, content: str, labels: tuple[str, ...] = ()
    ) -> Result[MemoryNode, StoreError]:
        if not content.strip():
            return ('err', StoreError(code='EMPTY_CONTENT'))
        try:
            embedding = await embed(content)
        except Exception as e:
            return ('err', StoreError(code='EMBED_FAILED', detail=str(e)))
        ts = now()
        node = MemoryNode(
            id=id,
            kind=kind,  # type: ignore[arg-type]
            content=content,
            event_time=ts,
            record_time=ts,
            last_accessed=ts,
            labels=labels,
            embedding=embedding,
        )
        return ('ok', node)

    return store_memory


def make_recall(search: SearchFn, embed: EmbedFn):
    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        if not query.text.strip():
            return ('err', RecallError(code='EMPTY_QUERY'))
        try:
            q_embedding = await embed(query.text)
            hits = await search(q_embedding, query.top_k)
        except Exception as e:
            return ('err', RecallError(code='SEARCH_FAILED', detail=str(e)))
        nodes = tuple(
            MemoryNode(
                id=hit_id,
                kind='fact',
                content='',
                event_time=0.0,
                record_time=0.0,
                last_accessed=0.0,
            )
            for hit_id, _score in hits
        )
        scores = tuple(score for _, score in hits)
        return ('ok', RecallResult(nodes=nodes, scores=scores))

    return recall
