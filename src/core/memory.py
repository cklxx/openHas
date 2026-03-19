"""Memory store & recall — pure functions with injected IO."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import MemoryNode, MemoryQuery, RecallResult
from src.domain_types.ports import (
    EmbedFn,
    ExpandContextFn,
    HydrateFn,
    NowFn,
    RerankFn,
    RewriteQueryFn,
    SearchFn,
    StoreNodeFn,
    UpdateNodeFn,
)
from src.domain_types.result import Result

logger = logging.getLogger(__name__)

_W_HYDE = 0.25
_RERANK_FETCH_FACTOR = 5
_HYDE_TIMEOUT = 5.0
_RERANK_TIMEOUT = 30.0
_REWRITE_TIMEOUT = 10.0


@dataclass(frozen=True, slots=True)
class StoreError:
    code: Literal['EMPTY_CONTENT', 'EMBED_FAILED', 'WRITE_FAILED']
    detail: str = ''


@dataclass(frozen=True, slots=True)
class RecallError:
    code: Literal['EMPTY_QUERY', 'SEARCH_FAILED']
    detail: str = ''


_RecallFn = Callable[[MemoryQuery], Awaitable[Result[RecallResult, RecallError]]]


@dataclass(frozen=True, slots=True)
class RecallDeps:
    """Bundled IO dependencies shared across recall variants."""
    search: SearchFn
    query_embed: EmbedFn
    hydrate: HydrateFn
    primary_search: SearchFn  # doc-side only — used by HyDE to avoid expansion contamination
    hyde_weight: float = _W_HYDE


# ── Module-level helpers ──────────────────────────────────────────────────────

async def _validate_embed(
    embed: EmbedFn, content: str
) -> Result[tuple[float, ...], StoreError]:
    if not content.strip():
        return ('err', StoreError(code='EMPTY_CONTENT'))
    try:
        return ('ok', await embed(content))
    except Exception as e:
        return ('err', StoreError(code='EMBED_FAILED', detail=str(e)))


async def _try_update_access(
    fn: UpdateNodeFn | None, ids: tuple[str, ...]
) -> None:
    if not fn or not ids:
        return
    try:
        await fn(ids)
    except Exception as exc:
        logger.warning("update_access failed (non-fatal): %s", exc)


def _merge_scores(
    searches: list[list[tuple[str, float]]]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hits in searches:
        for nid, score in hits:
            if score > scores.get(nid, -1.0):
                scores[nid] = score
    return scores


def _blend_scores(
    base_map: dict[str, float], hyde_map: dict[str, float], w_hyde: float
) -> dict[str, float]:
    all_nids = set(base_map) | set(hyde_map)
    return {
        nid: base_map.get(nid, 0.0) * (1.0 - w_hyde)
             + hyde_map.get(nid, 0.0) * w_hyde
        for nid in all_nids
    }


def _build_result(
    hydrated: dict[str, MemoryNode],
    ordered: list[str],
    scores: dict[str, float],
) -> RecallResult:
    pairs = [(hydrated[k], scores[k]) for k in ordered if k in hydrated]
    return RecallResult(
        nodes=tuple(n for n, _ in pairs),
        scores=tuple(s for _, s in pairs),
    )


def _blend_and_rank(
    base_hits: list[tuple[str, float]],
    hyde_map: dict[str, float],
    top_k: int,
    w_hyde: float = _W_HYDE,
) -> tuple[list[str], dict[str, float]]:
    blended = _blend_scores(dict(base_hits), hyde_map, w_hyde)
    top = sorted(blended, key=blended.__getitem__, reverse=True)[:top_k]
    return top, blended


async def _safe_hyde_snippets(
    hyde_fn: ExpandContextFn, query_text: str
) -> list[str]:
    try:
        return await asyncio.wait_for(hyde_fn(query_text), _HYDE_TIMEOUT)
    except Exception as exc:
        logger.warning("HyDE fallback (non-fatal): %s", exc)
        return []


async def _compute_hyde_scores(
    doc_embed: EmbedFn, search: SearchFn,
    snippets: list[str], query: MemoryQuery,
) -> dict[str, float]:
    fetch_k = query.top_k * _RERANK_FETCH_FACTOR
    embs = await asyncio.gather(*[doc_embed(s) for s in snippets])
    searches = await asyncio.gather(
        *[search(e, fetch_k, query.kinds, query.labels) for e in embs]
    )
    return _merge_scores(list(searches))


def _get_candidates(
    r: object, decayed_ids: frozenset[str]
) -> list[MemoryNode] | None:
    if r[0] == 'err' or not r[1].nodes:  # type: ignore[index]
        return None
    return [n for n in r[1].nodes if n.id not in decayed_ids]  # type: ignore[index]


async def _safe_rerank(
    rerank_fn: RerankFn,
    query_text: str,
    candidates: list[MemoryNode],
) -> list[str] | None:
    try:
        return await asyncio.wait_for(
            rerank_fn(query_text, candidates), _RERANK_TIMEOUT
        )
    except Exception as exc:
        logger.warning("Reranker fallback (non-fatal): %s", exc)
        return None


# ── Factory functions ─────────────────────────────────────────────────────────

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


def make_recall(
    search: SearchFn,
    embed: EmbedFn,
    hydrate: HydrateFn,
    update_access: UpdateNodeFn | None = None,
):
    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        if not query.text.strip():
            return ('err', RecallError(code='EMPTY_QUERY'))
        try:
            q_emb = await embed(query.text)
            hits = await search(q_emb, query.top_k, query.kinds, query.labels)
        except Exception as e:
            return ('err', RecallError(code='SEARCH_FAILED', detail=str(e)))
        hits_map = dict(hits)
        hydrated = await hydrate(tuple(hits_map))
        result = _build_result(hydrated, list(hits_map), hits_map)
        await _try_update_access(update_access, tuple(n.id for n in result.nodes))
        return ('ok', result)

    return recall


def make_hyde_recall(
    deps: RecallDeps,
    doc_embed: EmbedFn,
    hyde_fn: ExpandContextFn,
    update_access: UpdateNodeFn | None = None,
):
    """Wrap base KNN recall with HyDE score blending."""

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        if not query.text.strip():
            return ('err', RecallError(code='EMPTY_QUERY'))
        try:
            q_emb = await deps.query_embed(query.text)
            base_hits = await deps.search(
                q_emb, query.top_k * _RERANK_FETCH_FACTOR, query.kinds, query.labels,
            )
        except Exception as e:
            return ('err', RecallError(code='SEARCH_FAILED', detail=str(e)))
        snippets = await _safe_hyde_snippets(hyde_fn, query.text)
        hyde_map = await _compute_hyde_scores(doc_embed, deps.primary_search, snippets, query)
        top_ids, blended = _blend_and_rank(base_hits, hyde_map, query.top_k, deps.hyde_weight)
        hydrated = await deps.hydrate(tuple(top_ids))
        result = _build_result(hydrated, top_ids, blended)
        await _try_update_access(update_access, tuple(n.id for n in result.nodes))
        return ('ok', result)

    return recall


def make_iterative_recall(
    base_recall: _RecallFn,
    query_embed: EmbedFn,
    search: SearchFn,
    hydrate: HydrateFn,
):
    """Wrap a recall fn with a second retrieval round using round-1 context."""

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        r1 = await base_recall(query)
        if r1[0] == 'err' or not r1[1].nodes:
            return r1  # type: ignore[return-value]
        ctx = ' '.join(n.content for n in r1[1].nodes[:3])
        try:
            r2_emb = await query_embed(f"{query.text} {ctx}")
            r2_hits = await search(r2_emb, query.top_k, query.kinds, query.labels)
        except Exception:
            return r1  # type: ignore[return-value]
        r1_map = dict(zip((n.id for n in r1[1].nodes), r1[1].scores, strict=False))
        r2_map: dict[str, float] = dict(r2_hits)
        merged = {
            nid: max(r1_map.get(nid, 0.0), r2_map.get(nid, 0.0))
            for nid in set(r1_map) | set(r2_map)
        }
        top_ids = sorted(merged, key=merged.__getitem__, reverse=True)[:query.top_k]
        hydrated = await hydrate(tuple(top_ids))
        return ('ok', _build_result(hydrated, top_ids, merged))

    return recall


def make_reranked_recall(
    base_recall: _RecallFn,
    rerank_fn: RerankFn,
    hydrate: HydrateFn,
    decayed_ids: frozenset[str],
):
    """Wrap a recall fn with an LLM reranking pass over a wider candidate pool."""

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        wide = MemoryQuery(text=query.text, top_k=query.top_k * _RERANK_FETCH_FACTOR)
        r = await base_recall(wide)
        candidates = _get_candidates(r, decayed_ids)
        if not candidates:
            return r  # type: ignore[return-value]
        ranked_ids = await _safe_rerank(rerank_fn, query.text, candidates)
        if ranked_ids is None:
            return r  # type: ignore[return-value]
        hydrated = await hydrate(tuple(ranked_ids[:query.top_k]))
        score_map: dict[str, float] = dict(zip(
            (n.id for n in r[1].nodes), r[1].scores, strict=False  # type: ignore[index]
        ))
        return ('ok', _build_result(hydrated, ranked_ids[:query.top_k], score_map))

    return recall


def make_rewritten_recall(
    base_recall: _RecallFn,
    rewrite_fn: RewriteQueryFn,
):
    """Wrap a recall fn with query rewriting to surface implicit knowledge."""

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        try:
            rewritten = await asyncio.wait_for(
                rewrite_fn(query.text), _REWRITE_TIMEOUT
            )
        except Exception as exc:
            logger.warning("Rewrite fallback (non-fatal): %s", exc)
            rewritten = query.text
        return await base_recall(MemoryQuery(text=rewritten, top_k=query.top_k))

    return recall
