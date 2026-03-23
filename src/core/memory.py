"""Memory store & recall — pure functions with injected IO."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from src.domain_types.memory import MemoryNode, MemoryQuery, RecallResult
from src.domain_types.ports import (
    BM25SearchFn,
    DecayMapFn,
    DecomposeFn,
    EmbedFn,
    ExpandContextFn,
    GapCheckFn,
    GraphSearchFn,
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
_DECOMPOSE_TIMEOUT = 10.0
_GAP_CHECK_TIMEOUT = 10.0


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
) -> list[tuple[str, float]] | None:
    try:
        return await asyncio.wait_for(
            rerank_fn(query_text, candidates), _RERANK_TIMEOUT
        )
    except Exception as exc:
        logger.warning("Reranker fallback (non-fatal): %s", exc)
        return None


def _ranked_ids(ranked: list[tuple[str, float]]) -> list[str]:
    return [nid for nid, _ in ranked]


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
    rewrite_fn: RewriteQueryFn | None = None,
):
    """Wrap a recall fn with reranking over a wider candidate pool.

    When rewrite_fn is provided, both original and rewritten queries
    are scored by the reranker. Each document keeps its max score across
    both queries, preserving precision for KU queries while bridging
    vocabulary gaps for cue_trigger queries.
    """

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        wide = MemoryQuery(text=query.text, top_k=query.top_k * _RERANK_FETCH_FACTOR)
        r = await base_recall(wide)
        candidates = _get_candidates(r, decayed_ids)
        if not candidates:
            return r  # type: ignore[return-value]
        rewritten = await _safe_rewrite(rewrite_fn, query.text) if rewrite_fn else None
        ranked = await _dual_rerank(rerank_fn, query.text, rewritten, candidates)
        if ranked is None:
            return r  # type: ignore[return-value]
        return await _hydrate_ranked(hydrate, ranked[:query.top_k], r)

    return recall


async def _dual_rerank(
    rerank_fn: RerankFn, original: str,
    rewritten: str | None, candidates: list[MemoryNode],
) -> list[str] | None:
    """Rerank with both queries in parallel, merge by best rank."""
    if rewritten is None or rewritten == original:
        ranked = await _safe_rerank(rerank_fn, original, candidates)
        return _ranked_ids(ranked) if ranked else None
    ranked_orig, ranked_alt = await asyncio.gather(
        _safe_rerank(rerank_fn, original, candidates),
        _safe_rerank(rerank_fn, rewritten, candidates),
    )
    if ranked_orig is None:
        return None
    return _merge_rankings(ranked_orig, ranked_alt) if ranked_alt else _ranked_ids(ranked_orig)


def _merge_rankings(
    r1: list[tuple[str, float]], r2: list[tuple[str, float]],
) -> list[str]:
    """Merge two rankings by best (lowest) rank per ID."""
    best_rank: dict[str, int] = {}
    for rank, (nid, _) in enumerate(r1):
        best_rank[nid] = rank
    for rank, (nid, _) in enumerate(r2):
        best_rank[nid] = min(best_rank.get(nid, rank), rank)
    return sorted(best_rank, key=best_rank.__getitem__)


async def _hydrate_ranked(
    hydrate: HydrateFn, ranked_ids: list[str],
    r: Result[RecallResult, RecallError],
) -> Result[RecallResult, RecallError]:
    """Hydrate ranked IDs and build result with original scores."""
    hydrated = await hydrate(tuple(ranked_ids))
    score_map: dict[str, float] = dict(zip(
        (n.id for n in r[1].nodes), r[1].scores, strict=False  # type: ignore[index]
    ))
    return ('ok', _build_result(hydrated, ranked_ids, score_map))


_StreamResult = AsyncGenerator[Result[RecallResult, RecallError], None]


@dataclass(frozen=True, slots=True)
class RerankDeps:
    """Bundled dependencies for the reranking phase."""
    rerank_fn: RerankFn
    hydrate: HydrateFn
    decayed_ids: frozenset[str]


def make_streaming_recall(
    base_recall: _RecallFn, deps: RerankDeps,
):
    """Yield base results immediately, then upgraded reranked results.

    First yield: top_k from base recall (fast KNN + HyDE + iterative).
    Second yield (if different): reranked top_k from wider candidate pool.
    """

    async def recall(query: MemoryQuery) -> _StreamResult:
        wide = MemoryQuery(text=query.text, top_k=query.top_k * _RERANK_FETCH_FACTOR)
        r = await base_recall(wide)
        if r[0] == 'err':
            yield r  # type: ignore[misc]
            return
        initial = RecallResult(
            nodes=r[1].nodes[:query.top_k],
            scores=r[1].scores[:query.top_k],
        )
        yield ('ok', initial)
        upgraded = await _rerank_phase(r, deps, query)
        if upgraded is not None and upgraded.nodes != initial.nodes:
            yield ('ok', upgraded)

    return recall


async def _rerank_phase(
    r: object, deps: RerankDeps, query: MemoryQuery,
) -> RecallResult | None:
    candidates = _get_candidates(r, deps.decayed_ids)
    if not candidates:
        return None
    ranked = await _safe_rerank(deps.rerank_fn, query.text, candidates)
    if ranked is None:
        return None
    ranked_ids = _ranked_ids(ranked)
    hydrated = await deps.hydrate(tuple(ranked_ids[:query.top_k]))
    score_map: dict[str, float] = dict(zip(
        (n.id for n in r[1].nodes), r[1].scores, strict=False  # type: ignore[index]
    ))
    return _build_result(hydrated, ranked_ids[:query.top_k], score_map)


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


def make_augmented_recall(
    base_recall: _RecallFn,
    rewrite_fn: RewriteQueryFn,
):
    """Run original + rewritten query in parallel, merge by max score.

    Unlike make_rewritten_recall which replaces the query, this preserves
    the original query's precision while gaining the rewrite's coverage.
    """

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        rewritten = await _safe_rewrite(rewrite_fn, query.text)
        rewritten_q = MemoryQuery(text=rewritten, top_k=query.top_k)
        r_orig, r_rewrite = await asyncio.gather(
            base_recall(query), base_recall(rewritten_q),
        )
        return _merge_recall_results(r_orig, r_rewrite, query.top_k)

    return recall


async def _safe_rewrite(rewrite_fn: RewriteQueryFn, text: str) -> str:
    try:
        return await asyncio.wait_for(rewrite_fn(text), _REWRITE_TIMEOUT)
    except Exception as exc:
        logger.warning("Rewrite fallback (non-fatal): %s", exc)
        return text


def _collect_best_nodes(
    result: RecallResult, merged: dict[str, float], node_map: dict[str, MemoryNode],
) -> None:
    """Update merged scores and node_map with max-score entries from result."""
    for node, score in zip(result.nodes, result.scores, strict=True):
        if score > merged.get(node.id, -1.0):
            merged[node.id] = score
            node_map[node.id] = node


def _merge_recall_results(
    r1: Result[RecallResult, RecallError],
    r2: Result[RecallResult, RecallError],
    top_k: int,
) -> Result[RecallResult, RecallError]:
    """Merge two recall results, keeping max score per node."""
    if r1[0] == 'err':
        return r2
    if r2[0] == 'err':
        return r1
    merged: dict[str, float] = {}
    node_map: dict[str, MemoryNode] = {}
    _collect_best_nodes(r1[1], merged, node_map)
    _collect_best_nodes(r2[1], merged, node_map)
    top_ids = sorted(merged, key=merged.__getitem__, reverse=True)[:top_k]
    return ('ok', _build_result(node_map, top_ids, merged))


async def _safe_decompose(
    decompose_fn: DecomposeFn, text: str,
) -> list[str] | None:
    """Decompose query; return None on failure or single-query result."""
    try:
        subs = await asyncio.wait_for(decompose_fn(text), _DECOMPOSE_TIMEOUT)
    except Exception as exc:
        logger.warning("Decompose fallback (non-fatal): %s", exc)
        return None
    return subs if len(subs) > 1 else None


def _merge_multi_results(
    results: list[Result[RecallResult, RecallError]],
) -> dict[str, float]:
    """Merge multiple recall results by max score per node."""
    merged: dict[str, float] = {}
    for r in results:
        if r[0] == 'err' or not r[1].nodes:
            continue
        for node, score in zip(r[1].nodes, r[1].scores, strict=True):
            if score > merged.get(node.id, -1.0):
                merged[node.id] = score
    return merged


def make_decomposed_recall(
    base_recall: _RecallFn,
    decompose_fn: DecomposeFn,
    hydrate: HydrateFn,
):
    """Wrap a recall fn with query decomposition for multi-constraint queries."""

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        subs = await _safe_decompose(decompose_fn, query.text)
        if subs is None:
            return await base_recall(query)
        sub_queries = [MemoryQuery(text=s, top_k=query.top_k) for s in subs]
        results = await asyncio.gather(
            base_recall(query), *[base_recall(sq) for sq in sub_queries],
        )
        merged = _merge_multi_results(results)
        if not merged:
            return await base_recall(query)
        top_ids = sorted(merged, key=merged.__getitem__, reverse=True)[:query.top_k]
        hydrated = await hydrate(tuple(top_ids))
        return ('ok', _build_result(hydrated, top_ids, merged))

    return recall


_RRF_K = 60  # standard from Cormack et al. 2009
_TEMPORAL_DECAY_DEFAULT = 0.0001  # per-second decay factor for temporal re-score


def _rrf_fuse(
    channels: list[list[tuple[str, float]]],
    k: int = _RRF_K,
    weights: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: dict[str, float] = {}
    for i, channel in enumerate(channels):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, (nid, _score) in enumerate(channel):
            scores[nid] = scores.get(nid, 0.0) + w / (k + rank + 1)
    return scores


def _temporal_score(
    node: MemoryNode, now: float, decay: float = _TEMPORAL_DECAY_DEFAULT,
) -> float:
    """Time-aware boost: recent + frequently-accessed nodes score higher."""
    age = max(now - node.last_accessed, 0.0)
    recency = 1.0 / (1.0 + decay * age)
    frequency = 1.0 + 0.1 * min(node.access_count, 10)
    return recency * frequency


@dataclass(frozen=True, slots=True)
class HybridDeps:
    """Bundled IO dependencies for hybrid multi-channel recall."""
    dense_search: SearchFn
    bm25_search: BM25SearchFn
    graph_search: GraphSearchFn
    embed: EmbedFn
    hydrate: HydrateFn
    decay_map_fn: DecayMapFn


@dataclass(frozen=True, slots=True)
class HybridConfig:
    use_bm25: bool
    use_graph: bool
    use_temporal: bool
    temporal_decay: float
    rrf_k: int = _RRF_K
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    graph_weight: float = 1.0


async def _empty() -> list[tuple[str, float]]:
    return []


async def _collect_channels(
    deps: HybridDeps, cfg: HybridConfig,
    q_emb: tuple[float, ...], query: MemoryQuery, fetch_k: int,
) -> list[list[tuple[str, float]]]:
    """Run CH1 + CH2 parallel, CH3 sequential; return ranked lists."""
    ch1_coro = deps.dense_search(q_emb, fetch_k, query.kinds, query.labels)
    ch2_coro = deps.bm25_search(query.text, fetch_k) if cfg.use_bm25 else _empty()
    ch1_hits, ch2_hits = await asyncio.gather(ch1_coro, ch2_coro)
    channels: list[list[tuple[str, float]]] = [ch1_hits]
    if cfg.use_bm25 and ch2_hits:
        channels.append(ch2_hits)
    if cfg.use_graph:
        ch3_hits = await _safe_graph(deps, tuple(nid for nid, _ in ch1_hits[:10]))
        if ch3_hits:
            channels.append(ch3_hits)
    return channels


async def _safe_graph(
    deps: HybridDeps, seed_ids: tuple[str, ...],
) -> list[tuple[str, float]]:
    try:
        return await deps.graph_search(seed_ids)
    except Exception:
        return []


def _apply_temporal(
    fused: dict[str, float], hydrated: dict[str, MemoryNode],
    decay_map: dict[str, float], now: float, decay: float,
) -> dict[str, float]:
    """Return new score dict with temporal boost + decay factor applied."""
    result = dict(fused)
    for nid, node in hydrated.items():
        if nid in result:
            result[nid] *= _temporal_score(node, now, decay)
            result[nid] *= decay_map.get(nid, 1.0)
    return result


def _channel_weights(
    cfg: HybridConfig, channels: list[list[tuple[str, float]]],
) -> tuple[float, ...]:
    """Build weight tuple matching the order channels were appended."""
    weights: list[float] = [cfg.dense_weight]
    idx = 1
    if cfg.use_bm25 and idx < len(channels):
        weights.append(cfg.bm25_weight)
        idx += 1
    if cfg.use_graph and idx < len(channels):
        weights.append(cfg.graph_weight)
    return tuple(weights)


async def _hybrid_core(
    deps: HybridDeps, cfg: HybridConfig,
    now_fn: NowFn, query: MemoryQuery,
) -> Result[RecallResult, RecallError]:
    """Core hybrid pipeline: channels → RRF → temporal → result."""
    fetch_k = query.top_k * _RERANK_FETCH_FACTOR
    try:
        q_emb = await deps.embed(query.text)
        channels = await _collect_channels(deps, cfg, q_emb, query, fetch_k)
    except Exception as e:
        return ('err', RecallError(code='SEARCH_FAILED', detail=str(e)))
    channel_weights = _channel_weights(cfg, channels)
    fused = _rrf_fuse(channels, k=cfg.rrf_k, weights=channel_weights)
    if not fused:
        return ('ok', RecallResult(nodes=(), scores=()))
    hydrated = await deps.hydrate(tuple(fused.keys()))
    if cfg.use_temporal:
        decay_map = deps.decay_map_fn(list(hydrated.keys()))
        fused = _apply_temporal(fused, hydrated, decay_map, now_fn(), cfg.temporal_decay)
    final = sorted(fused, key=fused.__getitem__, reverse=True)[:query.top_k]
    return ('ok', _build_result(hydrated, final, fused))


def make_hybrid_recall(
    deps: HybridDeps, now_fn: NowFn,
    cfg: HybridConfig | None = None,
):  # type: ignore[return]
    """Multi-channel recall: dense + BM25 + graph → RRF → temporal re-score."""
    effective = cfg or HybridConfig(True, True, True, _TEMPORAL_DECAY_DEFAULT)

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        if not query.text.strip():
            return ('err', RecallError(code='EMPTY_QUERY'))
        return await _hybrid_core(deps, effective, now_fn, query)

    return recall


async def _safe_gap_check(
    gap_check_fn: GapCheckFn, query_text: str, facts: list[str],
) -> list[str] | None:
    """Run gap check; return None on failure or no gaps."""
    try:
        gaps = await asyncio.wait_for(
            gap_check_fn(query_text, facts), _GAP_CHECK_TIMEOUT,
        )
    except Exception as exc:
        logger.warning("Gap check fallback (non-fatal): %s", exc)
        return None
    return gaps or None


async def _fill_gaps(
    gap_queries: list[str], query_embed: EmbedFn,
    search: SearchFn, query: MemoryQuery,
) -> list[tuple[str, float]]:
    """Run embed+search for each gap query, collect all hits."""
    all_hits: list[tuple[str, float]] = []
    for gq in gap_queries:
        try:
            emb = await query_embed(gq)
            hits = await search(emb, query.top_k, query.kinds, query.labels)
            all_hits.extend(hits)
        except Exception:
            continue
    return all_hits


def _merge_gap_scores(
    base: dict[str, float], gap_hits: list[tuple[str, float]],
) -> dict[str, float]:
    """Merge base scores with gap-fill hits, keeping max per node."""
    merged = dict(base)
    for nid, score in gap_hits:
        if score > merged.get(nid, -1.0):
            merged[nid] = score
    return merged


def make_gap_filled_recall(
    base_recall: _RecallFn,
    gap_check_fn: GapCheckFn,
    query_embed: EmbedFn,
    search: SearchFn,
    hydrate: HydrateFn,
):
    """Wrap a recall fn with evidence-gap detection and filling.

    Inspired by MemR³ (arxiv:2512.20237) evidence-gap tracking.
    """

    async def recall(query: MemoryQuery) -> Result[RecallResult, RecallError]:
        r1 = await base_recall(query)
        if r1[0] == 'err' or not r1[1].nodes:
            return r1  # type: ignore[return-value]
        facts = [n.content for n in r1[1].nodes]
        gap_queries = await _safe_gap_check(gap_check_fn, query.text, facts)
        if gap_queries is None:
            return r1  # type: ignore[return-value]
        r1_map = dict(zip(
            (n.id for n in r1[1].nodes), r1[1].scores, strict=True,
        ))
        gap_hits = await _fill_gaps(gap_queries, query_embed, search, query)
        merged = _merge_gap_scores(r1_map, gap_hits)
        top_ids = sorted(merged, key=merged.__getitem__, reverse=True)[:query.top_k]
        hydrated = await hydrate(tuple(top_ids))
        return ('ok', _build_result(hydrated, top_ids, merged))

    return recall
