"""Tests for sqlite_vec_store adapter.

Integration tests (@pytest.mark.integration) require sqlite-vec installed
and sqlite-vec Python package available.
"""

import sys

import pytest
from src.domain_types.memory import ConsolidationAction, MemoryNode

_DECAYED_SCORE_MAX = 0.2
_CORPUS_SIZE = 3
_MISSING_EXCLUDED = 2
_EXPANSION_COUNT = 3

_EMBED_URL = "http://localhost:18080"


def _make_node(id: str, emb: tuple[float, ...] = ()) -> MemoryNode:
    return MemoryNode(
        id=id, kind='fact', content=f'content-{id}',
        event_time=1.0, record_time=1.0, last_accessed=1.0,
        embedding=emb,
    )


# ── Integration tests (require sqlite-vec) ───────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_and_recall_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import time

    from src.adapters.llama_embed import make_doc_embed_fn, make_query_embed_fn
    from src.adapters.sqlite_vec_store import (
        make_hydrate_fn,
        make_search_fn,
        make_store_fn,
        open_db,
    )
    from src.core.memory import make_recall, make_store_memory
    from src.domain_types.memory import MemoryQuery

    conn = open_db(str(tmp_path / "test.db"))
    store_node = make_store_fn(conn, 'alice')
    store = make_store_memory(make_doc_embed_fn(_EMBED_URL), time.time, store_node)
    assert (await store('diet-veg', 'fact', 'User follows a strict vegetarian diet'))[0] == 'ok'
    recall = make_recall(
        make_search_fn(conn, 'alice'),
        make_query_embed_fn(_EMBED_URL),
        make_hydrate_fn(conn, 'alice'),
    )
    r = await recall(MemoryQuery(text='what food restrictions does the user have?', top_k=3))
    assert r[0] == 'ok' and any(n.id == 'diet-veg' for n in r[1].nodes)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_id_isolation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nodes stored for alice are invisible to bob."""
    from src.adapters.sqlite_vec_store import make_hydrate_fn, make_store_fn, open_db

    conn = open_db(str(tmp_path / "test.db"))
    await make_store_fn(conn, 'alice')(_make_node('n1', (0.5,) * 1024))
    assert await make_hydrate_fn(conn, 'bob')(('n1',)) == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consolidation_decay(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Decay action sets decay_factor to 0.1, lowering search score."""
    from src.adapters.sqlite_vec_store import make_executor, make_search_fn, make_store_fn, open_db

    conn = open_db(str(tmp_path / "test.db"))
    emb: tuple[float, ...] = (1.0,) + (0.0,) * 1023
    await make_store_fn(conn, 'u')(MemoryNode(
        id='old', kind='fact', content='old fact',
        event_time=1.0, record_time=1.0, last_accessed=1.0, embedding=emb,
    ))
    decay_action = ConsolidationAction(action='decay', node_ids=('old',))
    result = await make_executor(conn, 'u')((decay_action,))
    assert result[0] == 'ok' and result[1] == 1
    hits = await make_search_fn(conn, 'u')(emb, 1)
    assert not hits or hits[0][1] < _DECAYED_SCORE_MAX


@pytest.mark.integration
@pytest.mark.asyncio
async def test_executor_skips_unknown_node(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Action on missing node_id should not crash."""
    from src.adapters.sqlite_vec_store import make_executor, open_db

    conn = open_db(str(tmp_path / "test.db"))
    result = await make_executor(conn, 'u')((
        ConsolidationAction(action='decay', node_ids=('ghost-id',)),
    ))
    assert result[0] == 'ok'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consolidation_remove(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Remove action deletes the node from the DB."""
    from src.adapters.sqlite_vec_store import (
        make_executor,
        make_hydrate_fn,
        make_store_fn,
        open_db,
    )

    conn = open_db(str(tmp_path / "test.db"))
    await make_store_fn(conn, 'u')(_make_node('to-remove', (0.1,) * 1024))
    remove_action = ConsolidationAction(action='remove', node_ids=('to-remove',))
    await make_executor(conn, 'u')((remove_action,))
    assert await make_hydrate_fn(conn, 'u')(('to-remove',)) == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hydrate_returns_full_content(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.adapters.sqlite_vec_store import make_hydrate_fn, make_store_fn, open_db

    conn = open_db(str(tmp_path / "test.db"))
    node = MemoryNode(
        id='n1', kind='preference', content='user is vegetarian',
        event_time=1.0, record_time=1.0, last_accessed=1.0, labels=('diet',),
    )
    await make_store_fn(conn, 'u')(node)
    hydrated = await make_hydrate_fn(conn, 'u')(('n1',))
    assert 'n1' in hydrated
    assert hydrated['n1'].content == 'user is vegetarian'
    assert hydrated['n1'].kind == 'preference'
    assert 'diet' in hydrated['n1'].labels


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_and_search_top_k_larger_than_corpus(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """top_k larger than corpus returns all available nodes."""
    from src.adapters.sqlite_vec_store import make_search_fn, make_store_fn, open_db

    conn = open_db(str(tmp_path / "test.db"))
    for i in range(_CORPUS_SIZE):
        emb: tuple[float, ...] = tuple(float(j == i) for j in range(1024))
        await make_store_fn(conn, 'u')(MemoryNode(
            id=f'n{i}', kind='fact', content=f'fact {i}',
            event_time=1.0, record_time=1.0, last_accessed=1.0, embedding=emb,
        ))
    hits = await make_search_fn(conn, 'u')((1.0,) + (0.0,) * 1023, 100)
    assert len(hits) == _CORPUS_SIZE


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_expansion_creates_multiple_vec_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.adapters.sqlite_vec_store import (
        make_store_expansion_fn,
        make_store_fn,
        open_db,
    )

    conn = open_db(str(tmp_path / "test.db"))
    await make_store_fn(conn, 'u')(_make_node('n1', (0.1,) * 1024))
    store_expansion = make_store_expansion_fn(conn, 'u')
    result = await store_expansion('n1', [(0.2,) * 1024, (0.3,) * 1024, (0.4,) * 1024])
    assert result[0] == 'ok' and result[1] == _EXPANSION_COUNT
    rows = conn.execute(
        "SELECT count(*) as c FROM vec_meta WHERE node_id = 'n1' AND user_id = 'u'"
    ).fetchone()
    assert rows['c'] >= _EXPANSION_COUNT


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_access_increments_count(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.adapters.sqlite_vec_store import (
        make_store_fn,
        make_update_access_fn,
        open_db,
    )

    conn = open_db(str(tmp_path / "test.db"))
    await make_store_fn(conn, 'u')(_make_node('n1', (0.1,) * 1024))
    update_access = make_update_access_fn(conn, 'u')
    result = await update_access(('n1',))
    assert result[0] == 'ok'
    row = conn.execute(
        "SELECT access_count FROM nodes WHERE id = 'n1' AND user_id = 'u'"
    ).fetchone()
    assert row['access_count'] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_access_empty_ids_is_noop(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.adapters.sqlite_vec_store import make_update_access_fn, open_db

    conn = open_db(str(tmp_path / "test.db"))
    update_access = make_update_access_fn(conn, 'u')
    result = await update_access(())
    assert result[0] == 'ok'


def test_open_db_without_sqlite_vec_raises() -> None:
    """Without sqlite-vec, open_db raises RuntimeError with install instructions."""
    import importlib

    import src.adapters.sqlite_vec_store as m

    original = sys.modules.pop('sqlite_vec', None)
    sys.modules['sqlite_vec'] = None  # type: ignore[assignment]
    try:
        importlib.reload(m)
        with pytest.raises(RuntimeError, match="sqlite-vec not installed"):
            m.open_db(":memory:")
    finally:
        sys.modules.pop('sqlite_vec', None)
        if original is not None:
            sys.modules['sqlite_vec'] = original
        importlib.reload(m)
