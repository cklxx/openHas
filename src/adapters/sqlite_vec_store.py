"""SQLite + sqlite-vec storage adapter.

Schema:
  nodes     — persisted MemoryNode rows (decay_factor applied at query time)
  edges     — typed relationships between nodes
  vec_meta  — maps vec_index rowid → (node_id, user_id) for user-scoped filtering
  vec_index — sqlite-vec virtual table for KNN search

All factory functions close over (conn, user_id) — no user_id in Protocol signatures.
"""

import json
import sqlite3
import struct
import time
from typing import Any

from src.domain_types.memory import (
    ConsolidationAction,
    Edge,
    ExecutorError,
    MemoryGraph,
    MemoryNode,
    NodeWriteError,
)
from src.domain_types.ports import (
    ConsolidationExecutorFn,
    HydrateFn,
    SearchFn,
    StoreNodeFn,
    UpdateNodeFn,
)
from src.domain_types.result import Result

_DECAY = 0.1  # decay_factor applied to superseded / decayed nodes


# ── Schema ───────────────────────────────────────────────────────────────────

def _schema(dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS nodes (
    id               TEXT NOT NULL,
    user_id          TEXT NOT NULL DEFAULT 'default',
    kind             TEXT NOT NULL,
    content          TEXT NOT NULL,
    event_time       REAL NOT NULL,
    record_time      REAL NOT NULL,
    last_accessed    REAL NOT NULL,
    permanence       TEXT NOT NULL DEFAULT 'unknown',
    access_count     INTEGER NOT NULL DEFAULT 0,
    decay_factor     REAL NOT NULL DEFAULT 1.0,
    labels           TEXT NOT NULL DEFAULT '[]',
    expansion_state  TEXT NOT NULL DEFAULT 'pending',
    PRIMARY KEY (id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_nodes_user ON nodes(user_id);
CREATE INDEX IF NOT EXISTS idx_nodes_expansion ON nodes(expansion_state, user_id);
CREATE TABLE IF NOT EXISTS edges (
    source_id  TEXT NOT NULL,
    target_id  TEXT NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'default',
    kind       TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (source_id, target_id, kind, user_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind, user_id);
CREATE TABLE IF NOT EXISTS vec_meta (
    rowid      INTEGER PRIMARY KEY,
    node_id    TEXT NOT NULL,
    user_id    TEXT NOT NULL DEFAULT 'default',
    is_primary INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_vec_meta_user ON vec_meta(user_id, node_id);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(embedding float[{dim}]);
"""


def open_db(db_path: str, dim: int = 1024) -> sqlite3.Connection:
    """Open (or create) the memory database and ensure schema exists."""
    try:
        import sqlite_vec  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("sqlite-vec not installed. Run: pip install sqlite-vec") from exc
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)  # type: ignore[reportUnknownMemberType]
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_schema(dim))
    conn.commit()
    return conn


# ── Row helpers ──────────────────────────────────────────────────────────────

def _row_to_node(row: dict[str, Any]) -> MemoryNode:
    return MemoryNode(
        id=row['id'],
        kind=row['kind'],  # type: ignore[arg-type]
        content=row['content'],
        event_time=row['event_time'],
        record_time=row['record_time'],
        last_accessed=row['last_accessed'],
        permanence=row['permanence'],  # type: ignore[arg-type]
        access_count=row['access_count'],
        labels=tuple(json.loads(row['labels'])),  # type: ignore[arg-type]
    )


def _row_to_edge(row: dict[str, Any]) -> Edge:
    return Edge(
        source_id=row['source_id'],
        target_id=row['target_id'],
        kind=row['kind'],  # type: ignore[arg-type]
        weight=row['weight'],
        created_at=row['created_at'],
    )


# ── KNN search helpers ───────────────────────────────────────────────────────

def _knn_rows(conn: sqlite3.Connection, emb_blob: bytes, n: int) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT rowid, distance FROM vec_index WHERE embedding MATCH ? AND k = ?",
        (emb_blob, n),
    ).fetchall()]


def _meta_rows(
    conn: sqlite3.Connection, rowids: list[int], user_id: str, primary_only: bool = False
) -> list[dict[str, Any]]:
    ph = ','.join('?' * len(rowids))
    clause = " AND is_primary = 1" if primary_only else ""
    return [dict(r) for r in conn.execute(
        f"SELECT rowid, node_id, is_primary FROM vec_meta"
        f" WHERE rowid IN ({ph}) AND user_id = ?{clause}",
        (*rowids, user_id),
    ).fetchall()]


def _decay_map(
    conn: sqlite3.Connection, node_ids: list[str], user_id: str
) -> dict[str, float]:
    ph = ','.join('?' * len(node_ids))
    rows = conn.execute(
        f"SELECT id, decay_factor FROM nodes WHERE id IN ({ph}) AND user_id = ?",
        (*node_ids, user_id),
    ).fetchall()
    return {r['id']: float(r['decay_factor']) for r in rows}


def _best(d: dict[str, float], nid: str, score: float) -> None:
    if score > d.get(nid, -1.0):
        d[nid] = score


def _dedup_scores(
    meta: list[dict[str, Any]],
    dist_map: dict[int, float],
    decay: dict[str, float],
) -> dict[str, float]:
    """Best score per node_id, applying decay_factor."""
    seen: dict[str, float] = {}
    for row in meta:
        nid: str = row['node_id']
        cosine = max(1.0 - dist_map[int(row['rowid'])] ** 2 / 2.0, 0.0)
        score = cosine * decay.get(nid, 1.0)
        _best(seen, nid, score)
    return seen


# ── Consolidation helper ─────────────────────────────────────────────────────

def _apply_action(
    conn: sqlite3.Connection, action: ConsolidationAction, user_id: str
) -> int:
    if action.action in ('decay', 'supersede'):
        conn.executemany(
            "UPDATE nodes SET decay_factor = ? WHERE id = ? AND user_id = ?",
            [(_DECAY, nid, user_id) for nid in action.node_ids],
        )
        return len(action.node_ids)
    if action.action == 'remove':
        conn.executemany(
            "DELETE FROM nodes WHERE id = ? AND user_id = ?",
            [(nid, user_id) for nid in action.node_ids],
        )
        return len(action.node_ids)
    return 0


# ── Public factories ─────────────────────────────────────────────────────────

def make_store_fn(conn: sqlite3.Connection, user_id: str) -> StoreNodeFn:
    """Return a StoreNodeFn scoped to user_id."""

    async def store(node: MemoryNode) -> Result[None, NodeWriteError]:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO nodes "
                "(id, user_id, kind, content, event_time, record_time, "
                "last_accessed, permanence, access_count, labels) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (node.id, user_id, node.kind, node.content,
                 node.event_time, node.record_time, node.last_accessed,
                 node.permanence, node.access_count, json.dumps(list(node.labels))),
            )
            if node.embedding:
                conn.execute(
                    "INSERT INTO vec_index(embedding) VALUES (?)",
                    (struct.pack(f'{len(node.embedding)}f', *node.embedding),),
                )
                conn.execute(
                    "INSERT INTO vec_meta(rowid, node_id, user_id) "
                    "VALUES (last_insert_rowid(), ?, ?)",
                    (node.id, user_id),
                )
            conn.commit()
        except sqlite3.OperationalError as exc:
            return ('err', NodeWriteError(code='WRITE_FAILED', detail=str(exc)))
        return ('ok', None)

    return store  # type: ignore[return-value]


def _make_search(
    conn: sqlite3.Connection, user_id: str, primary_only: bool
) -> SearchFn:
    async def search(
        embedding: tuple[float, ...], top_k: int
    ) -> list[tuple[str, float]]:
        emb_blob = struct.pack(f'{len(embedding)}f', *embedding)
        knn = _knn_rows(conn, emb_blob, top_k * 3)
        if not knn:
            return []
        dist_map: dict[int, float] = {int(r['rowid']): float(r['distance']) for r in knn}
        meta = _meta_rows(conn, list(dist_map.keys()), user_id, primary_only)
        if not meta:
            return []
        node_ids = list({str(r['node_id']) for r in meta})
        decay = _decay_map(conn, node_ids, user_id)
        scores = _dedup_scores(meta, dist_map, decay)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return search  # type: ignore[return-value]


def make_search_fn(conn: sqlite3.Connection, user_id: str) -> SearchFn:
    """Return a SearchFn scoped to user_id (blends primary + expansion scores)."""
    return _make_search(conn, user_id, primary_only=False)


def make_primary_search_fn(conn: sqlite3.Connection, user_id: str) -> SearchFn:
    """Return a SearchFn that searches only primary (doc-side) embeddings.

    Used by HyDE to avoid querying expansion (query-side) rows with a doc embedding.
    """
    return _make_search(conn, user_id, primary_only=True)


def make_hydrate_fn(conn: sqlite3.Connection, user_id: str) -> HydrateFn:
    """Return a HydrateFn scoped to user_id."""

    async def hydrate(ids: tuple[str, ...]) -> dict[str, MemoryNode]:
        if not ids:
            return {}
        ph = ','.join('?' * len(ids))
        rows = conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({ph}) AND user_id = ?",
            (*ids, user_id),
        ).fetchall()
        return {r['id']: _row_to_node(dict(r)) for r in rows}

    return hydrate  # type: ignore[return-value]


def make_executor(conn: sqlite3.Connection, user_id: str) -> ConsolidationExecutorFn:
    """Return a ConsolidationExecutorFn scoped to user_id."""

    async def execute(
        actions: tuple[ConsolidationAction, ...]
    ) -> Result[int, ExecutorError]:
        count = 0
        try:
            for action in actions:
                count += _apply_action(conn, action, user_id)
            conn.commit()
        except sqlite3.OperationalError as exc:
            return ('err', ExecutorError(code='WRITE_FAILED', detail=str(exc)))
        return ('ok', count)

    return execute  # type: ignore[return-value]


def make_update_access_fn(conn: sqlite3.Connection, user_id: str) -> UpdateNodeFn:
    """Return an UpdateNodeFn that increments access_count and sets last_accessed."""

    async def update_access(ids: tuple[str, ...]) -> Result[None, NodeWriteError]:
        if not ids:
            return ('ok', None)
        try:
            now = time.time()
            ph = ','.join('?' * len(ids))
            conn.execute(
                f"UPDATE nodes SET access_count = access_count + 1, last_accessed = ? "
                f"WHERE id IN ({ph}) AND user_id = ?",
                (now, *ids, user_id),
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            return ('err', NodeWriteError(code='WRITE_FAILED', detail=str(exc)))
        return ('ok', None)

    return update_access  # type: ignore[return-value]


def make_store_expansion_fn(
    conn: sqlite3.Connection, user_id: str
):  # type: ignore[return]
    """Return a function that stores expansion embeddings for a node.

    Inserts each embedding into vec_index + vec_meta, then marks the node
    expansion_state='expanded'. Returns Result[int, NodeWriteError] (count stored).
    """

    async def store_expansion(
        node_id: str, embeddings: list[tuple[float, ...]]
    ) -> Result[int, NodeWriteError]:
        try:
            count = 0
            for emb in embeddings:
                conn.execute(
                    "INSERT INTO vec_index(embedding) VALUES (?)",
                    (struct.pack(f'{len(emb)}f', *emb),),
                )
                conn.execute(
                    "INSERT INTO vec_meta(rowid, node_id, user_id, is_primary) "
                    "VALUES (last_insert_rowid(), ?, ?, 0)",
                    (node_id, user_id),
                )
                count += 1
            conn.execute(
                "UPDATE nodes SET expansion_state = 'expanded' "
                "WHERE id = ? AND user_id = ?",
                (node_id, user_id),
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            return ('err', NodeWriteError(code='WRITE_FAILED', detail=str(exc)))
        return ('ok', count)

    return store_expansion


def make_list_pending_fn(
    conn: sqlite3.Connection, user_id: str
):  # type: ignore[return]
    """Return a function that lists nodes with expansion_state='pending'."""

    async def list_pending() -> list[MemoryNode]:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE expansion_state = 'pending' AND user_id = ?",
            (user_id,),
        ).fetchall()
        return [_row_to_node(dict(r)) for r in rows]

    return list_pending


def load_graph(conn: sqlite3.Connection, user_id: str) -> MemoryGraph:
    """Load the full MemoryGraph for a user (for consolidation)."""
    node_rows = [dict(r) for r in conn.execute(
        "SELECT * FROM nodes WHERE user_id = ?", (user_id,)
    ).fetchall()]
    edge_rows = [dict(r) for r in conn.execute(
        "SELECT * FROM edges WHERE user_id = ?", (user_id,)
    ).fetchall()]
    return MemoryGraph(
        nodes=tuple(_row_to_node(r) for r in node_rows),
        edges=tuple(_row_to_edge(r) for r in edge_rows),
    )
