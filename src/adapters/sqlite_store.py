"""SQLite adapter — mutable IO boundary."""

import sqlite3

from src.domain_types.ports import QueryFn


def make_sqlite_query(db_path: str) -> QueryFn:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    async def query(sql: str, params: tuple[object, ...]) -> list[dict[str, object]]:
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    return query  # type: ignore[return-value]
