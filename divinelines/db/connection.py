"""SQLite access layer.

SQLite is a fine store for this workload, but it has two sharp edges: writer
contention and silent type coercion.  This module handles both:

* WAL journalling plus a process-wide write lock, so concurrent refresh jobs
  serialise their writes instead of raising ``database is locked``.
* A single ``connect()`` entry point that always applies the same PRAGMAs.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import settings
from ..logging_setup import get_logger

log = get_logger(__name__)

_WRITE_LOCK = threading.RLock()
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 2


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")


@contextmanager
def connect(db_path: str | Path | None = None, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Yield a configured connection.  Commits on success, rolls back on error."""
    path = Path(db_path or settings.paths.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def write_connection(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Serialised write access.  Use for every mutating operation."""
    with _WRITE_LOCK:
        with connect(db_path) as conn:
            yield conn


def init_db(db_path: str | Path | None = None) -> None:
    """Create the schema if absent.  Safe to call repeatedly."""
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with write_connection(db_path) as conn:
        conn.executescript(sql)
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
            (SCHEMA_VERSION,),
        )
    log.debug("schema ensured", extra={"db": str(db_path or settings.paths.db_path)})


def query_df(sql: str, params: Sequence[Any] | dict[str, Any] = (), db_path: str | Path | None = None) -> pd.DataFrame:
    with connect(db_path, readonly=True) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def fetch_all(sql: str, params: Sequence[Any] = (), db_path: str | Path | None = None) -> list[sqlite3.Row]:
    with connect(db_path, readonly=True) as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params: Sequence[Any] = (), db_path: str | Path | None = None) -> sqlite3.Row | None:
    with connect(db_path, readonly=True) as conn:
        return conn.execute(sql, params).fetchone()


def executemany(sql: str, rows: Iterable[Sequence[Any]], db_path: str | Path | None = None) -> int:
    """Batched write inside one transaction.  Returns the affected row count."""
    rows = list(rows)
    if not rows:
        return 0
    with write_connection(db_path) as conn:
        cur = conn.executemany(sql, rows)
        return cur.rowcount


def upsert_rows(
    table: str,
    rows: Sequence[dict[str, Any]],
    *,
    conflict_columns: Sequence[str] | None = None,
    update: bool = True,
    db_path: str | Path | None = None,
) -> int:
    """Insert dictionaries into ``table``.

    With ``conflict_columns`` an UPSERT is issued so re-running a refresh
    updates rather than duplicating.  Without them, conflicts are ignored.
    """
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)

    if conflict_columns and update:
        assignments = ", ".join(
            f"{c}=excluded.{c}" for c in columns if c not in conflict_columns
        )
        suffix = (
            f" ON CONFLICT({', '.join(conflict_columns)}) DO UPDATE SET {assignments}"
            if assignments
            else f" ON CONFLICT({', '.join(conflict_columns)}) DO NOTHING"
        )
        sql = f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders}){suffix}"
    else:
        sql = f"INSERT OR IGNORE INTO {table} ({column_sql}) VALUES ({placeholders})"

    payload = [tuple(row.get(col) for col in columns) for row in rows]
    with write_connection(db_path) as conn:
        conn.executemany(sql, payload)
    return len(payload)


def table_exists(name: str, db_path: str | Path | None = None) -> bool:
    row = fetch_one(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
        db_path,
    )
    return row is not None
