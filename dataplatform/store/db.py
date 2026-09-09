"""SQLite connection handling and schema.

Why SQLite rather than the JSON catalog for this data: the activity logs and
session table are written on nearly every request from FastAPI's threadpool,
and a read-modify-rewrite-whole-file cycle loses updates under concurrency.
SQLite in WAL mode gives real transactions - concurrent readers never block,
and a single writer is serialised by the database rather than by luck.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ..config import settings

# Applied to every connection. WAL itself is persistent (stored in the db
# header, set once at init); these are not - they reset with each connection.
_BUSY_TIMEOUT_MS = 5000


def db_path(path: Path | None = None) -> Path:
    """Resolve the database file.

    Read from `settings` lazily on every call rather than captured at import:
    the tests replace `config.settings` wholesale *after* importing this
    module, and an import-time read would send them at the real
    ~/.insight-platform/app.db.
    """
    return Path(path) if path is not None else Path(settings.db_path)


@contextmanager
def connect(path: Path | None = None):
    """A short-lived connection, one per operation.

    Deliberately not shared across requests: a connection carries transaction
    state, so sharing one would let concurrent requests see each other's
    half-finished writes. Per-operation connections also sidestep sqlite3's
    thread-affinity check entirely, since each is created and used on the
    same threadpool thread.
    """
    target = db_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(target),
        timeout=_BUSY_TIMEOUT_MS / 1000,
        # None = autocommit; we open transactions explicitly with BEGIN
        # IMMEDIATE where we need them. Python's legacy default ("") opens a
        # deferred transaction implicitly and does not autocommit, which
        # silently swallows writes when a commit is forgotten.
        isolation_level=None,
    )
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        # Per-connection and OFF by default - without this every FOREIGN KEY
        # in the schema below is inert.
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()


@contextmanager
def write_txn(path: Path | None = None):
    """A write transaction, opened with BEGIN IMMEDIATE.

    IMMEDIATE takes the write lock up front. A deferred transaction (the
    default) that reads and then writes can fail with SQLITE_BUSY at the point
    it tries to *upgrade* the lock, and busy_timeout does not retry lock
    upgrades - that is the classic "database is locked" no timeout ever fixes.
    """
    with connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL,
    -- normalised (NFKC + casefold) form, so Alice and alice cannot both exist
    username_key  TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    -- per-user workspace: catalog file lives here, and (DuckDB) the warehouse
    workspace_dir TEXT NOT NULL,
    -- per-user schema for SQL warehouses; NULL for DuckDB or the legacy default
    warehouse_schema TEXT,
    -- The user's own Anthropic key ("bring your own key"). NULL means fall
    -- back to the server's environment, so a single-user install keeps working
    -- exactly as before.
    anthropic_api_key TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    last_seen  TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- Activity logs. user_id is nullable ONLY so pre-auth rows captured at first
-- init can sit unclaimed until the first signup adopts them.
CREATE TABLE IF NOT EXISTS explore_activity (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    question      TEXT NOT NULL,
    mode          TEXT NOT NULL DEFAULT 'spec',
    sql           TEXT,
    row_count     INTEGER,
    published     INTEGER NOT NULL DEFAULT 0,
    dashboard_url TEXT,
    chart_url     TEXT,
    created_at    TEXT NOT NULL,
    dedupe_key    TEXT NOT NULL,
    UNIQUE(user_id, dedupe_key)
);

CREATE TABLE IF NOT EXISTS ask_activity (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    dataset    TEXT,
    created_at TEXT NOT NULL,
    dedupe_key TEXT NOT NULL,
    UNIQUE(user_id, dedupe_key)
);

CREATE TABLE IF NOT EXISTS dashboard_activity (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    action        TEXT NOT NULL,
    title         TEXT NOT NULL,
    request       TEXT NOT NULL DEFAULT '',
    datasets      TEXT NOT NULL DEFAULT '[]',
    dashboard_url TEXT,
    n_charts      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    dedupe_key    TEXT NOT NULL,
    UNIQUE(user_id, dedupe_key)
);

-- No dedupe_key and no cap: this one is append-only, matching the behaviour
-- the JSON catalog had. Every publish is its own historical record.
CREATE TABLE IF NOT EXISTS dashboard_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title         TEXT NOT NULL,
    dashboard_url TEXT,
    chart_url     TEXT,
    datasets      TEXT NOT NULL DEFAULT '[]',
    n_charts      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
"""


# (column, DDL) pairs applied to `users` if the column is missing. CREATE TABLE
# IF NOT EXISTS does nothing to a table that already exists, so a database
# created before a column was added needs this.
_USER_COLUMNS = (("anthropic_api_key", "ALTER TABLE users ADD COLUMN anthropic_api_key TEXT"),)


def init(path: Path | None = None) -> None:
    """Create the schema if absent. Safe to call on every startup."""
    with connect(path) as conn:
        # Persistent - stored in the database header, so this only has to
        # take effect once, but setting it again is harmless.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)

        have = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        for column, ddl in _USER_COLUMNS:
            if column not in have:
                conn.execute(ddl)


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM schema_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
