"""Per-user activity logs, backed by SQLite.

Replaces the four lists that used to live in catalog.json. Two behaviours are
carried over deliberately and one is fixed:

* Explore / Ask / Dashboard logs de-duplicate: re-running the same question
  refreshes the existing row rather than adding another. Previously this was a
  read-filter-rewrite in Python, which loses updates when two requests
  interleave; here it is a single atomic UPSERT.
* dashboard_history stays append-only and uncapped - every publish is its own
  record.
* Ordering is by created_at at *millisecond* precision. The UPSERT keeps the
  original row id, so "bump to top" cannot rely on id, and second-precision
  timestamps tie for anything re-run inside the same second.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..catalog.models import (
    AskActivityEntry,
    DashboardActivityEntry,
    DashboardHistoryEntry,
    ExploreActivityEntry,
)
from .db import connect, get_meta, set_meta, write_txn

MAX_ENTRIES = 200

_DEDUPED_TABLES = ("explore_activity", "ask_activity", "dashboard_activity")


def _now() -> str:
    """UTC, millisecond precision - see the module docstring on ordering."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _key(*parts: object) -> str:
    """Stable dedupe key.

    A sha256 rather than a delimiter-joined string: a dashboard title
    containing the delimiter would otherwise collide with a different entry.
    """
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def explore_key(question: str) -> str:
    return _key("explore", question.strip().lower())


def ask_key(question: str, dataset: str | None) -> str:
    return _key("ask", question.strip().lower(), (dataset or "").strip().lower())


def dashboard_key(title: str, request: str, datasets: list[str]) -> str:
    return _key(
        "dashboard",
        title.strip().lower(),
        request.strip().lower(),
        ",".join(sorted(d.strip().lower() for d in datasets)),
    )


def _cap(conn: sqlite3.Connection, table: str, user_id: int) -> None:
    """Trim to the newest MAX_ENTRIES rows for this user.

    The subquery form is required: DELETE ... ORDER BY ... LIMIT is a syntax
    error unless SQLite was compiled with SQLITE_ENABLE_UPDATE_DELETE_LIMIT,
    which Python's bundled build is not.
    """
    conn.execute(
        f"DELETE FROM {table} WHERE user_id = ? AND id NOT IN ("
        f"  SELECT id FROM {table} WHERE user_id = ?"
        f"  ORDER BY created_at DESC, id DESC LIMIT ?)",
        (user_id, user_id, MAX_ENTRIES),
    )


# ---------------------------------------------------------------- explore
def add_explore(user_id: int, entry: ExploreActivityEntry, path: Path | None = None) -> None:
    now = _now()
    with write_txn(path) as conn:
        conn.execute(
            "INSERT INTO explore_activity "
            "  (user_id, question, mode, sql, row_count, published, dashboard_url,"
            "   chart_url, created_at, dedupe_key) "
            "VALUES (:uid, :question, :mode, :sql, :row_count, :published, :dashboard_url,"
            "        :chart_url, :created_at, :dedupe_key) "
            # Every non-key column is refreshed, not just the timestamp: a
            # re-run that failed to publish must not keep the old chart_url.
            "ON CONFLICT(user_id, dedupe_key) DO UPDATE SET "
            "  question=excluded.question, mode=excluded.mode, sql=excluded.sql,"
            "  row_count=excluded.row_count, published=excluded.published,"
            "  dashboard_url=excluded.dashboard_url, chart_url=excluded.chart_url,"
            "  created_at=excluded.created_at",
            {
                "uid": user_id,
                "question": entry.question,
                "mode": entry.mode,
                "sql": entry.sql,
                "row_count": entry.row_count,
                "published": int(bool(entry.published)),
                "dashboard_url": entry.dashboard_url,
                "chart_url": entry.chart_url,
                "created_at": now,
                "dedupe_key": explore_key(entry.question),
            },
        )
        _cap(conn, "explore_activity", user_id)


def list_explore(user_id: int, path: Path | None = None) -> list[ExploreActivityEntry]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM explore_activity WHERE user_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
    return [
        ExploreActivityEntry(
            question=r["question"],
            mode=r["mode"],
            sql=r["sql"],
            row_count=r["row_count"],
            published=bool(r["published"]),
            dashboard_url=r["dashboard_url"],
            chart_url=r["chart_url"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


# -------------------------------------------------------------------- ask
def add_ask(user_id: int, entry: AskActivityEntry, path: Path | None = None) -> None:
    now = _now()
    with write_txn(path) as conn:
        conn.execute(
            "INSERT INTO ask_activity (user_id, question, dataset, created_at, dedupe_key) "
            "VALUES (:uid, :question, :dataset, :created_at, :dedupe_key) "
            "ON CONFLICT(user_id, dedupe_key) DO UPDATE SET "
            "  question=excluded.question, dataset=excluded.dataset,"
            "  created_at=excluded.created_at",
            {
                "uid": user_id,
                "question": entry.question,
                "dataset": entry.dataset,
                "created_at": now,
                "dedupe_key": ask_key(entry.question, entry.dataset),
            },
        )
        _cap(conn, "ask_activity", user_id)


def list_ask(user_id: int, path: Path | None = None) -> list[AskActivityEntry]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM ask_activity WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
    return [
        AskActivityEntry(question=r["question"], dataset=r["dataset"], created_at=r["created_at"])
        for r in rows
    ]


# -------------------------------------------------------------- dashboard
def add_dashboard(user_id: int, entry: DashboardActivityEntry, path: Path | None = None) -> None:
    now = _now()
    with write_txn(path) as conn:
        conn.execute(
            "INSERT INTO dashboard_activity "
            "  (user_id, action, title, request, datasets, dashboard_url, n_charts,"
            "   created_at, dedupe_key) "
            "VALUES (:uid, :action, :title, :request, :datasets, :dashboard_url, :n_charts,"
            "        :created_at, :dedupe_key) "
            "ON CONFLICT(user_id, dedupe_key) DO UPDATE SET "
            "  action=excluded.action, title=excluded.title, request=excluded.request,"
            "  datasets=excluded.datasets, dashboard_url=excluded.dashboard_url,"
            "  n_charts=excluded.n_charts, created_at=excluded.created_at",
            {
                "uid": user_id,
                "action": entry.action,
                "title": entry.title,
                "request": entry.request,
                "datasets": json.dumps(list(entry.datasets)),
                "dashboard_url": entry.dashboard_url,
                "n_charts": entry.n_charts,
                "created_at": now,
                "dedupe_key": dashboard_key(entry.title, entry.request, list(entry.datasets)),
            },
        )
        _cap(conn, "dashboard_activity", user_id)


def list_dashboard(user_id: int, path: Path | None = None) -> list[DashboardActivityEntry]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM dashboard_activity WHERE user_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
    return [
        DashboardActivityEntry(
            action=r["action"],
            title=r["title"],
            request=r["request"],
            datasets=json.loads(r["datasets"] or "[]"),
            dashboard_url=r["dashboard_url"],
            n_charts=r["n_charts"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


# ------------------------------------------------------------ history
def add_history(user_id: int, entry: DashboardHistoryEntry, path: Path | None = None) -> None:
    """Append-only: no dedupe, no cap. Matches the previous JSON behaviour."""
    with write_txn(path) as conn:
        conn.execute(
            "INSERT INTO dashboard_history "
            "  (user_id, title, dashboard_url, chart_url, datasets, n_charts, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                entry.title,
                entry.dashboard_url,
                entry.chart_url,
                json.dumps(list(entry.datasets)),
                entry.n_charts,
                _now(),
            ),
        )


def list_history(user_id: int, path: Path | None = None) -> list[DashboardHistoryEntry]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM dashboard_history WHERE user_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()
    return [
        DashboardHistoryEntry(
            title=r["title"],
            dashboard_url=r["dashboard_url"],
            chart_url=r["chart_url"],
            datasets=json.loads(r["datasets"] or "[]"),
            n_charts=r["n_charts"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


# -------------------------------------------------------------- clearing
def clear(user_id: int, table: str, path: Path | None = None) -> None:
    if table not in _DEDUPED_TABLES + ("dashboard_history",):
        raise ValueError(f"unknown activity table {table!r}")
    with write_txn(path) as conn:
        conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))


# ------------------------------------------------------- pre-auth capture
_IMPORT_FLAG = "pre_auth_import"


def _synth_timestamps(count: int) -> list[str]:
    """Descending stand-ins for entries whose created_at was never recorded.

    The source lists are newest-first, so index 0 must sort newest. Empty
    strings would sort last and be culled first by the cap.
    """
    base = datetime.now(timezone.utc)
    return [
        (base - timedelta(seconds=i)).isoformat(timespec="milliseconds")
        for i in range(count)
    ]


def import_pre_auth(catalog_path: Path, path: Path | None = None) -> int:
    """Capture the activity lists from a pre-auth catalog.json.

    Runs at database init, NOT at first signup: once the four fields are
    removed from CatalogState, the very next catalog.save() drops them from
    the file, and an /ingest can trigger that long before anyone registers.
    Rows land unowned (user_id IS NULL) and are adopted by the first signup.

    Returns the number of rows captured.
    """
    catalog_path = Path(catalog_path)
    with connect(path) as conn:
        if get_meta(conn, _IMPORT_FLAG):
            return 0

    if not catalog_path.exists():
        with write_txn(path) as conn:
            set_meta(conn, _IMPORT_FLAG, "no-catalog")
        return 0

    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        # A malformed legacy file must not make startup (or a signup) fail.
        # The flag is deliberately not set, so a repaired file still imports.
        return 0

    # Keep a copy before anything can rewrite the file without these fields.
    backup = catalog_path.with_suffix(".json.pre-auth.bak")
    if not backup.exists():
        try:
            shutil.copy2(catalog_path, backup)
        except OSError:
            pass

    explore = raw.get("explore_activity") or []
    ask = raw.get("ask_activity") or []
    dash = raw.get("dashboard_activity") or []
    hist = raw.get("dashboard_history") or []
    captured = 0

    with write_txn(path) as conn:
        for src, stamps in (
            (explore, _synth_timestamps(len(explore))),
            (ask, _synth_timestamps(len(ask))),
            (dash, _synth_timestamps(len(dash))),
            (hist, _synth_timestamps(len(hist))),
        ):
            for entry, fallback in zip(src, stamps):
                entry.setdefault("created_at", "")
                if not entry["created_at"]:
                    entry["created_at"] = fallback

        for e in explore:
            conn.execute(
                "INSERT OR IGNORE INTO explore_activity "
                "  (user_id, question, mode, sql, row_count, published, dashboard_url,"
                "   chart_url, created_at, dedupe_key) "
                "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    e.get("question", ""),
                    e.get("mode", "spec"),
                    e.get("sql"),
                    e.get("row_count"),
                    int(bool(e.get("published"))),
                    e.get("dashboard_url"),
                    e.get("chart_url"),
                    e["created_at"],
                    explore_key(e.get("question", "")),
                ),
            )
            captured += 1
        for e in ask:
            conn.execute(
                "INSERT OR IGNORE INTO ask_activity "
                "  (user_id, question, dataset, created_at, dedupe_key) "
                "VALUES (NULL, ?, ?, ?, ?)",
                (
                    e.get("question", ""),
                    e.get("dataset"),
                    e["created_at"],
                    ask_key(e.get("question", ""), e.get("dataset")),
                ),
            )
            captured += 1
        for e in dash:
            datasets = e.get("datasets") or []
            conn.execute(
                "INSERT OR IGNORE INTO dashboard_activity "
                "  (user_id, action, title, request, datasets, dashboard_url, n_charts,"
                "   created_at, dedupe_key) "
                "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    e.get("action", "preview"),
                    e.get("title", ""),
                    e.get("request", ""),
                    json.dumps(datasets),
                    e.get("dashboard_url"),
                    e.get("n_charts", 0),
                    e["created_at"],
                    dashboard_key(e.get("title", ""), e.get("request", ""), datasets),
                ),
            )
            captured += 1
        for e in hist:
            conn.execute(
                "INSERT OR IGNORE INTO dashboard_history "
                "  (user_id, title, dashboard_url, chart_url, datasets, n_charts, created_at) "
                "VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                (
                    e.get("title", ""),
                    e.get("dashboard_url"),
                    e.get("chart_url"),
                    json.dumps(e.get("datasets") or []),
                    e.get("n_charts", 0),
                    e["created_at"],
                ),
            )
            captured += 1

        set_meta(conn, _IMPORT_FLAG, f"imported:{captured}")
    return captured


def claim_unowned(conn: sqlite3.Connection, user_id: int) -> None:
    """Adopt the pre-auth rows. Called inside the first signup's transaction.

    UPDATE OR IGNORE plus a sweep, so that if this ever runs for a user who
    already holds a colliding dedupe_key the unclaimable rows are discarded
    rather than left orphaned forever.
    """
    for table in _DEDUPED_TABLES + ("dashboard_history",):
        conn.execute(
            f"UPDATE OR IGNORE {table} SET user_id = ? WHERE user_id IS NULL", (user_id,)
        )
        conn.execute(f"DELETE FROM {table} WHERE user_id IS NULL")
