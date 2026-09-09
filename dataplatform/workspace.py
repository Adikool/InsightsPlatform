"""Per-user workspaces.

Each user gets their own catalog file and their own warehouse storage, so
isolation happens at the storage layer rather than by prefixing table names.
That matters: because the catalog *is* the SQL allow-list and the dataset
namespace, handing a user their own catalog automatically scopes the compiler,
the guard and the NL layer with no changes to any of them - and table names
stay unqualified (`orders`, not `u3_orders`).

    DuckDB      one warehouse file per user
    SQL/Postgres one schema per user, bound via search_path
"""

from __future__ import annotations

import threading
from pathlib import Path

from . import config
from .auth import User
from .platform import Platform

# Platform holds an open engine/connection pool and an in-memory catalog, so
# it is cached rather than rebuilt per request. Keyed by user id.
_platforms: dict[int, Platform] = {}
_lock = threading.Lock()


def workspace_dir(user: User) -> Path:
    return Path(user.workspace_dir)


def catalog_path(user: User) -> Path:
    return workspace_dir(user) / "catalog.json"


def warehouse_uri(user: User) -> str:
    """The warehouse URI for this user.

    On DuckDB, isolation is a separate file inside the workspace. On a SQL
    warehouse everyone shares the server and is separated by schema instead,
    so the URI is unchanged and `warehouse_schema` does the work.
    """
    # Read through the module rather than a `from .config import settings`
    # binding: settings is a module-level singleton that gets replaced
    # wholesale (tests do exactly that), and a captured reference would keep
    # pointing at the old object - silently opening the wrong warehouse.
    settings = config.settings
    if settings.warehouse_uri.startswith("duckdb://"):
        return f"duckdb:///{(workspace_dir(user) / 'warehouse.duckdb').as_posix()}"
    return settings.warehouse_uri


def platform_for(user: User) -> Platform:
    """The Platform bound to this user's catalog and warehouse."""
    with _lock:
        existing = _platforms.get(user.id)
        if existing is not None:
            return existing

    directory = workspace_dir(user)
    directory.mkdir(parents=True, exist_ok=True)
    built = Platform(
        warehouse_uri=warehouse_uri(user),
        catalog_path=catalog_path(user),
        warehouse_schema=user.warehouse_schema,
        api_key=user.api_key,
    )

    with _lock:
        # Another thread may have built one while this was constructing; keep
        # whichever landed first so every request shares one catalog object
        # (and therefore one lock).
        return _platforms.setdefault(user.id, built)


def forget(user_id: int) -> None:
    """Drop a cached Platform.

    Required after the model key changes: the cached Platform built its NL2SQL
    (and its LLM client) around the old key, so without this the new one would
    not take effect until the process restarted.
    """
    with _lock:
        _platforms.pop(user_id, None)


def reset_cache() -> None:
    with _lock:
        _platforms.clear()
