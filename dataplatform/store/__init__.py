"""SQLite-backed storage for users, sessions and per-user activity logs.

Separate from `catalog/`, which stays JSON-backed and holds one user's
sources/datasets/metrics. This package holds the things that must be safe
under concurrent writes and scoped to a user id.
"""

from .db import connect, db_path, init, write_txn

__all__ = ["connect", "db_path", "init", "write_txn"]
