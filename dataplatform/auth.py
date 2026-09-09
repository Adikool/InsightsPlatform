"""Users, passwords and sessions.

Standard library only - no passlib, no JWT. Sessions are opaque random tokens
stored server-side, so there is no signing key to manage or leak: the cookie
value *is* the secret, and it is only meaningful next to its `sessions` row.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import settings
from .store import activity
from .store.db import connect, write_txn

# OWASP's floor for PBKDF2-HMAC-SHA256. Costly on purpose; see _throttle for
# why that cost also has to be rate-limited on unauthenticated routes.
_ITERATIONS = 600_000
_SALT_BYTES = 16

SESSION_COOKIE = "insight_session"

# A session row is only rewritten when it is this stale, so that a sliding
# expiry does not turn every single request into a database write.
_RENEW_AFTER = timedelta(hours=1)


class AuthError(Exception):
    """Bad credentials, a taken username, or too many attempts."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class User:
    id: int
    username: str
    workspace_dir: str
    warehouse_schema: str | None
    # The user's own model key, if they have supplied one. None means the
    # server's environment key is used instead.
    api_key: str | None = None


# ------------------------------------------------------------------ helpers
def normalize_username(username: str) -> str:
    """Fold to a single comparable form so Alice and alice cannot both exist."""
    return unicodedata.normalize("NFKC", username).strip().casefold()


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS).hex()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds")


# ------------------------------------------------------------------ throttle
class _Throttle:
    """In-process failure counter for the unauthenticated auth routes.

    Two reasons this is not optional. Brute force, obviously; but also that a
    600k-iteration hash on a sync route pins an anyio threadpool worker for
    ~0.3s, and the default pool is 40 - a few hundred login attempts a second
    would starve every other route in the app.
    """

    def __init__(self, limit: int = 10, window: float = 300.0) -> None:
        self._limit = limit
        self._window = window
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self._window]
            self._hits[key] = hits
            if len(hits) >= self._limit:
                raise AuthError("too many attempts; wait a few minutes", status=429)

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._hits.setdefault(key, []).append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


throttle = _Throttle()


# ------------------------------------------------------------------ users
def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        workspace_dir=row["workspace_dir"],
        warehouse_schema=row["warehouse_schema"],
        api_key=row["anthropic_api_key"] if "anthropic_api_key" in row.keys() else None,
    )


def _workspace_for(user_id: int, is_first: bool) -> tuple[str, str | None]:
    """Where this user's catalog and warehouse tables live.

    The first user adopts the pre-auth workspace - the existing catalog.json
    and the warehouse exactly as configured - so upgrading does not appear to
    delete everything already ingested. Everyone after gets a private
    directory, and on a SQL warehouse a private schema.
    """
    if is_first:
        return str(settings.home), None
    workspace = settings.users_dir / str(user_id)
    if settings.warehouse_uri.startswith("duckdb://"):
        return str(workspace), None
    return str(workspace), f"u{user_id}"


def create_user(username: str, password: str, path: Path | None = None) -> User:
    display = unicodedata.normalize("NFKC", username).strip()
    key = normalize_username(username)
    if not key:
        raise AuthError("username is required")
    if len(password) < settings.min_password_length:
        raise AuthError(f"password must be at least {settings.min_password_length} characters")

    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _hash(password, salt)

    with write_txn(path) as conn:
        first = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] == 0
        try:
            cur = conn.execute(
                "INSERT INTO users (username, username_key, password_hash, password_salt,"
                " workspace_dir, warehouse_schema, created_at) VALUES (?, ?, ?, ?, '', NULL, ?)",
                (display, key, digest, salt.hex(), _iso(_now())),
            )
        except sqlite3.IntegrityError:
            # Never check-then-insert: the UNIQUE index is the only thing that
            # can decide this without a race between two simultaneous signups.
            raise AuthError("that username is taken", status=409) from None

        user_id = int(cur.lastrowid)
        workspace_dir, schema = _workspace_for(user_id, first)
        conn.execute(
            "UPDATE users SET workspace_dir = ?, warehouse_schema = ? WHERE id = ?",
            (workspace_dir, schema, user_id),
        )
        if first:
            # Same transaction as the insert, so two racing signups cannot both
            # adopt the pre-auth rows.
            activity.claim_unowned(conn, user_id)

    return User(
        id=user_id, username=display, workspace_dir=workspace_dir, warehouse_schema=schema
    )


def verify_credentials(username: str, password: str, path: Path | None = None) -> User:
    key = normalize_username(username)
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username_key = ?", (key,)).fetchone()

    if row is None:
        # Hash anyway, with the same cost. Returning early here is what turns
        # a ~300ms difference into a username oracle.
        _hash(password, b"\x00" * _SALT_BYTES)
        raise AuthError("invalid username or password", status=401)

    expected = row["password_hash"]
    actual = _hash(password, bytes.fromhex(row["password_salt"]))
    if not hmac.compare_digest(expected, actual):
        raise AuthError("invalid username or password", status=401)
    return _row_to_user(row)


def get_user(user_id: int, path: Path | None = None) -> User | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def user_count(path: Path | None = None) -> int:
    with connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])


# ---------------------------------------------------------------- sessions
def create_session(user_id: int, path: Path | None = None) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    expires = now + timedelta(days=settings.session_ttl_days)
    with write_txn(path) as conn:
        # Opportunistic cleanup, so the table cannot grow without bound.
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_iso(now),))
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, last_seen, expires_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (token, user_id, _iso(now), _iso(now), _iso(expires)),
        )
    return token


def resolve_session(token: str | None, path: Path | None = None) -> User | None:
    """Return the signed-in user, refreshing the session's sliding expiry."""
    if not token:
        return None
    now = _now()
    with connect(path) as conn:
        row = conn.execute(
            "SELECT s.token, s.last_seen, s.expires_at, u.* FROM sessions s"
            " JOIN users u ON u.id = s.user_id WHERE s.token = ?",
            (token,),
        ).fetchone()
        if row is None or row["expires_at"] <= _iso(now):
            return None
        stale = _iso(now - _RENEW_AFTER)
        if row["last_seen"] < stale:
            expires = _iso(now + timedelta(days=settings.session_ttl_days))
            conn.execute(
                "UPDATE sessions SET last_seen = ?, expires_at = ? WHERE token = ?",
                (_iso(now), expires, token),
            )
    return _row_to_user(row)


def delete_session(token: str | None, path: Path | None = None) -> None:
    if not token:
        return
    with write_txn(path) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ------------------------------------------------------------------ api keys
def mask_api_key(key: str | None) -> str | None:
    """A recognisable stub, never the key itself.

    Enough to tell one key from another when confirming what is stored, and
    useless to anyone who intercepts it.
    """
    if not key:
        return None
    tail = key[-4:] if len(key) >= 4 else ""
    return f"...{tail}"


def set_api_key(user_id: int, key: str | None, path: Path | None = None) -> None:
    """Store (or clear, with None) this user's own model key."""
    cleaned = (key or "").strip() or None
    if cleaned is not None and len(cleaned) < 8:
        raise AuthError("that does not look like an API key")
    with write_txn(path) as conn:
        conn.execute(
            "UPDATE users SET anthropic_api_key = ? WHERE id = ?", (cleaned, user_id)
        )


def get_api_key(user_id: int, path: Path | None = None) -> str | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT anthropic_api_key FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return row["anthropic_api_key"] if row else None
