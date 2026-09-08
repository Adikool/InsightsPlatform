"""Password handling, session lifecycle, and the timing-safe login path."""

from __future__ import annotations

import time

import pytest

from dataplatform import auth
from dataplatform.store import activity, db


@pytest.fixture()
def dbfile(tmp_path, monkeypatch):
    path = tmp_path / "app.db"
    db.init(path)
    # Keep workspaces inside tmp_path rather than the real ~/.insight-platform.
    monkeypatch.setattr(auth.settings, "home", tmp_path, raising=False)
    monkeypatch.setattr(auth.settings, "min_password_length", 8, raising=False)
    return path


def test_signup_then_login(dbfile):
    created = auth.create_user("Aditya", "correct horse", dbfile)
    assert created.id == 1
    signed_in = auth.verify_credentials("Aditya", "correct horse", dbfile)
    assert signed_in.id == created.id


def test_username_is_case_and_whitespace_insensitive(dbfile):
    auth.create_user("Aditya", "correct horse", dbfile)
    assert auth.verify_credentials("  aditya ", "correct horse", dbfile).id == 1
    with pytest.raises(auth.AuthError) as exc:
        auth.create_user("ADITYA", "another password", dbfile)
    assert exc.value.status == 409


def test_wrong_password_is_rejected(dbfile):
    auth.create_user("a", "correct horse", dbfile)
    with pytest.raises(auth.AuthError) as exc:
        auth.verify_credentials("a", "wrong horse", dbfile)
    assert exc.value.status == 401


def test_short_password_is_rejected(dbfile):
    with pytest.raises(auth.AuthError):
        auth.create_user("a", "short", dbfile)


def test_unknown_user_costs_the_same_as_a_wrong_password(dbfile):
    """An early return on a missing user leaks which usernames exist."""
    auth.create_user("real", "correct horse", dbfile)

    def timed(username):
        start = time.perf_counter()
        with pytest.raises(auth.AuthError):
            auth.verify_credentials(username, "wrong horse", dbfile)
        return time.perf_counter() - start

    # min-of-N, because only the floor is meaningful: any sample can be inflated
    # by scheduling noise, but none can run faster than the work actually done.
    known = min(timed("real") for _ in range(5))
    unknown = min(timed("ghost") for _ in range(5))
    ratio = max(known, unknown) / max(min(known, unknown), 1e-9)

    # The bug this guards against is an early return that skips the 600k-round
    # hash entirely - that shows up as orders of magnitude, not a few percent.
    # A tight bound here would buy no extra detection and would flake whenever
    # the machine is loaded.
    assert ratio < 4.0, f"timing gap leaks user existence (known={known:.3f}s unknown={unknown:.3f}s)"


def test_session_round_trip(dbfile):
    user = auth.create_user("a", "correct horse", dbfile)
    token = auth.create_session(user.id, dbfile)
    assert auth.resolve_session(token, dbfile).id == user.id

    auth.delete_session(token, dbfile)
    assert auth.resolve_session(token, dbfile) is None
    with db.connect(dbfile) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"] == 0


def test_expired_session_is_rejected(dbfile):
    user = auth.create_user("a", "correct horse", dbfile)
    token = auth.create_session(user.id, dbfile)
    with db.write_txn(dbfile) as conn:
        conn.execute(
            "UPDATE sessions SET expires_at = ? WHERE token = ?",
            ("2000-01-01T00:00:00.000+00:00", token),
        )
    assert auth.resolve_session(token, dbfile) is None


def test_no_session_resolves_to_nobody(dbfile):
    assert auth.resolve_session(None, dbfile) is None
    assert auth.resolve_session("made-up", dbfile) is None


def test_throttle_locks_out_after_repeated_failures():
    t = auth._Throttle(limit=3, window=60.0)
    for _ in range(3):
        t.check("ip:1.2.3.4")
        t.record_failure("ip:1.2.3.4")
    with pytest.raises(auth.AuthError) as exc:
        t.check("ip:1.2.3.4")
    assert exc.value.status == 429
    t.clear("ip:1.2.3.4")
    t.check("ip:1.2.3.4")  # cleared on success


# --------------------------------------------------------------- workspaces
def test_first_user_adopts_the_existing_workspace(dbfile, tmp_path):
    user = auth.create_user("owner", "correct horse", dbfile)
    assert user.workspace_dir == str(tmp_path), "first user keeps the pre-auth catalog"
    assert user.warehouse_schema is None


def test_later_users_get_private_workspaces(dbfile, tmp_path, monkeypatch):
    monkeypatch.setattr(auth.settings, "warehouse_uri", "postgresql+psycopg://x/y", raising=False)
    auth.create_user("owner", "correct horse", dbfile)
    second = auth.create_user("second", "correct horse", dbfile)
    assert second.workspace_dir != str(tmp_path)
    assert second.warehouse_schema == f"u{second.id}"


def test_duckdb_users_get_a_directory_not_a_schema(dbfile, monkeypatch):
    monkeypatch.setattr(auth.settings, "warehouse_uri", "duckdb:///x.duckdb", raising=False)
    auth.create_user("owner", "correct horse", dbfile)
    second = auth.create_user("second", "correct horse", dbfile)
    assert second.warehouse_schema is None


def test_first_signup_adopts_pre_auth_activity(dbfile, tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"explore_activity": [{"question": "legacy"}]}', encoding="utf-8")
    activity.import_pre_auth(catalog, dbfile)

    user = auth.create_user("owner", "correct horse", dbfile)
    rows = activity.list_explore(user.id, dbfile)
    assert [r.question for r in rows] == ["legacy"]


def test_second_user_does_not_inherit_pre_auth_activity(dbfile, tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"explore_activity": [{"question": "legacy"}]}', encoding="utf-8")
    activity.import_pre_auth(catalog, dbfile)

    auth.create_user("owner", "correct horse", dbfile)
    second = auth.create_user("second", "correct horse", dbfile)
    assert activity.list_explore(second.id, dbfile) == []
