"""Bring-your-own-key: storage, isolation, and the effect on the model layer."""

from __future__ import annotations

import pytest

from dataplatform import auth
from dataplatform.nlp import llm
from dataplatform.store import db


@pytest.fixture()
def dbfile(tmp_path, monkeypatch):
    path = tmp_path / "app.db"
    db.init(path)
    monkeypatch.setattr(auth.settings, "home", tmp_path, raising=False)
    monkeypatch.setattr(auth.settings, "min_password_length", 8, raising=False)
    llm.reset_auth_failure()
    return path


def test_key_round_trips_and_is_scoped_to_one_user(dbfile):
    a = auth.create_user("a", "test-password", dbfile)
    b = auth.create_user("b", "test-password", dbfile)

    auth.set_api_key(a.id, "sk-ant-aaaaaaaaaaaa", dbfile)
    assert auth.get_api_key(a.id, dbfile) == "sk-ant-aaaaaaaaaaaa"
    assert auth.get_api_key(b.id, dbfile) is None, "one user's key must not reach another"
    assert auth.get_user(a.id, dbfile).api_key == "sk-ant-aaaaaaaaaaaa"


def test_key_can_be_replaced_and_cleared(dbfile):
    user = auth.create_user("a", "test-password", dbfile)
    auth.set_api_key(user.id, "sk-ant-first-key", dbfile)
    auth.set_api_key(user.id, "sk-ant-second-key", dbfile)
    assert auth.get_api_key(user.id, dbfile) == "sk-ant-second-key"

    auth.set_api_key(user.id, None, dbfile)
    assert auth.get_api_key(user.id, dbfile) is None


def test_obvious_nonsense_is_rejected(dbfile):
    user = auth.create_user("a", "test-password", dbfile)
    with pytest.raises(auth.AuthError):
        auth.set_api_key(user.id, "abc", dbfile)


def test_mask_never_reveals_the_key():
    masked = auth.mask_api_key("sk-ant-supersecret-tail")
    assert masked == "...tail"
    assert "supersecret" not in masked
    assert auth.mask_api_key(None) is None


def test_migration_adds_the_column_to_an_existing_database(tmp_path):
    """app.db predates this column, so init() must add it in place."""
    path = tmp_path / "app.db"
    db.init(path)
    with db.write_txn(path) as conn:
        conn.execute("ALTER TABLE users DROP COLUMN anthropic_api_key")
    with db.connect(path) as conn:
        before = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    assert "anthropic_api_key" not in before

    db.init(path)  # rerun, as a restart would
    with db.connect(path) as conn:
        after = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    assert "anthropic_api_key" in after


# ------------------------------------------------------------- model layer
def test_a_users_key_makes_the_model_layer_available(monkeypatch):
    """Availability is per user: their key counts even with no server key."""
    # has_llm is a property that reads the environment at call time.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert llm.available(None) is False
    assert llm.available("sk-ant-something") is True


def test_auth_failure_is_latched_per_key_not_globally():
    """One person's rejected key must not disable the model layer for everyone.

    The latch exists to stop re-uploading a large prompt for a call that cannot
    succeed - but keyed globally it would punish every other user.
    """
    llm.reset_auth_failure()
    llm._latch_auth_failure("sk-ant-broken", "rejected")

    assert llm.disabled_reason("sk-ant-broken") == "rejected"
    assert llm.disabled_reason("sk-ant-working") is None
    assert llm.disabled_reason(None) is None, "the server's env key is unaffected"

    llm.reset_auth_failure("sk-ant-broken")
    assert llm.disabled_reason("sk-ant-broken") is None


def test_client_prefers_the_users_key_over_the_environment(monkeypatch):
    captured = {}

    class _Fake:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    import types

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _Fake
    fake_mod.APIError = type("APIError", (Exception,), {})
    fake_mod.AuthenticationError = type("AuthenticationError", (fake_mod.APIError,), {})
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    llm.LLMClient(api_key="sk-ant-mine").client
    assert captured.get("api_key") == "sk-ant-mine"
