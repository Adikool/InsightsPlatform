"""The auth gate, checked against the real route table.

The middleware is easy to get subtly wrong, and a route added later that
quietly defaults to public would be invisible. So rather than testing the
allowlist in isolation, these tests enumerate `app.routes` and assert the
actual behaviour of every path.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# Reachable without a session, by design: the login page itself, the assets it
# needs to render, and the auth endpoints used to obtain a session.
PUBLIC = {"/", "/health", "/favicon.ico", "/static", "/auth/signup", "/auth/login", "/auth/logout", "/auth/me"}


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("DP_HOME", str(tmp_path))
    import dataplatform.config as config_module

    # Pin the warehouse to a throwaway DuckDB file. Left blank, Settings falls
    # back to DP_WAREHOUSE_URI from the developer's .env - so these tests would
    # open real connections to whatever database that points at, creating
    # schemas in it, and hanging for minutes when it is unreachable.
    config_module.settings = config_module.Settings(
        home=tmp_path,
        warehouse_uri=f"duckdb:///{(tmp_path / 'wh.duckdb').as_posix()}",
    )

    from dataplatform.api import main as api_main
    from dataplatform.store import db as store_db

    monkeypatch.setattr(api_main, "settings", config_module.settings)
    monkeypatch.setattr(store_db.settings, "home", tmp_path, raising=False)
    monkeypatch.setattr(api_main.auth.settings, "home", tmp_path, raising=False)
    store_db.init()
    api_main._platform = None

    from dataplatform import workspace

    # Keyed by user id, which every test restarts from 1 - without this a test
    # inherits the previous one's Platform, pointed at a deleted tmp_path.
    workspace.reset_cache()
    yield api_main
    workspace.reset_cache()


def _sample_paths(app):
    """Every registered path, with a concrete value for any path parameter."""
    seen = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or not methods:
            continue
        concrete = path.replace("{name}", "anything")
        for method in sorted(methods - {"HEAD", "OPTIONS"}):
            seen.append((method, path, concrete))
    return seen


def test_every_non_public_route_requires_a_session(app_module):
    client = TestClient(app_module.app)
    unguarded = []
    for method, declared, concrete in _sample_paths(app_module.app):
        if declared in PUBLIC or declared.startswith("/static"):
            continue
        response = client.request(method, concrete, json={})
        if response.status_code != 401:
            unguarded.append(f"{method} {declared} -> {response.status_code}")
    assert not unguarded, "these routes answered without a session: " + ", ".join(unguarded)


def test_public_routes_stay_reachable(app_module):
    client = TestClient(app_module.app)
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/auth/me").status_code == 401  # reachable, but says no


def test_health_does_not_leak_a_dataset_count(app_module):
    """It answers anonymously, so it must not describe the data."""
    body = TestClient(app_module.app).get("/health").json()
    assert "datasets" not in body


def test_signup_then_reach_a_guarded_route(app_module):
    client = TestClient(app_module.app)
    assert client.get("/datasets").status_code == 401
    assert client.post(
        "/auth/signup", json={"username": "a", "password": "test-password"}
    ).status_code == 200
    assert client.get("/datasets").status_code == 200


def test_logout_revokes_access(app_module):
    client = TestClient(app_module.app)
    client.post("/auth/signup", json={"username": "a", "password": "test-password"})
    assert client.get("/datasets").status_code == 200
    client.post("/auth/logout")
    assert client.get("/datasets").status_code == 401


def test_login_rejects_a_bad_password(app_module):
    client = TestClient(app_module.app)
    client.post("/auth/signup", json={"username": "a", "password": "test-password"})
    client.post("/auth/logout")
    bad = client.post("/auth/login", json={"username": "a", "password": "wrong-password"})
    assert bad.status_code == 401
    assert client.get("/datasets").status_code == 401


def test_duplicate_signup_is_rejected(app_module):
    client = TestClient(app_module.app)
    client.post("/auth/signup", json={"username": "a", "password": "test-password"})
    again = client.post("/auth/signup", json={"username": "a", "password": "test-password"})
    assert again.status_code == 409


def test_two_users_have_separate_activity(app_module):
    from dataplatform.catalog.models import ExploreActivityEntry
    from dataplatform.store import activity

    client = TestClient(app_module.app)
    client.post("/auth/signup", json={"username": "one", "password": "test-password"})
    activity.add_explore(1, ExploreActivityEntry(question="first user's question"))
    assert len(client.get("/explore-activity").json()) == 1

    other = TestClient(app_module.app)
    other.post("/auth/signup", json={"username": "two", "password": "test-password"})
    assert other.get("/explore-activity").json() == []


def test_two_users_have_separate_datasets(app_module):
    """The isolation that matters: user two must not see user one's tables."""
    client = TestClient(app_module.app)
    client.post("/auth/signup", json={"username": "one", "password": "test-password"})
    other = TestClient(app_module.app)
    other.post("/auth/signup", json={"username": "two", "password": "test-password"})

    assert client.get("/datasets").json() == []
    assert other.get("/datasets").json() == []
    # Distinct workspaces on disk is what enforces it.
    from dataplatform import auth

    assert auth.get_user(1).workspace_dir != auth.get_user(2).workspace_dir
