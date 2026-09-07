"""Activity store: dedupe, capping, ordering, and the pre-auth import."""

from __future__ import annotations

import json

import pytest

from dataplatform.catalog.models import (
    AskActivityEntry,
    DashboardActivityEntry,
    DashboardHistoryEntry,
    ExploreActivityEntry,
)
from dataplatform.store import activity, db


@pytest.fixture()
def dbfile(tmp_path):
    path = tmp_path / "app.db"
    db.init(path)
    # A user row must exist: the activity tables carry a FOREIGN KEY, and
    # PRAGMA foreign_keys is ON, so an orphan user_id would be rejected.
    with db.write_txn(path) as conn:
        conn.execute(
            "INSERT INTO users (id, username, username_key, password_hash, password_salt,"
            " workspace_dir, created_at) VALUES (1,'a','a','h','s','ws','now')"
        )
        conn.execute(
            "INSERT INTO users (id, username, username_key, password_hash, password_salt,"
            " workspace_dir, created_at) VALUES (2,'b','b','h','s','ws','now')"
        )
    return path


def test_rerunning_a_question_updates_in_place(dbfile):
    activity.add_explore(
        1, ExploreActivityEntry(question="revenue by region", sql="SELECT 1", row_count=5), dbfile
    )
    activity.add_explore(
        1,
        ExploreActivityEntry(
            question="revenue by region", sql="SELECT 2", row_count=9, published=True
        ),
        dbfile,
    )
    rows = activity.list_explore(1, dbfile)
    assert len(rows) == 1, "same question must not create a second row"
    # Every column refreshes, not just the timestamp.
    assert rows[0].sql == "SELECT 2"
    assert rows[0].row_count == 9
    assert rows[0].published is True


def test_rerun_bumps_to_the_top(dbfile):
    activity.add_explore(1, ExploreActivityEntry(question="first"), dbfile)
    activity.add_explore(1, ExploreActivityEntry(question="second"), dbfile)
    assert [r.question for r in activity.list_explore(1, dbfile)] == ["second", "first"]

    # The UPSERT keeps the original row id, so this only works if ordering is
    # driven by a timestamp fine-grained enough not to tie.
    activity.add_explore(1, ExploreActivityEntry(question="first"), dbfile)
    assert [r.question for r in activity.list_explore(1, dbfile)] == ["first", "second"]


def test_distinct_questions_stay_separate(dbfile):
    activity.add_explore(1, ExploreActivityEntry(question="one"), dbfile)
    activity.add_explore(1, ExploreActivityEntry(question="two"), dbfile)
    assert len(activity.list_explore(1, dbfile)) == 2


def test_ask_dedupes_on_question_and_dataset(dbfile):
    activity.add_ask(1, AskActivityEntry(question="total?", dataset="orders"), dbfile)
    activity.add_ask(1, AskActivityEntry(question="total?", dataset="orders"), dbfile)
    activity.add_ask(1, AskActivityEntry(question="total?", dataset="customers"), dbfile)
    rows = activity.list_ask(1, dbfile)
    assert len(rows) == 2, "same question against a different dataset is a distinct entry"


def test_dashboard_dedupes_on_title_request_and_datasets(dbfile):
    def entry(**kw):
        base = dict(action="preview", title="Sales", request="3 kpi", datasets=["orders"])
        base.update(kw)
        return DashboardActivityEntry(**base)

    activity.add_dashboard(1, entry(), dbfile)
    activity.add_dashboard(1, entry(action="publish", n_charts=6), dbfile)
    rows = activity.list_dashboard(1, dbfile)
    assert len(rows) == 1
    assert rows[0].action == "publish", "a later publish must supersede the preview"
    assert rows[0].n_charts == 6

    activity.add_dashboard(1, entry(request="weekly trend"), dbfile)
    assert len(activity.list_dashboard(1, dbfile)) == 2


def test_cap_keeps_the_newest_and_drops_the_oldest(dbfile):
    for i in range(activity.MAX_ENTRIES + 5):
        activity.add_explore(1, ExploreActivityEntry(question=f"q{i:04d}"), dbfile)
    rows = activity.list_explore(1, dbfile)
    assert len(rows) == activity.MAX_ENTRIES
    questions = {r.question for r in rows}
    assert "q0204" in questions, "newest must survive"
    for dropped in range(5):
        assert f"q{dropped:04d}" not in questions, "oldest must be culled"


def test_history_is_append_only_and_uncapped(dbfile):
    for _ in range(activity.MAX_ENTRIES + 3):
        activity.add_history(1, DashboardHistoryEntry(title="same", datasets=["orders"]), dbfile)
    rows = activity.list_history(1, dbfile)
    assert len(rows) == activity.MAX_ENTRIES + 3, "history keeps duplicates and is not capped"


def test_users_cannot_see_each_others_activity(dbfile):
    activity.add_explore(1, ExploreActivityEntry(question="mine"), dbfile)
    activity.add_explore(2, ExploreActivityEntry(question="theirs"), dbfile)
    assert [r.question for r in activity.list_explore(1, dbfile)] == ["mine"]
    assert [r.question for r in activity.list_explore(2, dbfile)] == ["theirs"]


def test_clear_only_affects_one_user(dbfile):
    activity.add_explore(1, ExploreActivityEntry(question="mine"), dbfile)
    activity.add_explore(2, ExploreActivityEntry(question="theirs"), dbfile)
    activity.clear(1, "explore_activity", dbfile)
    assert activity.list_explore(1, dbfile) == []
    assert len(activity.list_explore(2, dbfile)) == 1


# ------------------------------------------------------------- migration
def _legacy_catalog(tmp_path, **lists):
    path = tmp_path / "catalog.json"
    payload = {"sources": {}, "datasets": {}, "metrics": {}}
    payload.update(lists)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_import_preserves_order_and_is_claimed_by_first_user(tmp_path, dbfile):
    catalog = _legacy_catalog(
        tmp_path,
        explore_activity=[
            {"question": "newest", "created_at": "2026-01-03T00:00:00+00:00"},
            {"question": "middle", "created_at": "2026-01-02T00:00:00+00:00"},
            {"question": "oldest", "created_at": "2026-01-01T00:00:00+00:00"},
        ],
        dashboard_history=[{"title": "H1"}],
    )
    assert activity.import_pre_auth(catalog, dbfile) == 4

    # Unowned until someone signs up.
    assert activity.list_explore(1, dbfile) == []
    with db.write_txn(dbfile) as conn:
        activity.claim_unowned(conn, 1)

    assert [r.question for r in activity.list_explore(1, dbfile)] == ["newest", "middle", "oldest"]
    assert [r.title for r in activity.list_history(1, dbfile)] == ["H1"]


def test_import_backfills_missing_timestamps_newest_first(tmp_path, dbfile):
    catalog = _legacy_catalog(
        tmp_path,
        explore_activity=[{"question": "a"}, {"question": "b"}, {"question": "c"}],
    )
    activity.import_pre_auth(catalog, dbfile)
    with db.write_txn(dbfile) as conn:
        activity.claim_unowned(conn, 1)
    assert [r.question for r in activity.list_explore(1, dbfile)] == ["a", "b", "c"]


def test_import_runs_once(tmp_path, dbfile):
    catalog = _legacy_catalog(tmp_path, explore_activity=[{"question": "a"}])
    assert activity.import_pre_auth(catalog, dbfile) == 1
    assert activity.import_pre_auth(catalog, dbfile) == 0, "second call is a no-op"


def test_import_writes_a_backup(tmp_path, dbfile):
    catalog = _legacy_catalog(tmp_path, explore_activity=[{"question": "a"}])
    activity.import_pre_auth(catalog, dbfile)
    assert (tmp_path / "catalog.json.pre-auth.bak").exists()


def test_import_on_missing_catalog_is_a_noop(tmp_path, dbfile):
    assert activity.import_pre_auth(tmp_path / "nope.json", dbfile) == 0


def test_import_on_corrupt_catalog_is_a_noop(tmp_path, dbfile):
    bad = tmp_path / "catalog.json"
    bad.write_text("{not json", encoding="utf-8")
    # Must not raise: a malformed legacy file cannot be allowed to break startup.
    assert activity.import_pre_auth(bad, dbfile) == 0
