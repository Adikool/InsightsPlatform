"""Per-user schema binding on a SQL warehouse.

The subtle failure this guards against: `SET search_path` is transactional in
Postgres, so binding the schema with a `connect` listener meant the pool's
rollback-on-return silently reverted it. The first use of each physical
connection worked and every reuse could not find the user's tables - which
looks like data loss, not a configuration bug.
"""

from __future__ import annotations

import pytest

from dataplatform.errors import WarehouseError
from dataplatform.warehouse import open_warehouse
from dataplatform.warehouse.sql_backend import SQLWarehouse


def test_schema_is_a_connection_startup_option_not_a_set(monkeypatch):
    """Pinned structurally: a `SET` here would be undone by the pool."""
    captured = {}

    def fake_create_engine(uri, **kwargs):
        captured.update(kwargs)

        class _Engine:
            dialect = type("d", (), {"name": "postgresql"})()

            def begin(self):
                raise AssertionError("should not need a connection in this test")

        return _Engine()

    import dataplatform.warehouse.sql_backend as backend

    monkeypatch.setattr(backend, "create_engine", fake_create_engine)
    monkeypatch.setattr(SQLWarehouse, "_ensure_schema", lambda self, schema: None)

    SQLWarehouse("postgresql+psycopg://u:p@h/db", schema="u7")
    assert captured["connect_args"]["options"] == "-csearch_path=u7", (
        "the schema must be set at connection startup, where a rollback cannot revert it"
    )


def test_no_schema_leaves_the_connection_untouched(monkeypatch):
    captured = {}

    def fake_create_engine(uri, **kwargs):
        captured.update(kwargs)

        class _Engine:
            dialect = type("d", (), {"name": "postgresql"})()

        return _Engine()

    import dataplatform.warehouse.sql_backend as backend

    monkeypatch.setattr(backend, "create_engine", fake_create_engine)
    SQLWarehouse("postgresql+psycopg://u:p@h/db")
    assert captured.get("connect_args") == {}, "an unscoped warehouse must not pin a schema"


def test_schema_name_is_validated(monkeypatch):
    """The schema goes into a connection string and DDL, so it cannot be free text."""
    import dataplatform.warehouse.sql_backend as backend

    monkeypatch.setattr(backend, "create_engine", lambda *a, **k: None)
    for bad in ("u1; DROP SCHEMA public", "public schema", "u1'--"):
        with pytest.raises(WarehouseError):
            SQLWarehouse("postgresql+psycopg://u:p@h/db", schema=bad)


def test_duckdb_ignores_schema(tmp_path):
    """DuckDB isolates by file, so a schema argument must not change anything."""
    wh = open_warehouse(f"duckdb:///{(tmp_path / 'w.duckdb').as_posix()}", schema="u3")
    assert wh.dialect == "duckdb"
