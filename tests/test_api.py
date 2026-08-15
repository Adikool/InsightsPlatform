"""API surface and UI asset tests, driven through FastAPI's TestClient.

These matter because the UI is a *client* of these shapes: a route that returns
NaN or a Timestamp serialises fine in pytest and then breaks the page.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DP_HOME", str(tmp_path))

    import dataplatform.config as config_module

    config_module.settings = config_module.Settings(home=tmp_path)

    rng = np.random.default_rng(3)
    n = 900
    frame = pd.DataFrame(
        {
            "order_id": range(n),
            "order_date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "region": rng.choice(["North", "South"], n),
            "revenue": np.round(rng.lognormal(4, 0.6, n), 2),
            "cost": np.round(rng.lognormal(3, 0.6, n), 2),
        }
    )
    frame.loc[frame.index[:40], "cost"] = np.nan  # nulls must survive serialisation

    db = tmp_path / "shop.db"
    with sqlite3.connect(db) as conn:
        frame.to_sql("orders", conn, index=False)

    from dataplatform.api import main as api_main
    from dataplatform.platform import Platform

    platform = Platform(
        warehouse_uri=f"duckdb:///{(tmp_path / 'wh.duckdb').as_posix()}",
        catalog_path=tmp_path / "catalog.json",
        use_llm=False,
    )
    platform.add_source("shop", f"sqlite:///{db.as_posix()}", type="sql")
    platform.ingest("shop")

    api_main._platform = platform
    monkeypatch.setattr(api_main, "settings", config_module.settings)
    yield TestClient(api_main.app)
    api_main._platform = None


# ------------------------------------------------------------------ UI serve
def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Insight Platform" in response.text
    assert "/static/app.js" in response.text


@pytest.mark.parametrize("asset", ["app.js", "charts.js", "styles.css"])
def test_static_assets_are_served(client, asset):
    response = client.get(f"/static/{asset}")
    assert response.status_code == 200
    assert len(response.content) > 500


def test_ui_references_no_external_hosts():
    """The platform must run air-gapped, so nothing may be fetched off-box."""
    from pathlib import Path

    import dataplatform.api.main as api_main

    for path in Path(api_main.STATIC).glob("*"):
        text = path.read_text(encoding="utf-8")
        for marker in ("http://", "https://", "//cdn", "unpkg", "jsdelivr"):
            offending = [
                line
                for line in text.splitlines()
                if marker in line and "www.w3.org" not in line and "127.0.0.1" not in line
            ]
            assert not offending, f"{path.name} reaches off-box: {offending[:2]}"


# -------------------------------------------------------------------- config
def test_config_reports_the_environment(client):
    body = client.get("/config").json()
    assert "warehouse" in body
    assert body["llm_available"] in (True, False)
    assert "model" in body


def test_superset_check_never_raises(client):
    """Superset being down is a normal state, not a 500.

    Asserts the contract rather than the connectivity: whether a Superset happens
    to be running on this machine is not something the test suite should depend on
    (an earlier version of this test asserted `connected is False` and duly broke
    the moment a real Superset was started).
    """
    response = client.get("/superset/check")
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["connected"], bool)
    if body["connected"]:
        assert body["user"]
    else:
        assert body["error"]


# ------------------------------------------------------------------ payloads
def test_ask_payload_is_json_safe(client):
    response = client.post("/ask", json={"question": "total revenue by region"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "spec"
    assert body["columns"] == ["region", "sum_revenue"]
    assert len(body["rows"]) == 2
    json.dumps(body)  # nothing exotic slipped through


def test_preview_serialises_nulls_and_timestamps(client):
    body = client.get("/datasets/shop_orders/preview?limit=60").json()
    assert body["columns"]
    assert len(body["rows"]) == 60

    # NaN must arrive as null, not the bare token NaN, which JSON.parse rejects.
    raw = client.get("/datasets/shop_orders/preview?limit=60").text
    assert "NaN" not in raw
    assert any(row["cost"] is None for row in body["rows"])

    # dates land as ISO strings the UI can format
    assert isinstance(body["rows"][0]["order_date"], str)


def test_sql_route_returns_columns_for_the_table_renderer(client):
    body = client.post("/sql", json={"sql": "SELECT region, count(*) AS n FROM shop_orders GROUP BY region"}).json()
    assert body["columns"] == ["region", "n"]
    assert body["row_count"] == 2


def test_sql_route_rejects_writes_with_a_readable_message(client):
    response = client.post("/sql", json={"sql": "DROP TABLE shop_orders"})
    assert response.status_code == 400
    assert "SELECT" in response.json()["detail"]


def test_analyze_payload_matches_what_the_ui_reads(client):
    body = client.post("/analyze", json={"dataset": "shop_orders", "target": "revenue"}).json()
    assert body["task"] == "regression"
    assert {"profile", "patterns", "recommendations"} <= body.keys()
    for pattern in body["patterns"]:
        assert pattern["severity"] in ("strong", "notable", "info")
        assert pattern["kind"] and pattern["description"]
    for rec in body["recommendations"]:
        assert rec["rank"] >= 1
        assert rec["rationale"]
    json.dumps(body)


def test_unknown_dataset_is_a_404_not_a_500(client):
    assert client.get("/datasets/nope/preview").status_code == 404


def test_source_removal_clears_its_datasets(client):
    assert client.delete("/sources/shop").status_code == 200
    assert client.get("/datasets").json() == []
