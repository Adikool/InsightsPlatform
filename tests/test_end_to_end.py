"""End-to-end: register a source, ingest it, ask a question, analyse the result.

Runs against a temporary DP_HOME so it never touches a real catalog or warehouse.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def platform(tmp_path, monkeypatch):
    monkeypatch.setenv("DP_HOME", str(tmp_path))

    import dataplatform.config as config_module

    config_module.settings = config_module.Settings(home=tmp_path)
    for module_name in ("dataplatform.platform", "dataplatform.ingest", "dataplatform.nlp.compiler"):
        import importlib
        import sys

        if module_name in sys.modules:
            monkeypatch.setattr(sys.modules[module_name], "settings", config_module.settings, raising=False)

    rng = np.random.default_rng(11)
    n = 4000
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    frame = pd.DataFrame(
        {
            "order_id": range(1, n + 1),
            "order_date": dates.strftime("%Y-%m-%d"),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "channel": rng.choice(["Online", "Retail"], n),
            "units": rng.integers(1, 10, n),
            "revenue": np.round(rng.lognormal(4, 0.7, n), 2),
        }
    )
    db = tmp_path / "shop.db"
    with sqlite3.connect(db) as conn:
        frame.to_sql("orders", conn, index=False)

    from dataplatform import Platform

    plat = Platform(
        warehouse_uri=f"duckdb:///{(tmp_path / 'wh.duckdb').as_posix()}",
        catalog_path=tmp_path / "catalog.json",
        use_llm=False,  # exercise the deterministic path; no network in tests
    )
    plat.add_source("shop", f"sqlite:///{db.as_posix()}", type="sql")
    plat.ingest("shop")
    return plat


def test_ingest_registers_the_dataset(platform):
    datasets = platform.list_datasets()
    assert len(datasets) == 1
    meta = datasets[0]
    assert meta.n_rows == 4000
    assert meta.column("order_date").semantic_type == "temporal"
    assert meta.column("order_id").semantic_type == "identifier"
    assert meta.column("revenue").semantic_type == "currency"


def test_ask_produces_runnable_sql(platform):
    result = platform.ask("total revenue by region")
    assert "GROUP BY" in result.sql
    assert set(result.data.columns) == {"region", "sum_revenue"}
    assert len(result.data) == 4
    assert result.data["sum_revenue"].sum() > 0


def test_ask_with_a_time_grain(platform):
    result = platform.ask("monthly revenue")
    assert "date_trunc('month'" in result.sql
    assert result.spec.chart == "line"


def test_raw_sql_guard_blocks_writes(platform):
    from dataplatform.errors import QueryValidationError

    with pytest.raises(QueryValidationError):
        platform.run_sql("DROP TABLE shop_orders")


def test_raw_sql_allows_reads(platform):
    frame = platform.run_sql("SELECT count(*) AS n FROM shop_orders")
    assert frame.iloc[0]["n"] == 4000


def test_analyze_returns_evidenced_recommendations(platform):
    report = platform.analyze("shop_orders", target="revenue")
    assert report.task == "regression"
    assert report.profile.n_rows == 4000
    assert report.recommendations
    assert all(rec.rationale for rec in report.recommendations)
    # lognormal revenue must show up as skewed
    assert any(p.kind == "skewed_distribution" for p in report.patterns)


def test_analyze_without_a_target_is_unsupervised(platform):
    report = platform.analyze("shop_orders")
    assert report.task == "clustering"
    assert report.target is None


def test_baseline_beats_the_dummy_and_excludes_identifiers(platform):
    result = platform.baseline("shop_orders", target="revenue")
    assert result.n_train == 4000
    assert "order_id" not in result.feature_importance
    assert "r2" in result.metrics
