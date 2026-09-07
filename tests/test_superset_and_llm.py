"""Superset chart-parameter construction and LLM schema hardening.

Both are pure functions over local data, so they are tested without a Superset
server or an API key.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from dataplatform.catalog import Catalog
from dataplatform.catalog.models import CatalogState, ColumnMeta, DatasetMeta
from dataplatform.errors import PlatformError
from dataplatform.nlp import SQLCompiler, strict_schema
from dataplatform.nlp.chart import SUPERSET_VIZ, reconcile, suggest_chart
from dataplatform.nlp.spec import Dimension, Metric, QuerySpec
from dataplatform.superset.publisher import SupersetPublisher


@pytest.fixture()
def compiler(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.state = CatalogState(
        datasets={
            "orders": DatasetMeta(
                name="orders",
                source="demo",
                origin_object="orders",
                n_rows=500,
                columns=[
                    ColumnMeta(name="order_date", dtype="datetime64[ns]", semantic_type="temporal"),
                    ColumnMeta(name="region", dtype="str", semantic_type="geo"),
                    ColumnMeta(name="revenue", dtype="float64", semantic_type="currency"),
                ],
            )
        }
    )
    return SQLCompiler(catalog, dialect="duckdb")


def _publisher() -> SupersetPublisher:
    return SupersetPublisher(client=object())  # type: ignore[arg-type]  # never called


def test_timeseries_params_use_the_time_column_as_the_axis(compiler):
    spec = QuerySpec(
        dataset="orders",
        dimensions=[Dimension(column="order_date", time_grain="month"), Dimension(column="region")],
        metrics=[Metric(column="revenue", func="sum")],
        chart="line",
    )
    compiled = compiler.compile(spec)
    params = _publisher()._params(compiled, 42, SUPERSET_VIZ["line"], [])

    assert params["x_axis"] == "order_date_month"
    assert params["groupby"] == ["region"]
    assert params["time_grain_sqla"] == "P1M"
    assert params["metrics"][0]["aggregate"] == "SUM"
    assert params["datasource"] == "42__table"
    json.dumps(params)  # must survive serialisation into the chart body


def test_average_metrics_are_reaggregated_with_avg_and_flagged(compiler):
    spec = QuerySpec(
        dataset="orders",
        dimensions=[Dimension(column="region")],
        metrics=[Metric(column="revenue", func="avg")],
        chart="bar",
    )
    compiled = compiler.compile(spec)
    notes: list[str] = []
    params = _publisher()._params(compiled, 7, SUPERSET_VIZ["bar"], notes)

    assert params["metrics"][0]["aggregate"] == "AVG"
    assert any("AVG" in note for note in notes)  # the caveat is surfaced, not hidden


def test_big_number_params_carry_a_single_metric(compiler):
    spec = QuerySpec(
        dataset="orders", metrics=[Metric(column="revenue", func="sum")], chart="big_number"
    )
    compiled = compiler.compile(spec)
    params = _publisher()._params(compiled, 1, SUPERSET_VIZ["big_number"], [])
    assert params["metric"]["column"]["column_name"] == "sum_revenue"


def test_raw_table_params_when_there_is_no_metric(compiler):
    spec = QuerySpec(dataset="orders", dimensions=[Dimension(column="region")], chart="table")
    compiled = compiler.compile(spec)
    params = _publisher()._params(compiled, 1, "table", [])
    assert params["query_mode"] == "aggregate" if compiled.metric_aliases else True


# ------------------------------------------------------------------- charts
def test_chart_choice_follows_the_shape_of_the_result(compiler):
    time_spec = QuerySpec(
        dataset="orders",
        dimensions=[Dimension(column="order_date", time_grain="month")],
        metrics=[Metric(column="revenue", func="sum")],
    )
    compiled = compiler.compile(time_spec)
    meta = compiler.catalog.get_dataset("orders")
    assert suggest_chart(compiled, meta, n_rows=24) == "line"

    flat = compiler.compile(
        QuerySpec(dataset="orders", metrics=[Metric(column="revenue", func="sum")])
    )
    assert suggest_chart(flat, meta, n_rows=1) == "big_number"


def test_a_pie_chart_of_too_many_slices_is_overridden(compiler):
    spec = QuerySpec(
        dataset="orders",
        dimensions=[Dimension(column="region")],
        metrics=[Metric(column="revenue", func="sum")],
        chart="pie",
    )
    compiled = compiler.compile(spec)
    meta = compiler.catalog.get_dataset("orders")
    assert reconcile(compiled, meta, n_rows=400) != "pie"
    assert reconcile(compiled, meta, n_rows=5) == "pie"


# ---------------------------------------------------------------- llm schema
class _Nested(BaseModel):
    label: str
    weight: float = Field(default=1.0, description="ignored under strict mode")


class _Payload(BaseModel):
    name: str = Field(default="", description="a name")
    count: int = Field(default=0)
    items: list[_Nested] = Field(default_factory=list)


def test_strict_schema_closes_every_object():
    schema = strict_schema(_Payload)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node.get("properties", {}))
            assert "default" not in node
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)
    assert "$defs" in schema  # nested models stay as refs, which the API supports


def test_strict_schema_strips_unsupported_keywords():
    class Constrained(BaseModel):
        value: int = Field(ge=0, le=10)
        text: str = Field(min_length=2, max_length=5)

    dumped = json.dumps(strict_schema(Constrained))
    for keyword in ("minimum", "maximum", "minLength", "maxLength"):
        assert keyword not in dumped


def test_sql_warehouse_chunks_under_the_bind_parameter_limit():
    """Regression: a fixed chunksize with method='multi' overflows Postgres.

    `to_sql(chunksize=10_000, method="multi")` packs chunksize x n_columns bind
    parameters into one INSERT. At 15 columns that is 150,000, and the wire
    protocol caps at 65,535 — so it worked on DuckDB and on narrow tables, then
    failed on the first wide table against a real Postgres.
    """
    from dataplatform.warehouse.sql_backend import SQLWarehouse

    for n_columns in (1, 5, 15, 60, 300, 1000):
        chunk = SQLWarehouse._chunksize(SQLWarehouse, n_columns)
        assert chunk >= 1
        assert chunk * n_columns <= 65_535


def test_superset_uri_can_differ_from_our_own():
    """Superset runs elsewhere, so it may need a different address for the same DB."""
    import dataplatform.superset.publisher as publisher_module

    class FakeWarehouse:
        sqlalchemy_uri = "postgresql+psycopg://insight:insight@localhost:55432/insight"

    original = publisher_module.settings.superset_warehouse_uri
    try:
        publisher_module.settings.superset_warehouse_uri = ""
        assert SupersetPublisher.warehouse_uri_for_superset(FakeWarehouse()) == FakeWarehouse.sqlalchemy_uri

        container_uri = "postgresql+psycopg2://insight:insight@insight_warehouse:5432/insight"
        publisher_module.settings.superset_warehouse_uri = container_uri
        assert SupersetPublisher.warehouse_uri_for_superset(FakeWarehouse()) == container_uri
    finally:
        publisher_module.settings.superset_warehouse_uri = original


def test_publishing_a_local_duckdb_fails_with_the_real_reason():
    """A containerised Superset cannot open a host file path — say so up front."""
    from dataplatform.errors import SupersetError

    class DuckWarehouse:
        sqlalchemy_uri = "duckdb:///C:/Users/someone/warehouse.duckdb"

    publisher = SupersetPublisher(client=object())  # type: ignore[arg-type]
    with pytest.raises(SupersetError, match="DuckDB"):
        publisher.ensure_warehouse_database(DuckWarehouse())


def test_querySpec_survives_hardening():
    """The schema the NL layer actually sends must be well-formed."""
    schema = strict_schema(QuerySpec)
    assert schema["additionalProperties"] is False
    assert "dataset" in schema["required"]
    json.dumps(schema)


# ----------------------------------------------------------- model-call failures
# `LLMClient._create()` is the one place that actually calls the network. Every
# caller in the app catches `PlatformError` to mean "the LLM path didn't work,
# fall back to the deterministic one" — so an SDK-level failure (rate limit,
# dropped connection, auth) has to arrive as a `PlatformError`, not the raw
# `anthropic.APIError` subclass the SDK raises, or it skips every one of those
# catches and surfaces as a raw exception instead of a graceful degrade.

def _stub_llm_client(monkeypatch, error):
    """An LLMClient whose underlying SDK call always raises `error`."""
    import anthropic

    from dataplatform.nlp.llm import LLMClient

    client = LLMClient()
    monkeypatch.setattr(type(client), "client", property(lambda self: FakeSDKClient(error)))
    return client


class FakeSDKClient:
    def __init__(self, error):
        self.messages = FakeMessages(error)


class FakeMessages:
    def __init__(self, error):
        self._error = error

    def create(self, **kwargs):
        raise self._error


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: __import__("anthropic").APIConnectionError(
            message="connection reset",
            request=__import__("httpx").Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        lambda: __import__("anthropic").RateLimitError(
            "rate limited",
            response=__import__("httpx").Response(
                429, request=__import__("httpx").Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            body=None,
        ),
    ],
    ids=["connection_error", "rate_limit_error"],
)
def test_sdk_errors_are_translated_to_platform_error(monkeypatch, make_error):
    client = _stub_llm_client(monkeypatch, make_error())
    with pytest.raises(PlatformError):
        client.text(instructions="be terse", prompt="hello")


def test_dashboard_composer_falls_back_on_a_platform_error(tmp_path, monkeypatch):
    """Not just missing credentials — any PlatformError from the model call."""
    from dataplatform.reports import composer as composer_module

    catalog = Catalog(tmp_path / "catalog.json")
    catalog.state = CatalogState(
        datasets={
            "orders": DatasetMeta(
                name="orders",
                source="demo",
                origin_object="orders",
                n_rows=10,
                columns=[ColumnMeta(name="revenue", dtype="float64", semantic_type="currency")],
            )
        }
    )

    monkeypatch.setattr(composer_module, "available", lambda: True)
    monkeypatch.setattr(
        composer_module,
        "compose_with_llm",
        lambda *a, **k: (_ for _ in ()).throw(PlatformError("simulated rate limit")),
    )

    spec = composer_module.compose(catalog, ["orders"], request="3 kpi cards")
    assert spec.tiles  # fell all the way through to the deterministic composer


def test_nl2sql_falls_back_on_a_platform_error(tmp_path, monkeypatch):
    from dataplatform.catalog import Catalog
    from dataplatform.catalog.models import CatalogState, ColumnMeta, DatasetMeta
    from dataplatform.nlp import NL2SQL

    catalog = Catalog(tmp_path / "catalog.json")
    catalog.state = CatalogState(
        datasets={
            "orders": DatasetMeta(
                name="orders",
                source="demo",
                origin_object="orders",
                n_rows=10,
                columns=[ColumnMeta(name="revenue", dtype="float64", semantic_type="currency")],
            )
        }
    )

    nl2sql = NL2SQL(catalog, use_llm=True)
    monkeypatch.setattr(
        nl2sql,
        "_translate_with_llm",
        lambda *a, **k: (_ for _ in ()).throw(PlatformError("simulated overload")),
    )

    compiled = nl2sql.translate("total revenue")
    assert "revenue" in compiled.sql.lower()  # heuristic path answered instead of raising


# ------------------------------------------------- per-user schema publishing
def test_dataset_names_do_not_collide_between_users():
    """Two users publishing the same title must not share a Superset dataset.

    Datasets are looked up by name within a database, so without the schema in
    the name the second user would reuse the first user's dataset - and be
    shown their data.
    """
    from dataplatform.superset.publisher import _dataset_name

    assert _dataset_name("Sales Overview", None) == "vq_sales_overview"
    assert _dataset_name("Sales Overview", "u2") == "vq_u2_sales_overview"
    assert _dataset_name("Sales Overview", "u2") != _dataset_name("Sales Overview", "u3")


def test_publish_sends_the_warehouse_schema_to_superset():
    """A virtual dataset's SQL names tables unqualified.

    Without the schema, Superset resolves them against its default schema and
    fails with a bare "Fatal error" for tables it cannot see.
    """
    from dataplatform.superset.publisher import _warehouse_schema

    class FakeWarehouse:
        schema = "u7"

    class SchemaLess:
        pass

    assert _warehouse_schema(FakeWarehouse()) == "u7"
    assert _warehouse_schema(SchemaLess()) is None
