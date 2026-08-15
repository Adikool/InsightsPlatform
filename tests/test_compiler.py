"""Compiler and guard tests — the security boundary, so these are the ones that
must not regress."""

from __future__ import annotations

import pytest

from dataplatform.catalog import Catalog
from dataplatform.catalog.models import CatalogState, ColumnMeta, DatasetMeta
from dataplatform.errors import QueryValidationError
from dataplatform.nlp import SQLCompiler, validate_sql
from dataplatform.nlp.spec import Dimension, Filter, Metric, QuerySpec, Sort


@pytest.fixture()
def catalog(tmp_path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.json")
    cat.state = CatalogState(
        datasets={
            "orders": DatasetMeta(
                name="orders",
                source="demo",
                origin_object="orders",
                n_rows=1000,
                columns=[
                    ColumnMeta(name="order_id", dtype="int64", semantic_type="identifier"),
                    ColumnMeta(name="order_date", dtype="datetime64[ns]", semantic_type="temporal"),
                    ColumnMeta(name="region", dtype="str", semantic_type="geo",
                               sample_values=["North", "West"]),
                    ColumnMeta(name="revenue", dtype="float64", semantic_type="currency"),
                    ColumnMeta(name="delivery_days", dtype="int64", semantic_type="categorical"),
                ],
            )
        }
    )
    return cat


@pytest.fixture()
def compiler(catalog) -> SQLCompiler:
    return SQLCompiler(catalog, dialect="duckdb")


def test_basic_aggregate(compiler):
    spec = QuerySpec(
        dataset="orders",
        dimensions=[Dimension(column="region")],
        metrics=[Metric(column="revenue", func="sum")],
        limit=10,
    )
    compiled = compiler.compile(spec)
    assert 'SUM("revenue") AS "sum_revenue"' in compiled.sql
    assert 'GROUP BY "region"' in compiled.sql
    assert compiled.sql.rstrip().endswith("LIMIT 10")
    # a sensible default ordering is applied when none was asked for
    assert 'ORDER BY "sum_revenue" DESC' in compiled.sql


def test_time_grain_produces_date_trunc(compiler):
    spec = QuerySpec(
        dataset="orders",
        dimensions=[Dimension(column="order_date", time_grain="month")],
        metrics=[Metric(column="revenue", func="sum")],
    )
    compiled = compiler.compile(spec)
    assert "date_trunc('month', \"order_date\")" in compiled.sql
    assert compiled.time_column == "order_date_month"
    # time series sort chronologically, not by size
    assert 'ORDER BY "order_date_month" ASC' in compiled.sql


def test_unknown_column_is_rejected_with_a_suggestion(compiler):
    spec = QuerySpec(dataset="orders", dimensions=[Dimension(column="regionn")])
    with pytest.raises(QueryValidationError, match="Did you mean: region"):
        compiler.compile(spec)


def test_unknown_dataset_is_rejected(compiler):
    with pytest.raises(QueryValidationError, match="unknown dataset"):
        compiler.compile(QuerySpec(dataset="ordrs"))


def test_sum_of_identifier_is_rejected(compiler):
    spec = QuerySpec(dataset="orders", metrics=[Metric(column="order_id", func="sum")])
    with pytest.raises(QueryValidationError, match="meaningless"):
        compiler.compile(spec)


def test_avg_of_text_column_is_rejected(compiler):
    spec = QuerySpec(dataset="orders", metrics=[Metric(column="region", func="avg")])
    with pytest.raises(QueryValidationError, match="cannot apply avg"):
        compiler.compile(spec)


def test_avg_of_integer_categorical_is_allowed(compiler):
    """delivery_days is inferred categorical but averaging it is well defined."""
    spec = QuerySpec(dataset="orders", metrics=[Metric(column="delivery_days", func="avg")])
    assert 'AVG("delivery_days")' in compiler.compile(spec).sql


def test_string_literals_are_escaped(compiler):
    spec = QuerySpec(
        dataset="orders",
        metrics=[Metric(column="revenue", func="sum")],
        filters=[Filter(column="region", op="=", values=["O'Brien"])],
    )
    assert "'O''Brien'" in compiler.compile(spec).sql


def test_injection_through_a_filter_value_cannot_escape_the_literal(compiler):
    import re

    spec = QuerySpec(
        dataset="orders",
        metrics=[Metric(column="revenue", func="sum")],
        filters=[Filter(column="region", op="=", values=["x'; DROP TABLE orders; --"])],
    )
    sql = compiler.compile(spec).sql

    # The payload is allowed to appear *inside* the string literal — that is inert.
    # What matters is that nothing escapes it, so strip every quoted literal and
    # check the SQL that remains is still just the query we asked for.
    outside_literals = re.sub(r"'(?:[^']|'')*'", "''", sql)
    assert "DROP" not in outside_literals.upper()
    assert ";" not in outside_literals
    assert "--" not in outside_literals


def test_non_numeric_value_on_a_numeric_filter_is_rejected(compiler):
    spec = QuerySpec(
        dataset="orders",
        filters=[Filter(column="revenue", op=">", values=["abc"])],
        metrics=[Metric(column="revenue", func="sum")],
    )
    with pytest.raises(QueryValidationError, match="needs a number"):
        compiler.compile(spec)


def test_bad_date_on_a_temporal_filter_is_rejected(compiler):
    spec = QuerySpec(
        dataset="orders",
        filters=[Filter(column="order_date", op=">", values=["not-a-date"])],
        metrics=[Metric(column="revenue", func="sum")],
    )
    with pytest.raises(QueryValidationError, match="needs a date"):
        compiler.compile(spec)


def test_filter_on_a_metric_alias_becomes_having(compiler):
    spec = QuerySpec(
        dataset="orders",
        dimensions=[Dimension(column="region")],
        metrics=[Metric(column="revenue", func="sum", alias="total")],
        filters=[Filter(column="total", op=">", values=["1000"])],
    )
    sql = compiler.compile(spec).sql
    assert "HAVING SUM(\"revenue\") > 1000" in sql
    assert "WHERE" not in sql


def test_limit_is_capped_at_max_rows(compiler):
    from dataplatform.config import settings

    spec = QuerySpec(dataset="orders", metrics=[Metric(column="revenue", func="sum")], limit=10**9)
    assert f"LIMIT {settings.max_rows}" in compiler.compile(spec).sql


def test_sort_on_an_unproduced_column_is_dropped_with_a_warning(compiler):
    spec = QuerySpec(
        dataset="orders",
        dimensions=[Dimension(column="region")],
        metrics=[Metric(column="revenue", func="sum")],
        sort=[Sort(field="nonexistent")],
    )
    compiled = compiler.compile(spec)
    assert any("dropped sort" in w for w in compiled.warnings)


def test_time_grain_on_a_non_temporal_column_is_rejected(compiler):
    spec = QuerySpec(dataset="orders", dimensions=[Dimension(column="region", time_grain="month")])
    with pytest.raises(QueryValidationError, match="non-temporal"):
        compiler.compile(spec)


# --------------------------------------------------------------------- guard
@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE orders SET revenue = 0",
        "SELECT 1; DELETE FROM orders",
        "INSERT INTO orders VALUES (1)",
        "SELECT * FROM read_csv('/etc/passwd')",
        "ATTACH 'evil.db' AS evil",
        "SELECT * FROM other_database",
    ],
)
def test_guard_rejects(statement, catalog):
    with pytest.raises(QueryValidationError):
        validate_sql(statement, catalog)


def test_guard_allows_a_select_and_injects_a_limit(catalog):
    out = validate_sql("SELECT region, count(*) FROM orders GROUP BY region", catalog)
    assert "LIMIT" in out


def test_guard_allows_a_cte(catalog):
    out = validate_sql("WITH t AS (SELECT * FROM orders) SELECT count(*) FROM t", catalog)
    assert out.lower().startswith("with")


def test_guard_keeps_an_existing_limit(catalog):
    out = validate_sql("SELECT * FROM orders LIMIT 5", catalog)
    assert out.count("LIMIT") == 1


def test_guard_ignores_keywords_inside_comments(catalog):
    out = validate_sql("SELECT * FROM orders -- DROP TABLE orders", catalog)
    assert "orders" in out
