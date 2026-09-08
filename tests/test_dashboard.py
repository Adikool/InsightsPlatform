"""Dashboard composition and Superset layout.

The layout tests matter disproportionately: a malformed `position_json` is
accepted by Superset's API and then renders as a blank page, so a broken layout
looks like a successful publish.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dataplatform.catalog import Catalog
from dataplatform.catalog.models import CatalogState, ColumnMeta, DatasetMeta
from dataplatform.nlp.compiler import SQLCompiler
from dataplatform.nlp.spec import Filter
from dataplatform.reports import compose, compose_default, compose_multi, validate
from dataplatform.reports.spec import GRID_COLUMNS, Tile
from dataplatform.superset.layout import build_position_json, pack_rows


def _column(name, dtype, semantic, n_unique=10, top=None, samples=None):
    return ColumnMeta(
        name=name, dtype=dtype, semantic_type=semantic, n_unique=n_unique,
        pct_top_value=top, sample_values=samples or [],
    )


@pytest.fixture()
def orders() -> DatasetMeta:
    return DatasetMeta(
        name="orders",
        source="demo",
        origin_object="orders",
        n_rows=50_000,
        columns=[
            _column("order_id", "int64", "identifier", 50_000),
            _column("customer_id", "int64", "identifier", 9_000),
            _column("order_date", "datetime64[ns]", "temporal", 1_000),
            _column("region", "str", "geo", 5, samples=["North", "South", "East", "West", "Central"]),
            _column("channel", "str", "categorical", 3, samples=["Online", "Retail", "Partner"]),
            _column("product_category", "str", "categorical", 5,
                    samples=["Electronics", "Apparel", "Home", "Grocery", "Sports"]),
            # degenerate: two values, 94% of rows share one of them
            _column("notes", "str", "categorical", 2, top=0.94, samples=["", "expedited"]),
            _column("revenue", "float64", "currency"),
            _column("unit_price", "float64", "currency"),
            _column("units", "int64", "numeric"),
        ],
    )


@pytest.fixture()
def catalog(tmp_path, orders) -> Catalog:
    cat = Catalog(tmp_path / "catalog.json")
    cat.state = CatalogState(datasets={"orders": orders})
    return cat


# ------------------------------------------------------------------ composer
def test_default_dashboard_has_the_conventional_shape(orders):
    spec = compose_default(orders)
    roles = [tile.role for tile in spec.ordered]

    assert roles.count("kpi") >= 3
    assert "trend" in roles
    assert "breakdown" in roles
    assert roles[-1] == "detail", "the detail table belongs at the bottom"
    assert roles[: roles.count("kpi")] == ["kpi"] * roles.count("kpi"), "KPIs lead"


def test_kpi_tiles_are_single_numbers(orders):
    for tile in compose_default(orders).by_role("kpi"):
        assert tile.spec.chart == "big_number"
        assert not tile.spec.dimensions
        assert len(tile.spec.metrics) == 1


def test_trend_uses_the_temporal_column_with_a_grain(orders):
    trend = compose_default(orders).by_role("trend")[0]
    assert trend.spec.dimensions[0].column == "order_date"
    assert trend.spec.dimensions[0].time_grain != "none"
    assert trend.spec.chart in ("line", "area")


def test_degenerate_columns_are_not_used_as_breakdowns(orders):
    """`notes` is 94% one value — grouping by it yields a single visible bar."""
    spec = compose_default(orders)
    grouped = {
        dimension.column
        for tile in spec.by_role("breakdown")
        for dimension in tile.spec.dimensions
    }
    assert "notes" not in grouped
    assert grouped & {"region", "channel", "product_category"}


def test_identifiers_are_never_summed(orders):
    for tile in compose_default(orders).tiles:
        for metric in tile.spec.metrics:
            if metric.column in ("order_id", "customer_id"):
                assert metric.func == "count_distinct"


def test_every_tile_compiles(catalog, orders):
    spec = compose_default(orders)
    kept, problems = validate(spec, SQLCompiler(catalog, dialect="postgresql"))
    assert not problems
    assert len(kept) == len(spec.tiles)


def test_dataset_without_measures_is_refused(tmp_path):
    from dataplatform.errors import QueryValidationError

    bare = DatasetMeta(
        name="labels", source="x", origin_object="x", n_rows=10,
        columns=[_column("name", "str", "text", 10)],
    )
    with pytest.raises(QueryValidationError, match="measure"):
        compose_default(bare)


# ---------------------------------------------------------- multi-table compose
@pytest.fixture()
def customers() -> DatasetMeta:
    return DatasetMeta(
        name="customers",
        source="demo",
        origin_object="customers",
        n_rows=9_000,
        columns=[
            _column("customer_id", "int64", "identifier", 9_000),
            _column("segment", "str", "categorical", 3, samples=["Consumer", "Corporate", "Home Office"]),
            _column("lifetime_revenue", "float64", "currency"),
        ],
    )


def test_compose_multi_with_one_dataset_matches_compose_default(orders):
    """A single-table call should not behave differently through the multi path."""
    single = compose_multi([orders])
    direct = compose_default(orders)
    assert [t.title for t in single.tiles] == [t.title for t in direct.tiles]
    assert single.title == direct.title


def test_compose_multi_merges_tiles_from_every_dataset(orders, customers):
    spec = compose_multi([orders, customers])
    dataset_names = {tile.spec.dataset for tile in spec.tiles}
    assert dataset_names == {"orders", "customers"}


def test_compose_multi_labels_tiles_by_table(orders, customers):
    """Two tables can each produce a 'Total revenue' card — the title has to say which."""
    spec = compose_multi([orders, customers])
    titles = [tile.title for tile in spec.tiles]
    assert any(t.startswith("Orders — ") for t in titles)
    assert any(t.startswith("Customers — ") for t in titles)


def test_compose_multi_default_title_names_both_tables(orders, customers):
    spec = compose_multi([orders, customers])
    assert "Orders" in spec.title
    assert "Customers" in spec.title


def test_compose_multi_respects_an_explicit_title(orders, customers):
    spec = compose_multi([orders, customers], title="Sales Overview")
    assert spec.title == "Sales Overview"


def test_compose_multi_interpretation_covers_both_tables(orders, customers):
    spec = compose_multi([orders, customers])
    assert "[orders]" in spec.interpretation
    assert "[customers]" in spec.interpretation


def test_compose_multi_every_tile_still_compiles(tmp_path, orders, customers):
    cat = Catalog(tmp_path / "catalog.json")
    cat.state = CatalogState(datasets={"orders": orders, "customers": customers})
    spec = compose_multi([orders, customers])
    kept, problems = validate(spec, SQLCompiler(cat, dialect="postgresql"))
    assert not problems
    assert len(kept) == len(spec.tiles)


def test_compose_dispatches_to_multi_for_several_datasets(tmp_path, orders, customers):
    cat = Catalog(tmp_path / "catalog.json")
    cat.state = CatalogState(datasets={"orders": orders, "customers": customers})
    spec = compose(cat, ["orders", "customers"], use_llm=False)
    assert {tile.spec.dataset for tile in spec.tiles} == {"orders", "customers"}


# -------------------------------------------------------------------- layout
def _tiles(*widths) -> list[Tile]:
    from dataplatform.nlp.spec import Metric, QuerySpec

    return [
        Tile(
            spec=QuerySpec(dataset="orders", metrics=[Metric(column="revenue", func="sum")], title=f"t{i}"),
            width=w,
        )
        for i, w in enumerate(widths)
    ]


def test_rows_never_exceed_twelve_columns():
    for row in pack_rows(_tiles(3, 3, 3, 3, 12, 6, 6, 12)):
        assert sum(tile.width for tile in row) <= GRID_COLUMNS


def test_packing_preserves_order():
    """Reordering to fill rows tighter would move the detail table off the bottom."""
    tiles = _tiles(3, 3, 3, 3, 12, 6, 6, 12)
    flattened = [tile for row in pack_rows(tiles) for tile in row]
    assert flattened == tiles


def test_four_kpis_share_one_row():
    rows = pack_rows(_tiles(3, 3, 3, 3, 12))
    assert len(rows[0]) == 4
    assert len(rows[1]) == 1


def test_position_json_spine_is_intact():
    tiles = _tiles(3, 3, 3, 3, 12, 6, 6, 12)
    position = build_position_json("Sales", tiles, list(range(101, 109)))

    assert position["DASHBOARD_VERSION_KEY"] == "v2"
    assert position["ROOT_ID"]["children"] == ["GRID_ID"]
    assert position["GRID_ID"]["parents"] == ["ROOT_ID"]

    rows = [k for k in position if k.startswith("ROW-")]
    charts = [k for k in position if k.startswith("CHART-")]
    assert len(charts) == len(tiles)
    assert set(position["GRID_ID"]["children"]) == set(rows)

    for key in charts:
        node = position[key]
        # A CHART node missing its ancestor chain saves fine and renders blank.
        assert node["parents"][:2] == ["ROOT_ID", "GRID_ID"]
        assert node["parents"][2] in rows
        assert key in position[node["parents"][2]]["children"]
        assert node["meta"]["chartId"] in range(101, 109)
        assert 1 <= node["meta"]["width"] <= GRID_COLUMNS


def test_chart_ids_map_to_the_right_tiles():
    tiles = _tiles(6, 6)
    tiles[0].spec.title, tiles[1].spec.title = "First", "Second"
    position = build_position_json("D", tiles, [55, 66])

    by_name = {
        node["meta"]["sliceName"]: node["meta"]["chartId"]
        for key, node in position.items()
        if key.startswith("CHART-")
    }
    assert by_name == {"First": 55, "Second": 66}


def test_mismatched_ids_are_rejected():
    with pytest.raises(ValueError):
        build_position_json("D", _tiles(6, 6), [1])


# --------------------------------------------------------------------- hints
@pytest.mark.parametrize(
    "question",
    [
        "create a dashboard with kpi cards, graphs on sales and a detail table",
        "build me a report with charts and tables",
        "sales scorecard",
    ],
)
def test_multi_chart_requests_are_flagged(question):
    from dataplatform.platform import _multi_chart_hint

    assert _multi_chart_hint(question), "a dashboard request should not silently return one chart"


@pytest.mark.parametrize(
    "question",
    ["monthly revenue by region", "top 5 categories by revenue", "average delivery days"],
)
def test_ordinary_questions_are_not_flagged(question):
    from dataplatform.platform import _multi_chart_hint

    assert not _multi_chart_hint(question)


def test_ask_offers_a_dashboard_action_not_just_advice(tmp_path, monkeypatch):
    """The suggestion must be structured so the UI can render a button.

    Telling someone to go and use a different view is a worse answer than
    offering to do it for them, and a warning string cannot carry an action.
    """
    import dataplatform.config as config_module

    monkeypatch.setenv("DP_HOME", str(tmp_path))
    config_module.settings = config_module.Settings(home=tmp_path)

    import sqlite3

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "order_id": range(400),
        "order_date": pd.date_range("2024-01-01", periods=400, freq="D").strftime("%Y-%m-%d"),
        "region": rng.choice(["N", "S"], 400),
        "revenue": rng.lognormal(4, 0.5, 400).round(2),
    })
    db = tmp_path / "s.db"
    with sqlite3.connect(db) as conn:
        frame.to_sql("orders", conn, index=False)

    from dataplatform import Platform

    platform = Platform(
        warehouse_uri=f"duckdb:///{(tmp_path / 'w.duckdb').as_posix()}",
        catalog_path=tmp_path / "catalog.json",
        use_llm=False,
    )
    platform.add_source("s", f"sqlite:///{db.as_posix()}", type="sql")
    platform.ingest("s")

    result = platform.ask("create a sales dashboard with kpi cards, graphs and a detail table")
    assert result.suggestion is not None
    assert result.suggestion["type"] == "dashboard"
    assert result.suggestion["dataset"] in {d.name for d in platform.list_datasets()}
    assert result.suggestion["action"]

    plain = platform.ask("total revenue by region")
    assert plain.suggestion is None


def test_tie_break_prefers_the_fact_table(catalog, orders):
    """A question naming no column should not be decided by insertion order."""
    from dataplatform.catalog.models import CatalogState
    from dataplatform.nlp.heuristic import pick_dataset

    lookup = DatasetMeta(
        name="regions", source="demo", origin_object="regions", n_rows=5,
        columns=[_column("region", "str", "geo", 5), _column("manager", "str", "categorical", 5)],
    )
    # `regions` is inserted first, so insertion order alone would pick it.
    catalog.state = CatalogState(datasets={"regions": lookup, "orders": orders})

    assert pick_dataset("build a dashboard with kpi cards and graphs", catalog).name == "orders"
    assert pick_dataset("", catalog).name == "orders"
    # An explicit signal still wins over richness.
    assert pick_dataset("manager by region", catalog).name == "regions"


# -------------------------------------------------------- request directives
def test_requested_kpi_count_is_honoured(orders):
    """Asking for 3 cards must not produce 4.

    The deterministic composer used to ignore the request text entirely, which
    made the "what should it show?" field a false promise.
    """
    for wanted in (1, 2, 3, 5):
        spec = compose_default(orders, request=f"dashboard with {wanted} kpi cards")
        assert len(spec.by_role("kpi")) == wanted


@pytest.mark.parametrize(
    "phrase,grain",
    [("weekly sales", "week"), ("quarterly revenue", "quarter"), ("daily trend", "day")],
)
def test_requested_time_grain_is_honoured(orders, phrase, grain):
    trend = compose_default(orders, request=phrase).by_role("trend")[0]
    assert trend.spec.dimensions[0].time_grain == grain


def test_sections_can_be_excluded(orders):
    assert not compose_default(orders, request="no detail table").by_role("detail")
    assert not compose_default(orders, request="without a trend").by_role("trend")


def test_named_dimensions_drive_the_breakdowns(orders):
    spec = compose_default(orders, request="revenue by product category")
    grouped = {d.column for t in spec.by_role("breakdown") for d in t.spec.dimensions}
    assert grouped == {"product_category"}


def test_named_measure_leads(orders):
    spec = compose_default(orders, request="dashboard for units")
    assert spec.by_role("kpi")[0].spec.metrics[0].column == "units"


def test_interpretation_reports_what_was_understood(orders):
    spec = compose_default(orders, request="3 kpi cards, weekly, no detail table")
    assert "3 KPI cards" in spec.interpretation
    assert "week" in spec.interpretation


def test_unsupported_directives_are_named_not_silently_dropped(orders):
    spec = compose_default(orders, request="revenue dashboard filtered to the West only")
    assert "filters" in spec.interpretation
    assert "ANTHROPIC_API_KEY" in spec.interpretation


def test_dataset_name_outweighs_a_column_in_another_table(tmp_path, orders):
    """`orders` has a customer_id column, but `customers` is the customer table."""
    from dataplatform.catalog.models import CatalogState
    from dataplatform.nlp.heuristic import pick_dataset

    customers = DatasetMeta(
        name="sales_db_customers", source="demo", origin_object="customers", n_rows=9_000,
        columns=[
            _column("customer_id", "int64", "identifier", 9_000),
            _column("lifetime_revenue", "float64", "currency"),
        ],
    )
    renamed = orders.model_copy(update={"name": "sales_db_orders"})
    cat = Catalog(tmp_path / "c.json")
    cat.state = CatalogState(datasets={"sales_db_orders": renamed, "sales_db_customers": customers})

    assert pick_dataset("sales customer dashboard with 3 kpi cards", cat).name == "sales_db_customers"
    assert pick_dataset("monthly revenue by region", cat).name == "sales_db_orders"


# --------------------------------------------------------- per-value KPI cards
def test_named_values_produce_one_kpi_per_value_in_typed_order(orders):
    """"3 kpi cards, one for north, one for east, one for west" was previously
    read as "make 3 generic cards" — the count survived, the actual ask (which
    three, filtered how) was silently dropped."""
    spec = compose_default(
        orders, request="3 kpi cards, one for north, one for east, one for west"
    )
    kpis = spec.by_role("kpi")
    assert len(kpis) == 3
    assert [tile.title for tile in kpis] == ["North revenue", "East revenue", "West revenue"]
    for tile, region in zip(kpis, ["North", "East", "West"]):
        assert tile.spec.filters == [Filter(column="region", op="=", values=[region])]
        assert tile.spec.metrics[0].column == "revenue"
        assert tile.spec.metrics[0].func == "sum"


def test_named_values_work_without_an_explicit_count(orders):
    spec = compose_default(orders, request="kpi cards for north, east and west")
    assert [t.title for t in spec.by_role("kpi")] == ["North revenue", "East revenue", "West revenue"]


def test_a_single_named_value_does_not_trigger_per_value_mode(orders):
    """One value named is ambiguous with just mentioning it in passing; two or
    more values naming the same column is what signals 'one card per value'."""
    spec = compose_default(orders, request="revenue for north")
    assert not any(tile.spec.filters for tile in spec.by_role("kpi"))
    assert len(spec.by_role("kpi")) > 1


def test_per_value_kpi_rows_fill_the_grid_width(orders):
    for values, expected_row_width in ((("North", "East"), 6), (("North", "East", "West"), 4)):
        text = "kpi cards for " + ", ".join(values)
        spec = compose_default(orders, request=text)
        kpis = spec.by_role("kpi")
        assert sum(tile.width for tile in kpis) == GRID_COLUMNS
        assert all(tile.width == expected_row_width for tile in kpis)


def test_per_value_kpis_are_not_truncated_by_a_conflicting_count(orders):
    """A named-value list is the more specific instruction; an explicit count
    that disagrees should not silently drop one of the named regions."""
    spec = compose_default(orders, request="2 kpi cards, one for north, one for east, one for west")
    assert len(spec.by_role("kpi")) == 3


def test_readback_names_the_regions_not_just_a_count(orders):
    spec = compose_default(orders, request="3 kpi cards, one for north, one for east, one for west")
    assert "North" in spec.interpretation
    assert "East" in spec.interpretation
    assert "West" in spec.interpretation


def test_per_value_kpi_sql_is_correctly_filtered(catalog, orders):
    spec = compose_default(orders, request="kpi cards for north, east, west")
    compiler = SQLCompiler(catalog, dialect="postgresql")
    for tile in spec.by_role("kpi"):
        compiled = compiler.compile(tile.spec)
        assert "WHERE" in compiled.sql
        region = tile.spec.filters[0].values[0]
        assert f"'{region}'" in compiled.sql


def test_heuristic_explanation_reads_back_the_query(catalog):
    from dataplatform.nlp import heuristic

    spec = heuristic.parse("monthly revenue by region", catalog)
    assert spec.explanation.startswith("Read as:")
    assert "revenue" in spec.explanation
    assert "region" in spec.explanation
    # The old wording led with the absence of a model, which read as an error.
    assert "without a language model" not in spec.explanation


# ------------------------------------------------------- replacing a dashboard
class _StubClient:
    """Enough of SupersetClient to exercise the publish decision, no network."""

    def __init__(self, existing_titles=()):
        self.titles = {t: i + 1 for i, t in enumerate(existing_titles)}
        self.created = []
        self.put_calls = []

    # -- lookups the replace decision depends on
    def find_dashboard(self, title):
        return {"id": self.titles[title]} if title in self.titles else None

    def dashboard_is_occupied(self, dashboard_id):
        return True  # the case that matters: someone already laid it out

    def unique_dashboard_title(self, title, limit=50):
        if title not in self.titles:
            return title
        for n in range(2, limit):
            if f"{title} ({n})" not in self.titles:
                return f"{title} ({n})"
        raise AssertionError("no free title")

    def ensure_dashboard(self, title):
        if title not in self.titles:
            self.titles[title] = len(self.titles) + 1
            self.created.append(title)
        return self.titles[title]

    # -- the rest, stubbed to no-ops
    def find_dataset(self, name, database_id=None):
        return {"id": 1}

    def create_dataset(self, *a, **k):
        return 1

    def refresh_dataset(self, dataset_id):
        pass

    def create_chart(self, **k):
        return 99

    def attach_chart(self, chart_id, dashboard_id):
        pass

    def dashboard_url(self, dashboard_id):
        return f"http://superset/dashboard/{dashboard_id}/"

    def chart_url(self, chart_id):
        return f"http://superset/chart/{chart_id}/"

    def put(self, path, payload):
        self.put_calls.append((path, payload))
        return {}


def _publish_with(stub, title, monkeypatch):
    """Run publish_dashboard against the stub and return (title used, notes)."""
    from dataplatform.superset import publisher as publisher_module
    from dataplatform.superset.publisher import SupersetPublisher, publish_dashboard

    pub = SupersetPublisher(client=stub)
    monkeypatch.setattr(pub, "ensure_warehouse_database", lambda *a, **k: 1)

    # Real Tile/QuerySpec rather than stand-ins: the layout builder reads fields
    # (height, width) that a hand-rolled stub silently lacks.
    from dataplatform.nlp.spec import Metric, QuerySpec

    query = QuerySpec(
        dataset="orders", title="Total", chart="big_number",
        metrics=[Metric(func="sum", column="revenue")],
    )
    tile = Tile(spec=query, role="kpi", width=3)
    spec = SimpleNamespace(title=title, tiles=[tile])
    compiled = SimpleNamespace(sql="SELECT 1", spec=query)
    monkeypatch.setattr(publisher_module.SupersetPublisher, "_params", lambda *a, **k: {})

    class _Warehouse:
        schema = None

    result = publish_dashboard(pub, spec, [(tile, compiled)], _Warehouse(),
                               replace=stub.replace_flag)
    return result, stub


def test_publishing_over_an_existing_dashboard_renames_by_default(monkeypatch):
    """Taking over a hand-built layout orphans its charts, so never do it silently."""
    stub = _StubClient(existing_titles=["Sales"])
    stub.replace_flag = False
    result, stub = _publish_with(stub, "Sales", monkeypatch)

    assert "Sales (2)" in stub.created, "should publish beside the existing dashboard"
    assert any("already exists" in n for n in result.notes)
    # The note must point at something the user can actually do.
    assert any("Replace existing" in n for n in result.notes)
    assert not any("replace=True" in n for n in result.notes), "no Python kwargs in UI copy"


def test_replace_overwrites_the_existing_dashboard(monkeypatch):
    """With replace on, the same dashboard is reused rather than duplicated."""
    stub = _StubClient(existing_titles=["Sales"])
    stub.replace_flag = True
    result, stub = _publish_with(stub, "Sales", monkeypatch)

    assert stub.created == [], "must reuse, not create a numbered copy"
    assert result.dashboard_id == 1
    assert not any("already exists" in n for n in result.notes)
