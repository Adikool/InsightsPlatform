"""Turn a CompiledQuery into a Superset dataset + chart (+ dashboard).

The compiled SQL becomes a *virtual dataset*, so the numbers on the dashboard are
produced by exactly the query that answered the question — there is no second
definition to drift.

That does mean the dataset arrives pre-aggregated, so the chart re-aggregates with
`SUM` over the same grouping. Summing an already-grouped column by the same keys is
a no-op, and it keeps Superset's own controls (filters, drilldown) working normally.
The one case where that could be wrong is an average: averaging pre-averaged
groups of unequal size is not the overall average. Charts group by every
dimension the query grouped by, so each group holds a single row and the
re-aggregation is a no-op - the published chart is exact. The caveat is
recorded in the chart description because it becomes real if someone later
removes a grouping in Superset, and it is raised as a publish note only for
the chart types that genuinely do collapse groups.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

from ..config import settings
from ..errors import DashboardExists, SupersetError
from ..nlp.chart import SUPERSET_VIZ
from ..nlp.spec import CompiledQuery
from .client import SupersetClient

_TIME_GRAIN_SQLA = {
    "day": "P1D",
    "week": "P1W",
    "month": "P1M",
    "quarter": "P3M",
    "year": "P1Y",
}


@dataclass
class PublishResult:
    dataset_id: int
    chart_id: int
    chart_url: str
    dashboard_id: int | None = None
    dashboard_url: str | None = None
    notes: list[str] = field(default_factory=list)


def _adhoc_metric(column: str, aggregate: str = "SUM", label: str | None = None) -> dict:
    return {
        "expressionType": "SIMPLE",
        "column": {"column_name": column},
        "aggregate": aggregate,
        "label": label or column,
        "optionName": f"metric_{uuid.uuid4().hex[:12]}",
        "hasCustomLabel": True,
    }


class SupersetPublisher:
    def __init__(self, client: SupersetClient | None = None) -> None:
        self.client = client or SupersetClient()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def warehouse_uri_for_superset(warehouse) -> str:
        """The URI to register in Superset, which may differ from ours.

        Superset usually runs in a container, so it resolves paths and hostnames
        in its own namespace: a Windows DuckDB path becomes `/app/C:/Users/...`
        and our `localhost` is its loopback, not the host's. `DP_SUPERSET_WAREHOUSE_URI`
        is the address that works from Superset's side.
        """
        return settings.superset_warehouse_uri or warehouse.sqlalchemy_uri

    def ensure_warehouse_database(self, warehouse, name: str = "insight_warehouse") -> int:
        uri = self.warehouse_uri_for_superset(warehouse)
        if uri.startswith("duckdb://") and not settings.superset_warehouse_uri:
            # Fail here with the actual cause rather than letting Superset return
            # a confusing "No such file or directory" for a path that exists.
            raise SupersetError(
                "Superset cannot read a local DuckDB file unless it runs on this "
                "same filesystem — a containerised Superset resolves the path "
                "inside its own container. Point the warehouse at a networked "
                "database (DP_WAREHOUSE_URI=postgresql+psycopg://...) and set "
                "DP_SUPERSET_WAREHOUSE_URI to the address Superset can reach it at."
            )
        return self.client.ensure_database(name, uri)

    # ------------------------------------------------------------------- main
    def publish(
        self,
        compiled: CompiledQuery,
        warehouse,
        chart_name: str | None = None,
        dashboard: str | None = None,
        database_name: str = "insight_warehouse",
    ) -> PublishResult:
        notes: list[str] = []
        database_id = self.ensure_warehouse_database(warehouse, database_name)

        title = chart_name or compiled.spec.title or f"Query on {compiled.spec.dataset}"
        schema = _warehouse_schema(warehouse)
        dataset_name = _dataset_name(title, schema)

        existing = self.client.find_dataset(dataset_name, database_id)
        if existing:
            dataset_id = int(existing["id"])
            notes.append(f"reused existing dataset {dataset_name!r}")
        else:
            # The schema must go with it: the SQL names tables unqualified, so
            # without it Superset resolves them against its default schema and
            # answers "Fatal error" for tables it cannot see.
            dataset_id = self.client.create_dataset(
                database_id, dataset_name, sql=compiled.sql, schema=schema
            )
        self.client.refresh_dataset(dataset_id)

        viz_type = SUPERSET_VIZ.get(compiled.spec.chart, "table")
        caveats: list[str] = []
        params = self._params(compiled, dataset_id, viz_type, notes, caveats)

        chart_id = self.client.create_chart(
            name=title,
            viz_type=viz_type,
            dataset_id=dataset_id,
            params=params,
            description=" ".join(filter(None, [compiled.spec.explanation, *caveats])),
        )

        result = PublishResult(
            dataset_id=dataset_id,
            chart_id=chart_id,
            chart_url=self.client.chart_url(chart_id),
            notes=notes,
        )

        if dashboard:
            dashboard_id = self.client.ensure_dashboard(dashboard)
            self.client.attach_chart(chart_id, dashboard_id)
            result.dashboard_id = dashboard_id
            result.dashboard_url = self.client.dashboard_url(dashboard_id)
        return result

    # ----------------------------------------------------------------- params
    @staticmethod
    def _collapses_groups(viz_type: str, dims: list[str]) -> bool:
        """Does this chart aggregate across the grouping the query already applied?

        Only then can re-aggregating a pre-computed average actually skew the
        number. Most charts group by every dimension in `dims`, so each group
        holds exactly one row and the aggregate returns it unchanged.
        """
        if not dims:
            return False  # one row in, one row out
        if viz_type == "big_number_total":
            return True  # ignores dimensions entirely
        if viz_type == "pie":
            return len(dims) > 1  # only the first dimension is used
        if viz_type == "heatmap_v2":
            return len(dims) > 2  # x_axis + one groupby
        return False

    def _params(
        self,
        compiled: CompiledQuery,
        dataset_id: int,
        viz_type: str,
        notes: list[str],
        caveats: list[str] | None = None,
    ) -> dict:
        spec = compiled.spec
        dims = list(compiled.dimension_aliases)
        metric_aliases = list(compiled.metric_aliases)

        # Choose the re-aggregation that is safe for each metric.
        aggregates: dict[str, str] = {}
        for metric, alias in zip(spec.metrics, metric_aliases):
            if metric.func in ("avg", "median"):
                aggregates[alias] = "AVG"
                caveat = (
                    f"{alias!r} is already aggregated per group with {metric.func.upper()}. "
                    "Superset re-aggregates "
                    "it with AVG, which is exact as published; it would skew only if a "
                    "grouping were removed here so unequal groups were averaged together."
                )
                # The description travels with the chart, which is where this
                # matters later. The publish note is reserved for the charts
                # that collapse groups now, so it stops being noise on every
                # publish that is in fact exact.
                (caveats if caveats is not None else notes).append(caveat)
                if caveats is not None and self._collapses_groups(viz_type, dims):
                    notes.append(
                        f"{alias!r} is a pre-computed {metric.func} and this chart aggregates "
                        "across groups, so the value shown is an unweighted average of averages"
                    )
            elif metric.func == "min":
                aggregates[alias] = "MIN"
            elif metric.func == "max":
                aggregates[alias] = "MAX"
            else:
                aggregates[alias] = "SUM"

        metrics = [_adhoc_metric(alias, aggregates.get(alias, "SUM")) for alias in metric_aliases]

        base: dict = {
            "datasource": f"{dataset_id}__table",
            "viz_type": viz_type,
            "row_limit": min(spec.limit, settings.max_rows),
            "adhoc_filters": [],
        }

        time_col = compiled.time_column
        grain = next((d.time_grain for d in spec.dimensions if d.time_grain != "none"), "none")

        if viz_type == "big_number_total":
            base["metric"] = metrics[0] if metrics else None
            return base

        if viz_type == "pie":
            base.update({"groupby": dims[:1], "metric": metrics[0] if metrics else None})
            return base

        if viz_type == "table":
            if metrics:
                base.update(
                    {"query_mode": "aggregate", "groupby": dims, "metrics": metrics}
                )
            else:
                base.update({"query_mode": "raw", "all_columns": compiled.columns})
            return base

        if viz_type == "heatmap_v2":
            base.update(
                {
                    "x_axis": dims[0] if dims else None,
                    "groupby": dims[1] if len(dims) > 1 else None,
                    "metric": metrics[0] if metrics else None,
                }
            )
            return base

        # echarts timeseries family: line / area / bar / scatter
        x_axis = time_col or (dims[0] if dims else None)
        series = [d for d in dims if d != x_axis]
        base.update(
            {
                "x_axis": x_axis,
                "groupby": series,
                "metrics": metrics,
                "x_axis_sort_asc": True,
                "orientation": "horizontal" if spec.chart == "horizontal_bar" else "vertical",
            }
        )
        if time_col and grain in _TIME_GRAIN_SQLA:
            base["time_grain_sqla"] = _TIME_GRAIN_SQLA[grain]
        return base


@dataclass
class DashboardResult:
    dashboard_id: int
    dashboard_url: str
    chart_ids: list[int] = field(default_factory=list)
    tiles: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def publish_dashboard(
    publisher: "SupersetPublisher",
    spec,
    compiled_tiles: list,
    warehouse,
    database_name: str = "insight_warehouse",
    replace: bool = False,
    on_conflict: str = "rename",
) -> DashboardResult:
    """Create one chart per tile, then lay them out on a single dashboard.

    Charts are created first because the layout has to reference their ids, and
    the dashboard is updated in one call at the end — a half-written
    `position_json` renders as a blank dashboard rather than a partial one.
    """
    from .layout import build_position_json

    client = publisher.client
    notes: list[str] = []
    database_id = publisher.ensure_warehouse_database(warehouse, database_name)

    # Never silently take over a dashboard that already has a layout — that
    # orphans every chart on it. Fall back to a suffixed title unless the caller
    # explicitly asked to replace.
    title = spec.title
    existing = client.find_dashboard(title)
    if existing and not replace:
        if client.dashboard_is_occupied(int(existing["id"])):
            if on_conflict == "ask":
                # Don't decide for the caller. An interface that can prompt
                # should offer to open the existing dashboard, overwrite it, or
                # rename - all better than silently leaving a numbered copy
                # behind for someone to find later.
                raise DashboardExists(
                    title=title,
                    existing_url=client.dashboard_url(int(existing["id"])),
                    suggested_title=client.unique_dashboard_title(title),
                )
            title = client.unique_dashboard_title(title)
            notes.append(
                f"a dashboard named {spec.title!r} already exists with its own layout; "
                f"published to {title!r} instead."
            )
    dashboard_id = client.ensure_dashboard(title)
    chart_ids: list[int] = []
    placed: list = []

    schema = _warehouse_schema(warehouse)
    for tile, compiled in compiled_tiles:
        dataset_name = _dataset_name(f"{spec.title} {tile.title}", schema)
        existing = client.find_dataset(dataset_name, database_id)
        if existing:
            dataset_id = int(existing["id"])
        else:
            dataset_id = client.create_dataset(
                database_id, dataset_name, sql=compiled.sql, schema=schema
            )
        client.refresh_dataset(dataset_id)

        viz_type = SUPERSET_VIZ.get(compiled.spec.chart, "table")
        caveats: list[str] = []
        params = publisher._params(compiled, dataset_id, viz_type, notes, caveats)
        chart_id = client.create_chart(
            name=tile.title,
            viz_type=viz_type,
            dataset_id=dataset_id,
            params=params,
            description=" ".join(filter(None, [compiled.spec.explanation, *caveats])),
        )
        client.attach_chart(chart_id, dashboard_id)
        chart_ids.append(chart_id)
        placed.append(tile)

    position = build_position_json(title, placed, chart_ids)
    client.put(
        f"/dashboard/{dashboard_id}",
        {
            "dashboard_title": title,
            "position_json": json.dumps(position),
            "published": True,
        },
    )

    return DashboardResult(
        dashboard_id=dashboard_id,
        dashboard_url=client.dashboard_url(dashboard_id),
        chart_ids=chart_ids,
        tiles=[tile.title for tile in placed],
        notes=notes,
    )


def _dataset_name(title: str, schema: str | None = None) -> str:
    """Superset-side name for a virtual dataset.

    The schema is part of the name because Superset datasets are looked up by
    name within a database: without it, two users who publish a dashboard with
    the same title would resolve to the *same* dataset, and the second would be
    shown the first one's data.
    """
    slug = re.sub(r"[^0-9A-Za-z]+", "_", title).strip("_").lower() or "query"
    prefix = f"vq_{schema}_" if schema else "vq_"
    return f"{prefix}{slug}"[:60]


def _warehouse_schema(warehouse) -> str | None:
    """The schema this user's tables live in, if the backend has one."""
    return getattr(warehouse, "schema", None)


def check_connection(client: SupersetClient | None = None) -> dict:
    return (client or SupersetClient()).ping()
