"""Chart-type selection from the shape of a query.

Used to (a) fill in `chart` when the model omits it or picks something the data
cannot support, and (b) drive the Superset `viz_type`. Shape beats preference: a
pie chart of 400 categories is wrong no matter who asked for it.
"""

from __future__ import annotations

from ..catalog.models import DatasetMeta
from .spec import ChartType, CompiledQuery

_MAX_PIE_SLICES = 8


def suggest_chart(compiled: CompiledQuery, dataset: DatasetMeta, n_rows: int | None = None) -> ChartType:
    n_dims = len(compiled.dimension_aliases)
    n_metrics = len(compiled.metric_aliases)
    has_time = bool(compiled.time_column)

    if n_metrics == 0:
        return "table"
    if n_dims == 0:
        return "big_number"
    if has_time:
        return "area" if n_metrics > 1 and n_dims == 1 else "line"
    if n_dims == 1:
        if n_metrics == 1 and n_rows is not None and n_rows <= _MAX_PIE_SLICES:
            return "pie"
        if n_rows is not None and n_rows > 25:
            return "horizontal_bar"
        return "bar"
    if n_dims == 2 and n_metrics == 1:
        return "heatmap"
    return "table"


def reconcile(compiled: CompiledQuery, dataset: DatasetMeta, n_rows: int | None = None) -> ChartType:
    """Keep the requested chart when the data supports it, else fall back."""
    requested = compiled.spec.chart
    suggested = suggest_chart(compiled, dataset, n_rows)

    n_dims = len(compiled.dimension_aliases)
    n_metrics = len(compiled.metric_aliases)

    supportable = {
        "table": True,
        "big_number": n_metrics >= 1 and n_dims == 0,
        "line": n_metrics >= 1 and n_dims >= 1,
        "area": n_metrics >= 1 and n_dims >= 1,
        "bar": n_metrics >= 1 and n_dims >= 1,
        "horizontal_bar": n_metrics >= 1 and n_dims >= 1,
        "pie": n_metrics == 1 and n_dims == 1 and (n_rows is None or n_rows <= _MAX_PIE_SLICES),
        "scatter": n_metrics >= 2,
        "heatmap": n_dims >= 2 and n_metrics >= 1,
    }
    return requested if supportable.get(requested, False) else suggested


SUPERSET_VIZ = {
    "table": "table",
    "line": "echarts_timeseries_line",
    "area": "echarts_area",
    "bar": "echarts_timeseries_bar",
    "horizontal_bar": "echarts_timeseries_bar",
    "pie": "pie",
    "big_number": "big_number_total",
    "scatter": "echarts_timeseries_scatter",
    "heatmap": "heatmap_v2",
}
