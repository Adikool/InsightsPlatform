"""The QuerySpec — the interface between natural language and SQL.

Everything the language layer produces lands in one of these. The schema is
deliberately closed (fixed enums, no free-form expression fields) so that a
malformed or adversarial model output cannot express anything the compiler will
not first validate against the catalog.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TimeGrain = Literal["none", "day", "week", "month", "quarter", "year"]

AggFunc = Literal[
    "sum", "avg", "min", "max", "count", "count_distinct", "median", "stddev"
]

FilterOp = Literal[
    "=", "!=", ">", ">=", "<", "<=",
    "in", "not_in",
    "contains", "starts_with",
    "between",
    "is_null", "is_not_null",
    "last_n_days", "last_n_months",
]

ChartType = Literal[
    "table", "line", "bar", "horizontal_bar", "area", "pie",
    "big_number", "scatter", "heatmap",
]


class Dimension(BaseModel):
    column: str = Field(description="Column to group by. Must exist in the dataset.")
    time_grain: TimeGrain = Field(
        default="none",
        description="Truncation for temporal columns; 'none' for non-temporal columns.",
    )
    alias: str = Field(default="", description="Output name. Empty to auto-generate.")


class Metric(BaseModel):
    column: str = Field(description="Column to aggregate, or '*' with func='count'.")
    func: AggFunc = Field(description="Aggregate function.")
    alias: str = Field(default="", description="Output name. Empty to auto-generate.")


class Filter(BaseModel):
    column: str = Field(description="Column to filter on, or a metric alias for HAVING.")
    op: FilterOp = Field(description="Comparison operator.")
    values: list[str] = Field(
        default_factory=list,
        description=(
            "Literal operands as strings. One for scalar comparisons, two for "
            "'between', many for 'in'/'not_in', one integer for 'last_n_*', "
            "none for null checks."
        ),
    )


class Sort(BaseModel):
    field: str = Field(description="An output alias produced by this query.")
    descending: bool = True


class QuerySpec(BaseModel):
    dataset: str = Field(description="Table name from the catalog.")
    dimensions: list[Dimension] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    sort: list[Sort] = Field(default_factory=list)
    limit: int = Field(default=1000, description="Row cap, 1..50000.")
    chart: ChartType = Field(default="table", description="How to visualise the result.")
    title: str = Field(default="", description="Short human title for the chart.")
    explanation: str = Field(
        default="", description="One sentence on how the question was interpreted."
    )

    @property
    def is_aggregate(self) -> bool:
        return bool(self.metrics)


class CompiledQuery(BaseModel):
    """A validated spec plus the SQL it produced."""

    spec: QuerySpec
    sql: str
    columns: list[str]
    dimension_aliases: list[str] = Field(default_factory=list)
    metric_aliases: list[str] = Field(default_factory=list)
    time_column: str = ""
    warnings: list[str] = Field(default_factory=list)
