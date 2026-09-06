"""Catalog data model.

The catalog is the contract between every layer: connectors write into it,
the NL layer reads it to ground the model, the SQL compiler uses it as an
allow-list, and Superset publishing uses it to name datasets.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SemanticType = Literal[
    "identifier",
    "categorical",
    "numeric",
    "currency",
    "temporal",
    "boolean",
    "text",
    "geo",
    "unknown",
]


class ColumnMeta(BaseModel):
    name: str
    dtype: str
    semantic_type: SemanticType = "unknown"
    nullable: bool = True
    n_unique: int | None = None
    null_fraction: float | None = None
    min: Any | None = None
    max: Any | None = None
    sample_values: list[str] = Field(default_factory=list)
    # Share of non-null rows held by the single commonest value. A column that is
    # 94% one value (an empty note, a default flag) is technically categorical and
    # useless to group by, and cardinality alone does not reveal that.
    pct_top_value: float | None = None
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)

    @property
    def is_measure(self) -> bool:
        return self.semantic_type in ("numeric", "currency")

    @property
    def is_numeric_dtype(self) -> bool:
        """Can this column be arithmetic-aggregated at all?

        Distinct from `is_measure`: an integer column with 11 distinct values is
        inferred categorical (you group by it), but averaging it — mean delivery
        days — is still perfectly well defined.
        """
        return self.dtype.lower().startswith(("int", "uint", "float", "decimal", "number"))

    @property
    def is_dimension(self) -> bool:
        return self.semantic_type in ("categorical", "boolean", "temporal", "geo", "identifier")


class DatasetMeta(BaseModel):
    """One physical table in the warehouse, plus everything the NL layer needs."""

    name: str
    source: str
    origin_object: str
    n_rows: int = 0
    columns: list[ColumnMeta] = Field(default_factory=list)
    description: str = ""
    grain: str = ""
    ingested_at: str = ""

    def column(self, name: str) -> ColumnMeta | None:
        lowered = name.lower()
        for col in self.columns:
            if col.name.lower() == lowered:
                return col
        return None

    def resolve(self, name: str) -> str | None:
        """Case-insensitive / synonym-aware column resolution -> canonical name."""
        lowered = name.strip().lower()
        normalised = lowered.replace(" ", "_")
        fallback: str | None = None
        for col in self.columns:
            col_lower = col.name.lower()
            if col_lower == lowered:
                return col.name  # exact match wins immediately
            if fallback is None:
                if col_lower.replace(" ", "_") == normalised:
                    fallback = col.name
                elif any(syn.lower() == lowered for syn in col.synonyms):
                    fallback = col.name
        return fallback

    @property
    def measures(self) -> list[ColumnMeta]:
        return [c for c in self.columns if c.is_measure]

    @property
    def dimensions(self) -> list[ColumnMeta]:
        return [c for c in self.columns if c.is_dimension]

    @property
    def temporal_columns(self) -> list[ColumnMeta]:
        return [c for c in self.columns if c.semantic_type == "temporal"]


class SourceMeta(BaseModel):
    name: str
    type: Literal["sql", "excel", "csv", "parquet"]
    uri: str
    options: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class DashboardHistoryEntry(BaseModel):
    title: str
    dashboard_url: str | None = None
    chart_url: str | None = None
    datasets: list[str] = Field(default_factory=list)
    n_charts: int = 0
    created_at: str = ""


class DashboardActivityEntry(BaseModel):
    """One explicit dashboard action — a "Preview plan" or "Publish" click.

    Distinct from `DashboardHistoryEntry`: this logs every explicit action, not
    just successful publishes, so the user has a full record of what they asked
    for, not only what made it to Superset. Live-typing recomposition is not
    logged here — only the button clicks are.
    """

    action: Literal["preview", "publish"]
    title: str
    request: str = ""
    datasets: list[str] = Field(default_factory=list)
    dashboard_url: str | None = None
    n_charts: int = 0
    created_at: str = ""


class ExploreActivityEntry(BaseModel):
    """One question submitted from the Explore tab (`/ask`)."""

    question: str
    mode: Literal["spec", "agent"] = "spec"
    sql: str | None = None
    row_count: int | None = None
    published: bool = False
    dashboard_url: str | None = None
    # Publishing without a dashboard title creates a standalone chart, which
    # has no dashboard_url — chart_url is the fallback so a published entry
    # always has *some* link to show, instead of looking unpublished.
    chart_url: str | None = None
    created_at: str = ""


class AskActivityEntry(BaseModel):
    """One question submitted from the Ask tab (`/ask-nlp`)."""

    question: str
    dataset: str | None = None
    created_at: str = ""


class CatalogState(BaseModel):
    sources: dict[str, SourceMeta] = Field(default_factory=dict)
    datasets: dict[str, DatasetMeta] = Field(default_factory=dict)
    metrics: dict[str, str] = Field(default_factory=dict)
    dashboard_history: list[DashboardHistoryEntry] = Field(default_factory=list)
    dashboard_activity: list[DashboardActivityEntry] = Field(default_factory=list)
    explore_activity: list[ExploreActivityEntry] = Field(default_factory=list)
    ask_activity: list[AskActivityEntry] = Field(default_factory=list)
