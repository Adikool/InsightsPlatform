"""Result types for the data-science layer.

Everything the layer produces is a plain pydantic object: serialisable to the API,
printable by the CLI, and — importantly — carrying the *evidence* alongside every
claim, so a recommendation can be argued with rather than merely accepted.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "notable", "strong"]

TaskType = Literal[
    "regression",
    "binary_classification",
    "multiclass_classification",
    "time_series_forecasting",
    "clustering",
    "anomaly_detection",
    "dimensionality_reduction",
    "association_rules",
    "text_classification",
    "survival_analysis",
    "none",
]


class ColumnStats(BaseModel):
    name: str
    dtype: str
    semantic_type: str
    n_missing: int = 0
    pct_missing: float = 0.0
    n_unique: int = 0
    pct_unique: float = 0.0
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    maximum: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    pct_zero: float | None = None
    pct_negative: float | None = None
    top_value: str | None = None
    pct_top_value: float | None = None
    is_constant: bool = False
    is_high_cardinality: bool = False


class DatasetProfile(BaseModel):
    dataset: str
    n_rows: int
    n_columns: int
    n_duplicate_rows: int = 0
    memory_mb: float = 0.0
    columns: list[ColumnStats] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    temporal_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    candidate_keys: list[str] = Field(default_factory=list)

    def stats(self, column: str) -> ColumnStats | None:
        return next((c for c in self.columns if c.name == column), None)


class Pattern(BaseModel):
    kind: str
    severity: Severity = "info"
    columns: list[str] = Field(default_factory=list)
    statistic: float | None = None
    p_value: float | None = None
    description: str = ""
    implication: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        head = f"[{self.severity}] {self.kind}"
        if self.columns:
            head += f" ({', '.join(self.columns)})"
        return f"{head}: {self.description}"


class Recommendation(BaseModel):
    rank: int
    task: TaskType
    algorithm: str
    library: str = "scikit-learn"
    confidence: Literal["low", "medium", "high"] = "medium"
    rationale: list[str] = Field(default_factory=list)
    preprocessing: list[str] = Field(default_factory=list)
    evaluation: str = ""
    caveats: list[str] = Field(default_factory=list)
    starter_code: str = ""

    def __str__(self) -> str:
        return f"{self.rank}. {self.algorithm} ({self.task}, confidence: {self.confidence})"


class BaselineResult(BaseModel):
    algorithm: str
    task: TaskType
    metrics: dict[str, float] = Field(default_factory=dict)
    cv_folds: int = 0
    n_train: int = 0
    n_features: int = 0
    feature_importance: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    dataset: str
    target: str | None = None
    task: TaskType = "none"
    profile: DatasetProfile
    patterns: list[Pattern] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    narrative: str = ""

    def top(self, n: int = 3) -> list[Recommendation]:
        return self.recommendations[:n]

    def patterns_of(self, kind: str) -> list[Pattern]:
        return [p for p in self.patterns if p.kind == kind]
