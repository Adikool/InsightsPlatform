"""Semantic type inference.

pandas gives us storage types; the NL layer needs *meaning*. An int64 column named
`order_id` with 100% distinct values is an identifier, not a measure you should sum.
Getting this wrong is the single biggest source of nonsense answers, so the rules
below combine dtype, cardinality, and name evidence rather than any one signal.
"""

from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as ptypes

from .models import ColumnMeta, SemanticType

_ID_NAME = re.compile(r"(^|_)(id|uuid|guid|key|code|no|number|sk|pk)$", re.I)
_CURRENCY_NAME = re.compile(
    r"(price|cost|revenue|amount|sales|total|salary|spend|profit|margin|fee|usd|eur|inr|gbp)",
    re.I,
)
_TIME_NAME = re.compile(r"(date|time|timestamp|_at$|_on$|month|year|quarter|week|day)", re.I)
_GEO_NAME = re.compile(r"(country|region|state|city|province|zip|postal|lat|lon|lng|geo)", re.I)

_MAX_SAMPLES = 10


def _looks_temporal(series: pd.Series) -> bool:
    """Try to parse an object column as dates without spamming warnings."""
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return parsed.notna().mean() > 0.9


def infer_semantic_type(name: str, series: pd.Series) -> SemanticType:
    non_null = series.dropna()
    n = len(non_null)
    n_unique = int(non_null.nunique()) if n else 0
    uniqueness = n_unique / n if n else 0.0

    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(series) or ptypes.is_timedelta64_dtype(series):
        return "temporal"

    if ptypes.is_numeric_dtype(series):
        # An id-named integer column is a key — primary if near-unique, foreign if
        # it repeats. Either way summing it is meaningless, so it is never a measure.
        if _ID_NAME.search(name) and ptypes.is_integer_dtype(series):
            return "identifier"
        if _CURRENCY_NAME.search(name):
            return "currency"
        # Small-cardinality integers are usually codes/flags used for grouping.
        if ptypes.is_integer_dtype(series) and n_unique <= 12 and n > 50:
            return "categorical"
        if _TIME_NAME.search(name) and n_unique < 200:
            return "categorical"
        return "numeric"

    # object / string from here on
    if _TIME_NAME.search(name) and _looks_temporal(series):
        return "temporal"
    if _ID_NAME.search(name) and uniqueness > 0.8:
        return "identifier"
    if _GEO_NAME.search(name):
        return "geo"
    if _looks_temporal(series):
        return "temporal"

    avg_len = float(non_null.astype(str).str.len().mean()) if n else 0.0
    # Long strings are prose whether or not they repeat; a canned support note that
    # appears 4,000 times is still text, and grouping by it is not useful.
    if avg_len > 60 or (uniqueness > 0.6 and avg_len > 40):
        return "text"
    if n_unique <= max(50, 0.2 * n):
        return "categorical"
    return "text"


def profile_column(name: str, series: pd.Series) -> ColumnMeta:
    non_null = series.dropna()
    semantic = infer_semantic_type(name, series)

    minimum = maximum = None
    if semantic in ("numeric", "currency") and not non_null.empty:
        minimum, maximum = float(non_null.min()), float(non_null.max())
    elif semantic == "temporal" and not non_null.empty:
        try:
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            if parsed.notna().any():
                minimum = str(parsed.min().date())
                maximum = str(parsed.max().date())
        except (ValueError, TypeError):
            pass

    samples: list[str] = []
    top_share: float | None = None
    if semantic in ("categorical", "boolean", "geo", "text", "identifier") and not non_null.empty:
        counts = non_null.value_counts()
        samples = [str(v) for v in counts.head(_MAX_SAMPLES).index]
        top_share = float(counts.iloc[0] / len(non_null))

    return ColumnMeta(
        name=name,
        dtype=str(series.dtype),
        semantic_type=semantic,
        nullable=bool(series.isna().any()),
        n_unique=int(non_null.nunique()) if len(non_null) else 0,
        null_fraction=float(series.isna().mean()) if len(series) else 0.0,
        min=minimum,
        max=maximum,
        sample_values=samples,
        pct_top_value=top_share,
    )


def profile_frame(df: pd.DataFrame, sample: int | None = None) -> list[ColumnMeta]:
    frame = df.head(sample) if sample and len(df) > sample else df
    return [profile_column(str(col), frame[col]) for col in frame.columns]


def guess_grain(columns: list[ColumnMeta], n_rows: int) -> str:
    """A one-line human description of what one row represents."""
    ids = [c.name for c in columns if c.semantic_type == "identifier"]
    times = [c.name for c in columns if c.semantic_type == "temporal"]
    unique_ids = [c.name for c in columns if c.n_unique and n_rows and c.n_unique >= n_rows * 0.98]
    if unique_ids:
        return f"one row per {unique_ids[0]}"
    if ids and times:
        return f"one row per {ids[0]} per {times[0]}"
    if ids:
        return f"one row per {ids[0]} (repeating)"
    return "unspecified"
