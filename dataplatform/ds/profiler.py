"""Statistical profiling — the substrate every pattern detector reads."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ..catalog.inference import infer_semantic_type
from .models import ColumnStats, DatasetProfile

HIGH_CARDINALITY_RATIO = 0.5
HIGH_CARDINALITY_ABS = 50


def profile(df: pd.DataFrame, name: str = "dataset") -> DatasetProfile:
    n_rows = len(df)
    columns: list[ColumnStats] = []

    numeric: list[str] = []
    categorical: list[str] = []
    temporal: list[str] = []
    text: list[str] = []
    candidate_keys: list[str] = []

    for raw_name in df.columns:
        col = str(raw_name)
        series = df[raw_name]
        non_null = series.dropna()
        n_unique = int(non_null.nunique()) if len(non_null) else 0
        semantic = infer_semantic_type(col, series)

        stats = ColumnStats(
            name=col,
            dtype=str(series.dtype),
            semantic_type=semantic,
            n_missing=int(series.isna().sum()),
            pct_missing=float(series.isna().mean()) if n_rows else 0.0,
            n_unique=n_unique,
            pct_unique=float(n_unique / n_rows) if n_rows else 0.0,
            is_constant=n_unique <= 1,
            is_high_cardinality=(
                semantic in ("categorical", "text", "identifier")
                and n_unique > HIGH_CARDINALITY_ABS
                and (n_unique / max(n_rows, 1)) > HIGH_CARDINALITY_RATIO * 0.2
            ),
        )

        if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
            values = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(values):
                stats.mean = float(values.mean())
                stats.std = float(values.std()) if len(values) > 1 else 0.0
                stats.minimum = float(values.min())
                stats.p25 = float(values.quantile(0.25))
                stats.median = float(values.median())
                stats.p75 = float(values.quantile(0.75))
                stats.maximum = float(values.max())
                stats.skew = float(values.skew()) if len(values) > 2 else None
                stats.kurtosis = float(values.kurtosis()) if len(values) > 3 else None
                stats.pct_zero = float((values == 0).mean())
                stats.pct_negative = float((values < 0).mean())

        if len(non_null) and semantic in ("categorical", "boolean", "geo", "identifier", "text"):
            counts = non_null.value_counts()
            stats.top_value = str(counts.index[0])
            stats.pct_top_value = float(counts.iloc[0] / len(non_null))

        # bucket the column
        if semantic in ("numeric", "currency"):
            numeric.append(col)
        elif semantic == "temporal":
            temporal.append(col)
        elif semantic == "text":
            text.append(col)
        elif semantic in ("categorical", "boolean", "geo"):
            categorical.append(col)

        if n_rows and n_unique >= n_rows * 0.99 and stats.pct_missing == 0:
            candidate_keys.append(col)

        columns.append(stats)

    try:
        duplicates = int(df.duplicated().sum())
    except TypeError:  # unhashable cells (lists/dicts) — not worth failing over
        duplicates = 0

    return DatasetProfile(
        dataset=name,
        n_rows=n_rows,
        n_columns=len(df.columns),
        n_duplicate_rows=duplicates,
        memory_mb=float(df.memory_usage(deep=True).sum() / 1_048_576),
        columns=columns,
        numeric_columns=numeric,
        categorical_columns=categorical,
        temporal_columns=temporal,
        text_columns=text,
        candidate_keys=candidate_keys,
    )


def numeric_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Numeric frame with constant and all-null columns dropped."""
    if not columns:
        return pd.DataFrame(index=df.index)
    frame = df[columns].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in frame.columns if frame[c].notna().sum() > 2 and frame[c].nunique() > 1]
    return frame[keep]


def infer_frequency(index: pd.DatetimeIndex) -> tuple[str, float]:
    """Return (pandas offset alias, median gap in days) for an irregular index."""
    if len(index) < 3:
        return "D", 1.0
    deltas = pd.Series(index.sort_values()).diff().dropna().dt.total_seconds() / 86400.0
    median_days = float(deltas.median()) if len(deltas) else 1.0
    if median_days <= 1.5:
        return "D", median_days
    if median_days <= 9:
        return "W", median_days
    if median_days <= 45:
        return "MS", median_days
    if median_days <= 120:
        return "QS", median_days
    return "YS", median_days


SEASONAL_PERIODS = {"D": [7, 30, 365], "W": [4, 13, 52], "MS": [3, 6, 12], "QS": [4], "YS": []}


def build_series(
    df: pd.DataFrame, time_column: str, value_column: str, how: str = "sum"
) -> tuple[pd.Series, str]:
    """Regularly-spaced series for the time-series detectors."""
    frame = df[[time_column, value_column]].dropna()
    if frame.empty:
        return pd.Series(dtype=float), "D"
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame = frame.dropna(subset=[time_column]).set_index(time_column).sort_index()
    freq, _ = infer_frequency(pd.DatetimeIndex(frame.index))
    values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
    if values.empty:
        return pd.Series(dtype=float), freq
    resampled = values.resample(freq).agg(how)
    return resampled.astype(float), freq


def ols_slope(y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares slope of y against its index.

    Returns (slope, r_squared, t_statistic). Written out rather than pulled from
    scipy so the core detectors work without the optional ml extra.
    """
    n = len(y)
    if n < 3:
        return 0.0, 0.0, 0.0
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = float((x_centered**2).sum())
    if denom == 0:
        return 0.0, 0.0, 0.0
    slope = float((x_centered * y_centered).sum() / denom)
    fitted = y.mean() + slope * x_centered
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float((y_centered**2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if n > 2 and ss_res > 0:
        se = np.sqrt(ss_res / (n - 2) / denom)
        t_stat = slope / se if se > 0 else 0.0
    else:
        t_stat = 0.0
    return slope, r2, float(t_stat)
