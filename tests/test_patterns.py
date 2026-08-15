"""Detector tests, written against data with a known ground truth.

Every case here plants exactly one property and asserts the detector finds it —
and, just as important, that the detectors which should stay quiet do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataplatform.ds import detect_all, infer_task, profile, recommend
from dataplatform.ds.patterns import (
    _changepoint,
    detect_correlations,
    detect_distributions,
    detect_target_patterns,
    detect_time_patterns,
)

RNG = np.random.default_rng(7)


def daily(n: int = 730) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq="D")


# ----------------------------------------------------------------- changepoint
def test_changepoint_found_where_planted():
    n = 600
    values = np.concatenate([np.full(300, 100.0), np.full(300, 130.0)])
    values = values + RNG.normal(0, 3, n)
    found = _changepoint(values)
    assert found is not None
    index, _f, relative = found
    assert 280 <= index <= 320
    assert 0.25 <= relative <= 0.35


def test_no_changepoint_on_a_pure_trend():
    """The failure this replaced: a trend read as a step."""
    values = np.linspace(100, 300, 600) + RNG.normal(0, 5, 600)
    assert _changepoint(values) is None


def test_no_changepoint_on_pure_seasonality():
    t = np.arange(730)
    values = 100 + 20 * np.sin(2 * np.pi * t / 365) + RNG.normal(0, 2, 730)
    assert _changepoint(values, periods=[365]) is None


def test_no_changepoint_on_multiplicative_growth():
    """Growing seasonal amplitude previously registered as a level shift."""
    t = np.arange(1095)
    values = (100 + 0.2 * t) * (1 + 0.4 * np.sin(2 * np.pi * t / 365))
    values = values * RNG.normal(1, 0.05, 1095)
    assert _changepoint(values, periods=[7, 30, 365]) is None


def test_tiny_step_is_not_reported():
    """Statistically certain but 1% — below the level anyone would act on."""
    values = np.concatenate([np.full(2000, 100.0), np.full(2000, 101.0)])
    values = values + RNG.normal(0, 0.5, 4000)
    assert _changepoint(values) is None


# -------------------------------------------------------------------- trend
def test_trend_detected():
    frame = pd.DataFrame(
        {"day": daily(), "revenue": np.linspace(100, 500, 730) + RNG.normal(0, 10, 730)}
    )
    patterns = detect_time_patterns(frame, profile(frame))
    trend = [p for p in patterns if p.kind == "trend"]
    assert trend and trend[0].statistic > 0.8


def test_seasonality_detected():
    t = np.arange(730)
    frame = pd.DataFrame(
        {"day": daily(), "sales": 100 + 30 * np.sin(2 * np.pi * t / 7) + RNG.normal(0, 3, 730)}
    )
    patterns = detect_time_patterns(frame, profile(frame))
    seasonal = [p for p in patterns if p.kind == "seasonality"]
    assert seasonal and seasonal[0].evidence["period"] == 7


def test_flat_noise_has_no_trend_or_seasonality():
    frame = pd.DataFrame({"day": daily(), "value": RNG.normal(100, 5, 730)})
    kinds = {p.kind for p in detect_time_patterns(frame, profile(frame))}
    assert "trend" not in kinds
    assert "seasonality" not in kinds
    assert "changepoint" not in kinds


# ------------------------------------------------------------- distributions
def test_skew_and_outliers():
    frame = pd.DataFrame({"amount": RNG.lognormal(3, 1.2, 5000)})
    patterns = detect_distributions(frame, profile(frame))
    kinds = {p.kind for p in patterns}
    assert "skewed_distribution" in kinds
    assert "outliers" in kinds


def test_symmetric_data_is_not_flagged_skewed():
    frame = pd.DataFrame({"value": RNG.normal(50, 5, 5000)})
    kinds = {p.kind for p in detect_distributions(frame, profile(frame))}
    assert "skewed_distribution" not in kinds


def test_zero_inflation():
    values = np.where(RNG.random(3000) < 0.7, 0.0, RNG.lognormal(2, 1, 3000))
    frame = pd.DataFrame({"claim_amount": values})
    kinds = {p.kind for p in detect_distributions(frame, profile(frame))}
    assert "zero_inflation" in kinds


# ---------------------------------------------------------------- relations
def test_correlation_and_nonlinearity():
    x = RNG.uniform(1, 100, 2000)
    frame = pd.DataFrame({"x": x, "linear": 3 * x + RNG.normal(0, 5, 2000), "curved": x**4})
    patterns = detect_correlations(frame, profile(frame))
    kinds = {p.kind for p in patterns}
    assert "correlation" in kinds
    assert "nonlinear_relationship" in kinds


def test_independent_columns_are_not_correlated():
    frame = pd.DataFrame({"a": RNG.normal(size=2000), "b": RNG.normal(size=2000)})
    assert not [p for p in detect_correlations(frame, profile(frame)) if p.kind == "correlation"]


# ------------------------------------------------------------------- target
def test_leakage_and_imbalance():
    n = 5000
    revenue = RNG.lognormal(4, 0.6, n)
    frame = pd.DataFrame(
        {
            "revenue": revenue,
            "revenue_restated": revenue * 1.0001,
            "units": RNG.integers(1, 5, n),
            "churn": (RNG.random(n) < 0.02).astype(int),
        }
    )
    stats = profile(frame)
    leakage = [p for p in detect_target_patterns(frame, stats, "revenue") if p.kind == "target_leakage"]
    assert leakage and "revenue_restated" in leakage[0].columns

    imbalance = [p for p in detect_target_patterns(frame, stats, "churn") if p.kind == "class_imbalance"]
    assert imbalance and imbalance[0].severity == "strong"


def test_group_effect():
    n = 3000
    group = RNG.choice(["a", "b", "c"], n)
    value = np.where(group == "a", 10.0, np.where(group == "b", 50.0, 90.0)) + RNG.normal(0, 3, n)
    frame = pd.DataFrame({"segment": group, "value": value})
    patterns = detect_target_patterns(frame, profile(frame), "value")
    effects = [p for p in patterns if p.kind == "group_effect"]
    assert effects and effects[0].statistic > 0.8


# ----------------------------------------------------------------- task/recs
@pytest.mark.parametrize(
    "target,expected",
    [
        ("price", "regression"),
        ("is_fraud", "binary_classification"),
        ("tier", "multiclass_classification"),
        (None, "clustering"),
    ],
)
def test_task_inference(target, expected):
    n = 1000
    frame = pd.DataFrame(
        {
            "price": RNG.lognormal(3, 0.5, n),
            "weight": RNG.normal(10, 2, n),
            "is_fraud": RNG.integers(0, 2, n),
            "tier": RNG.choice(["bronze", "silver", "gold"], n),
        }
    )
    assert infer_task(frame, profile(frame), target) == expected


def test_recommendations_are_ranked_and_evidenced():
    n = 2000
    frame = pd.DataFrame(
        {
            "amount": RNG.lognormal(4, 1.1, n),
            "units": RNG.integers(1, 20, n),
            "region": RNG.choice(["N", "S", "E", "W"], n),
        }
    )
    stats = profile(frame)
    patterns = detect_all(frame, stats, target="amount")
    recs = recommend(frame, stats, patterns, target="amount")

    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))
    assert "Dummy" in recs[0].algorithm  # the baseline always comes first
    assert all(r.rationale for r in recs)  # nothing is recommended without a reason
    assert all(r.evaluation for r in recs)


def test_outlier_heavy_data_recommends_robust_regression():
    n = 2000
    amount = RNG.normal(100, 10, n)
    amount[:100] = RNG.normal(1000, 200, 100)  # a contaminated tail
    frame = pd.DataFrame({"amount": amount, "driver": RNG.normal(size=n)})
    stats = profile(frame)
    recs = recommend(frame, stats, detect_all(frame, stats, target="amount"), target="amount")
    assert any("Huber" in r.algorithm for r in recs)
