"""Pattern detection.

Each detector answers one question about the data and returns zero or more
`Pattern` objects carrying the statistic that produced them. The recommender
consumes these — that is the whole point of the layer: the algorithm advice is
*derived from* measured properties rather than from the shape of the file.

scipy is optional. Where it is present we report exact p-values; where it is not
we fall back to normal approximations and say so by leaving `p_value` unset.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd

from .models import DatasetProfile, Pattern
from .profiler import SEASONAL_PERIODS, build_series, numeric_matrix, ols_slope

try:  # optional
    from scipy import stats as _scipy_stats

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _scipy_stats = None
    HAVE_SCIPY = False

# --- thresholds, gathered here so they can be argued with in one place -------
MISSING_NOTABLE = 0.05
MISSING_STRONG = 0.30
SKEW_NOTABLE = 1.0
SKEW_STRONG = 2.0
OUTLIER_NOTABLE = 0.01
OUTLIER_STRONG = 0.05
CORR_NOTABLE = 0.6
CORR_STRONG = 0.85
NONLINEAR_GAP_INFO = 0.10
NONLINEAR_GAP_NOTABLE = 0.20
# 0.95 rather than 0.99: a feature this correlated with the target is nearly always
# derived from it (gross_margin = revenue - cost sits at 0.98). A genuine predictor
# occasionally lands here, which is why the finding says "verify" and the baseline
# reports what it dropped rather than dropping it silently.
LEAKAGE_CORR = 0.95
VIF_NOTABLE = 10.0
TREND_R2 = 0.25
ACF_NOTABLE = 0.3
CHANGEPOINT_MIN_STEP = 0.05  # a shift smaller than 5% of the level is not worth reporting
IMBALANCE_NOTABLE = 5.0
IMBALANCE_STRONG = 20.0
MIN_SERIES_POINTS = 12


def _p_from_t(t_stat: float, dof: int) -> float | None:
    if dof <= 0:
        return None
    if HAVE_SCIPY:
        return float(2 * _scipy_stats.t.sf(abs(t_stat), dof))
    # Normal approximation is adequate for the dof we see in practice (>30).
    return float(math.erfc(abs(t_stat) / math.sqrt(2))) if dof > 30 else None


# ---------------------------------------------------------------- data hygiene
def detect_missingness(df: pd.DataFrame, profile: DatasetProfile) -> list[Pattern]:
    found: list[Pattern] = []
    for stats in profile.columns:
        if stats.pct_missing < MISSING_NOTABLE:
            continue
        severity = "strong" if stats.pct_missing >= MISSING_STRONG else "notable"
        found.append(
            Pattern(
                kind="missing_data",
                severity=severity,
                columns=[stats.name],
                statistic=round(stats.pct_missing, 4),
                description=f"{stats.name} is {stats.pct_missing:.1%} null",
                implication=(
                    "Drop the column or use a learner with native missing support "
                    "(HistGradientBoosting, LightGBM); simple mean imputation at this "
                    "rate will distort the distribution."
                    if severity == "strong"
                    else "Impute; add a missingness indicator if the nulls may be informative."
                ),
            )
        )

    # Is missingness itself structured? If two columns go missing together the
    # nulls are a mechanism, not noise, and imputation must respect that.
    missing_cols = [s.name for s in profile.columns if 0 < s.pct_missing < 1]
    if len(missing_cols) >= 2:
        indicator = df[missing_cols].isna().astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = indicator.corr()
        for i, a in enumerate(missing_cols):
            for b in missing_cols[i + 1 :]:
                value = corr.loc[a, b]
                if pd.notna(value) and abs(value) > 0.8:
                    found.append(
                        Pattern(
                            kind="structured_missingness",
                            severity="notable",
                            columns=[a, b],
                            statistic=round(float(value), 3),
                            description=f"{a} and {b} are missing together (r={value:.2f})",
                            implication=(
                                "Nulls are not at random — model the mechanism or use "
                                "indicator variables rather than imputing independently."
                            ),
                        )
                    )
    return found


def detect_structural_issues(df: pd.DataFrame, profile: DatasetProfile) -> list[Pattern]:
    found: list[Pattern] = []

    constants = [s.name for s in profile.columns if s.is_constant]
    if constants:
        found.append(
            Pattern(
                kind="constant_columns",
                severity="notable",
                columns=constants,
                statistic=float(len(constants)),
                description=f"{len(constants)} column(s) carry a single value",
                implication="No information for any model; drop before training.",
            )
        )

    if profile.n_rows and profile.n_duplicate_rows:
        fraction = profile.n_duplicate_rows / profile.n_rows
        if fraction > 0.001:
            found.append(
                Pattern(
                    kind="duplicate_rows",
                    severity="strong" if fraction > 0.05 else "notable",
                    statistic=round(fraction, 4),
                    description=f"{profile.n_duplicate_rows:,} duplicate rows ({fraction:.1%})",
                    implication=(
                        "Duplicates leak between train and test folds and inflate every "
                        "score. De-duplicate before splitting, or confirm they are genuine."
                    ),
                )
            )

    high_card = [s.name for s in profile.columns if s.is_high_cardinality]
    if high_card:
        found.append(
            Pattern(
                kind="high_cardinality",
                severity="notable",
                columns=high_card,
                statistic=float(len(high_card)),
                description=f"high-cardinality categoricals: {', '.join(high_card[:5])}",
                implication=(
                    "One-hot encoding will explode the feature space. Use target/ordinal "
                    "encoding, hashing, or a learner with native categorical support "
                    "(CatBoost, LightGBM)."
                ),
            )
        )

    if profile.text_columns:
        found.append(
            Pattern(
                kind="free_text",
                severity="info",
                columns=profile.text_columns,
                description=f"free-text column(s): {', '.join(profile.text_columns[:5])}",
                implication=(
                    "Unused by tabular models. TF-IDF + a linear model is the cheap "
                    "baseline; sentence embeddings if the text carries real signal."
                ),
            )
        )
    return found


def detect_distributions(df: pd.DataFrame, profile: DatasetProfile) -> list[Pattern]:
    found: list[Pattern] = []
    for stats in profile.columns:
        if stats.name not in profile.numeric_columns or stats.skew is None:
            continue

        if abs(stats.skew) >= SKEW_NOTABLE:
            severity = "strong" if abs(stats.skew) >= SKEW_STRONG else "notable"
            positive = stats.minimum is not None and stats.minimum >= 0
            found.append(
                Pattern(
                    kind="skewed_distribution",
                    severity=severity,
                    columns=[stats.name],
                    statistic=round(stats.skew, 3),
                    description=f"{stats.name} is {'right' if stats.skew > 0 else 'left'}-skewed (skew={stats.skew:.2f})",
                    implication=(
                        "Apply log1p/Box-Cox before any distance- or variance-based method "
                        "(linear models, KMeans, PCA). Tree ensembles are unaffected."
                        if positive
                        else "Consider a Yeo-Johnson transform; the column has negative values "
                        "so log is unavailable."
                    ),
                )
            )

        if stats.pct_zero is not None and stats.pct_zero > 0.5:
            found.append(
                Pattern(
                    kind="zero_inflation",
                    severity="notable",
                    columns=[stats.name],
                    statistic=round(stats.pct_zero, 3),
                    description=f"{stats.name} is {stats.pct_zero:.0%} zeros",
                    implication=(
                        "A single conditional-mean model fits this badly. Consider a "
                        "two-part (hurdle) model, Tweedie objective, or zero-inflated count model."
                    ),
                )
            )

    # Outliers by the IQR rule, which does not assume normality.
    for column in profile.numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) < 20:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        fraction = float(mask.mean())
        if fraction >= OUTLIER_NOTABLE:
            found.append(
                Pattern(
                    kind="outliers",
                    severity="strong" if fraction >= OUTLIER_STRONG else "notable",
                    columns=[column],
                    statistic=round(fraction, 4),
                    description=f"{column} has {fraction:.1%} points outside 1.5×IQR",
                    evidence={
                        "lower_fence": float(q1 - 1.5 * iqr),
                        "upper_fence": float(q3 + 1.5 * iqr),
                        "n_outliers": int(mask.sum()),
                    },
                    implication=(
                        "Squared-error losses will chase these points. Use robust "
                        "regression (Huber/quantile), RobustScaler, or winsorise — "
                        "but check first whether they are the phenomenon of interest."
                    ),
                )
            )
    return found


# ------------------------------------------------------------------ relations
def detect_correlations(df: pd.DataFrame, profile: DatasetProfile) -> list[Pattern]:
    frame = numeric_matrix(df, profile.numeric_columns)
    if frame.shape[1] < 2:
        return []

    found: list[Pattern] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = frame.corr(method="pearson")
        spearman = frame.corr(method="spearman")

    columns = list(frame.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            r = pearson.loc[a, b]
            rho = spearman.loc[a, b]
            if pd.isna(r):
                continue
            n = int(frame[[a, b]].dropna().shape[0])

            # These are two independent questions, not an either/or: a pair can be
            # both strongly correlated and badly served by a straight line. The gap
            # between the two coefficients is how much of a perfectly monotone
            # relationship a linear fit throws away, so it sets the severity rather
            # than gating the finding at one arbitrary cutoff.
            gap = abs(rho) - abs(r) if pd.notna(rho) else 0.0
            if pd.notna(rho) and abs(rho) >= CORR_NOTABLE and gap >= NONLINEAR_GAP_INFO:
                found.append(
                    Pattern(
                        kind="nonlinear_relationship",
                        severity="notable" if gap >= NONLINEAR_GAP_NOTABLE else "info",
                        columns=[a, b],
                        statistic=round(float(rho), 3),
                        evidence={"pearson": round(float(r), 3)},
                        description=(
                            f"{a} and {b} are monotonically but not linearly related "
                            f"(spearman={rho:.2f} vs pearson={r:.2f})"
                        ),
                        implication=(
                            "A linear model will understate this. Use splines, a monotone "
                            "transform, or a tree ensemble."
                        ),
                    )
                )

            if abs(r) >= CORR_NOTABLE:
                dof = n - 2
                t_stat = (
                    abs(r) * math.sqrt(dof / max(1e-12, 1 - r**2)) if dof > 0 and abs(r) < 1 else 0.0
                )
                found.append(
                    Pattern(
                        kind="correlation",
                        severity="strong" if abs(r) >= CORR_STRONG else "notable",
                        columns=[a, b],
                        statistic=round(float(r), 3),
                        p_value=_p_from_t(t_stat, dof),
                        description=f"{a} and {b} move together (pearson r={r:.2f}, n={n:,})",
                        evidence={"spearman": round(float(rho), 3) if pd.notna(rho) else None},
                        implication=(
                            "Near-duplicate information: keep one, or combine them, before "
                            "fitting a linear model."
                            if abs(r) >= CORR_STRONG
                            else "A usable predictor pair; watch for collinearity in linear models."
                        ),
                    )
                )
    return found


def detect_multicollinearity(df: pd.DataFrame, profile: DatasetProfile) -> list[Pattern]:
    """VIF via the R² of regressing each column on the others."""
    frame = numeric_matrix(df, profile.numeric_columns).dropna()
    if frame.shape[1] < 3 or len(frame) < frame.shape[1] + 5:
        return []
    if frame.shape[1] > 40:  # the full O(k) least-squares sweep stops being cheap
        frame = frame.iloc[:, :40]

    found: list[Pattern] = []
    matrix = frame.to_numpy(dtype=float)
    for index, column in enumerate(frame.columns):
        y = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        try:
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = y - design @ coefficients
        ss_res = float((residual**2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot <= 0:
            continue
        r2 = 1 - ss_res / ss_tot
        vif = 1 / max(1e-9, 1 - r2)
        if vif >= VIF_NOTABLE:
            exact = r2 > 0.9999
            found.append(
                Pattern(
                    kind="multicollinearity",
                    severity="strong" if vif >= 30 else "notable",
                    columns=[str(column)],
                    statistic=round(min(float(vif), 1e6), 2),
                    description=(
                        f"{column} is an exact linear combination of the other numeric columns"
                        if exact
                        else f"{column} is {r2:.0%} explained by the other numeric columns (VIF={vif:.1f})"
                    ),
                    implication=(
                        "Coefficients from OLS/logistic regression will be unstable and "
                        "uninterpretable. Use Ridge/ElasticNet, drop one of the pair, or "
                        "reduce with PCA."
                    ),
                )
            )
    return found


# ------------------------------------------------------------------ time series
def detect_time_patterns(
    df: pd.DataFrame, profile: DatasetProfile, max_series: int = 6
) -> list[Pattern]:
    if not profile.temporal_columns or not profile.numeric_columns:
        return []

    time_column = profile.temporal_columns[0]
    found: list[Pattern] = []

    # Coverage and gaps first — they qualify everything else.
    times = pd.to_datetime(df[time_column], errors="coerce").dropna().sort_values()
    if len(times) >= 3:
        span_days = (times.iloc[-1] - times.iloc[0]).days
        gaps = times.diff().dropna()
        median_gap = gaps.median()
        big_gaps = int((gaps > median_gap * 5).sum()) if median_gap.total_seconds() > 0 else 0
        found.append(
            Pattern(
                kind="temporal_coverage",
                severity="info",
                columns=[time_column],
                statistic=float(span_days),
                description=(
                    f"{time_column} spans {span_days:,} days "
                    f"({times.iloc[0].date()} to {times.iloc[-1].date()})"
                ),
                evidence={"median_gap_days": median_gap.total_seconds() / 86400, "large_gaps": big_gaps},
                implication=(
                    "Enough history for a seasonal model."
                    if span_days > 730
                    else "Short history — a seasonal model cannot be validated on this span; "
                    "prefer simple trend/regression methods."
                ),
            )
        )
        if big_gaps:
            found.append(
                Pattern(
                    kind="temporal_gaps",
                    severity="notable",
                    columns=[time_column],
                    statistic=float(big_gaps),
                    description=f"{big_gaps} unusually large gap(s) in {time_column}",
                    implication=(
                        "Resample to a fixed frequency and decide explicitly whether gaps "
                        "mean zero or unknown — forecasting models cannot tell the difference."
                    ),
                )
            )

    for value_column in profile.numeric_columns[:max_series]:
        series, freq = build_series(
            df, time_column, value_column, how=_series_aggregation(value_column)
        )
        series = series.dropna()
        if len(series) < MIN_SERIES_POINTS:
            continue
        values = series.to_numpy(dtype=float)

        # --- trend ---------------------------------------------------------
        slope, r2, t_stat = ols_slope(values)
        p_value = _p_from_t(t_stat, len(values) - 2)
        if r2 >= TREND_R2 and abs(t_stat) > 2:
            direction = "rising" if slope > 0 else "falling"
            per_period = slope
            found.append(
                Pattern(
                    kind="trend",
                    severity="strong" if r2 >= 0.6 else "notable",
                    columns=[time_column, value_column],
                    statistic=round(float(r2), 3),
                    p_value=p_value,
                    description=(
                        f"{value_column} is {direction} over time "
                        f"({per_period:+,.2f} per {freq} period, R²={r2:.2f})"
                    ),
                    evidence={"slope": float(slope), "freq": freq, "n_periods": len(values)},
                    implication=(
                        "Non-stationary. Difference the series or include a time index as a "
                        "feature; a model trained on levels will extrapolate badly."
                    ),
                )
            )

        # --- seasonality ---------------------------------------------------
        detrended = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1), np.arange(len(values)))
        threshold = max(ACF_NOTABLE, 2 / math.sqrt(len(values)))
        # Collect every cycle present, not just the first: a daily series can carry
        # a weekly *and* an annual cycle, and the changepoint test has to have both
        # removed before it can see a genuine level shift.
        seasonal_periods: list[int] = []
        reported_seasonality = False
        for period in SEASONAL_PERIODS.get(freq, []):
            if len(detrended) < period * 2 + 2:
                continue
            acf = _autocorrelation(detrended, period)
            if acf is not None and acf >= threshold:
                seasonal_periods.append(period)
                if reported_seasonality:
                    continue
                reported_seasonality = True
                found.append(
                    Pattern(
                        kind="seasonality",
                        severity="strong" if acf >= 0.5 else "notable",
                        columns=[time_column, value_column],
                        statistic=round(float(acf), 3),
                        description=(
                            f"{value_column} repeats every {period} {freq}-periods "
                            f"(autocorrelation={acf:.2f})"
                        ),
                        evidence={"period": period, "freq": freq},
                        implication=(
                            "Use a seasonal model (SARIMA, ETS, Prophet) or engineer "
                            f"lag-{period} and calendar features for a gradient-boosted model."
                        ),
                    )
                )

        # --- level shift ---------------------------------------------------
        # Control for every cycle the frequency could carry, not only the ones that
        # cleared the reporting threshold: an annual cycle too weak to report is
        # still strong enough to masquerade as a step.
        shift = _changepoint(values, periods=SEASONAL_PERIODS.get(freq, []))
        if shift:
            index, f_stat, relative_step = shift
            found.append(
                Pattern(
                    kind="changepoint",
                    severity="strong" if relative_step >= 0.15 else "notable",
                    columns=[time_column, value_column],
                    statistic=round(float(relative_step), 4),
                    description=(
                        f"{value_column} steps by {relative_step:.0%} around "
                        f"{series.index[index].date()}, over and above the trend "
                        f"(Chow F={f_stat:,.0f})"
                    ),
                    evidence={
                        "index": int(index),
                        "date": str(series.index[index].date()),
                        "f_statistic": round(float(f_stat), 1),
                        "relative_step": round(float(relative_step), 4),
                    },
                    implication=(
                        "Something changed — a pricing change, a tracking change, a merger. "
                        "Training across the break will average two regimes; either add a "
                        "regime indicator or train on the post-break period only."
                    ),
                )
            )

        # --- variance drift -------------------------------------------------
        half = len(values) // 2
        first_std, second_std = float(np.std(values[:half])), float(np.std(values[half:]))
        if first_std > 0 and second_std > 0:
            ratio = max(first_std, second_std) / min(first_std, second_std)
            if ratio > 2.0:
                found.append(
                    Pattern(
                        kind="variance_drift",
                        severity="notable",
                        columns=[time_column, value_column],
                        statistic=round(float(ratio), 2),
                        description=f"{value_column} volatility changes {ratio:.1f}× between halves",
                        implication=(
                            "Heteroscedastic. Model the log, use a multiplicative "
                            "decomposition, or fit prediction intervals that widen over time."
                        ),
                    )
                )
    return found


_RATE_NAME = re.compile(
    r"(price|rate|ratio|pct|percent|share|score|avg|average|mean|days|age|index|level)", re.I
)


def _series_aggregation(column: str) -> str:
    """Sum is wrong for rates.

    Rolling a price up to a daily total makes the series track order *volume*, so
    the volume's weekly cycle shows up as "seasonality in unit_price". Rates,
    percentages, and durations are averaged; quantities and amounts are summed.
    """
    return "mean" if _RATE_NAME.search(column) else "sum"


def _autocorrelation(values: np.ndarray, lag: int) -> float | None:
    if lag >= len(values):
        return None
    centered = values - values.mean()
    denominator = float((centered**2).sum())
    if denominator == 0:
        return None
    return float((centered[lag:] * centered[:-lag]).sum() / denominator)


def _seasonal_design(n: int, periods: list[int], harmonics: int = 2) -> np.ndarray | None:
    """Fourier terms for each candidate cycle.

    Preferred over subtracting per-phase means: an annual cycle in three years of
    daily data has only three observations per phase, so phase-mean removal eats a
    third of the signal and leaves artifacts. Two sine/cosine pairs describe a
    smooth yearly shape using four degrees of freedom instead of 364.
    """
    columns: list[np.ndarray] = []
    t = np.arange(n, dtype=float)
    for period in periods:
        if period < 3 or period > n / 2:
            continue
        for k in range(1, harmonics + 1):
            if period / k < 2:
                break
            angle = 2 * np.pi * k * t / period
            columns.append(np.sin(angle))
            columns.append(np.cos(angle))
    return np.column_stack(columns) if columns else None


def _changepoint(
    values: np.ndarray, periods: list[int] | None = None
) -> tuple[int, float, float] | None:
    """Chow test for a step in the level, on top of a linear trend.

    Asking "where is the biggest difference in means" is the wrong question on a
    series that trends and cycles — the answer is always the trend, and any
    difference-of-means z-score grows with n so everything looks significant.
    The right question is whether adding a step term at some point *explains
    variance the trend alone does not*, which is an F-test between the two nested
    fits. Seasonality is removed first, or a seasonal peak reads as a level shift.

    Returns (index, F statistic, step size relative to the mean level).
    """
    n = len(values)
    min_segment = max(8, int(0.1 * n))
    if n < 2 * min_segment + 4:
        return None

    # Business series are usually multiplicative: the trend scales the seasonal
    # swing rather than adding to it. Fitting an additive trend+step to that finds
    # a spurious "step" wherever the amplitude has grown. Working in log space
    # makes the structure additive, and the step coefficient falls out as a
    # percentage change, which is what a reader wants anyway.
    positive = bool((values > 0).all())
    series = np.log(values.astype(float)) if positive else values.astype(float)

    # The null carries a quadratic as well as a slope. Two reasons: real series bend
    # (growth accelerates, saturation sets in), and the log transform above turns a
    # perfectly straight line into a concave one — without the curvature term the
    # transform itself would manufacture a "step" on trending data.
    x = np.arange(n, dtype=float)
    x = (x - x.mean()) / max(x.std(), 1e-9)  # centred and scaled, so x² stays conditioned
    null_design = np.column_stack([np.ones(n), x, x**2])
    seasonal = _seasonal_design(n, periods or [])
    if seasonal is not None:
        null_design = np.column_stack([null_design, seasonal])

    n_params = null_design.shape[1]
    dof = n - n_params - 1
    if dof <= 2:
        return None

    coefficients, *_ = np.linalg.lstsq(null_design, series, rcond=None)
    rss_null = float(((series - null_design @ coefficients) ** 2).sum())
    if rss_null <= 0:
        return None

    level = float(np.abs(series).mean()) or 1.0
    best = None
    for i in range(min_segment, n - min_segment):
        indicator = np.zeros(n)
        indicator[i:] = 1.0
        design = np.column_stack([null_design, indicator])
        try:
            beta, *_ = np.linalg.lstsq(design, series, rcond=None)
        except np.linalg.LinAlgError:
            continue
        rss_step = float(((series - design @ beta) ** 2).sum())
        if rss_step <= 0:
            continue
        f_stat = (rss_null - rss_step) / (rss_step / dof)
        step = float(beta[-1])
        relative = abs(math.expm1(step)) if positive else abs(step) / level
        if best is None or f_stat > best[1]:
            best = (i, f_stat, relative)

    if best is None:
        return None

    index, f_stat, relative_step = best
    significant = (
        float(_scipy_stats.f.sf(f_stat, 1, dof)) < 1e-4 if HAVE_SCIPY else f_stat > 30.0
    )
    # Significance alone is not enough: on a long series a 0.5% shift clears any
    # p-value threshold and means nothing to the person reading the report.
    if significant and relative_step >= CHANGEPOINT_MIN_STEP:
        return index, f_stat, relative_step
    return None


# --------------------------------------------------------------------- target
def detect_target_patterns(
    df: pd.DataFrame, profile: DatasetProfile, target: str
) -> list[Pattern]:
    if target not in df.columns:
        return []
    found: list[Pattern] = []
    y = df[target]
    stats = profile.stats(target)
    is_numeric = target in profile.numeric_columns

    # --- imbalance --------------------------------------------------------
    if not is_numeric or (stats and stats.n_unique <= 20):
        counts = y.dropna().value_counts()
        if 1 < len(counts) <= 50:
            ratio = float(counts.iloc[0] / counts.iloc[-1])
            if ratio >= IMBALANCE_NOTABLE:
                found.append(
                    Pattern(
                        kind="class_imbalance",
                        severity="strong" if ratio >= IMBALANCE_STRONG else "notable",
                        columns=[target],
                        statistic=round(ratio, 2),
                        description=(
                            f"{target} is imbalanced {ratio:.0f}:1 "
                            f"({counts.index[0]}={counts.iloc[0]:,} vs {counts.index[-1]}={counts.iloc[-1]:,})"
                        ),
                        evidence={str(k): int(v) for k, v in counts.head(10).items()},
                        implication=(
                            "Accuracy is meaningless here — a constant predictor scores "
                            f"{counts.iloc[0] / counts.sum():.1%}. Use PR-AUC or balanced "
                            "accuracy, set class_weight='balanced', and stratify every split."
                        ),
                    )
                )
        if len(counts) > 0 and counts.iloc[-1] < 10:
            rare = [str(k) for k, v in counts.items() if v < 10]
            found.append(
                Pattern(
                    kind="rare_classes",
                    severity="notable",
                    columns=[target],
                    statistic=float(len(rare)),
                    description=f"{len(rare)} class(es) with fewer than 10 rows",
                    evidence={"classes": rare[:20]},
                    implication="Merge into an 'other' bucket or drop; k-fold cannot stratify these.",
                )
            )

    # --- leakage ----------------------------------------------------------
    if is_numeric:
        numeric = numeric_matrix(df, [c for c in profile.numeric_columns if c != target])
        y_num = pd.to_numeric(y, errors="coerce")
        for column in numeric.columns:
            joined = pd.concat([numeric[column], y_num], axis=1).dropna()
            if len(joined) < 10:
                continue
            r = float(joined.corr().iloc[0, 1])
            if pd.isna(r):
                continue
            if abs(r) >= LEAKAGE_CORR:
                found.append(
                    Pattern(
                        kind="target_leakage",
                        severity="strong",
                        columns=[str(column), target],
                        statistic=round(r, 4),
                        description=f"{column} is almost perfectly correlated with {target} (r={r:.3f})",
                        implication=(
                            "Probable leakage: this is the target restated, or something "
                            "computed from it. Check how it is produced — if it is not "
                            "available at prediction time, remove it or the model's score "
                            "is fiction."
                        ),
                    )
                )
            elif abs(r) >= 0.4:
                found.append(
                    Pattern(
                        kind="predictive_feature",
                        severity="notable",
                        columns=[str(column), target],
                        statistic=round(r, 3),
                        description=f"{column} carries signal for {target} (r={r:.2f})",
                        implication="A strong candidate feature.",
                    )
                )

        # Categorical -> numeric target: correlation ratio (eta squared).
        for column in profile.categorical_columns:
            joined = pd.concat([df[column], y_num], axis=1).dropna()
            if len(joined) < 20 or joined[column].nunique() > 30:
                continue
            value_column = joined.columns[1]
            grand_mean = joined[value_column].mean()
            groups = joined.groupby(column, observed=True)[value_column]
            between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups)
            total = float(((joined[value_column] - grand_mean) ** 2).sum())
            if total <= 0:
                continue
            eta_squared = float(between / total)
            if eta_squared >= 0.15:
                found.append(
                    Pattern(
                        kind="group_effect",
                        severity="strong" if eta_squared >= 0.5 else "notable",
                        columns=[column, target],
                        statistic=round(eta_squared, 3),
                        description=f"{column} explains {eta_squared:.0%} of the variance in {target}",
                        implication=(
                            "A genuine grouping effect. Keep the column, and consider a "
                            "mixed-effects or per-group model if the groups are few and stable."
                        ),
                    )
                )
    return found


def detect_cluster_tendency(df: pd.DataFrame, profile: DatasetProfile) -> list[Pattern]:
    """Is there structure to find without a target? Silhouette over small k."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return []

    frame = numeric_matrix(df, profile.numeric_columns).dropna()
    if frame.shape[0] < 50 or frame.shape[1] < 2:
        return []
    sample = frame.sample(min(5000, len(frame)), random_state=0)
    scaled = StandardScaler().fit_transform(sample)

    best_k, best_score = 0, -1.0
    for k in range(2, min(7, len(sample) // 10 + 1)):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(scaled)
            score = float(silhouette_score(scaled, labels, sample_size=min(2000, len(scaled)), random_state=0))
        except Exception:
            continue
        if score > best_score:
            best_k, best_score = k, score

    if best_k and best_score >= 0.25:
        return [
            Pattern(
                kind="cluster_structure",
                severity="strong" if best_score >= 0.5 else "notable",
                columns=list(frame.columns),
                statistic=round(best_score, 3),
                description=f"{best_k} reasonably separated groups in the numeric columns (silhouette={best_score:.2f})",
                evidence={"k": best_k},
                implication=(
                    "Segmentation is worth pursuing. Confirm the clusters are stable and "
                    "interpretable before acting on them."
                ),
            )
        ]
    return [
        Pattern(
            kind="cluster_structure",
            severity="info",
            statistic=round(best_score, 3) if best_score > -1 else None,
            description="no well-separated clusters in the numeric columns",
            implication=(
                "Segmentation will produce arbitrary boundaries. Prefer supervised "
                "framing, or cluster on engineered behavioural features instead."
            ),
        )
    ]


# ------------------------------------------------------------------ entrypoint
_SEVERITY_ORDER = {"strong": 0, "notable": 1, "info": 2}


def detect_all(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target: str | None = None,
    include_clustering: bool = True,
) -> list[Pattern]:
    found: list[Pattern] = []
    found += detect_missingness(df, profile)
    found += detect_structural_issues(df, profile)
    found += detect_distributions(df, profile)
    found += detect_correlations(df, profile)
    found += detect_multicollinearity(df, profile)
    found += detect_time_patterns(df, profile)
    if target:
        found += detect_target_patterns(df, profile, target)
    if include_clustering and not target:
        found += detect_cluster_tendency(df, profile)

    found.sort(key=lambda p: (_SEVERITY_ORDER.get(p.severity, 3), p.kind))
    return found
