"""Algorithm recommendation.

A transparent rule engine, not a model. Every recommendation names the measured
properties that produced it, so a reader can check the reasoning and reject it.
The ordering is by suitability *given the evidence*, not by generic popularity —
and the first recommendation for any supervised task is always the trivial
baseline, because a model that cannot beat it has not been shown to work.
"""

from __future__ import annotations

import pandas as pd

from .models import DatasetProfile, Pattern, Recommendation, TaskType

SMALL_DATA = 500
MEDIUM_DATA = 10_000
LARGE_DATA = 500_000


def _strong(patterns: list[Pattern], kind: str) -> bool:
    return any(p.kind == kind and p.severity == "strong" for p in patterns)


def _present(patterns: list[Pattern], kind: str) -> bool:
    return any(p.kind == kind for p in patterns)


def _notable(patterns: list[Pattern], kind: str) -> bool:
    """Present *and* worth acting on — info-level findings are context, not evidence."""
    return any(p.kind == kind and p.severity in ("notable", "strong") for p in patterns)


# --------------------------------------------------------------------- task
def infer_task(
    df: pd.DataFrame, profile: DatasetProfile, target: str | None
) -> TaskType:
    if target is None:
        # "No target" almost always means segmentation. A date column is not
        # evidence of a time series — a customer table has first_order/last_order
        # and is still one row per person. Forecasting is added alongside by
        # `recommend()` only when trend or seasonality was actually measured.
        return "clustering" if profile.numeric_columns else "none"

    if target not in df.columns:
        return "none"

    stats = profile.stats(target)
    n_unique = stats.n_unique if stats else int(df[target].nunique())

    if target in profile.text_columns:
        return "text_classification"

    if target in profile.numeric_columns:
        # A numeric column with a handful of levels is a label, not a quantity.
        if n_unique == 2:
            return "binary_classification"
        if n_unique <= 15 and (stats is None or stats.pct_unique < 0.05):
            return "multiclass_classification"
        return "regression"

    if n_unique == 2:
        return "binary_classification"
    if 2 < n_unique <= 100:
        return "multiclass_classification"
    return "none"


def is_forecastable(profile: DatasetProfile, patterns: list[Pattern]) -> bool:
    """Is there enough of a time axis to justify a forecasting model at all?"""
    if not profile.temporal_columns:
        return False
    coverage = next((p for p in patterns if p.kind == "temporal_coverage"), None)
    if coverage and coverage.statistic is not None and coverage.statistic < 60:
        return False
    return bool(profile.numeric_columns)


# ------------------------------------------------------------ recommendations
def recommend(
    df: pd.DataFrame,
    profile: DatasetProfile,
    patterns: list[Pattern],
    target: str | None = None,
    task: TaskType | None = None,
) -> list[Recommendation]:
    task = task or infer_task(df, profile, target)
    builders = {
        "regression": _regression,
        "binary_classification": _classification,
        "multiclass_classification": _classification,
        "text_classification": _text_classification,
        "time_series_forecasting": _forecasting,
        "clustering": _clustering,
    }
    builder = builders.get(task)
    items: list[Recommendation] = builder(df, profile, patterns, target, task) if builder else []

    # Cross-cutting additions that stand alongside the primary task.
    if task in ("regression", "binary_classification", "multiclass_classification"):
        if is_forecastable(profile, patterns) and _present(patterns, "seasonality"):
            items += _forecasting(df, profile, patterns, target, "time_series_forecasting")[:1]
    if task == "clustering" and is_forecastable(profile, patterns):
        if _present(patterns, "trend") or _present(patterns, "seasonality"):
            items += _forecasting(df, profile, patterns, target, "time_series_forecasting")[:3]
    if task == "clustering" or target is None:
        items += _anomaly_detection(df, profile, patterns)
        items += _dimensionality_reduction(df, profile, patterns)
        items += _association_rules(df, profile, patterns)

    for index, item in enumerate(items, start=1):
        item.rank = index
        # Several rationale/preprocessing entries are conditional expressions that
        # collapse to "" when their condition is false; drop them here rather than
        # littering every builder with filters.
        item.rationale = [line for line in item.rationale if line.strip()]
        item.preprocessing = [line for line in item.preprocessing if line.strip()]
        item.caveats = [line for line in item.caveats if line.strip()]
    return items


# ------------------------------------------------------------------ builders
def _size_notes(profile: DatasetProfile) -> tuple[str, list[str]]:
    n = profile.n_rows
    if n < SMALL_DATA:
        return "small", [
            f"only {n:,} rows — high-variance models will overfit; prefer regularised "
            "linear models and report cross-validated intervals, not point scores"
        ]
    if n < MEDIUM_DATA:
        return "medium", [f"{n:,} rows supports tree ensembles with modest depth"]
    if n < LARGE_DATA:
        return "large", [f"{n:,} rows is comfortable for gradient boosting"]
    return "very_large", [
        f"{n:,} rows — consider subsampling for iteration, and histogram-based "
        "implementations (HistGradientBoosting / LightGBM) over exact ones"
    ]


def _common_preprocessing(profile: DatasetProfile, patterns: list[Pattern]) -> list[str]:
    steps: list[str] = []
    if _present(patterns, "missing_data"):
        steps.append("impute missing values (median for numeric, most-frequent for categorical)")
    if _present(patterns, "duplicate_rows"):
        steps.append("drop duplicate rows before splitting, or folds will leak")
    if _present(patterns, "constant_columns"):
        steps.append("drop constant columns")
    if profile.categorical_columns:
        steps.append(
            "target-encode high-cardinality categoricals; one-hot the rest"
            if _present(patterns, "high_cardinality")
            else "one-hot encode categoricals"
        )
    if _present(patterns, "target_leakage"):
        steps.append("REMOVE the leaking columns flagged above before anything else")
    return steps


def _regression(df, profile, patterns, target, task) -> list[Recommendation]:
    size, size_notes = _size_notes(profile)
    base = _common_preprocessing(profile, patterns)
    items: list[Recommendation] = []

    items.append(
        Recommendation(
            rank=0,
            task=task,
            algorithm="DummyRegressor (predict the mean) — the bar to beat",
            confidence="high",
            rationale=["every score below is only meaningful relative to this"],
            preprocessing=[],
            evaluation="MAE and R²; a model with R² near 0 has learned nothing",
            caveats=["not a deliverable, a control"],
            starter_code=(
                "from sklearn.dummy import DummyRegressor\n"
                "from sklearn.model_selection import cross_val_score\n"
                f"cross_val_score(DummyRegressor(), X, y, cv=5, scoring='r2')"
            ),
        )
    )

    heavy_outliers = _strong(patterns, "outliers")
    nonlinear = _notable(patterns, "nonlinear_relationship")
    collinear = _present(patterns, "multicollinearity")
    missing = _present(patterns, "missing_data")
    skewed = _present(patterns, "skewed_distribution")
    zero_inflated = _present(patterns, "zero_inflation")

    if size != "small":
        rationale = ["tabular data with mixed column types is gradient boosting's home ground"]
        rationale += size_notes
        if nonlinear:
            rationale.append("monotone-but-not-linear relationships were detected — trees capture these without a transform")
        if missing:
            rationale.append("handles NaN natively, so imputation choices stop mattering")
        if skewed:
            rationale.append("splits on rank, so the skew flagged above needs no transform")
        items.append(
            Recommendation(
                rank=0,
                task=task,
                algorithm="HistGradientBoostingRegressor",
                confidence="high",
                rationale=rationale,
                preprocessing=[s for s in base if "impute" not in s],
                evaluation="5-fold MAE + R²; hold out the most recent period if the data is time-ordered",
                caveats=[
                    "coefficients are not interpretable — use permutation importance or SHAP",
                    "extrapolates flat beyond the training range, so it cannot forecast a trend",
                ],
                starter_code=(
                    "from sklearn.ensemble import HistGradientBoostingRegressor\n"
                    "from sklearn.model_selection import cross_validate\n"
                    "model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)\n"
                    "cross_validate(model, X, y, cv=5, scoring=['neg_mean_absolute_error', 'r2'])"
                ),
            )
        )

    if collinear or size == "small" or not nonlinear:
        rationale = []
        if collinear:
            rationale.append("multicollinearity was detected — the L2 penalty is what makes coefficients stable here")
        if size == "small":
            rationale += size_notes
        if not nonlinear:
            rationale.append("no strong non-linearity was found, so the linear form is not obviously wrong")
        rationale.append("gives interpretable, signed coefficients — often the actual deliverable")
        items.append(
            Recommendation(
                rank=0,
                task=task,
                algorithm="RidgeCV (or ElasticNetCV if you also want feature selection)",
                confidence="high" if collinear else "medium",
                rationale=rationale,
                preprocessing=base + ["StandardScaler — penalised models are scale-sensitive"],
                evaluation="5-fold MAE/RMSE; inspect coefficients only after scaling",
                caveats=[
                    "assumes additive, linear effects",
                    "sensitive to outliers" if heavy_outliers else "check residuals for structure",
                ],
                starter_code=(
                    "from sklearn.linear_model import RidgeCV\n"
                    "from sklearn.pipeline import make_pipeline\n"
                    "from sklearn.preprocessing import StandardScaler\n"
                    "model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100]))"
                ),
            )
        )

    if heavy_outliers:
        items.append(
            Recommendation(
                rank=0,
                task=task,
                algorithm="HuberRegressor / QuantileRegressor (robust regression)",
                confidence="medium",
                rationale=[
                    f"{sum(1 for p in patterns if p.kind == 'outliers')} column(s) carry heavy outliers",
                    "squared error lets a handful of extreme rows dictate the fit; Huber caps their influence",
                    "quantile regression additionally answers 'what is a bad case', not just 'what is typical'",
                ],
                preprocessing=base + ["RobustScaler instead of StandardScaler"],
                evaluation="median absolute error, and MAE at the 10th/90th quantiles",
                caveats=[
                    "if the outliers ARE the phenomenon (fraud, incidents, spikes), do not "
                    "downweight them — reframe as anomaly detection instead"
                ],
                starter_code=(
                    "from sklearn.linear_model import HuberRegressor, QuantileRegressor\n"
                    "robust = HuberRegressor(epsilon=1.35)\n"
                    "p90 = QuantileRegressor(quantile=0.9, alpha=0.01)"
                ),
            )
        )

    if zero_inflated:
        items.append(
            Recommendation(
                rank=0,
                task=task,
                algorithm="Two-part (hurdle) model, or a Tweedie/Poisson GLM",
                confidence="medium",
                rationale=[
                    "the target is mostly zeros with a continuous positive tail",
                    "a single conditional-mean model splits the difference and fits neither part",
                ],
                preprocessing=base,
                evaluation="separate the two questions: AUC on 'is it non-zero', MAE on the positives",
                caveats=["two models means two sets of errors to combine when reporting"],
                starter_code=(
                    "from sklearn.linear_model import TweedieRegressor\n"
                    "model = TweedieRegressor(power=1.5, link='log')  # compound Poisson-gamma\n"
                    "# or: classifier on (y > 0), then a regressor fitted on y[y > 0]"
                ),
            )
        )

    if _present(patterns, "high_cardinality") and size in ("large", "very_large"):
        items.append(
            Recommendation(
                rank=0,
                task=task,
                algorithm="CatBoostRegressor / LGBMRegressor",
                library="catboost / lightgbm",
                confidence="medium",
                rationale=[
                    "high-cardinality categorical columns were detected",
                    "both handle categoricals natively — CatBoost with ordered target statistics, "
                    "LightGBM with a partition-based split — avoiding a one-hot blow-up",
                ],
                preprocessing=[s for s in base if "encode" not in s],
                evaluation="5-fold MAE; use early stopping on a held-out fold",
                caveats=["an extra dependency; only worth it if one-hot is genuinely infeasible"],
                starter_code=(
                    "from catboost import CatBoostRegressor\n"
                    "model = CatBoostRegressor(cat_features=CATEGORICAL_COLS, verbose=0)"
                ),
            )
        )
    return items


def _classification(df, profile, patterns, target, task) -> list[Recommendation]:
    size, size_notes = _size_notes(profile)
    base = _common_preprocessing(profile, patterns)
    imbalance = next((p for p in patterns if p.kind == "class_imbalance"), None)
    binary = task == "binary_classification"

    metric = (
        "average precision (PR-AUC) and balanced accuracy — NOT accuracy"
        if imbalance
        else ("ROC-AUC and F1" if binary else "macro-F1 and a confusion matrix")
    )
    imbalance_steps = (
        [
            "set class_weight='balanced' (cheap, no resampling)",
            "stratify every split, including cross-validation folds",
            "tune the decision threshold on a validation fold — 0.5 is almost never right here",
        ]
        if imbalance
        else []
    )

    items: list[Recommendation] = [
        Recommendation(
            rank=0,
            task=task,
            algorithm="DummyClassifier(strategy='prior') — the bar to beat",
            confidence="high",
            rationale=(
                [
                    f"the classes are imbalanced {imbalance.statistic:.0f}:1, so always "
                    "predicting the majority already looks accurate — this quantifies that"
                ]
                if imbalance and imbalance.statistic
                else ["establishes the floor before anything is claimed"]
            ),
            preprocessing=[],
            evaluation=metric,
            caveats=["not a deliverable, a control"],
            starter_code=(
                "from sklearn.dummy import DummyClassifier\n"
                "from sklearn.model_selection import cross_val_score\n"
                "cross_val_score(DummyClassifier(strategy='prior'), X, y, cv=5, scoring='average_precision')"
            ),
        )
    ]

    rationale = ["strongest default for tabular classification"] + size_notes
    if _present(patterns, "missing_data"):
        rationale.append("handles NaN natively")
    if _present(patterns, "nonlinear_relationship"):
        rationale.append("non-linear relationships were detected — captured without manual transforms")
    items.append(
        Recommendation(
            rank=0,
            task=task,
            algorithm="HistGradientBoostingClassifier",
            confidence="high",
            rationale=rationale,
            preprocessing=[s for s in base if "impute" not in s] + imbalance_steps,
            evaluation=metric,
            caveats=[
                "probabilities are not calibrated out of the box — wrap in CalibratedClassifierCV "
                "if you will threshold on them",
            ],
            starter_code=(
                "from sklearn.ensemble import HistGradientBoostingClassifier\n"
                "model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,\n"
                "                                       class_weight='balanced')"
            ),
        )
    )

    items.append(
        Recommendation(
            rank=0,
            task=task,
            algorithm="LogisticRegression",
            confidence="high" if size == "small" else "medium",
            rationale=[
                "interpretable log-odds coefficients, which is often what a stakeholder wants",
                "well-calibrated probabilities by construction",
            ]
            + (size_notes if size == "small" else []),
            preprocessing=base + ["StandardScaler"] + imbalance_steps,
            evaluation=metric,
            caveats=(
                ["multicollinearity was detected — the coefficients will be unstable; use L2 and do not over-read them"]
                if _present(patterns, "multicollinearity")
                else ["assumes a linear decision boundary in the feature space"]
            ),
            starter_code=(
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.pipeline import make_pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "model = make_pipeline(StandardScaler(),\n"
                "    LogisticRegression(max_iter=2000, class_weight='balanced'))"
            ),
        )
    )

    if imbalance and imbalance.statistic and imbalance.statistic >= 20:
        items.append(
            Recommendation(
                rank=0,
                task=task,
                algorithm="Reframe as anomaly detection (IsolationForest / One-Class SVM)",
                confidence="medium",
                rationale=[
                    f"the minority class is {imbalance.statistic:.0f}× rarer than the majority",
                    "past roughly 20:1 there is often too little minority signal to learn a "
                    "decision boundary; modelling 'normal' and flagging deviations can work better",
                ],
                preprocessing=base + ["fit on majority-class rows only"],
                evaluation="precision@k — how many of the top k flags are real",
                caveats=[
                    "gives up on using the labels you do have; try the supervised route first "
                    "and only switch if recall stays near zero"
                ],
                starter_code=(
                    "from sklearn.ensemble import IsolationForest\n"
                    "model = IsolationForest(contamination=0.01, random_state=0)\n"
                    "model.fit(X[y == majority_class])"
                ),
            )
        )
    return items


def _text_classification(df, profile, patterns, target, task) -> list[Recommendation]:
    return [
        Recommendation(
            rank=0,
            task="text_classification",
            algorithm="TF-IDF + LinearSVC",
            confidence="high",
            rationale=[
                f"free-text column(s) detected: {', '.join(profile.text_columns[:3])}",
                "a strong, fast baseline that a transformer has to justify beating",
            ],
            preprocessing=["lowercase, strip accents", "word 1-2 grams, min_df=2"],
            evaluation="macro-F1 with stratified 5-fold",
            caveats=["no word order or negation handling beyond bigrams"],
            starter_code=(
                "from sklearn.feature_extraction.text import TfidfVectorizer\n"
                "from sklearn.svm import LinearSVC\n"
                "from sklearn.pipeline import make_pipeline\n"
                "model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LinearSVC())"
            ),
        ),
        Recommendation(
            rank=0,
            task="text_classification",
            algorithm="Sentence embeddings + LogisticRegression",
            library="sentence-transformers + scikit-learn",
            confidence="medium",
            rationale=["captures meaning where TF-IDF only captures vocabulary overlap"],
            preprocessing=["embed once and cache — encoding dominates the runtime"],
            evaluation="macro-F1 against the TF-IDF baseline; keep it only if it wins",
            caveats=["heavier dependency; needs a GPU to be fast on large corpora"],
            starter_code=(
                "from sentence_transformers import SentenceTransformer\n"
                "emb = SentenceTransformer('all-MiniLM-L6-v2').encode(texts)"
            ),
        ),
    ]


def _forecasting(df, profile, patterns, target, task) -> list[Recommendation]:
    seasonal = next((p for p in patterns if p.kind == "seasonality"), None)
    trend = next((p for p in patterns if p.kind == "trend"), None)
    changepoint = _present(patterns, "changepoint")
    coverage = next((p for p in patterns if p.kind == "temporal_coverage"), None)
    span_days = coverage.statistic if coverage and coverage.statistic else 0
    period = seasonal.evidence.get("period") if seasonal else None

    items: list[Recommendation] = [
        Recommendation(
            rank=0,
            task="time_series_forecasting",
            algorithm=("Seasonal naive (last season's value)" if seasonal else "Naive (last value)"),
            library="none — three lines of pandas",
            confidence="high",
            rationale=[
                "the honest baseline for any forecast",
                "seasonal naive is surprisingly hard to beat on strongly seasonal series"
                if seasonal
                else "if a model cannot beat 'tomorrow looks like today', it is not a forecast",
            ],
            preprocessing=["resample to a fixed frequency and decide what a gap means"],
            evaluation="MASE (scaled against this baseline by construction) or MAPE",
            caveats=["carries no uncertainty estimate"],
            starter_code=(
                f"forecast = series.shift({period})" if period else "forecast = series.shift(1)"
            ),
        )
    ]

    if seasonal and span_days > 730:
        items.append(
            Recommendation(
                rank=0,
                task="time_series_forecasting",
                algorithm=f"SARIMA (seasonal ARIMA, m={period})",
                library="statsmodels",
                confidence="high",
                rationale=[
                    f"seasonality detected at period {period} (autocorrelation={seasonal.statistic})",
                    f"{span_days / 365:.1f} years of history — enough to estimate a seasonal cycle and validate it",
                    "trend detected, which the (d) differencing term handles" if trend else "",
                ],
                preprocessing=[
                    "regular frequency, gaps filled explicitly",
                    "difference to stationarity; confirm with ADF/KPSS",
                    "log-transform first" if _present(patterns, "variance_drift") else "",
                ],
                evaluation="rolling-origin backtest — never a random split on time series",
                caveats=[
                    "one series at a time; does not scale to thousands of SKUs",
                    "assumes a stable seasonal shape",
                ],
                starter_code=(
                    "from statsmodels.tsa.statespace.sarimax import SARIMAX\n"
                    f"model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,{period})).fit()\n"
                    "model.get_forecast(steps=12).summary_frame()"
                ),
            )
        )

    if seasonal or trend:
        items.append(
            Recommendation(
                rank=0,
                task="time_series_forecasting",
                algorithm="Exponential smoothing (Holt-Winters / ETS)",
                library="statsmodels",
                confidence="high" if span_days > 365 else "medium",
                rationale=[
                    "trend present" if trend else "",
                    f"seasonality present at period {period}" if seasonal else "",
                    "fewer parameters than SARIMA, so it degrades more gracefully on short history",
                ],
                preprocessing=[
                    "multiplicative seasonality if the swings scale with the level"
                    if _present(patterns, "variance_drift")
                    else "additive seasonality"
                ],
                evaluation="rolling-origin backtest, MASE against seasonal naive",
                caveats=["no exogenous regressors — promotions and price changes cannot be included"],
                starter_code=(
                    "from statsmodels.tsa.holtwinters import ExponentialSmoothing\n"
                    f"model = ExponentialSmoothing(series, trend='add', seasonal='add',\n"
                    f"                             seasonal_periods={period or 12}).fit()"
                ),
            )
        )

    if changepoint:
        items.append(
            Recommendation(
                rank=0,
                task="time_series_forecasting",
                algorithm="Prophet",
                library="prophet",
                confidence="medium",
                rationale=[
                    "a level shift was detected — Prophet models changepoints explicitly rather than averaging across them",
                    "tolerates missing periods and irregular spacing",
                    "holiday and event regressors are first-class",
                ],
                preprocessing=["rename to the required ds/y columns"],
                evaluation="prophet.diagnostics.cross_validation with a rolling horizon",
                caveats=[
                    "a curve-fitter, not a stochastic process model — the intervals are optimistic",
                    "frequently loses to ETS on clean, regular series",
                ],
                starter_code=(
                    "from prophet import Prophet\n"
                    "model = Prophet(changepoint_prior_scale=0.1, yearly_seasonality=True)\n"
                    "model.fit(frame.rename(columns={'date': 'ds', 'value': 'y'}))"
                ),
            )
        )

    if len(profile.numeric_columns) > 2 or profile.categorical_columns:
        items.append(
            Recommendation(
                rank=0,
                task="time_series_forecasting",
                algorithm="Gradient boosting on lag + calendar features",
                confidence="medium",
                rationale=[
                    "other columns are available as exogenous drivers, which the classical models cannot use",
                    "one model can cover many series at once (a global model), unlike per-series SARIMA",
                    f"engineer lag-1, lag-{period}, rolling means and calendar flags" if period else "engineer lags and rolling means",
                ],
                preprocessing=[
                    "build lags strictly from the past — no leakage across the split point",
                    "TimeSeriesSplit, never KFold",
                ],
                evaluation="rolling-origin backtest with the same horizon you will deploy at",
                caveats=[
                    "cannot extrapolate a trend beyond the training range — de-trend first "
                    "or predict differences rather than levels",
                ],
                starter_code=(
                    "from sklearn.ensemble import HistGradientBoostingRegressor\n"
                    "from sklearn.model_selection import TimeSeriesSplit\n"
                    "frame['lag_1'] = frame['y'].shift(1)\n"
                    f"frame['lag_s'] = frame['y'].shift({period or 12})\n"
                    "frame['roll_mean'] = frame['y'].shift(1).rolling(7).mean()\n"
                    "cv = TimeSeriesSplit(n_splits=5)"
                ),
            )
        )
    return items


def _clustering(df, profile, patterns, target, task) -> list[Recommendation]:
    structure = next((p for p in patterns if p.kind == "cluster_structure"), None)
    separated = bool(structure and structure.severity != "info")
    k = (structure.evidence.get("k") if structure else None) or 4
    items: list[Recommendation] = []

    items.append(
        Recommendation(
            rank=0,
            task="clustering",
            algorithm=f"KMeans (k≈{k})",
            confidence="high" if separated else "low",
            rationale=(
                [f"silhouette {structure.statistic} at k={k} — the groups are reasonably separated"]
                if separated
                else [
                    "no well-separated structure was found, so any clustering here will be a "
                    "partition of a continuum rather than a discovery",
                    "listed because it is still the right first thing to try",
                ]
            ),
            preprocessing=[
                "StandardScaler — KMeans is Euclidean, so unscaled columns dominate",
                "log-transform the skewed columns flagged above" if _present(patterns, "skewed_distribution") else "",
                "PCA first" if _present(patterns, "multicollinearity") else "",
            ],
            evaluation="silhouette + a stability check: re-cluster on bootstrap samples and compare",
            caveats=[
                "assumes spherical, equal-sized clusters",
                "k is your choice, not the data's — validate that the segments mean something",
            ],
            starter_code=(
                "from sklearn.cluster import KMeans\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.pipeline import make_pipeline\n"
                f"model = make_pipeline(StandardScaler(), KMeans(n_clusters={k}, n_init=10))"
            ),
        )
    )

    if _present(patterns, "outliers") or not separated:
        items.append(
            Recommendation(
                rank=0,
                task="clustering",
                algorithm="DBSCAN / HDBSCAN",
                library="scikit-learn / hdbscan",
                confidence="medium",
                rationale=[
                    "outliers were detected — density methods label them noise instead of "
                    "forcing them into a cluster and dragging its centroid",
                    "finds non-spherical shapes and does not need k up front",
                ],
                preprocessing=["scale", "tune eps from a k-distance plot"],
                evaluation="fraction labelled noise + silhouette on the clustered points only",
                caveats=["struggles when clusters have very different densities"],
                starter_code=(
                    "from sklearn.cluster import DBSCAN\n"
                    "labels = DBSCAN(eps=0.7, min_samples=10).fit_predict(X_scaled)"
                ),
            )
        )

    items.append(
        Recommendation(
            rank=0,
            task="clustering",
            algorithm="GaussianMixture",
            confidence="medium",
            rationale=[
                "gives soft memberships and elliptical clusters, which fit correlated features better than KMeans spheres",
                "BIC over k is a principled way to choose the number of components",
            ],
            preprocessing=["scale"],
            evaluation="BIC across k, then silhouette on the chosen model",
            caveats=["needs more rows per cluster than KMeans; can converge to a degenerate fit"],
            starter_code=(
                "from sklearn.mixture import GaussianMixture\n"
                "bic = {k: GaussianMixture(k, random_state=0).fit(X).bic(X) for k in range(2, 9)}"
            ),
        )
    )
    return items


def _anomaly_detection(df, profile, patterns) -> list[Recommendation]:
    if not _present(patterns, "outliers") and not _present(patterns, "changepoint"):
        return []
    reasons = []
    if _present(patterns, "outliers"):
        columns = [c for p in patterns if p.kind == "outliers" for c in p.columns]
        reasons.append(f"outliers were found in {', '.join(columns[:4])}")
    if _present(patterns, "changepoint"):
        reasons.append("a level shift was detected, so 'normal' is not constant over time")
    return [
        Recommendation(
            rank=0,
            task="anomaly_detection",
            algorithm="IsolationForest (+ residual thresholding for the time series)",
            confidence="medium",
            rationale=reasons + ["works unsupervised, which matters because you have no anomaly labels"],
            preprocessing=[
                "set contamination from how many alerts you can actually action, not from a default",
                "for the time series: model the seasonal level, then threshold the residuals",
            ],
            evaluation="precision@k reviewed by a human — there is no ground truth to score against",
            caveats=[
                "will flag rare-but-legitimate rows; budget for triage",
                "if the level shift is a real regime change, retrain rather than alerting forever",
            ],
            starter_code=(
                "from sklearn.ensemble import IsolationForest\n"
                "scores = IsolationForest(contamination=0.01, random_state=0).fit(X).score_samples(X)"
            ),
        )
    ]


def _dimensionality_reduction(df, profile, patterns) -> list[Recommendation]:
    if not _present(patterns, "multicollinearity") and len(profile.numeric_columns) < 15:
        return []
    return [
        Recommendation(
            rank=0,
            task="dimensionality_reduction",
            algorithm="PCA (then UMAP for visualisation only)",
            confidence="medium",
            rationale=[
                f"{len(profile.numeric_columns)} numeric columns"
                + (" with multicollinearity" if _present(patterns, "multicollinearity") else ""),
                "redundant dimensions inflate variance in every downstream model",
            ],
            preprocessing=["StandardScaler before PCA, always"],
            evaluation="cumulative explained variance; keep the components reaching ~90%",
            caveats=[
                "components are linear blends and usually not interpretable",
                "never fit UMAP coordinates as model features — it does not preserve distances",
            ],
            starter_code=(
                "from sklearn.decomposition import PCA\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "pca = PCA(n_components=0.9).fit(StandardScaler().fit_transform(X))"
            ),
        )
    ]


def _association_rules(df, profile, patterns) -> list[Recommendation]:
    # Transaction shape: a repeating id plus a categorical item column.
    repeating = [
        s.name
        for s in profile.columns
        if s.semantic_type == "identifier" and 0 < s.pct_unique < 0.5
    ]
    if not repeating or not profile.categorical_columns:
        return []
    return [
        Recommendation(
            rank=0,
            task="association_rules",
            algorithm="FP-Growth / Apriori",
            library="mlxtend",
            confidence="low",
            rationale=[
                f"{repeating[0]} repeats across rows alongside categorical columns — a basket shape",
                "surfaces co-occurrence rules without needing a target",
            ],
            preprocessing=[f"pivot to one row per {repeating[0]}, one boolean column per item"],
            evaluation="lift > 1 with enough support to be actionable, not just statistically present",
            caveats=[
                "produces a great many rules, most of them trivial",
                "co-occurrence is not causation — a rule is a hypothesis for a test, not a decision",
            ],
            starter_code=(
                "from mlxtend.frequent_patterns import fpgrowth, association_rules\n"
                "sets = fpgrowth(basket_bool, min_support=0.02, use_colnames=True)\n"
                "association_rules(sets, metric='lift', min_threshold=1.2)"
            ),
        )
    ]
