"""Fit the recommendation.

A recommendation is a claim; this checks it. It trains the top-ranked model on the
actual data with an honest split and reports cross-validated scores next to the
trivial baseline's — so "gradient boosting is applicable here" becomes a number
you can disagree with.

Deliberately not a tuning harness: no search, no stacking, fixed hyperparameters.
The point is a floor and a signal check, not a production model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..errors import PlatformError
from .models import BaselineResult, DatasetProfile, TaskType
from .patterns import detect_target_patterns

MAX_TRAIN_ROWS = 200_000
MAX_ONEHOT_CARDINALITY = 30


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise PlatformError(
            "baseline training needs scikit-learn: pip install 'insight-platform[ml]'"
        ) from exc


def select_features(
    df: pd.DataFrame, profile: DatasetProfile, target: str, drop: list[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """(numeric, categorical, excluded) — identifiers and free text are excluded.

    Dropping identifiers is not an optimisation: a near-unique key lets a tree
    memorise the training rows and produces a beautiful, meaningless score.
    """
    excluded = set(drop or []) | {target}
    numeric: list[str] = []
    categorical: list[str] = []

    for stats in profile.columns:
        name = stats.name
        if name in excluded or name not in df.columns:
            continue
        if stats.is_constant:
            excluded.add(name)
            continue
        if stats.semantic_type == "identifier" and stats.pct_unique > 0.5:
            excluded.add(name)
            continue
        if stats.semantic_type == "text":
            excluded.add(name)
            continue
        if stats.semantic_type == "temporal":
            excluded.add(name)  # handled via ordering, not as a raw feature
            continue
        if name in profile.numeric_columns:
            numeric.append(name)
        elif stats.n_unique <= MAX_ONEHOT_CARDINALITY:
            categorical.append(name)
        else:
            excluded.add(name)

    return numeric, categorical, sorted(excluded - {target})


def _encode_labels(y: pd.Series) -> tuple[pd.Series, dict[int, object]]:
    """Map class labels onto integer codes, rarest class last.

    Two reasons this is not just `astype(str)`. Binary scorers (`f1`, `roc_auc`,
    `average_precision`) default to `pos_label=1`, which errors outright on string
    labels like '0'/'1'. And ordering by descending frequency puts the *minority*
    class at code 1, so those defaults score the class you actually care about on
    an imbalanced problem instead of the uninformative majority.
    """
    counts = y.value_counts()
    ordered = list(counts.index)
    if len(ordered) == 2:
        ordered = [counts.index[0], counts.index[-1]]  # majority -> 0, minority -> 1
    mapping = {label: code for code, label in enumerate(ordered)}
    return y.map(mapping).astype(int), {code: label for label, code in mapping.items()}


def _build_pipeline(task: TaskType, numeric: list[str], categorical: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    try:  # sklearn >= 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)
    except TypeError:  # pragma: no cover - older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", encoder),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise PlatformError("no usable feature columns after excluding ids, text and constants")

    pre = ColumnTransformer(transformers, remainder="drop")
    if task == "regression":
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=0, class_weight="balanced"
        )
    return Pipeline([("prep", pre), ("model", model)])


def train_baseline(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target: str,
    task: TaskType,
    time_column: str | None = None,
    cv_folds: int = 5,
) -> BaselineResult:
    _require_sklearn()

    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate

    if target not in df.columns:
        raise PlatformError(f"target {target!r} is not a column of the dataset")
    if task not in ("regression", "binary_classification", "multiclass_classification"):
        raise PlatformError(f"baseline training does not cover task {task!r}")

    notes: list[str] = []
    frame = df.dropna(subset=[target]).copy()
    if len(frame) < len(df):
        notes.append(f"dropped {len(df) - len(frame):,} rows with a missing target")

    # Order by time before sampling, so a time-aware split stays meaningful.
    if time_column and time_column in frame.columns:
        frame = frame.sort_values(time_column)

    if len(frame) > MAX_TRAIN_ROWS:
        frame = frame.tail(MAX_TRAIN_ROWS) if time_column else frame.sample(MAX_TRAIN_ROWS, random_state=0)
        notes.append(f"trained on {MAX_TRAIN_ROWS:,} rows for speed")

    # Leaking columns must go before the split, not after the score. A baseline that
    # quietly keeps them reports a number that says nothing about the real problem.
    leaking = sorted(
        {
            column
            for pattern in detect_target_patterns(frame, profile, target)
            if pattern.kind == "target_leakage"
            for column in pattern.columns
            if column != target
        }
    )
    if leaking:
        notes.append(f"dropped as target leakage: {', '.join(leaking)}")

    numeric, categorical, excluded = select_features(frame, profile, target, drop=leaking)
    if excluded:
        notes.append(f"excluded as features: {', '.join(excluded[:8])}")

    X = frame[numeric + categorical]
    y = frame[target]
    if task != "regression":
        y, label_map = _encode_labels(y)
        notes.append(
            "class codes: " + ", ".join(f"{code}={label}" for code, label in label_map.items())
            + (" (code 1 is the minority class, which the f1/AP scores refer to)"
               if len(label_map) == 2 else "")
        )

    pipeline = _build_pipeline(task, numeric, categorical)

    if time_column and time_column in frame.columns:
        splitter = TimeSeriesSplit(n_splits=min(cv_folds, 5))
        notes.append("used TimeSeriesSplit — a random split would leak the future into training")
    elif task == "regression":
        splitter = min(cv_folds, 5)
    else:
        counts = y.value_counts()
        folds = int(min(cv_folds, max(2, counts.min())))
        if folds < cv_folds:
            notes.append(f"reduced to {folds} folds — the rarest class has only {counts.min()} rows")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

    if task == "regression":
        scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
        dummy = DummyRegressor(strategy="mean")
    elif task == "binary_classification":
        scoring = ["roc_auc", "average_precision", "balanced_accuracy", "f1"]
        dummy = DummyClassifier(strategy="prior")
    else:
        scoring = ["balanced_accuracy", "f1_macro"]
        dummy = DummyClassifier(strategy="prior")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            scores = cross_validate(pipeline, X, y, cv=splitter, scoring=scoring, error_score="raise")
        except Exception as exc:
            raise PlatformError(f"baseline training failed: {exc}") from exc

        dummy_pipeline = _build_pipeline(task, numeric, categorical)
        dummy_pipeline.steps[-1] = ("model", dummy)
        try:
            dummy_scores = cross_validate(dummy_pipeline, X, y, cv=splitter, scoring=scoring)
        except Exception:
            dummy_scores = {}

    metrics: dict[str, float] = {}
    for name in scoring:
        key = f"test_{name}"
        if key in scores:
            value = float(np.mean(scores[key]))
            metrics[name.replace("neg_", "")] = abs(value) if name.startswith("neg_") else value
        if key in dummy_scores:
            value = float(np.mean(dummy_scores[key]))
            metrics[f"baseline_{name.replace('neg_', '')}"] = (
                abs(value) if name.startswith("neg_") else value
            )

    importance = _feature_importance(pipeline, X, y, numeric, categorical)

    return BaselineResult(
        algorithm=(
            "HistGradientBoostingRegressor"
            if task == "regression"
            else "HistGradientBoostingClassifier"
        ),
        task=task,
        metrics=metrics,
        cv_folds=splitter.n_splits if hasattr(splitter, "n_splits") else int(splitter),
        n_train=len(X),
        n_features=len(numeric) + len(categorical),
        feature_importance=importance,
        notes=notes,
    )


def _feature_importance(
    pipeline, X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]
) -> dict[str, float]:
    """Permutation importance on the raw columns.

    Deliberately measured on the *input* columns rather than the encoded ones, so a
    one-hot-exploded categorical is reported as one number an analyst recognises.
    """
    from sklearn.inspection import permutation_importance

    sample = X.sample(min(5000, len(X)), random_state=0)
    y_sample = y.loc[sample.index]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(sample, y_sample)
            result = permutation_importance(
                pipeline, sample, y_sample, n_repeats=5, random_state=0, n_jobs=1
            )
    except Exception:
        return {}

    columns = list(sample.columns)
    scores = {col: float(value) for col, value in zip(columns, result.importances_mean)}
    total = sum(abs(v) for v in scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
    return {name: round(value / total, 4) for name, value in ranked}
