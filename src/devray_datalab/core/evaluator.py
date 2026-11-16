"""Evaluation metrics and cross-validation for classification and regression.

Metrics are computed with scikit-learn and include clear explanations so
clients can interpret results without guessing. Cross-validation is performed
with preprocessing fitted independently inside each fold, preventing data
leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance as _sk_permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from devray_datalab.core.preprocessing import (
    PreprocessingArtifact,
    build_preprocessor,
    transform_features,
)
from devray_datalab.utils.errors import TrainingError, ValidationError


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    f1_macro: float
    f1_weighted: float
    roc_auc: float | None
    pr_auc: float | None
    log_loss: float | None
    brier_score: float | None
    support: int
    confusion_matrix: list[list[int]] | None
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 6),
            "balanced_accuracy": round(self.balanced_accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "f1_macro": round(self.f1_macro, 6),
            "f1_weighted": round(self.f1_weighted, 6),
            "roc_auc": None if self.roc_auc is None else round(self.roc_auc, 6),
            "pr_auc": None if self.pr_auc is None else round(self.pr_auc, 6),
            "log_loss": None if self.log_loss is None else round(self.log_loss, 6),
            "brier_score": None if self.brier_score is None else round(self.brier_score, 6),
            "support": self.support,
            "confusion_matrix": self.confusion_matrix,
            "explanation": self.explanation,
        }


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    median_absolute_error: float
    mean_absolute_percentage_error: float | None
    support: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae": round(self.mae, 6),
            "rmse": round(self.rmse, 6),
            "r2": round(self.r2, 6),
            "median_absolute_error": round(self.median_absolute_error, 6),
            "mean_absolute_percentage_error": (
                None
                if self.mean_absolute_percentage_error is None
                else round(self.mean_absolute_percentage_error, 6)
            ),
            "support": self.support,
            "explanation": self.explanation,
        }


@dataclass
class CVResult:
    """Cross-validation result with per-fold metrics and diagnostics."""

    strategy: str
    n_folds: int
    scoring: str
    fold_metrics: list[dict[str, Any]]
    mean: float
    std: float
    fold_sizes: list[int]
    runtime_seconds: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_folds": self.n_folds,
            "scoring": self.scoring,
            "fold_metrics": self.fold_metrics,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "fold_sizes": self.fold_sizes,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "explanation": self.explanation,
        }


def classification_explanation() -> str:
    return (
        "Accuracy = fraction of correct predictions. Balanced accuracy = average "
        "recall per class (useful for imbalanced data). Precision = fraction of "
        "positive predictions that are correct. Recall = fraction of actual "
        "positives found. F1 = harmonic mean of precision and recall. "
        "ROC-AUC = ability to rank positives above negatives (binary only). "
        "PR-AUC = area under the precision-recall curve. Log loss = probabilistic "
        "quality (lower is better). Brier score = mean squared error of predicted "
        "probabilities (lower is better). Confusion matrix = per-class counts."
    )


def regression_explanation() -> str:
    return (
        "MAE = average absolute error (same units as target). RMSE = root mean "
        "squared error (penalizes large errors). R^2 = fraction of target "
        "variance explained (1 = perfect, 0 = mean baseline). "
        "Median absolute error = robust typical error (same units as target). "
        "MAPE = mean absolute percentage error (undefined near zero targets)."
    )


def _safe_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    try:
        return float(log_loss(y_true, y_proba))
    except Exception:  # noqa: BLE001
        return None


def _safe_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    try:
        return float(brier_score_loss(y_true, y_proba[:, 1]))
    except Exception:  # noqa: BLE001
        return None


def _safe_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(y_true, y_proba[:, 1]))
    except Exception:  # noqa: BLE001
        return None


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> ClassificationMetrics:
    """Compute classification metrics with zero-division safety."""
    classes = np.unique(y_true)
    is_binary = len(classes) == 2
    average = "binary" if is_binary else "macro"
    zero_division = 0

    try:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        confusion = cm.tolist()
    except Exception:  # noqa: BLE001
        confusion = None

    roc_auc: float | None = None
    pr_auc: float | None = None
    log_loss_val: float | None = None
    brier_val: float | None = None
    if y_proba is not None:
        proba = np.asarray(y_proba)
        if is_binary and proba.ndim == 2 and proba.shape[1] >= 2:
            pos = proba[:, 1]
            try:
                roc_auc = float(roc_auc_score(y_true, pos))
            except Exception:  # noqa: BLE001
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(y_true, pos))
            except Exception:  # noqa: BLE001
                pr_auc = None
            log_loss_val = _safe_log_loss(y_true, proba)
            brier_val = _safe_brier(y_true, proba)
        elif proba.ndim == 1:
            try:
                pr_auc = float(average_precision_score(y_true, proba))
            except Exception:  # noqa: BLE001
                pr_auc = None

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(
            __import__(
                "sklearn.metrics", fromlist=["balanced_accuracy_score"]
            ).balanced_accuracy_score(  # noqa: E501
                y_true, y_pred
            )
        ),
        precision=float(
            precision_score(y_true, y_pred, average=average, zero_division=zero_division)
        ),
        recall=float(recall_score(y_true, y_pred, average=average, zero_division=zero_division)),
        f1=float(f1_score(y_true, y_pred, average=average, zero_division=zero_division)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=zero_division)),
        f1_weighted=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        log_loss=log_loss_val,
        brier_score=brier_val,
        support=int(len(y_true)),
        confusion_matrix=confusion,
        explanation=classification_explanation(),
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
        if not np.isfinite(mape):
            mape = None
    except Exception:  # noqa: BLE001
        mape = None
    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        median_absolute_error=medae,
        mean_absolute_percentage_error=mape,
        support=int(len(y_true)),
        explanation=regression_explanation(),
    )


def _cv_splitter(
    strategy: str,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
    time_column: pd.Series | None = None,
):
    """Return a scikit-learn CV splitter for the requested strategy."""
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy == "group":
        if groups is None:
            raise ValidationError("GroupKFold requires a group column.")
        return GroupKFold(n_splits=n_splits)
    if strategy == "stratified_group":
        if groups is None:
            raise ValidationError("StratifiedGroupKFold requires a group column.")
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValidationError(f"Unknown CV strategy: {strategy}")


def _split(
    splitter: Any,
    df: pd.DataFrame,
    y_all: np.ndarray,
    groups: np.ndarray | None,
):
    """Yield (train_idx, test_idx) handling per-strategy split arguments."""
    from sklearn.model_selection import (
        GroupKFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    if isinstance(splitter, (StratifiedKFold, StratifiedGroupKFold)):
        return splitter.split(df, y=y_all, groups=groups)
    if isinstance(splitter, GroupKFold):
        return splitter.split(df, groups=groups)
    return splitter.split(df)


def _fit_preprocess_and_split(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PreprocessingArtifact]:
    """Split df by index and fit preprocessing on the train portion only."""
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    y_train = train_df[target_column].to_numpy()
    y_test = test_df[target_column].to_numpy()
    X_train = train_df.drop(columns=[target_column])
    X_test = test_df.drop(columns=[target_column])

    artifact = build_preprocessor(
        train_df.assign(**{target_column: train_df[target_column]}),
        target_column=target_column,
        task_type=task_type,
    )
    X_train_t = transform_features(artifact, X_train)
    X_test_t = transform_features(artifact, X_test)
    return X_train_t, X_test_t, y_train, y_test, artifact


def cross_validate_with_preprocessing(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    model_cls: type[BaseEstimator],
    model_kwargs: dict[str, Any],
    strategy: str = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str | None = None,
    time_column: str | None = None,
    scoring: str = "auto",
) -> CVResult:
    """Run cross-validation with preprocessing fitted independently per fold.

    This prevents data leakage: the preprocessing pipeline is fit only on the
    training portion of each fold, never on the held-out fold.
    """
    import time

    from devray_datalab.utils.validation import validate_task_type

    validate_task_type(task_type)

    if target_column not in df.columns:
        raise ValidationError(f"Target column '{target_column}' not found.")

    n_samples = len(df)
    if n_splits < 2:
        raise ValidationError("n_splits must be at least 2.")
    if n_splits > n_samples:
        raise ValidationError(
            f"n_splits ({n_splits}) cannot exceed number of samples ({n_samples})."
        )

    groups: np.ndarray | None = None
    time_series: pd.Series | None = None
    if group_column:
        if group_column not in df.columns:
            raise ValidationError(f"Group column '{group_column}' not found.")
        groups = df[group_column].to_numpy()
    if time_column:
        if time_column not in df.columns:
            raise ValidationError(f"Time column '{time_column}' not found.")
        time_series = df[time_column]

    splitter = _cv_splitter(
        strategy, n_splits, random_state, groups=groups, time_column=time_series
    )

    is_classification = task_type == "classification"

    fold_metrics: list[dict[str, Any]] = []
    scores: list[float] = []
    fold_sizes: list[int] = []

    y_all = df[target_column].to_numpy()
    start = time.perf_counter()

    for fold_idx, (train_idx, test_idx) in enumerate(_split(splitter, df, y_all, groups)):
        fold_sizes.append(int(len(test_idx)))
        X_train_t, X_test_t, y_train, y_test, _artifact = _fit_preprocess_and_split(
            df, target_column, task_type, train_idx, test_idx
        )

        model = model_cls(**model_kwargs)
        try:
            model.fit(X_train_t, y_train)
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"Model training failed on fold {fold_idx + 1}: {exc}") from exc

        y_pred = model.predict(X_test_t)
        y_proba = None
        if is_classification and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_t)
            except Exception:  # pragma: no cover - best effort
                y_proba = None

        if is_classification:
            m = evaluate_classification(np.asarray(y_test), np.asarray(y_pred), y_proba)
            m_dict = m.to_dict()
            score = (
                m_dict["roc_auc"]
                if scoring in ("auto", "roc_auc") and m_dict["roc_auc"] is not None
                else m_dict["f1"]
            )
            fold_metrics.append(
                {
                    "fold": fold_idx + 1,
                    "n_test": int(len(y_test)),
                    "accuracy": m_dict["accuracy"],
                    "balanced_accuracy": m_dict["balanced_accuracy"],
                    "precision": m_dict["precision"],
                    "recall": m_dict["recall"],
                    "f1": m_dict["f1"],
                    "f1_macro": m_dict["f1_macro"],
                    "f1_weighted": m_dict["f1_weighted"],
                    "roc_auc": m_dict["roc_auc"],
                    "pr_auc": m_dict["pr_auc"],
                    "log_loss": m_dict["log_loss"],
                    "brier_score": m_dict["brier_score"],
                    "confusion_matrix": m_dict["confusion_matrix"],
                }
            )
        else:
            m = evaluate_regression(np.asarray(y_test), np.asarray(y_pred))
            m_dict = m.to_dict()
            score = m_dict["r2"] if scoring in ("auto", "r2") else m_dict["mae"]
            fold_metrics.append(
                {
                    "fold": fold_idx + 1,
                    "n_test": int(len(y_test)),
                    "mae": m_dict["mae"],
                    "rmse": m_dict["rmse"],
                    "r2": m_dict["r2"],
                    "median_absolute_error": m_dict["median_absolute_error"],
                    "mean_absolute_percentage_error": m_dict["mean_absolute_percentage_error"],
                }
            )
        scores.append(score if score is not None else float("nan"))

    runtime = time.perf_counter() - start
    valid_scores = [s for s in scores if s is not None and np.isfinite(s)]
    mean_score = float(np.mean(valid_scores)) if valid_scores else float("nan")
    std_score = float(np.std(valid_scores)) if valid_scores else float("nan")

    strategy_label = strategy
    if strategy == "stratified" and is_classification:
        strategy_label = "StratifiedKFold"
    elif strategy == "group":
        strategy_label = "GroupKFold"
    elif strategy == "stratified_group":
        strategy_label = "StratifiedGroupKFold"
    elif strategy == "timeseries":
        strategy_label = "TimeSeriesSplit"
    elif strategy == "kfold":
        strategy_label = "KFold"

    scoring_label = scoring if scoring != "auto" else ("roc_auc" if is_classification else "r2")

    return CVResult(
        strategy=strategy_label,
        n_folds=n_splits,
        scoring=scoring_label,
        fold_metrics=fold_metrics,
        mean=mean_score,
        std=std_score,
        fold_sizes=fold_sizes,
        runtime_seconds=runtime,
        explanation=(
            f"{strategy_label} with {n_splits} folds. Preprocessing was fit on "
            "training data only within each fold to prevent data leakage. "
            f"Mean {scoring_label} = {mean_score:.4f} (std {std_score:.4f})."
        ),
    )


def cross_validate_classification(
    df: pd.DataFrame,
    target_column: str,
    model_cls: type[BaseEstimator],
    model_kwargs: dict[str, Any],
    strategy: str = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str | None = None,
    time_column: str | None = None,
    scoring: str = "auto",
) -> CVResult:
    """Cross-validate a classification model with leakage-safe preprocessing."""
    return cross_validate_with_preprocessing(
        df=df,
        target_column=target_column,
        task_type="classification",
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        strategy=strategy,
        n_splits=n_splits,
        random_state=random_state,
        group_column=group_column,
        time_column=time_column,
        scoring=scoring,
    )


def cross_validate_regression(
    df: pd.DataFrame,
    target_column: str,
    model_cls: type[BaseEstimator],
    model_kwargs: dict[str, Any],
    strategy: str = "kfold",
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str | None = None,
    time_column: str | None = None,
    scoring: str = "auto",
) -> CVResult:
    """Cross-validate a regression model with leakage-safe preprocessing."""
    return cross_validate_with_preprocessing(
        df=df,
        target_column=target_column,
        task_type="regression",
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        strategy=strategy,
        n_splits=n_splits,
        random_state=random_state,
        group_column=group_column,
        time_column=time_column,
        scoring=scoring,
    )


def calibrate_model(
    model: BaseEstimator,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "sigmoid",
    cv: int = 5,
) -> BaseEstimator:
    """Probability-calibrate a fitted classifier using calibration data.

    Calibration is fit on a held-out calibration set (never the training data)
    to avoid leakage. Supported methods: "sigmoid" (Platt scaling) and
    "isotonic" (isotonic regression). Returns a calibrated classifier that
    exposes ``predict_proba`` with better-calibrated probabilities.
    """
    if method not in ("sigmoid", "isotonic"):
        raise ValidationError(f"Unsupported calibration method: {method}.")
    if not hasattr(model, "predict_proba"):
        raise ValidationError("Model does not expose predict_proba; cannot calibrate.")
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    calibrated.fit(X_calib, y_calib)
    return calibrated


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    objective: str = "f1",
    precision_min: float | None = None,
    recall_min: float | None = None,
    cost_matrix: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Optimize a classification decision threshold on validation probabilities.

    Searches thresholds in (0, 1) and returns the threshold maximizing the
    chosen objective. Optionally enforces minimum precision/recall or applies
    a user-supplied cost matrix with keys "tp", "fp", "tn", "fn".

    Results are computed on validation probabilities only; never tune against
    protected holdout data.
    """
    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        scores = y_proba[:, 1]
    elif y_proba.ndim == 1:
        scores = y_proba
    else:
        raise ValidationError("Unsupported probability shape.")

    thresholds = np.linspace(0.001, 0.999, 999)
    best: dict[str, Any] | None = None

    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        if cost_matrix is not None:
            cost = (
                tp * float(cost_matrix.get("tp", 0))
                + fp * float(cost_matrix.get("fp", 0))
                + tn * float(cost_matrix.get("tn", 0))
                + fn * float(cost_matrix.get("fn", 0))
            )
            score = -cost
        elif objective == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif objective == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:  # f1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if precision_min is not None:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if prec < precision_min:
                continue
        if recall_min is not None:
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if rec < recall_min:
                continue

        if best is None or score > best["score"]:
            best = {
                "threshold": float(thr),
                "score": float(score),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            }
            p = best["precision"]
            r = best["recall"]
            best["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    if best is None:
        raise TrainingError("No threshold satisfied the constraints.")

    return {
        "threshold": best["threshold"],
        "objective": objective,
        "score": best["score"],
        "tp": best["tp"],
        "fp": best["fp"],
        "tn": best["tn"],
        "fn": best["fn"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
    }


def permutation_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "auto",
) -> dict[str, Any]:
    """Permutation feature importance for a fitted model.

    Measures the drop in model performance when each feature's values are
    randomly shuffled. Features whose shuffling causes a large performance
    drop are important. Computed on the supplied (validation/holdout) data
    only; never on training data, to avoid overstated importance.

    Returns per-feature importances (mean and std over repeats) and the
    baseline score before any shuffling.
    """
    from sklearn.metrics import (
        accuracy_score,
        r2_score,
    )

    is_classification = hasattr(model, "predict_proba")

    if scoring == "auto":
        scoring = "accuracy" if is_classification else "r2"

    def _score(estimator: BaseEstimator, X_in: np.ndarray, y_in: np.ndarray) -> float:
        y_pred = estimator.predict(X_in)
        if is_classification:
            return float(accuracy_score(y_in, y_pred))
        return float(r2_score(y_in, y_pred))

    baseline = _score(model, X, y)
    result = _sk_permutation_importance(
        model,
        X,
        y,
        scoring=_score,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    importances = result.importances_mean
    stds = result.importances_std
    feature_names = [f"feature_{i}" for i in range(len(importances))]

    return {
        "baseline_score": float(baseline),
        "scoring": scoring,
        "n_repeats": int(n_repeats),
        "importances": [
            {
                "feature": feature_names[i],
                "importance_mean": float(importances[i]),
                "importance_std": float(stds[i]),
            }
            for i in range(len(importances))
        ],
    }
