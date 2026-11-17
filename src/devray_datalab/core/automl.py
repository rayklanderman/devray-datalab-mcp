"""Automated Machine Learning engine.

Bounds a practical AutoML flow: inspect the dataset, select suitable candidate
models, train each with leakage-safe cross-validation, rank by objective, and
(optionally) tune the best model with hyperparameter optimization.

Optuna is used when available for tuned search; otherwise a small deterministic
grid search is used. All training happens locally on the user's CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from devray_datalab.core.evaluator import cross_validate_with_preprocessing
from devray_datalab.core.trainer import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    _model_init_kwargs,
)
from devray_datalab.utils.errors import ValidationError

try:
    import optuna  # type: ignore

    _OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _OPTUNA_AVAILABLE = False


@dataclass
class TrialResult:
    """Result of a single AutoML trial (one model configuration)."""

    model: str
    params: dict[str, Any]
    metric: str
    value: float
    std: float
    n_folds: int
    runtime_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "params": self.params,
            "metric": self.metric,
            "value": round(self.value, 6),
            "std": round(self.std, 6),
            "n_folds": self.n_folds,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "error": self.error,
        }


@dataclass
class AutoMLResult:
    """Complete AutoML run result."""

    task_type: str
    metric: str
    dataset_shape: tuple[int, int]
    n_features: int
    n_classes: int | None
    trials: list[TrialResult] = field(default_factory=list)
    best_model: str | None = None
    best_params: dict[str, Any] | None = None
    best_value: float | None = None
    runtime_seconds: float = 0.0
    optuna_used: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "metric": self.metric,
            "dataset_shape": list(self.dataset_shape),
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "trials": [t.to_dict() for t in self.trials],
            "best_model": self.best_model,
            "best_params": self.best_params,
            "best_value": None if self.best_value is None else round(self.best_value, 6),
            "runtime_seconds": round(self.runtime_seconds, 4),
            "optuna_used": self.optuna_used,
            "error": self.error,
        }


def _default_models(task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        return {
            "logistic_regression": CLASSIFICATION_MODELS["logistic_regression"],
            "decision_tree": CLASSIFICATION_MODELS["decision_tree"],
            "random_forest": CLASSIFICATION_MODELS["random_forest"],
            "extra_trees": CLASSIFICATION_MODELS["extra_trees"],
            "knn": CLASSIFICATION_MODELS["knn"],
        }
    return {
        "linear_regression": REGRESSION_MODELS["linear_regression"],
        "ridge_regression": REGRESSION_MODELS["ridge_regression"],
        "decision_tree": REGRESSION_MODELS["decision_tree"],
        "random_forest": REGRESSION_MODELS["random_forest"],
        "extra_trees": REGRESSION_MODELS["extra_trees"],
        "knn": REGRESSION_MODELS["knn"],
    }


def _model_param_grid(model_cls: Any, task_type: str) -> dict[str, list[Any]]:
    """Small deterministic hyperparameter grid per model (no Optuna fallback)."""
    grid: dict[str, list[Any]] = {}
    name = model_cls.__name__
    if task_type == "classification":
        if name == "LogisticRegression":
            grid = {"C": [0.1, 1.0, 10.0], "max_iter": [1000]}
        elif name == "DecisionTreeClassifier":
            grid = {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]}
        elif name == "RandomForestClassifier":
            grid = {"n_estimators": [50, 100], "max_depth": [None, 10]}
        elif name == "ExtraTreesClassifier":
            grid = {"n_estimators": [50, 100], "max_depth": [None, 10]}
        elif name == "KNeighborsClassifier":
            grid = {"n_neighbors": [3, 5, 7]}
        elif name == "HistGradientBoostingClassifier":
            grid = {"max_iter": [50, 100], "learning_rate": [0.05, 0.1]}
        elif name == "SVC":
            grid = {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}
    else:
        if name == "LinearRegression":
            grid = {}
        elif name == "Ridge":
            grid = {"alpha": [0.1, 1.0, 10.0]}
        elif name == "DecisionTreeRegressor":
            grid = {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]}
        elif name == "RandomForestRegressor":
            grid = {"n_estimators": [50, 100], "max_depth": [None, 10]}
        elif name == "ExtraTreesRegressor":
            grid = {"n_estimators": [50, 100], "max_depth": [None, 10]}
        elif name == "KNeighborsRegressor":
            grid = {"n_neighbors": [3, 5, 7]}
        elif name == "HistGradientBoostingRegressor":
            grid = {"max_iter": [50, 100], "learning_rate": [0.05, 0.1]}
        elif name == "SVR":
            grid = {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}
    return grid


def _optuna_suggest(trial: Any, model_cls: Any, task_type: str) -> dict[str, Any]:
    """Suggest hyperparameters via Optuna for a model."""
    name = model_cls.__name__
    params: dict[str, Any] = {}
    if task_type == "classification":
        if name == "LogisticRegression":
            params["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
            params["max_iter"] = 1000
        elif name == "DecisionTreeClassifier":
            params["max_depth"] = trial.suggest_int("max_depth", 1, 16)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        elif name == "RandomForestClassifier":
            params["n_estimators"] = trial.suggest_int("n_estimators", 10, 200)
            params["max_depth"] = trial.suggest_int("max_depth", 1, 16)
        elif name == "ExtraTreesClassifier":
            params["n_estimators"] = trial.suggest_int("n_estimators", 10, 200)
            params["max_depth"] = trial.suggest_int("max_depth", 1, 16)
        elif name == "KNeighborsClassifier":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 15)
        elif name == "HistGradientBoostingClassifier":
            params["max_iter"] = trial.suggest_int("max_iter", 50, 200)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2)
        elif name == "SVC":
            params["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
            params["kernel"] = trial.suggest_categorical("kernel", ["rbf", "linear"])
    else:
        if name == "LinearRegression":
            params = {}
        elif name == "Ridge":
            params["alpha"] = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        elif name == "DecisionTreeRegressor":
            params["max_depth"] = trial.suggest_int("max_depth", 1, 16)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        elif name == "RandomForestRegressor":
            params["n_estimators"] = trial.suggest_int("n_estimators", 10, 200)
            params["max_depth"] = trial.suggest_int("max_depth", 1, 16)
        elif name == "ExtraTreesRegressor":
            params["n_estimators"] = trial.suggest_int("n_estimators", 10, 200)
            params["max_depth"] = trial.suggest_int("max_depth", 1, 16)
        elif name == "KNeighborsRegressor":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 15)
        elif name == "HistGradientBoostingRegressor":
            params["max_iter"] = trial.suggest_int("max_iter", 50, 200)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2)
        elif name == "SVR":
            params["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
            params["kernel"] = trial.suggest_categorical("kernel", ["rbf", "linear"])
    return params


def _score_for_metric(metrics: dict[str, Any], metric: str) -> float | None:
    """Extract the optimization score from a CV metrics dict."""
    if metric == "auto":
        # Prefer roc_auc (classification) / r2 (regression), else f1 / mae.
        if metrics.get("roc_auc") is not None:
            return metrics["roc_auc"]
        if metrics.get("r2") is not None:
            return metrics["r2"]
        if metrics.get("f1") is not None:
            return metrics["f1"]
        if metrics.get("mae") is not None:
            return -metrics["mae"]
        return None
    if metric in metrics:
        val = metrics[metric]
        # Error-like metrics: minimize -> negate for maximization consistency.
        if metric in ("mae", "rmse", "log_loss", "median_absolute_error"):
            return -val if val is not None else None
        return val
    return None


def _direction_for_metric(metric: str) -> str:
    if metric in ("mae", "rmse", "log_loss", "median_absolute_error"):
        return "minimize"
    return "maximize"


def run_automl(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    metric: str = "auto",
    max_trials: int = 10,
    max_time_seconds: int = 600,
    random_state: int = 42,
    model_subset: list[str] | None = None,
    n_splits: int = 3,
    exclude_columns: list[str] | None = None,
    cv_strategy: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
) -> AutoMLResult:
    """Run an automated machine learning search.

    Trains candidate models with leakage-safe cross-validation, ranks by the
    selected objective, and optionally tunes the best model with Optuna.

    Never tunes against the protected holdout; all evaluation is via CV.
    """
    import time as _time

    from devray_datalab.utils.validation import validate_task_type

    validate_task_type(task_type)

    if target_column not in df.columns:
        raise ValidationError(f"Target column '{target_column}' not found.")

    if max_trials < 1:
        raise ValidationError("max_trials must be at least 1.")
    if max_trials > 50:
        raise ValidationError("max_trials must not exceed 50.")

    start = _time.time()

    candidates = _default_models(task_type)
    if model_subset:
        candidates = {k: v for k, v in candidates.items() if k in model_subset}
    if not candidates:
        raise ValidationError("No candidate models selected.")

    y = df[target_column]
    n_classes = int(y.nunique()) if task_type == "classification" else None

    exclude_columns = exclude_columns or []
    strategy = cv_strategy or ("stratified" if task_type == "classification" else "kfold")

    result = AutoMLResult(
        task_type=task_type,
        metric=metric,
        dataset_shape=(int(len(df)), int(df.shape[1] - 1)),
        n_features=int(df.shape[1] - 1),
        n_classes=n_classes,
    )

    trials: list[TrialResult] = []
    direction = _direction_for_metric(metric)

    # Phase 1: train each candidate with CV. Enforce the max_trials cap.
    trials_run = 0
    for model_name, model_cls in candidates.items():
        if trials_run >= max_trials:
            break
        if _time.time() - start > max_time_seconds:
            break
        trials_run += 1
        trial_start = _time.time()
        try:
            cv = cross_validate_with_preprocessing(
                df=df,
                target_column=target_column,
                task_type=task_type,
                model_cls=model_cls,
                model_kwargs=_model_init_kwargs(model_cls, random_state),
                strategy=strategy,
                n_splits=n_splits,
                random_state=random_state,
                group_column=group_column,
                time_column=time_column,
                scoring=metric if metric != "auto" else "auto",
                exclude_columns=exclude_columns,
            )
            # Aggregate mean of the primary metric across folds.
            key = (
                metric
                if metric != "auto"
                else ("roc_auc" if task_type == "classification" else "r2")
            )
            vals = [fm.get(key) for fm in cv.fold_metrics if fm.get(key) is not None]
            mean_val = float(np.mean(vals)) if vals else float("nan")
            std_val = float(np.std(vals)) if vals else float("nan")
            if metric in ("mae", "rmse", "log_loss", "median_absolute_error"):
                score_for_rank = -mean_val
            else:
                score_for_rank = mean_val
            trials.append(
                TrialResult(
                    model=model_name,
                    params={},
                    metric=key if metric == "auto" else metric,
                    value=score_for_rank,
                    std=std_val,
                    n_folds=cv.n_folds,
                    runtime_seconds=_time.time() - trial_start,
                )
            )
        except Exception as exc:  # noqa: BLE001
            trials.append(
                TrialResult(
                    model=model_name,
                    params={},
                    metric=metric,
                    value=float("nan"),
                    std=float("nan"),
                    n_folds=0,
                    runtime_seconds=_time.time() - trial_start,
                    error=str(exc),
                )
            )

    # Phase 2: optionally tune the best model with Optuna.
    optuna_used = False
    valid_trials = [t for t in trials if t.error is None and t.value == t.value]
    if valid_trials and _OPTUNA_AVAILABLE and max_trials > len(trials):
        best_trial = (
            max(valid_trials, key=lambda t: t.value)
            if direction == "maximize"
            else min(valid_trials, key=lambda t: t.value)
        )
        model_cls = candidates[best_trial.model]

        def objective(trial: Any) -> float:
            params = _optuna_suggest(trial, model_cls, task_type)
            try:
                cv = cross_validate_with_preprocessing(
                    df=df,
                    target_column=target_column,
                    task_type=task_type,
                    model_cls=model_cls,
                    model_kwargs=params,
                    strategy="stratified" if task_type == "classification" else "kfold",
                    n_splits=n_splits,
                    random_state=random_state,
                    scoring=metric if metric != "auto" else "auto",
                )
                key = (
                    metric
                    if metric != "auto"
                    else ("roc_auc" if task_type == "classification" else "r2")
                )
                vals = [fm.get(key) for fm in cv.fold_metrics if fm.get(key) is not None]
                val = float(np.mean(vals)) if vals else float("nan")
                if metric in ("mae", "rmse", "log_loss", "median_absolute_error"):
                    return -val
                return val
            except Exception:  # noqa: BLE001
                return float("nan")

        remaining = max_trials - len(trials)
        study = optuna.create_study(direction=direction)  # type: ignore[attr-defined]
        study.optimize(objective, n_trials=remaining)  # type: ignore[attr-defined]
        optuna_used = True

        best = study.best_trial
        best_params = best.params
        best_value = best.value
        if metric in ("mae", "rmse", "log_loss", "median_absolute_error"):
            best_value = -best_value
        best_model = best_trial.model
    elif valid_trials:
        best_trial = (
            max(valid_trials, key=lambda t: t.value)
            if direction == "maximize"
            else min(valid_trials, key=lambda t: t.value)
        )
        best_model = best_trial.model
        best_params = {}
        best_value = best_trial.value
    else:
        best_model = None
        best_params = None
        best_value = None

    result.trials = trials
    result.best_model = best_model
    result.best_params = best_params
    result.best_value = best_value
    result.runtime_seconds = _time.time() - start
    result.optuna_used = optuna_used
    return result
