"""Shared task-aware metric policy for model comparison, AutoML, and tuning.

Defines, for each task type, which metrics are supported and whether larger
or smaller values are better. Resolves the ``"auto"`` metric to a documented,
task-appropriate concrete metric before any scoring or ranking happens, so
comparison, fallback AutoML, tuning, and leaderboards all apply identical
selection rules.
"""

from __future__ import annotations

AUTO_METRIC = "auto"

# Classification metrics where larger is better.
CLASSIFICATION_MAXIMIZE: frozenset[str] = frozenset(
    {
        "roc_auc",
        "pr_auc",
        "accuracy",
        "f1",
        "f1_macro",
        "f1_weighted",
        "balanced_accuracy",
        "precision",
        "recall",
    }
)
# Classification metrics where smaller is better.
CLASSIFICATION_MINIMIZE: frozenset[str] = frozenset({"log_loss", "brier_score"})
CLASSIFICATION_METRICS: frozenset[str] = CLASSIFICATION_MAXIMIZE | CLASSIFICATION_MINIMIZE

# Regression metrics where larger is better.
REGRESSION_MAXIMIZE: frozenset[str] = frozenset({"r2"})
# Regression metrics where smaller is better.
REGRESSION_MINIMIZE: frozenset[str] = frozenset(
    {"mae", "rmse", "median_absolute_error", "mean_absolute_percentage_error"}
)
REGRESSION_METRICS: frozenset[str] = REGRESSION_MAXIMIZE | REGRESSION_MINIMIZE

# Task-type registry: (valid metrics, maximize set, minimize set).
_METRIC_POLICY: dict[str, tuple[frozenset[str], frozenset[str], frozenset[str]]] = {
    "classification": (CLASSIFICATION_METRICS, CLASSIFICATION_MAXIMIZE, CLASSIFICATION_MINIMIZE),
    "regression": (REGRESSION_METRICS, REGRESSION_MAXIMIZE, REGRESSION_MINIMIZE),
}

# Task-appropriate default when the caller leaves the metric unresolved.
_AUTO_DEFAULT: dict[str, str] = {"classification": "roc_auc", "regression": "r2"}


def valid_metrics(task_type: str) -> frozenset[str]:
    """Return the supported metric names for a task type."""
    if task_type not in _METRIC_POLICY:
        raise ValueError(f"Unknown task type: {task_type}")
    return _METRIC_POLICY[task_type][0]


def direction_for_metric(task_type: str, metric: str) -> str:
    """Return ``"minimize"`` or ``"maximize"`` for a task/metric pair.

    Raw losses (log-loss, MAE, RMSE, median absolute error, MAPE, brier score)
    are minimized. R², ROC-AUC, PR-AUC, accuracy, F1 variants, balanced
    accuracy, precision and recall are maximized.
    """
    if task_type not in _METRIC_POLICY:
        raise ValueError(f"Unknown task type: {task_type}")
    _, maximize, minimize = _METRIC_POLICY[task_type]
    if metric in minimize:
        return "minimize"
    if metric in maximize:
        return "maximize"
    raise ValueError(f"Unsupported metric '{metric}' for task type '{task_type}'.")


def is_valid_metric(task_type: str, metric: str) -> bool:
    """True when ``metric`` is supported for ``task_type`` (``auto`` included)."""
    if task_type not in _METRIC_POLICY:
        return False
    if metric == AUTO_METRIC:
        return True
    return metric in _METRIC_POLICY[task_type][0]


def resolve_metric(task_type: str, metric: str, n_classes: int | None = None) -> str:
    """Resolve ``metric`` to a concrete, task-appropriate metric name.

    ``"auto"`` resolves to ROC-AUC for binary classification, F1 for
    multiclass classification (binary ROC-AUC is unavailable there), and R²
    for regression. Any other metric is returned unchanged after validation.
    """
    if task_type not in _METRIC_POLICY:
        raise ValueError(f"Unknown task type: {task_type}")
    if metric == AUTO_METRIC:
        if task_type == "classification":
            # Binary ROC-AUC is well defined; multiclass falls back to F1 so
            # every classification task has a usable default objective.
            return "roc_auc" if n_classes == 2 else "f1"
        return "r2"
    valid, _, _ = _METRIC_POLICY[task_type]
    if metric not in valid:
        raise ValueError(f"Unsupported metric '{metric}' for task type '{task_type}'.")
    return metric
