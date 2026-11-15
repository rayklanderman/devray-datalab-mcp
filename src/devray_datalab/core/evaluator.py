"""Evaluation metrics for classification and regression.

Metrics are computed with scikit-learn and include clear explanations so
clients can interpret results without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    support: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "roc_auc": None if self.roc_auc is None else round(self.roc_auc, 6),
            "support": self.support,
            "explanation": self.explanation,
        }


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    support: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae": round(self.mae, 6),
            "rmse": round(self.rmse, 6),
            "r2": round(self.r2, 6),
            "support": self.support,
            "explanation": self.explanation,
        }


def classification_explanation() -> str:
    return (
        "Accuracy = fraction of correct predictions. Precision = fraction of "
        "positive predictions that are correct. Recall = fraction of actual "
        "positives found. F1 = harmonic mean of precision and recall. "
        "ROC-AUC = ability to rank positives above negatives (binary only)."
    )


def regression_explanation() -> str:
    return (
        "MAE = average absolute error (same units as target). RMSE = root mean "
        "squared error (penalizes large errors). R^2 = fraction of target "
        "variance explained (1 = perfect, 0 = mean baseline, negative = worse "
        "than mean)."
    )


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> ClassificationMetrics:
    """Compute classification metrics with zero-division safety."""
    average = "binary" if len(np.unique(y_true)) == 2 else "macro"
    zero_division = 0

    try:
        roc_auc: float | None = None
        if y_proba is not None and len(np.unique(y_true)) == 2:
            # Use probability of the positive class when available.
            proba = np.asarray(y_proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                roc_auc = float(roc_auc_score(y_true, proba[:, 1]))
            elif proba.ndim == 1:
                roc_auc = float(roc_auc_score(y_true, proba))
    except Exception:  # noqa: BLE001
        roc_auc = None

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(
            precision_score(y_true, y_pred, average=average, zero_division=zero_division)
        ),
        recall=float(recall_score(y_true, y_pred, average=average, zero_division=zero_division)),
        f1=float(f1_score(y_true, y_pred, average=average, zero_division=zero_division)),
        roc_auc=roc_auc,
        support=int(len(y_true)),
        explanation=classification_explanation(),
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        support=int(len(y_true)),
        explanation=regression_explanation(),
    )


def cross_validate_classification(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict[str, float]:
    """Run stratified cross-validation and return mean/std per metric."""
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cv_splitter = StratifiedKFold(n_splits=min(cv, len(np.unique(y))), shuffle=True)
    scores = cross_validate(estimator, X, y, cv=cv_splitter, scoring=scoring, n_jobs=1)
    key = f"test_{scoring}"
    return {
        "mean": float(np.mean(scores[key])),
        "std": float(np.std(scores[key])),
    }
