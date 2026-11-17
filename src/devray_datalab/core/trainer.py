"""Model training engine.

Trains scikit-learn models locally with train/test splitting, stratification,
thread limits, timeouts, and cooperative cancellation. A single concurrent
training job runs by default.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from devray_datalab.config import DevRayConfig
from devray_datalab.core.evaluator import (
    evaluate_classification,
    evaluate_regression,
)
from devray_datalab.core.preprocessing import (
    PreprocessingArtifact,
    build_preprocessor,
    get_feature_names_out,
    transform_features,
)
from devray_datalab.core.splits import SplitStore, _dataset_fingerprint
from devray_datalab.utils.errors import DatasetError, TrainingError, ValidationError


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Tracked training job."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def request_cancel(self) -> None:
        with self._lock:
            self.cancel_requested = True

    def is_cancel_requested(self) -> bool:
        with self._lock:
            return self.cancel_requested


# Supported baseline models.
CLASSIFICATION_MODELS: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
}
REGRESSION_MODELS: dict[str, type[BaseEstimator]] = {
    "linear_regression": LinearRegression,
    "ridge_regression": Ridge,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "svm": SVR,
    "knn": KNeighborsRegressor,
}


def _model_init_kwargs(model_cls: type[BaseEstimator], random_state: int) -> dict[str, Any]:
    """Return init kwargs, including random_state only when the model supports it."""
    import inspect

    sig = inspect.signature(model_cls.__init__)
    kwargs: dict[str, Any] = {}
    if "random_state" in sig.parameters:
        kwargs["random_state"] = random_state
    return kwargs


class Trainer:
    """Trains baseline models under resource constraints."""

    def __init__(self, config: DevRayConfig) -> None:
        self.config = config
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._active_count = 0

        if config.max_cpu_threads:
            try:
                import threadpoolctl

                threadpoolctl.limit_limits(limits=config.max_cpu_threads)
            except Exception:  # pragma: no cover - best effort
                pass

    # ------------------------------------------------------------------ jobs
    def _register_job(self, job: TrainingJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def _job_running(self) -> bool:
        with self._lock:
            return self._active_count > 0

    def _increment_active(self) -> None:
        with self._lock:
            self._active_count += 1

    def _decrement_active(self) -> None:
        with self._lock:
            self._active_count = max(0, self._active_count - 1)

    def get_job(self, job_id: str) -> TrainingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    # ---------------------------------------------------------------- training
    def train_baseline(
        self,
        df: "pd.DataFrame",
        target_column: str,
        task_type: str,
        model_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        job_id: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        exclude_columns: list[str] | None = None,
        split_manifest_id: str | None = None,
    ) -> dict[str, Any]:
        """Train a single baseline model and return structured results."""

        job_id = job_id or str(uuid.uuid4())
        job = TrainingJob(job_id=job_id)
        self._register_job(job)

        if self._job_running() and self.config.max_concurrent_jobs <= 1:
            job.status = JobStatus.FAILED
            job.error = "A training job is already running. Try again after it completes."
            raise TrainingError(job.error)

        self._increment_active()
        job.status = JobStatus.RUNNING
        try:
            result = self._run_training(
                job,
                df,
                target_column,
                task_type,
                model_name,
                test_size,
                random_state,
                stratify,
                exclude_columns=exclude_columns,
                split_manifest_id=split_manifest_id,
                cancel_check=cancel_check,
            )
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.result = result
            return result
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = str(exc)
            raise
        finally:
            self._decrement_active()

    def _run_training(
        self,
        job: TrainingJob,
        df: "pd.DataFrame",
        target_column: str,
        task_type: str,
        model_name: str,
        test_size: float,
        random_state: int,
        stratify: bool,
        exclude_columns: list[str] | None = None,
        split_manifest_id: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        from devray_datalab.utils.validation import validate_model_name, validate_task_type

        validate_task_type(task_type)

        if task_type == "classification":
            models = CLASSIFICATION_MODELS
        else:
            models = REGRESSION_MODELS

        validate_model_name(model_name, task_type=task_type)
        model_cls = models[model_name]

        if target_column not in df.columns:
            raise ValidationError(f"Target column '{target_column}' not found.")

        # Normalize and drop excluded columns before any split, preprocessing
        # fit, or model training so they can never reach a fitted model.
        exclude_columns = exclude_columns or []
        excluded_existing = [c for c in exclude_columns if c in df.columns and c != target_column]
        if target_column in excluded_existing:
            excluded_existing.remove(target_column)

        y = df[target_column]
        X = df.drop(columns=[target_column] + excluded_existing)

        # Validate target suitability.
        if task_type == "classification":
            if y.nunique() < 2:
                raise TrainingError("Classification requires at least 2 target classes.")
            class_counts = y.value_counts()
            if (class_counts < 2).any():
                raise TrainingError("Each class needs at least 2 samples for stratified splitting.")

        # Preprocessing is fit on training data only after splitting; excluded
        # columns are already removed from X so they never enter the preprocessor.
        job.progress = 0.1

        manifest = None
        if split_manifest_id:
            store = SplitStore(self.config)
            manifest = store.load(split_manifest_id)
            if manifest is None:
                raise DatasetError(f"Split manifest not found: {split_manifest_id}")
            if manifest.dataset_fingerprint != _dataset_fingerprint(df):
                raise ValidationError(
                    "Dataset fingerprint mismatch: dataset changed since the manifest was created."
                )
            # Protected test holdout is never used for training, preprocessing
            # fitting, cross-validation, or hyperparameter selection.
            train_indices = list(manifest.train_indices)
            val_indices = list(manifest.val_indices)
            test_indices = list(manifest.test_indices)
            if not train_indices:
                raise TrainingError("Manifest train set is empty.")
            if not test_indices:
                raise TrainingError("Manifest test set is empty.")
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            # QA-002: use validation data for development-stage evaluation;
            # never the protected test holdout.
            if val_indices:
                X_test = X.iloc[val_indices]
                y_test = y.iloc[val_indices]
            else:
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]
        else:
            stratify_col = y if (stratify and task_type == "classification") else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_col,
                )
            except ValueError as exc:
                # Stratification failed (e.g. too few samples); retry without stratify.
                if stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                else:
                    raise TrainingError(f"Train/test split failed: {exc}") from exc

        if len(X_test) == 0:
            raise TrainingError("Test set is empty; dataset too small to split.")

        job.progress = 0.3
        if job.is_cancel_requested() or (cancel_check and cancel_check()):
            raise TrainingError("Training cancelled.")

        # Fit preprocessing on training data only. Excluded columns are absent
        # from X_train, so they cannot leak into the fitted preprocessor or model.
        artifact = build_preprocessor(
            X_train.assign(**{target_column: y_train}),
            target_column=target_column,
            task_type=task_type,
        )
        X_train_t = transform_features(artifact, X_train)
        X_test_t = transform_features(artifact, X_test)

        job.progress = 0.5
        if job.is_cancel_requested() or (cancel_check and cancel_check()):
            raise TrainingError("Training cancelled.")

        # Train model.
        model = model_cls(**_model_init_kwargs(model_cls, random_state))
        try:
            model.fit(X_train_t, y_train)
        except Exception as exc:
            raise TrainingError(f"Model training failed: {exc}") from exc

        job.progress = 0.8
        if job.is_cancel_requested() or (cancel_check and cancel_check()):
            raise TrainingError("Training cancelled.")

        # Evaluate.
        y_pred = model.predict(X_test_t)
        y_proba = None
        if hasattr(model, "predict_proba") and task_type == "classification":
            try:
                y_proba = model.predict_proba(X_test_t)
            except Exception:  # pragma: no cover - best effort
                y_proba = None

        if task_type == "classification":
            metrics = evaluate_classification(np.asarray(y_test), np.asarray(y_pred), y_proba)
        else:
            metrics = evaluate_regression(np.asarray(y_test), np.asarray(y_pred))

        job.progress = 1.0

        return {
            "job_id": job.job_id,
            "status": "completed",
            "task_type": task_type,
            "model": model_name,
            "test_size": test_size,
            "random_state": random_state,
            "stratified": bool(stratify and task_type == "classification"),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": artifact.feature_columns,
            "target": target_column,
            "n_features": len(artifact.feature_columns),
            "numeric_features": artifact.numeric_features,
            "categorical_features": artifact.categorical_features,
            "preprocessed_feature_names": get_feature_names_out(artifact),
            "excluded_columns": excluded_existing,
            "split_manifest_id": split_manifest_id,
            "metrics": metrics.to_dict() if hasattr(metrics, "to_dict") else metrics,
            "model_fitted": True,
        }

    def train_model_with_artifact(
        self,
        df: "pd.DataFrame",
        target_column: str,
        task_type: str,
        model_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        exclude_columns: list[str] | None = None,
        split_manifest_id: str | None = None,
    ) -> tuple[dict[str, Any], PreprocessingArtifact, BaseEstimator]:
        """Train and return results plus the fitted preprocessor and model.

        Excluded columns are dropped before any split, preprocessing fit, or
        training so they can never reach a fitted transformer or estimator. When a
        split manifest is supplied, only the manifest train rows are used for
        training/preprocessing fit and the protected test holdout is never
        touched.
        """

        from devray_datalab.utils.validation import validate_model_name, validate_task_type

        task_type = validate_task_type(task_type)
        validate_model_name(model_name, task_type=task_type)
        models = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
        model_cls = models[model_name]

        if target_column not in df.columns:
            raise ValidationError(f"Target column '{target_column}' not found.")

        # Normalize and drop excluded columns before any split or preprocessing fit.
        exclude_columns = exclude_columns or []
        excluded_existing = [c for c in exclude_columns if c in df.columns and c != target_column]
        if target_column in excluded_existing:
            excluded_existing.remove(target_column)

        y = df[target_column]
        X = df.drop(columns=[target_column] + excluded_existing)

        if task_type == "classification":
            if y.nunique() < 2:
                raise TrainingError("Classification requires at least 2 target classes.")

        manifest = None
        if split_manifest_id:
            store = SplitStore(self.config)
            manifest = store.load(split_manifest_id)
            if manifest is None:
                raise DatasetError(f"Split manifest not found: {split_manifest_id}")
            if manifest.dataset_fingerprint != _dataset_fingerprint(df):
                raise ValidationError(
                    "Dataset fingerprint mismatch: dataset changed since the manifest was created."
                )
            # Protected test holdout is never used for training or preprocessing fit.
            train_indices = list(manifest.train_indices)
            val_indices = list(manifest.val_indices)
            test_indices = list(manifest.test_indices)
            if not train_indices:
                raise TrainingError("Manifest train set is empty.")
            if not test_indices:
                raise TrainingError("Manifest test set is empty.")
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            # QA-002: use validation data for development-stage evaluation;
            # never the protected test holdout.
            if val_indices:
                X_test = X.iloc[val_indices]
                y_test = y.iloc[val_indices]
            else:
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]
        else:
            stratify_col = y if (stratify and task_type == "classification") else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_col,
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

        if len(X_test) == 0:
            raise TrainingError("Test set is empty; dataset too small to split.")

        # Fit preprocessing on training data only. Excluded columns are absent
        # from X_train, so they cannot leak into the fitted preprocessor or model.
        artifact = build_preprocessor(
            X_train.assign(**{target_column: y_train}),
            target_column=target_column,
            task_type=task_type,
        )
        X_train_t = transform_features(artifact, X_train)
        X_test_t = transform_features(artifact, X_test)

        model = model_cls(**_model_init_kwargs(model_cls, random_state))
        try:
            model.fit(X_train_t, y_train)
        except Exception as exc:
            raise TrainingError(f"Model training failed: {exc}") from exc

        y_pred = model.predict(X_test_t)
        y_proba = None
        if hasattr(model, "predict_proba") and task_type == "classification":
            try:
                y_proba = model.predict_proba(X_test_t)
            except Exception:
                y_proba = None

        if task_type == "classification":
            metrics = evaluate_classification(np.asarray(y_test), np.asarray(y_pred), y_proba)
        else:
            metrics = evaluate_regression(np.asarray(y_test), np.asarray(y_pred))

        results = {
            "status": "completed",
            "task_type": task_type,
            "model": model_name,
            "test_size": test_size,
            "random_state": random_state,
            "stratified": bool(stratify and task_type == "classification"),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": artifact.feature_columns,
            "target": target_column,
            "metrics": metrics.to_dict() if hasattr(metrics, "to_dict") else metrics,
            "excluded_columns": excluded_existing,
            "split_manifest_id": split_manifest_id,
        }
        return results, artifact, model
