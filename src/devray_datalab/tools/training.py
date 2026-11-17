"""Machine learning tools: prepare_dataset, train_baseline, compare_models."""

from __future__ import annotations

import importlib.metadata as _md
import uuid
from typing import Any

import numpy as np
import pandas as pd
from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.evaluator import cross_validate_with_preprocessing
from devray_datalab.core.experiments import ExperimentStore
from devray_datalab.core.metric_policy import (
    AUTO_METRIC,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    direction_for_metric,
    resolve_metric,
)
from devray_datalab.core.preprocessing import build_preprocessor
from devray_datalab.core.splits import (
    SplitStore,
    validate_manifest_for_consumption,
)
from devray_datalab.core.trainer import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    Trainer,
    _model_init_kwargs,
)
from devray_datalab.tools.storage import (
    save_artifact,
)
from devray_datalab.utils.errors import (
    DevRayError,
    TrainingError,
    ValidationError,
    sanitize_for_json,
)
from devray_datalab.utils.validation import (
    validate_model_name,
    validate_probability,
    validate_task_type,
)


def _json(data: Any) -> str:
    import json

    return json.dumps(sanitize_for_json(data), default=str, allow_nan=False)


def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=_json({"ok": True, "data": data}))]


def _err(message: str, code: str = "error") -> list[TextContent]:

    return [
        TextContent(
            type="text", text=_json({"ok": False, "error": {"message": message, "code": code}})
        )
    ]


# ---------------------------------------------------------------------------
# Module-level trainer (single shared instance enforcing one concurrent job).
# ---------------------------------------------------------------------------
_trainer: Trainer | None = None


def _get_trainer(config: DevRayConfig) -> Trainer:
    global _trainer
    if _trainer is None or _trainer.config is not config:
        _trainer = Trainer(config)
    return _trainer


def _record_training_experiment(
    config: DevRayConfig,
    df: pd.DataFrame,
    dataset_name: str,
    task_type: str,
    algorithm: str,
    result: dict[str, Any],
) -> None:
    """Record a training run in the experiment store (best-effort)."""
    store = ExperimentStore(config.experiments_db_path)
    model_cls = (CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS)[
        algorithm
    ]
    hyperparameters = _model_init_kwargs(model_cls, int(result.get("random_state", 42)))
    feature_columns = result.get("features", [])
    feature_schema = {
        "feature_columns": feature_columns,
        "target_column": result.get("target"),
        "task_type": task_type,
        "n_features": len(feature_columns),
    }
    library_versions = {
        pkg: _md.version(pkg) or "unknown"
        for pkg in ("scikit-learn", "pandas", "numpy", "scipy", "joblib", "mcp")
    }
    store.record(
        experiment_id=str(uuid.uuid4()),
        dataset_name=dataset_name,
        df_bytes=df.to_csv(index=False).encode("utf-8"),
        task_type=task_type,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        library_versions=library_versions,
        metrics=result.get("metrics", {}),
        cv_result=None,
        n_train=int(result.get("n_train", 0)),
        n_test=int(result.get("n_test", 0)),
        feature_schema=feature_schema,
        training_duration_seconds=float(result.get("training_duration_seconds", 0.0)),
        artifact_name=None,
        status=result.get("status", "completed"),
        error=None,
    )


def _record_comparison_experiment(
    config: DevRayConfig,
    df: pd.DataFrame,
    dataset_name: str,
    task_type: str,
    algorithm: str,
    cv_result: dict[str, Any],
    metrics: dict[str, Any],
    random_state: int,
) -> None:
    """Record a model-comparison run (CV) in the experiment store (best-effort)."""
    store = ExperimentStore(config.experiments_db_path)
    model_cls = (CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS)[
        algorithm
    ]
    hyperparameters = _model_init_kwargs(model_cls, int(random_state))
    feature_schema = {
        "target_column": None,
        "task_type": task_type,
        "n_folds": cv_result.get("n_folds"),
        "strategy": cv_result.get("strategy"),
    }
    library_versions = {
        pkg: _md.version(pkg) or "unknown"
        for pkg in ("scikit-learn", "pandas", "numpy", "scipy", "joblib", "mcp")
    }
    store.record(
        experiment_id=str(uuid.uuid4()),
        dataset_name=dataset_name,
        df_bytes=df.to_csv(index=False).encode("utf-8"),
        task_type=task_type,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        library_versions=library_versions,
        metrics=metrics,
        cv_result=cv_result,
        n_train=None,
        n_test=None,
        feature_schema=feature_schema,
        training_duration_seconds=float(cv_result.get("runtime_seconds", 0.0)),
        artifact_name=None,
        status="completed",
        error=None,
    )


async def prepare_dataset_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Define target and task type, fit preprocessing on training data only.

    Returns a preprocessing handle id that can be referenced by train tools.
    The fitted preprocessor is stored under a generated handle in the artifacts
    directory so prediction can reuse the exact same transformation.
    """
    name = arguments.get("name")
    target_column = arguments.get("target_column")
    task_type = arguments.get("task_type")
    if not name or not target_column or not task_type:
        return _err(
            "Required arguments: 'name', 'target_column', 'task_type'.",
            "validation_error",
        )
    try:
        validate_task_type(str(task_type))
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    exclude_columns = arguments.get("exclude_columns") or []
    delimiter = arguments.get("delimiter")

    try:
        from devray_datalab.core.loader import load_dataset

        df = load_dataset(config, str(name), delimiter=delimiter)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load dataset: {exc}", "dataset_error")

    if target_column not in df.columns:
        return _err(f"Target column '{target_column}' not found.", "validation_error")

    # Excluded columns are dropped before preprocessing so they never enter
    # the fitted preprocessor or any downstream model.
    excluded_existing = [c for c in exclude_columns if c in df.columns and c != target_column]
    if excluded_existing:
        df = df.drop(columns=excluded_existing)

    # Test/test split so preprocessing is fit on training data only.
    test_size = float(arguments.get("test_size", 0.2))
    try:
        validate_probability(test_size, "test_size")
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    random_state = int(arguments.get("random_state", 42))
    stratify = bool(arguments.get("stratify", True)) and task_type == "classification"

    from sklearn.model_selection import train_test_split

    y = df[target_column]
    X = df.drop(columns=[target_column])
    stratify_col = y if stratify else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    try:
        artifact = build_preprocessor(
            X_train.assign(**{target_column: y_train}),
            target_column=target_column,
            task_type=task_type,
        )
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    handle = f"prep_{abs(hash((name, target_column, task_type, random_state))) % 10**10}"
    payload = {
        "artifact": artifact,
        "feature_columns": artifact.feature_columns,
        "target_column": target_column,
        "task_type": task_type,
        "numeric_features": artifact.numeric_features,
        "categorical_features": artifact.categorical_features,
    }
    try:
        save_artifact(config, handle, payload, kind="preprocessing")
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    return _ok(
        {
            "handle": handle,
            "dataset": name,
            "target_column": target_column,
            "task_type": task_type,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "feature_columns": artifact.feature_columns,
            "numeric_features": artifact.numeric_features,
            "categorical_features": artifact.categorical_features,
            "n_features": len(artifact.feature_columns),
            "excluded_columns": excluded_existing,
            "note": "Preprocessing fit on training data only; excluded columns dropped first.",
        }
    )


async def train_baseline_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Train a single baseline classification or regression model."""
    name = arguments.get("name")
    target_column = arguments.get("target_column")
    task_type = arguments.get("task_type")
    model_name = arguments.get("model_name")
    if not all([name, target_column, task_type, model_name]):
        return _err(
            "Required arguments: 'name', 'target_column', 'task_type', 'model_name'.",
            "validation_error",
        )
    try:
        task_type = validate_task_type(str(task_type))
        validate_model_name(str(model_name), task_type=task_type)
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    test_size = float(arguments.get("test_size", 0.2))
    try:
        validate_probability(test_size, "test_size")
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    random_state = int(arguments.get("random_state", 42))
    stratify = bool(arguments.get("stratify", True))

    exclude_columns = arguments.get("exclude_columns") or []
    split_manifest_id = arguments.get("split_manifest_id")

    delimiter = arguments.get("delimiter")

    try:
        from devray_datalab.core.loader import load_dataset

        df = load_dataset(config, str(name), delimiter=delimiter)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load dataset: {exc}", "dataset_error")

    # QA-001: enforce shared mandatory manifest validation before any
    # indexing, preprocessing, fitting, or model selection.
    if split_manifest_id:
        try:
            manifest_validation = validate_manifest_for_consumption(
                config=config,
                manifest_id=str(split_manifest_id),
                df=df,
                dataset_name=str(name),
            )
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Manifest validation failed: {exc}", "split_error")
        if not manifest_validation["valid"]:
            return _err(
                "Invalid split manifest: " + "; ".join(manifest_validation["errors"]),
                "validation_error",
            )

    trainer = _get_trainer(config)
    try:
        result = trainer.train_baseline(
            df=df,
            target_column=str(target_column),
            task_type=str(task_type),
            model_name=str(model_name),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            exclude_columns=exclude_columns,
            split_manifest_id=split_manifest_id,
        )
    except TrainingError as exc:
        return _err(str(exc), exc.code if hasattr(exc, "code") else "training_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Training failed: {exc}", "training_error")

    # Record the experiment.
    try:
        _record_training_experiment(
            config=config,
            df=df,
            dataset_name=str(name),
            task_type=str(task_type),
            algorithm=str(model_name),
            result=result,
        )
    except Exception:  # noqa: BLE001 - recording must not fail the tool
        pass

    return _ok(result)


async def compare_models_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Train and compare multiple models using consistent splits and CV."""
    name = arguments.get("name")
    target_column = arguments.get("target_column")
    task_type = arguments.get("task_type")
    if not all([name, target_column, task_type]):
        return _err(
            "Required arguments: 'name', 'target_column', 'task_type'.",
            "validation_error",
        )
    try:
        validate_task_type(str(task_type))
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    models_arg = arguments.get("models")
    metric_arg = str(arguments.get("metric", "auto"))
    test_size = float(arguments.get("test_size", 0.2))
    try:
        validate_probability(test_size, "test_size")
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    random_state = int(arguments.get("random_state", 42))
    cv = int(arguments.get("cv", 5))
    cv = max(2, min(cv, 10))

    exclude_columns = arguments.get("exclude_columns") or []
    split_manifest_id = arguments.get("split_manifest_id")

    delimiter = arguments.get("delimiter")

    try:
        from devray_datalab.core.loader import load_dataset

        df = load_dataset(config, str(name), delimiter=delimiter)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load dataset: {exc}", "dataset_error")

    # QA-001: enforce shared mandatory manifest validation before any
    # indexing, preprocessing, CV, fitting, or model selection.
    manifest = None
    if split_manifest_id:
        try:
            manifest_validation = validate_manifest_for_consumption(
                config=config,
                manifest_id=str(split_manifest_id),
                df=df,
                dataset_name=str(name),
            )
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Manifest validation failed: {exc}", "split_error")
        if not manifest_validation["valid"]:
            return _err(
                "Invalid split manifest: " + "; ".join(manifest_validation["errors"]),
                "validation_error",
            )
        store = SplitStore(config)
        manifest = store.load(str(split_manifest_id))
        cv_indices = list(manifest.train_indices) + list(manifest.val_indices)
        if not cv_indices:
            return _err("Manifest has no train/validation rows for comparison.", "validation_error")
        df = df.iloc[cv_indices].reset_index(drop=True)

    # Resolve the selection objective before any model is scored. ``auto``
    # resolves to ROC-AUC for binary classification, F1 for multiclass, and
    # R² for regression; unknown metrics are rejected before fitting.
    if task_type == "classification":
        model_names = list(CLASSIFICATION_MODELS.keys())
        _n_classes = int(df[target_column].nunique())
    else:
        model_names = list(REGRESSION_MODELS.keys())
        _n_classes = None
    if metric_arg != AUTO_METRIC:
        _allowed = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
        if metric_arg not in _allowed:
            return _err(
                f"Unsupported metric '{metric_arg}' for task type '{task_type}'.",
                "validation_error",
            )
    metric = resolve_metric(task_type, metric_arg, _n_classes)
    _selection_direction = direction_for_metric(task_type, metric)
    if models_arg:
        model_names = [m for m in models_arg if m in model_names]
    if not model_names:
        return _err("No valid models selected for comparison.", "validation_error")

    from devray_datalab.utils.errors import TrainingError

    # QA-003: derive CV strategy from manifest split type. Chronological
    # manifests use forward-chaining temporal CV; group manifests use
    # group-aware CV; stratified/random manifests use existing strategies.
    # No protected-test rows appear in any CV fold (already restricted to
    # train+val above).
    if manifest is not None:
        if manifest.split_type == "time":
            cv_strategy = "timeseries"
        elif manifest.split_type == "group":
            cv_strategy = "group"
        elif manifest.split_type == "stratified":
            cv_strategy = "stratified"
        else:
            cv_strategy = "kfold"
    else:
        cv_strategy = "stratified" if task_type == "classification" else "kfold"

    results: list[dict[str, Any]] = []

    for model_name in model_names:
        try:
            if task_type == "classification":
                model_cls = CLASSIFICATION_MODELS[model_name]
            else:
                model_cls = REGRESSION_MODELS[model_name]
        except KeyError:
            results.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error": f"Unsupported model: {model_name}.",
                }
            )
            continue

        model_kwargs = _model_init_kwargs(model_cls, random_state)
        try:
            cv_result = cross_validate_with_preprocessing(
                df=df,
                target_column=str(target_column),
                task_type=str(task_type),
                model_cls=model_cls,
                model_kwargs=model_kwargs,
                strategy=cv_strategy,
                n_splits=cv,
                random_state=random_state,
                group_column=manifest.group_column if manifest is not None else None,
                time_column=manifest.time_column if manifest is not None else None,
                scoring=metric,
                exclude_columns=exclude_columns,
            )
        except (TrainingError, ValidationError) as exc:
            results.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue
        except DevRayError as exc:
            results.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error": exc.message,
                }
            )
            continue

        # Aggregate fold metrics into a single metrics dict for result display.
        fold_metrics = cv_result.fold_metrics
        agg: dict[str, Any] = {}
        if fold_metrics:
            keys = fold_metrics[0].keys()
            for key in keys:
                if key in ("fold", "n_test", "confusion_matrix"):
                    continue
                vals = [fm.get(key) for fm in fold_metrics]
                numeric_vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
                if numeric_vals:
                    agg[key] = float(np.mean(numeric_vals))
                    agg[f"{key}_std"] = float(np.std(numeric_vals))
                else:
                    agg[key] = vals[0] if vals else None

        results.append(
            {
                "model": model_name,
                "status": "completed",
                "cv": cv_result.to_dict(),
                "metrics": agg,
                "stratified": cv_strategy in ("stratified", "stratified_group"),
                "n_folds": cv_result.n_folds,
                "strategy": cv_result.strategy,
            }
        )

        # Record the experiment (best-effort).
        try:
            _record_comparison_experiment(
                config=config,
                df=df,
                dataset_name=str(name),
                task_type=str(task_type),
                algorithm=model_name,
                cv_result=cv_result.to_dict(),
                metrics=agg,
                random_state=random_state,
            )
        except Exception:  # noqa: BLE001 - recording must not fail the tool
            pass

    # Select best model using the resolved objective and its direction: losses
    # are minimized, everything else is maximized. Reported metric values keep
    # their natural sign (losses are never negated).
    completed = [r for r in results if r["status"] == "completed"]
    best = None
    if completed:
        if _selection_direction == "minimize":
            best = min(
                completed,
                key=lambda r: (
                    r["metrics"].get(metric)
                    if r["metrics"].get(metric) is not None
                    else float("inf")
                ),
            )
        else:
            best = max(
                completed,
                key=lambda r: (
                    r["metrics"].get(metric)
                    if r["metrics"].get(metric) is not None
                    else float("-inf")
                ),
            )

    return _ok(
        {
            "dataset": name,
            "target_column": target_column,
            "task_type": task_type,
            "test_size": test_size,
            "random_state": random_state,
            "selection_metric": metric,
            "models_compared": len(results),
            "results": results,
            "best_model": best["model"] if best else None,
            "best_metrics": best["metrics"] if best else None,
            "excluded_columns": exclude_columns,
            "split_manifest_id": split_manifest_id,
            "note": "Compared with consistent splits; protected holdout excluded with a manifest.",
        }
    )
