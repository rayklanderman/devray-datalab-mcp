"""Machine learning tools: prepare_dataset, train_baseline, compare_models."""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.preprocessing import build_preprocessor
from devray_datalab.core.trainer import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    Trainer,
)
from devray_datalab.tools.storage import (
    save_artifact,
)
from devray_datalab.utils.errors import DevRayError, TrainingError
from devray_datalab.utils.validation import (
    validate_model_name,
    validate_probability,
    validate_task_type,
)


def _json(data: Any) -> str:
    import json

    return json.dumps(data, default=str)


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

    try:
        from devray_datalab.core.loader import load_dataset

        df = load_dataset(config, str(name))
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load dataset: {exc}", "dataset_error")

    if target_column not in df.columns:
        return _err(f"Target column '{target_column}' not found.", "validation_error")

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
            "note": "Preprocessing fit on training data only.",
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
        validate_task_type(str(task_type))
        validate_model_name(str(model_name))
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    test_size = float(arguments.get("test_size", 0.2))
    try:
        validate_probability(test_size, "test_size")
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    random_state = int(arguments.get("random_state", 42))
    stratify = bool(arguments.get("stratify", True))

    try:
        from devray_datalab.core.loader import load_dataset

        df = load_dataset(config, str(name))
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load dataset: {exc}", "dataset_error")

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
        )
    except TrainingError as exc:
        return _err(str(exc), exc.code if hasattr(exc, "code") else "training_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Training failed: {exc}", "training_error")

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
    metric = str(arguments.get("metric", "auto"))
    test_size = float(arguments.get("test_size", 0.2))
    try:
        validate_probability(test_size, "test_size")
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    random_state = int(arguments.get("random_state", 42))
    cv = int(arguments.get("cv", 5))
    cv = max(2, min(cv, 10))

    try:
        from devray_datalab.core.loader import load_dataset

        df = load_dataset(config, str(name))
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load dataset: {exc}", "dataset_error")

    if task_type == "classification":
        model_names = list(CLASSIFICATION_MODELS.keys())
    else:
        model_names = list(REGRESSION_MODELS.keys())
    if models_arg:
        model_names = [m for m in models_arg if m in model_names]
    if not model_names:
        return _err("No valid models selected for comparison.", "validation_error")

    from devray_datalab.utils.errors import TrainingError

    trainer = _get_trainer(config)
    results: list[dict[str, Any]] = []

    for model_name in model_names:
        try:
            result = trainer.train_baseline(
                df=df,
                target_column=str(target_column),
                task_type=str(task_type),
                model_name=model_name,
                test_size=test_size,
                random_state=random_state,
                stratify=True,
            )
        except TrainingError as exc:
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

        metrics = result.get("metrics", {})
        results.append(
            {
                "model": model_name,
                "status": "completed",
                "n_train": result.get("n_train"),
                "n_test": result.get("n_test"),
                "metrics": metrics,
                "stratified": result.get("stratified"),
            }
        )

    # Select best model.
    completed = [r for r in results if r["status"] == "completed"]
    best = None
    if completed:
        if metric == "auto":
            if task_type == "classification":
                # Prefer ROC-AUC if available, else F1, else accuracy.
                for key in ("roc_auc", "f1", "accuracy"):
                    if key in completed[0]["metrics"] and completed[0]["metrics"][key] is not None:
                        metric = key
                        break
                else:
                    metric = "accuracy"
        best = max(
            completed,
            key=lambda r: (
                r["metrics"].get(metric) if r["metrics"].get(metric) is not None else float("-inf")
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
            "note": "Compared using consistent train/test splits and one random seed.",
        }
    )
