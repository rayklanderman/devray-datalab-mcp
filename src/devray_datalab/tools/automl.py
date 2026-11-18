"""AutoML tools: automl_run, tune_model, get_model_leaderboard."""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.automl import run_automl
from devray_datalab.core.splits import (
    development_rows_from_manifest,
    validate_manifest_for_consumption,
)
from devray_datalab.utils.errors import DevRayError, sanitize_for_json


def _json(data: Any) -> str:
    import json

    return json.dumps(sanitize_for_json(data), default=str, allow_nan=False)


def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=_json({"ok": True, "data": data}))]


def _err(message: str, code: str = "error") -> list[TextContent]:
    return [
        TextContent(
            type="text",
            text=_json({"ok": False, "error": {"message": message, "code": code}}),
        )
    ]


async def automl_run_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Run an automated machine learning search and return a ranked leaderboard."""
    return await _run_search_tool(config, arguments, leaderboard_only=False)


async def _run_search_tool(
    config: DevRayConfig, arguments: dict, *, leaderboard_only: bool
) -> list[TextContent]:
    name = arguments.get("name")
    target_column = arguments.get("target_column")
    task_type = arguments.get("task_type")
    if not all([name, target_column, task_type]):
        return _err(
            "Required arguments: 'name', 'target_column', 'task_type'.",
            "validation_error",
        )

    try:
        from devray_datalab.core.loader import load_dataset
        from devray_datalab.utils.validation import validate_task_type

        task_type = validate_task_type(str(task_type))
        metric = str(arguments.get("metric", "auto"))
        max_trials = int(arguments.get("max_trials", 10))
        max_time = int(arguments.get("max_time_seconds", 600))
        random_state = int(arguments.get("random_state", 42))
        n_splits = int(arguments.get("cv", 3))
        model_subset = arguments.get("models")
        if "models" in arguments and model_subset is None:
            return _err("models must be a non-empty list of model names.", "validation_error")

        try:
            df = load_dataset(config, str(name), delimiter=arguments.get("delimiter"))
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Failed to load dataset: {exc}", "dataset_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    exclude_columns = arguments.get("exclude_columns") or []
    split_manifest_id = arguments.get("split_manifest_id")

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
        try:
            df, manifest = development_rows_from_manifest(config, df, split_manifest_id)
        except DevRayError as exc:
            return _err(exc.message, exc.code)
    cv_strategy = (
        "timeseries"
        if manifest is not None and manifest.split_type == "time"
        else (
            "group"
            if manifest is not None and manifest.split_type == "group"
            else ("stratified" if task_type == "classification" else "kfold")
        )
    )

    try:
        result = run_automl(
            df=df,
            target_column=str(target_column),
            task_type=str(task_type),
            metric=metric,
            max_trials=max_trials,
            max_time_seconds=max_time,
            random_state=random_state,
            model_subset=model_subset,
            n_splits=n_splits,
            exclude_columns=exclude_columns,
            cv_strategy=cv_strategy,
            group_column=manifest.group_column if manifest is not None else None,
            time_column=manifest.time_column if manifest is not None else None,
        )
    except (DevRayError, ValueError) as exc:
        return _err(str(exc), exc.code if isinstance(exc, DevRayError) else "validation_error")
    except Exception as exc:  # noqa: BLE001
        operation = "Leaderboard" if leaderboard_only else "AutoML run"
        return _err(f"{operation} failed: {exc}", "automl_error")

    data = result.to_dict()
    if leaderboard_only:
        data = {
            key: data[key]
            for key in (
                "task_type",
                "metric",
                "best_model",
                "best_params",
                "best_value",
                "trials",
                "optuna_used",
            )
        }
    return _ok(data)


async def tune_model_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Tune hyperparameters of a single model with cross-validation."""
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
        from devray_datalab.core.loader import load_dataset
        from devray_datalab.utils.validation import (
            validate_model_name,
            validate_task_type,
        )

        task_type = validate_task_type(str(task_type))
        validate_model_name(str(model_name), task_type=task_type)
        max_trials = int(arguments.get("max_trials", 20))
        max_time = int(arguments.get("max_time_seconds", 600))
        random_state = int(arguments.get("random_state", 42))
        n_splits = int(arguments.get("cv", 3))
        metric = str(arguments.get("metric", "auto"))

        try:
            df = load_dataset(config, str(name), delimiter=arguments.get("delimiter"))
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Failed to load dataset: {exc}", "dataset_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    exclude_columns = arguments.get("exclude_columns") or []
    split_manifest_id = arguments.get("split_manifest_id")

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
        try:
            df, manifest = development_rows_from_manifest(config, df, split_manifest_id)
        except DevRayError as exc:
            return _err(exc.message, exc.code)
    cv_strategy = (
        "timeseries"
        if manifest is not None and manifest.split_type == "time"
        else (
            "group"
            if manifest is not None and manifest.split_type == "group"
            else ("stratified" if task_type == "classification" else "kfold")
        )
    )

    try:
        result = run_automl(
            df=df,
            target_column=str(target_column),
            task_type=str(task_type),
            metric=metric,
            max_trials=max_trials,
            max_time_seconds=max_time,
            random_state=random_state,
            model_subset=[model_name],
            n_splits=n_splits,
            exclude_columns=exclude_columns,
            cv_strategy=cv_strategy,
            group_column=manifest.group_column if manifest is not None else None,
            time_column=manifest.time_column if manifest is not None else None,
        )
    except (DevRayError, ValueError) as exc:
        return _err(str(exc), exc.code if isinstance(exc, DevRayError) else "validation_error")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Tuning failed: {exc}", "automl_error")

    return _ok(
        {
            "model": model_name,
            "best_params": result.best_params,
            "best_value": result.best_value,
            "metric": result.metric,
            "trials": [t.to_dict() for t in result.trials],
            "optuna_used": result.optuna_used,
            "runtime_seconds": result.runtime_seconds,
            "error": result.error,
        }
    )


async def get_model_leaderboard_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Run the same fresh AutoML search and return only its ranking and winner."""
    return await _run_search_tool(config, arguments, leaderboard_only=True)
