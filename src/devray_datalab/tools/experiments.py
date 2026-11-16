"""Experiment tracking tools."""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.experiments import ExperimentStore
from devray_datalab.tools.storage import load_artifact
from devray_datalab.utils.errors import DevRayError, ModelError


def _json(data: Any) -> str:
    import json

    return json.dumps(data, default=str)


def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=_json({"ok": True, "data": data}))]


def _err(message: str, code: str = "error") -> list[TextContent]:
    return [
        TextContent(
            type="text",
            text=_json({"ok": False, "error": {"message": message, "code": code}}),
        )
    ]


def _get_store(config: DevRayConfig) -> ExperimentStore:
    return ExperimentStore(config.experiments_db_path)


def _record_to_dict(record) -> dict[str, Any]:
    d = record.to_dict()
    d["created_at"] = record.created_at
    d["training_duration_seconds"] = record.training_duration_seconds
    return d


async def list_experiments_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """List recorded training experiments, newest first."""
    limit = int(arguments.get("limit", 50))
    try:
        store = _get_store(config)
        records = store.list_experiments(limit=limit)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to list experiments: {exc}", "experiment_error")
    return _ok(
        {
            "experiments": [_record_to_dict(r) for r in records],
            "count": len(records),
        }
    )


async def get_experiment_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Get a single recorded experiment by ID."""
    experiment_id = arguments.get("experiment_id")
    if not experiment_id:
        return _err("Required argument: 'experiment_id'.", "validation_error")
    try:
        store = _get_store(config)
        record = store.get_experiment(str(experiment_id))
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to get experiment: {exc}", "experiment_error")
    if record is None:
        return _err(f"Experiment not found: {experiment_id}", "not_found")
    return _ok(_record_to_dict(record))


async def compare_experiments_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Compare multiple recorded experiments by ID."""
    experiment_ids = arguments.get("experiment_ids")
    if experiment_ids is None or not isinstance(experiment_ids, list):
        return _err(
            "Required argument: 'experiment_ids' (a list of experiment IDs).",
            "validation_error",
        )
    try:
        store = _get_store(config)
        records = store.compare_experiments([str(e) for e in experiment_ids])
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to compare experiments: {exc}", "experiment_error")
    found = [r.experiment_id for r in records]
    missing = [e for e in experiment_ids if str(e) not in found]
    return _ok(
        {
            "experiments": [_record_to_dict(r) for r in records],
            "requested": [str(e) for e in experiment_ids],
            "found": found,
            "missing": missing,
        }
    )


async def list_models_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """List experiments that produced a model artifact, newest first."""
    try:
        store = _get_store(config)
        records = store.list_models()
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to list models: {exc}", "experiment_error")
    return _ok(
        {
            "models": [_record_to_dict(r) for r in records],
            "count": len(records),
        }
    )


async def load_model_handle_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Load a previously exported model artifact by experiment ID or artifact name.

    Returns the model's metadata, feature schema, and metrics so the model can
    be reused for prediction via the ``predict`` tool.
    """
    experiment_id = arguments.get("experiment_id")
    artifact_name = arguments.get("artifact_name")
    if not experiment_id and not artifact_name:
        return _err(
            "Required argument: 'experiment_id' or 'artifact_name'.",
            "validation_error",
        )
    try:
        store = _get_store(config)
        if experiment_id:
            record = store.get_experiment(str(experiment_id))
            if record is None:
                return _err(f"Experiment not found: {experiment_id}", "not_found")
            artifact_name = record.artifact_name
        if not artifact_name:
            return _err("Experiment has no exported model artifact.", "not_found")
        bundle = load_artifact(config, str(artifact_name))
    except ModelError as exc:
        return _err(exc.message, exc.code)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load model: {exc}", "model_error")

    if not isinstance(bundle, dict) or "model" not in bundle:
        return _err("Stored artifact is not a valid DevRay model bundle.", "model_error")

    return _ok(
        {
            "artifact_name": artifact_name,
            "experiment_id": experiment_id,
            "model": type(bundle["model"]).__name__,
            "feature_schema": bundle.get("feature_schema", {}),
            "hyperparameters": bundle.get("hyperparameters", {}),
            "metrics": bundle.get("metrics"),
            "train_info": bundle.get("train_info"),
            "metadata": bundle.get("metadata", {}),
        }
    )
