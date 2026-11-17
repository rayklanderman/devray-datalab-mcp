"""Non-blocking training job tools.

start_training_job / get_training_status / cancel_training_job / list_training_jobs
run training outside the MCP request loop so the server stays responsive.
"""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.jobs import JobManager
from devray_datalab.core.splits import validate_manifest_for_consumption
from devray_datalab.tools import training
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


# Module-level manager (single shared instance).
_manager: JobManager | None = None


def _get_manager(config: DevRayConfig) -> JobManager:
    global _manager
    if _manager is None or _manager.config is not config:
        _manager = JobManager(config, training._get_trainer(config))
    return _manager


async def start_training_job_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Submit a training job and return its id immediately (non-blocking)."""
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
            validate_probability,
            validate_task_type,
        )

        task_type = validate_task_type(str(task_type))
        validate_model_name(str(model_name), task_type=task_type)
        test_size = float(arguments.get("test_size", 0.2))
        validate_probability(test_size, "test_size")
        random_state = int(arguments.get("random_state", 42))
        stratify = bool(arguments.get("stratify", True))
        exclude_columns = arguments.get("exclude_columns") or []
        split_manifest_id = arguments.get("split_manifest_id")

        delimiter = arguments.get("delimiter")

        try:
            df = load_dataset(config, str(name), delimiter=delimiter)
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Failed to load dataset: {exc}", "dataset_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    # QA-001: enforce shared mandatory manifest validation before submitting
    # a background job. Invalid manifests must fail without executing training.
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

    manager = _get_manager(config)
    job_id = manager.submit(
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
    return _ok(
        {
            "job_id": job_id,
            "status": "queued",
            "dataset": str(name),
            "target_column": str(target_column),
            "task_type": str(task_type),
            "model": str(model_name),
            "exclude_columns": exclude_columns,
            "split_manifest_id": split_manifest_id,
            "note": "Job submitted. Poll get_training_status with the job_id for results.",
        }
    )


async def get_training_status_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Return the status and result of a training job."""
    job_id = arguments.get("job_id")
    if not job_id:
        return _err("Required argument: 'job_id'.", "validation_error")

    manager = _get_manager(config)
    job = manager.get_job(str(job_id))
    if job is None:
        return _err(f"Job not found: {job_id}", "not_found")

    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }
    if job.result is not None:
        payload["result"] = job.result
    if job.error is not None:
        payload["error"] = job.error
    return _ok(payload)


async def cancel_training_job_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Request cancellation of a queued or running training job."""
    job_id = arguments.get("job_id")
    if not job_id:
        return _err("Required argument: 'job_id'.", "validation_error")

    manager = _get_manager(config)
    try:
        cancelled = manager.cancel_job(str(job_id))
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to cancel job: {exc}", "job_error")

    return _ok({"job_id": str(job_id), "cancelled": cancelled})


async def list_training_jobs_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """List training jobs, newest first."""
    limit = int(arguments.get("limit", 50))
    manager = _get_manager(config)
    try:
        jobs = manager.list_jobs(limit=limit)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to list jobs: {exc}", "job_error")

    return _ok(
        {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status.value,
                    "progress": j.progress,
                    "created_at": j.created_at,
                    "started_at": j.started_at,
                    "completed_at": j.completed_at,
                    "cancel_requested": j.cancel_requested,
                }
                for j in jobs
            ],
            "count": len(jobs),
            "active": manager.active_count(),
        }
    )
