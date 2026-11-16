"""Split manifest tools: create_split_manifest, validate_split_manifest, list_split_manifests."""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.splits import (
    create_split_manifest,
    list_split_manifests,
    validate_split_manifest,
)
from devray_datalab.utils.errors import DevRayError


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


async def create_split_manifest_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Create a reproducible train/validation/test split manifest.

    The test set is a protected holdout and should not be used for training
    or hyperparameter tuning.
    """
    name = arguments.get("name")
    if not name:
        return _err("Required argument: 'name' (dataset filename).", "validation_error")

    try:
        from devray_datalab.core.loader import load_dataset
        from devray_datalab.utils.validation import validate_task_type

        task_type = str(arguments.get("task_type", "classification"))
        validate_task_type(task_type)
        split_type = str(arguments.get("split_type", "random"))
        test_size = float(arguments.get("test_size", 0.2))
        val_size = float(arguments.get("val_size", 0.0))
        random_state = int(arguments.get("random_state", 42))
        stratify_column = arguments.get("stratify_column")
        group_column = arguments.get("group_column")
        time_column = arguments.get("time_column")
        protected_holdout = bool(arguments.get("protected_holdout", True))

        try:
            df = load_dataset(config, str(name))
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Failed to load dataset: {exc}", "dataset_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    try:
        manifest = create_split_manifest(
            config=config,
            df=df,
            dataset_name=str(name),
            split_type=split_type,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify_column=stratify_column,
            group_column=group_column,
            time_column=time_column,
            protected_holdout=protected_holdout,
        )
    except (DevRayError, ValueError) as exc:
        return _err(str(exc), exc.code if isinstance(exc, DevRayError) else "validation_error")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Split manifest creation failed: {exc}", "split_error")

    return _ok(
        {
            "manifest_id": manifest.manifest_id,
            "dataset_name": manifest.dataset_name,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "split_type": manifest.split_type,
            "n_train": manifest.n_train,
            "n_val": manifest.n_val,
            "n_test": manifest.n_test,
            "stratify_column": manifest.stratify_column,
            "group_column": manifest.group_column,
            "time_column": manifest.time_column,
            "protected_holdout": manifest.protected_holdout,
            "note": "The test set is a protected holdout; do not use it for training or tuning.",
        }
    )


async def validate_split_manifest_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Validate a stored split manifest against a dataset."""
    manifest_id = arguments.get("manifest_id")
    name = arguments.get("name")
    if not manifest_id or not name:
        return _err(
            "Required arguments: 'manifest_id' and 'name' (dataset filename).",
            "validation_error",
        )

    try:
        from devray_datalab.core.loader import load_dataset

        try:
            df = load_dataset(config, str(name))
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Failed to load dataset: {exc}", "dataset_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    try:
        report = validate_split_manifest(
            config=config,
            manifest_id=str(manifest_id),
            df=df,
            dataset_name=str(name),
        )
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Manifest validation failed: {exc}", "split_error")

    return _ok(report)


async def list_split_manifests_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """List all stored split manifests, newest first."""
    try:
        manifests = list_split_manifests(config)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to list manifests: {exc}", "split_error")

    return _ok(
        {
            "manifests": [
                {
                    "manifest_id": m.manifest_id,
                    "dataset_name": m.dataset_name,
                    "dataset_fingerprint": m.dataset_fingerprint,
                    "split_type": m.split_type,
                    "n_train": m.n_train,
                    "n_val": m.n_val,
                    "n_test": m.n_test,
                    "created_at": m.created_at,
                    "protected_holdout": m.protected_holdout,
                }
                for m in manifests
            ],
            "count": len(manifests),
        }
    )
