"""Dataset tools: list_datasets, inspect_dataset."""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.loader import (
    list_datasets,
    load_dataset,
    preview_dataframe,
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
            type="text", text=_json({"ok": False, "error": {"message": message, "code": code}})
        )
    ]


async def list_datasets_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """List available datasets in the configured workspace."""
    try:
        datasets = list_datasets(config.workspace_dir)
        return _ok(
            {
                "workspace": str(config.workspace_dir),
                "count": len(datasets),
                "datasets": datasets,
            }
        )
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc), "unexpected_error")


async def inspect_dataset_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Load and inspect a CSV dataset."""
    name = arguments.get("name")
    if not name:
        return _err("Missing required argument: 'name' (dataset filename).", "validation_error")
    try:
        from devray_datalab.utils.validation import validate_dataset_name

        validate_dataset_name(str(name))
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    max_rows = arguments.get("max_rows", config.max_preview_rows)
    # QA-009: reject unsupported types and invalid ranges with clear errors.
    # Defaults apply only when the optional parameter is omitted.
    if max_rows is not None:
        if isinstance(max_rows, bool) or not isinstance(max_rows, (int, float)):
            return _err(
                f"max_rows must be a number, got {type(max_rows).__name__}.",
                "validation_error",
            )
        max_rows = int(max_rows)
        if max_rows < 1 or max_rows > 100:
            return _err(
                f"max_rows must be between 1 and 100, got {max_rows}.",
                "validation_error",
            )
    else:
        max_rows = config.max_preview_rows

    delimiter = arguments.get("delimiter")

    try:
        df = load_dataset(config, str(name), delimiter=delimiter)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to inspect dataset: {exc}", "dataset_error")

    # Basic metadata.
    n_rows, n_cols = df.shape
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = df.isna().sum().to_dict()
    missing_total = int(sum(missing.values()))
    duplicate_rows = int(df.duplicated().sum())

    # Numerical statistics.
    numeric = df.select_dtypes(include="number")
    numeric_stats: dict[str, Any] = {}
    if not numeric.empty:
        desc = numeric.describe()
        for col in desc.columns:
            numeric_stats[col] = {
                "count": float(desc.loc["count", col]) if "count" in desc.index else None,
                "mean": desc.loc["mean", col] if "mean" in desc.index else None,
                "std": desc.loc["std", col] if "std" in desc.index else None,
                "min": desc.loc["min", col] if "min" in desc.index else None,
                "25%": desc.loc["25%", col] if "25%" in desc.index else None,
                "50%": desc.loc["50%", col] if "50%" in desc.index else None,
                "75%": desc.loc["75%", col] if "75%" in desc.index else None,
                "max": desc.loc["max", col] if "max" in desc.index else None,
            }

    # Categorical summaries.
    categorical = df.select_dtypes(exclude="number")
    categorical_summaries: dict[str, Any] = {}
    for col in categorical.columns:
        vc = categorical[col].value_counts(dropna=True)
        categorical_summaries[col] = {
            "unique": int(categorical[col].nunique(dropna=True)),
            "top": str(vc.index[0]) if len(vc) else None,
            "freq": int(vc.iloc[0]) if len(vc) else 0,
            "missing": int(categorical[col].isna().sum()),
        }

    preview = preview_dataframe(df, max_rows=max_rows)

    return _ok(
        {
            "filename": name,
            "rows": n_rows,
            "columns": n_cols,
            "column_names": list(df.columns),
            "data_types": dtypes,
            "missing_values": missing,
            "missing_values_total": missing_total,
            "duplicate_rows": duplicate_rows,
            "numeric_statistics": numeric_stats,
            "categorical_summaries": categorical_summaries,
            "preview": preview,
            "preview_rows": len(preview),
        }
    )
