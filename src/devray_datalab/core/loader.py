"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from devray_datalab.config import DevRayConfig
from devray_datalab.utils.errors import DatasetError
from devray_datalab.utils.validation import (
    MAX_PREVIEW_ROWS,
    validate_dataset_path,
    validate_file_size,
)


def list_datasets(workspace: Path) -> list[dict[str, Any]]:
    """List datasets in the workspace with filenames, formats, and sizes."""
    if not workspace.is_dir():
        return []

    datasets: list[dict[str, Any]] = []
    for entry in sorted(workspace.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in (".csv",):
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        datasets.append(
            {
                "filename": entry.name,
                "format": entry.suffix.lower().lstrip("."),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
            }
        )
    return datasets


def load_dataset(config: DevRayConfig, name: str) -> pd.DataFrame:
    """Load a CSV dataset from the workspace, validating path and size."""
    path = validate_dataset_path(config.workspace_dir, name, must_exist=True)
    validate_file_size(path, config.max_dataset_size_bytes)

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface a clear dataset error
        raise DatasetError(f"Failed to load dataset '{name}': {exc}") from exc

    if df.empty:
        raise DatasetError(f"Dataset '{name}' is empty.")
    return df


def preview_dataframe(df: pd.DataFrame, max_rows: int = MAX_PREVIEW_ROWS) -> list[dict[str, Any]]:
    """Return the first ``max_rows`` rows as a list of dicts."""
    return df.head(max_rows).to_dict(orient="records")


def dataset_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute numerical statistics for a dataframe."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {}
    stats = numeric.describe().to_dict()
    return stats
