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


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        import pyarrow  # noqa: F401  (engine availability check)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DatasetError(
            "Parquet support requires the optional 'pyarrow' package. "
            "Install with: pip install 'devray-datalab-mcp[parquet]'"
        ) from exc
    return pd.read_parquet(path)


def _read_excel(path: Path) -> pd.DataFrame:
    try:
        import openpyxl  # noqa: F401  (engine availability check)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DatasetError(
            "Excel support requires the optional 'openpyxl' package. "
            "Install with: pip install 'devray-datalab-mcp[excel]'"
        ) from exc
    return pd.read_excel(path)


DATASET_READERS: dict[str, Any] = {
    ".csv": lambda p: pd.read_csv(p),
    ".parquet": _read_parquet,
    ".xlsx": _read_excel,
    ".xls": _read_excel,
}


def list_datasets(workspace: Path) -> list[dict[str, Any]]:
    """List datasets in the workspace with filenames, formats, and sizes."""
    if not workspace.is_dir():
        return []

    datasets: list[dict[str, Any]] = []
    for entry in sorted(workspace.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in DATASET_READERS:
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
    """Load a dataset from the workspace, validating path and size.

    Supports CSV, Parquet, and Excel (.xlsx/.xls) formats. Parquet and Excel
    require optional dependencies ('pyarrow' / 'openpyxl').
    """
    path = validate_dataset_path(config.workspace_dir, name, must_exist=True)
    validate_file_size(path, config.max_dataset_size_bytes)

    ext = path.suffix.lower()
    reader = DATASET_READERS.get(ext)
    if reader is None:
        raise DatasetError(
            f"Unsupported file extension '{ext}'. Supported: .csv, .parquet, .xlsx, .xls"
        )

    try:
        df = reader(path)
    except DatasetError:
        raise
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
