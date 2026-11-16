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

SUPPORTED_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")
SAMPLE_LINES: int = 50
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024


def _detect_delimiter(path: Path) -> str:
    """Detect the CSV delimiter by sampling the first lines of the file.

    Tests comma, semicolon, tab, and pipe delimiters. Returns the delimiter
    that produces the most consistent multi-column count across sampled lines.
    Falls back to comma if no delimiter produces more than one column.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            sample_lines: list[str] = []
            for _ in range(SAMPLE_LINES):
                line = f.readline()
                if not line:
                    break
                sample_lines.append(line.rstrip("\n\r"))
    except UnicodeDecodeError:
        # Try latin-1 as fallback for non-UTF-8 files
        try:
            with open(path, "r", encoding="latin-1", errors="replace") as f:
                sample_lines = []
                for _ in range(SAMPLE_LINES):
                    line = f.readline()
                    if not line:
                        break
                    sample_lines.append(line.rstrip("\n\r"))
        except OSError as exc:
            raise DatasetError(f"Failed to read dataset '{path.name}': {exc}") from exc
    except OSError as exc:
        raise DatasetError(f"Failed to read dataset '{path.name}': {exc}") from exc

    if not sample_lines:
        return ","

    best_delimiter: str = ","
    best_score: int = 0

    for delim in SUPPORTED_DELIMITERS:
        col_counts: list[int] = []
        for line in sample_lines:
            # Count fields using a simple CSV-aware split
            # Use Python's csv module for proper quoted field handling
            import csv
            import io

            reader = csv.reader(io.StringIO(line), delimiter=delim)
            try:
                row = next(reader, [])
            except csv.Error:
                row = []
            col_counts.append(len(row))

        if not col_counts:
            continue

        # Score: number of lines that produce >1 column, weighted by consistency
        multi_col_count = sum(1 for c in col_counts if c > 1)
        if multi_col_count == 0:
            continue

        # Prefer delimiters where most lines agree on the same column count
        from collections import Counter

        count_freq = Counter(col_counts)
        most_common_count, most_common_freq = count_freq.most_common(1)[0]

        # Score: prioritize consistency (most_common_freq) then column count
        score = most_common_freq * 100 + most_common_count

        if score > best_score:
            best_score = score
            best_delimiter = delim

    return best_delimiter


def _read_csv_with_detection(
    path: Path,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Read a CSV file with automatic delimiter detection.

    Args:
        path: Path to the CSV file.
        delimiter: Explicit delimiter (comma, semicolon, tab, pipe). If None,
            auto-detection is performed.
        encoding: Text encoding. If None, tries UTF-8 then falls back to
            Latin-1.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        DatasetError: If the file cannot be read or parsed.
    """
    # Determine delimiter
    if delimiter is None:
        detected_delimiter = _detect_delimiter(path)
    else:
        if delimiter not in SUPPORTED_DELIMITERS:
            raise DatasetError(
                f"Unsupported delimiter '{delimiter}'. "
                f"Supported: {', '.join(repr(d) for d in SUPPORTED_DELIMITERS)}."
            )
        detected_delimiter = delimiter

    # Determine encoding
    encodings = [encoding] if encoding else ["utf-8", "latin-1"]

    last_error: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                delimiter=detected_delimiter,
                encoding=enc,
                engine="python",
            )
            # Detect suspicious single-column result
            if df.shape[1] == 1 and len(df.columns) == 1:
                # Check if the single column contains the delimiter, suggesting
                # the wrong delimiter was auto-detected
                col_name = str(df.columns[0])
                sample_val = str(df.iloc[0, 0]) if len(df) > 0 else ""
                if detected_delimiter in sample_val or detected_delimiter in col_name:
                    raise DatasetError(
                        f"Failed to parse dataset '{path.name}': "
                        f"detected delimiter {detected_delimiter!r} produced "
                        f"a single column, which may indicate the wrong delimiter. "
                        f"Try specifying the delimiter explicitly."
                    )
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DatasetError(f"Dataset '{path.name}' is empty.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetError(f"Failed to parse dataset '{path.name}': {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if last_error is not None:
        raise DatasetError(f"Failed to read dataset '{path.name}': {last_error}") from last_error

    raise DatasetError(f"Failed to read dataset '{path.name}' with any encoding.")


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
    ".csv": lambda p: _read_csv_with_detection(p),
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


def load_dataset(
    config: DevRayConfig,
    name: str,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Load a dataset from the workspace, validating path and size.

    Supports CSV, Parquet, and Excel (.xlsx/.xls) formats. Parquet and Excel
    require optional dependencies ('pyarrow' / 'openpyxl').

    Args:
        config: Server configuration.
        name: Dataset filename.
        delimiter: Optional CSV delimiter (comma, semicolon, tab, pipe).
            If None, auto-detection is performed.
        encoding: Optional text encoding. If None, tries UTF-8 then Latin-1.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        DatasetError: If the file cannot be found, read, or parsed.
    """
    path = validate_dataset_path(config.workspace_dir, name, must_exist=True)
    validate_file_size(path, config.max_dataset_size_bytes)

    ext = path.suffix.lower()
    if ext == ".csv":
        return _read_csv_with_detection(path, delimiter=delimiter, encoding=encoding)

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
