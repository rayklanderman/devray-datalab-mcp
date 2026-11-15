"""Input validation and security helpers.

All filesystem access is confined to the configured workspace so that the
MCP server can never read or write outside of it, preventing path traversal
and unauthorized access.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from devray_datalab.utils.errors import DatasetError, SecurityError, ValidationError

# Allowed dataset file extensions.
ALLOWED_DATASET_EXTENSIONS: Final[tuple[str, ...]] = (".csv",)

# Maximum dataset file size in bytes (200 MB by default).
MAX_DATASET_SIZE_BYTES: Final[int] = 200 * 1024 * 1024

# Maximum number of preview rows returned by inspect_dataset.
MAX_PREVIEW_ROWS: Final[int] = 25

# Safe filename pattern: alphanumerics, dash, underscore, dot.
SAFE_FILENAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]+$")

# Dangerous substrings that must never appear in a resolved workspace-relative path.
FORBIDDEN_PATH_SEGMENTS: Final[tuple[str, ...]] = ("..",)


def ensure_within_workspace(workspace: Path, target: Path) -> Path:
    """Resolve ``target`` and guarantee it lives inside ``workspace``.

    Returns the resolved absolute path. Raises :class:`SecurityError` if the
    path escapes the workspace.
    """
    workspace_resolved = workspace.resolve()
    target_resolved = target.resolve()

    try:
        target_resolved.relative_to(workspace_resolved)
    except ValueError as exc:
        raise SecurityError(f"Path escapes the configured workspace: {target_resolved}") from exc

    # Reject any ``..`` segments in the relative path as an extra guard.
    relative = target_resolved.relative_to(workspace_resolved)
    for part in relative.parts:
        if part in FORBIDDEN_PATH_SEGMENTS:
            raise SecurityError(f"Forbidden path segment in target: {target_resolved}")

    return target_resolved


def validate_dataset_name(name: str) -> str:
    """Validate a dataset filename, returning it if safe."""
    if not name or not isinstance(name, str):
        raise ValidationError("Dataset name must be a non-empty string.")
    if not SAFE_FILENAME_RE.match(name):
        raise ValidationError(
            "Dataset name contains unsafe characters. Use letters, digits, "
            "dashes, underscores, and dots only."
        )
    if name in (".", ".."):
        raise ValidationError("Invalid dataset name.")
    return name


def validate_dataset_path(workspace: Path, name: str, must_exist: bool = True) -> Path:
    """Validate and resolve a dataset path inside the workspace."""
    validate_dataset_name(name)
    target = workspace / name
    resolved = ensure_within_workspace(workspace, target)

    ext = resolved.suffix.lower()
    if ext not in ALLOWED_DATASET_EXTENSIONS:
        raise DatasetError(
            f"Unsupported file extension '{ext}'. Allowed: {', '.join(ALLOWED_DATASET_EXTENSIONS)}"
        )

    if must_exist and not resolved.is_file():
        raise DatasetError(f"Dataset not found: {name}")

    return resolved


def validate_file_size(path: Path, max_size_bytes: int) -> None:
    """Ensure the file at ``path`` does not exceed ``max_size_bytes``."""
    size = path.stat().st_size
    if size > max_size_bytes:
        raise DatasetError(
            f"File too large: {size / (1024 * 1024):.1f} MB. "
            f"Limit is {max_size_bytes / (1024 * 1024):.1f} MB."
        )


def validate_positive_int(value: int, name: str, minimum: int = 1) -> int:
    """Validate an integer is at least ``minimum``."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer.")
    if value < minimum:
        raise ValidationError(f"{name} must be at least {minimum}.")
    return value


def validate_probability(value: float, name: str) -> float:
    """Validate a value is a probability in [0, 1]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be a number.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValidationError(f"{name} must be between 0 and 1.")
    return float(value)


def validate_task_type(task_type: str) -> str:
    """Validate a machine learning task type."""
    valid = ("classification", "regression")
    if task_type not in valid:
        raise ValidationError(
            f"Invalid task type '{task_type}'. Must be one of: {', '.join(valid)}."
        )
    return task_type


def validate_model_name(model_name: str) -> str:
    """Validate a supported baseline model name."""
    valid = (
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "linear_regression",
        "ridge_regression",
    )
    if model_name not in valid:
        raise ValidationError(f"Unsupported model '{model_name}'. Supported: {', '.join(valid)}.")
    return model_name


def sanitize_artifact_name(name: str) -> str:
    """Sanitize a model artifact directory name."""
    validate_dataset_name(name)
    return name
