"""Local artifact storage using joblib.

All artifacts are stored inside the configured artifacts directory. Path
traversal is prevented so artifacts can never be written outside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from devray_datalab.config import DevRayConfig
from devray_datalab.utils.errors import ModelError
from devray_datalab.utils.validation import ensure_within_workspace, sanitize_artifact_name

ARTIFAKIND_PREPROCESSING = "preprocessing"
ARTIFAKIND_MODEL = "model"


def _sanitize_handle(handle: str) -> str:
    return sanitize_artifact_name(handle)


def artifact_path_for(config: DevRayConfig, handle: str, kind: str) -> Path:
    """Return the storage path for an artifact, confined to the artifacts dir."""
    safe_handle = _sanitize_handle(handle)
    kind_dir = config.artifacts_dir / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    path = kind_dir / f"{safe_handle}.joblib"
    return ensure_within_workspace(config.artifacts_dir, path)


def save_artifact(config: DevRayConfig, handle: str, payload: Any, kind: str = "model") -> Path:
    """Persist a payload to disk with joblib and return its path."""
    path = artifact_path_for(config, handle, kind)
    try:
        joblib.dump(payload, path)
    except Exception as exc:  # noqa: BLE001
        raise ModelError(f"Failed to save artifact '{handle}': {exc}") from exc
    return path


def load_artifact(config: DevRayConfig, handle: str, kind: str = "model") -> Any:
    """Load a previously saved artifact. Only artifacts from this app are trusted."""
    path = artifact_path_for(config, handle, kind)
    if not path.is_file():
        raise ModelError(f"Artifact not found: {handle}")
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ModelError(f"Failed to load artifact '{handle}': {exc}") from exc
