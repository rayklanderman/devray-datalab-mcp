"""Server configuration.

Settings are read from environment variables with sensible defaults so the
server runs on modest hardware without any cloud credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass
class DevRayConfig:
    """Runtime configuration for the DevRay DataLab MCP server."""

    # Filesystem locations.
    workspace_dir: Path = field(
        default_factory=lambda: Path(_env_devray("DEVRAY_WORKSPACE_DIR", "datasets"))
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path(_env_devray("DEVRAY_ARTIFACTS_DIR", "artifacts"))
    )

    # Dataset limits.
    max_dataset_size_bytes: int = _env_int("DEVRAY_MAX_DATASET_SIZE_BYTES", 200 * 1024 * 1024)
    allowed_extensions: tuple[str, ...] = (".csv",)

    # Preview limits.
    max_preview_rows: int = _env_int("DEVRAY_MAX_PREVIEW_ROWS", 25)

    # Training / resource limits.
    max_cpu_threads: int | None = _env_int("DEVRAY_MAX_CPU_THREADS", 0) or None
    training_memory_budget_mb: int | None = _env_int("DEVRAY_TRAINING_MEMORY_BUDGET_MB", 0) or None
    max_concurrent_jobs: int = _env_int("DEVRAY_MAX_CONCURRENT_JOBS", 1)
    training_timeout_seconds: int = _env_int("DEVRAY_TRAINING_TIMEOUT_SECONDS", 1800)

    # Logging.
    log_level: str = _env_str("DEVRAY_LOG_LEVEL", "INFO")

    def __post_init__(self) -> None:
        self.workspace_dir = self.workspace_dir.resolve()
        self.artifacts_dir = self.artifacts_dir.resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _env_devray(name: str, default: str) -> str:
    return os.environ.get(name, default)
