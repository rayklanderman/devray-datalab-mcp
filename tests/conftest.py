"""Pytest configuration and fixtures for DevRay DataLab MCP tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from devray_datalab.config import DevRayConfig


@pytest.fixture
def config(tmp_path: Path) -> DevRayConfig:
    """A DevRayConfig using a temporary workspace and artifacts dir."""
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    experiments_db = tmp_path / "experiments.db"
    workspace.mkdir()
    artifacts.mkdir()
    experiments_db.parent.mkdir(parents=True, exist_ok=True)
    # Use monkeypatch-free env via direct env var override for the duration.
    import os

    old_ws = os.environ.get("DEVRAY_WORKSPACE_DIR")
    old_ar = os.environ.get("DEVRAY_ARTIFACTS_DIR")
    old_edb = os.environ.get("DEVRAY_EXPERIMENTS_DB_PATH")
    os.environ["DEVRAY_WORKSPACE_DIR"] = str(workspace)
    os.environ["DEVRAY_ARTIFACTS_DIR"] = str(artifacts)
    os.environ["DEVRAY_EXPERIMENTS_DB_PATH"] = str(experiments_db)
    cfg = DevRayConfig()
    yield cfg
    if old_ws is None:
        os.environ.pop("DEVRAY_WORKSPACE_DIR", None)
    else:
        os.environ["DEVRAY_WORKSPACE_DIR"] = old_ws
    if old_ar is None:
        os.environ.pop("DEVRAY_ARTIFACTS_DIR", None)
    else:
        os.environ["DEVRAY_ARTIFACTS_DIR"] = old_ar
    if old_edb is None:
        os.environ.pop("DEVRAY_EXPERIMENTS_DB_PATH", None)
    else:
        os.environ["DEVRAY_EXPERIMENTS_DB_PATH"] = old_edb


@pytest.fixture
def example_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def classification_csv(config: DevRayConfig, example_dir: Path) -> Path:
    dest = config.workspace_dir / "sample_classification.csv"
    shutil.copy(example_dir / "sample_classification.csv", dest)
    return dest


@pytest.fixture
def regression_csv(config: DevRayConfig, example_dir: Path) -> Path:
    dest = config.workspace_dir / "sample_regression.csv"
    shutil.copy(example_dir / "sample_regression.csv", dest)
    return dest
