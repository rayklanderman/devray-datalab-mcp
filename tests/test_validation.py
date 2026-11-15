"""Tests for validation and security helpers."""

from __future__ import annotations

import pytest

from devray_datalab.utils.errors import DatasetError, DevRayError, SecurityError, ValidationError
from devray_datalab.utils.validation import (
    ensure_within_workspace,
    validate_dataset_name,
    validate_dataset_path,
    validate_file_size,
    validate_probability,
    validate_task_type,
)


def test_validate_dataset_name_ok():
    assert validate_dataset_name("data.csv") == "data.csv"
    assert validate_dataset_name("my-data_1.csv") == "my-data_1.csv"


def test_validate_dataset_name_bad():
    for bad in ["", "..", "..\\..\\evil", "foo/bar.csv", "bad;rm -rf"]:
        with pytest.raises(ValidationError):
            validate_dataset_name(bad)


def test_ensure_within_workspace_blocks_traversal(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        ensure_within_workspace(workspace, workspace / ".." / "secret.txt")
        raise AssertionError("expected SecurityError")
    except SecurityError:
        pass


def test_validate_dataset_path_blocks_traversal(config):
    with pytest.raises((ValidationError, SecurityError, DatasetError)):
        validate_dataset_path(config.workspace_dir, "../../outside.csv")


def test_validate_file_size(tmp_path):
    f = tmp_path / "small.csv"
    f.write_text("a,b\n1,2\n")
    validate_file_size(f, 1024)
    big = tmp_path / "big.csv"
    big.write_text("x" * 1000)
    try:
        validate_file_size(big, 10)
        raise AssertionError("expected DevRayError")
    except DevRayError:
        pass


def test_validate_probability():
    assert validate_probability(0.5, "x") == 0.5
    assert validate_probability(0, "x") == 0.0
    assert validate_probability(1, "x") == 1.0
    for bad in (-0.1, 1.1, "nope"):
        with pytest.raises(ValidationError):
            validate_probability(bad, "x")


def test_validate_task_type():
    assert validate_task_type("classification") == "classification"
    assert validate_task_type("regression") == "regression"
    with pytest.raises(ValidationError):
        validate_task_type("clustering")


def test_validate_dataset_path_ok(config, classification_csv):
    resolved = validate_dataset_path(config.workspace_dir, "sample_classification.csv")
    assert resolved.is_file()
