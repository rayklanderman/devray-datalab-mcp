"""Tests for dataset format support (CSV, Parquet, Excel)."""

from __future__ import annotations

from devray_datalab.core.loader import list_datasets, load_dataset
from devray_datalab.utils.errors import DatasetError, DevRayError


def test_list_datasets_includes_parquet_and_excel(config):
    # Create parquet and excel placeholders to confirm list_datasets lists them
    # without validating content.
    (config.workspace_dir / "data.parquet").write_bytes(b"")
    (config.workspace_dir / "data.xlsx").write_bytes(b"")

    datasets = list_datasets(config.workspace_dir)
    names = {d["filename"] for d in datasets}
    assert "data.parquet" in names
    assert "data.xlsx" in names
    assert all(
        d["format"] in ("parquet", "xlsx")
        for d in datasets
        if d["filename"] in ("data.parquet", "data.xlsx")
    )


def test_load_dataset_unsupported_extension(config):
    (config.workspace_dir / "data.parquet").write_bytes(b"")
    try:
        load_dataset(config, "data.parquet")
        raise AssertionError("expected DatasetError for missing parquet engine")
    except DatasetError as exc:
        assert "pyarrow" in exc.message


def test_load_dataset_unsupported_extension_xlsx(config):
    (config.workspace_dir / "data.xlsx").write_bytes(b"")
    try:
        load_dataset(config, "data.xlsx")
        raise AssertionError("expected DatasetError for missing excel engine")
    except DatasetError as exc:
        assert "openpyxl" in exc.message


def test_load_dataset_path_traversal_still_blocked(config):
    try:
        load_dataset(config, "../../outside.csv")
        raise AssertionError("expected DevRayError for path traversal")
    except DevRayError:
        pass
