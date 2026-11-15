"""Tests for dataset loading and inspection."""

from __future__ import annotations

from devray_datalab.core.loader import list_datasets, load_dataset, preview_dataframe
from devray_datalab.utils.errors import DatasetError, DevRayError


def test_list_datasets_empty(config):
    assert list_datasets(config.workspace_dir) == []


def test_list_datasets(config, classification_csv, regression_csv):
    datasets = list_datasets(config.workspace_dir)
    names = {d["filename"] for d in datasets}
    assert names == {"sample_classification.csv", "sample_regression.csv"}
    for d in datasets:
        assert d["format"] == "csv"
        assert d["size_bytes"] > 0
        assert "size_mb" in d


def test_load_dataset(config, classification_csv):
    df = load_dataset(config, "sample_classification.csv")
    assert not df.empty
    assert "purchased" in df.columns
    assert len(df) == 20


def test_load_dataset_missing(config):
    try:
        load_dataset(config, "does_not_exist.csv")
        raise AssertionError("expected DatasetError")
    except DatasetError:
        pass


def test_load_dataset_path_traversal(config):
    try:
        load_dataset(config, "../evil.csv")
        raise AssertionError("expected DevRayError")
    except DevRayError:
        pass


def test_preview_dataframe(config, classification_csv):

    df = load_dataset(config, "sample_classification.csv")
    preview = preview_dataframe(df, max_rows=3)
    assert len(preview) == 3
    assert isinstance(preview[0], dict)
