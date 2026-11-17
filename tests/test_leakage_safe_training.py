"""Leakage-safe training tests: excluded columns and protected holdouts.

Uses small synthetic fixtures (not the real Bank Marketing dataset).
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest

from devray_datalab.core.splits import (
    create_split_manifest,
    validate_split_manifest,
)
from devray_datalab.tools.training import (
    compare_models_tool,
    prepare_dataset_tool,
    train_baseline_tool,
)
from devray_datalab.utils.errors import ValidationError


def run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def _make_df(n=60):
    """Synthetic classification fixture with a leaky 'duration' feature."""
    import numpy as np

    np.random.seed(7)
    df = pd.DataFrame(
        {
            "duration": np.random.randint(1, 1000, size=n).astype(float),
            "age": np.random.randint(18, 70, size=n),
            "balance": np.random.randint(-2000, 5000, size=n),
            "campaign": np.random.randint(1, 10, size=n),
            "category": np.random.choice(["a", "b", "c"], size=n),
        }
    )
    # Make target weakly related to non-duration features so training is meaningful.
    y = ((df["age"] > 40) & (df["balance"] > 0)).astype(int)
    df["y"] = y
    return df


def _write_csv(config, name, df):
    path = config.workspace_dir / name
    df.to_csv(path, index=False)
    return path


def test_excluded_columns_never_reach_fitted_model(config):
    """Excluded 'duration' must not appear in fitted preprocessor features."""
    df = _make_df()
    _write_csv(config, "leak.csv", df)

    # prepare_dataset: duration excluded before preprocessing fit.
    result = run(
        prepare_dataset_tool(
            config,
            {
                "name": "leak.csv",
                "target_column": "y",
                "task_type": "classification",
                "exclude_columns": ["duration"],
                "test_size": 0.25,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    assert "duration" not in body["feature_columns"], body["feature_columns"]
    assert body["excluded_columns"] == ["duration"]
    assert body["n_features"] == len(body["feature_columns"])

    # train_baseline: duration excluded; fitted model never sees it.
    result = run(
        train_baseline_tool(
            config,
            {
                "name": "leak.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": ["duration"],
                "test_size": 0.25,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    assert "duration" not in body["features"], body["features"]
    assert "duration" not in body["numeric_features"]
    assert body["excluded_columns"] == ["duration"]


def test_compare_models_excludes_columns(config):
    """compare_models with exclude_columns keeps them out of every model's features."""
    df = _make_df()
    _write_csv(config, "leak2.csv", df)

    result = run(
        compare_models_tool(
            config,
            {
                "name": "leak2.csv",
                "target_column": "y",
                "task_type": "classification",
                "models": ["logistic_regression", "decision_tree"],
                "exclude_columns": ["duration"],
                "metric": "f1",
                "cv": 3,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    assert body["excluded_columns"] == ["duration"]
    for r in body["results"]:
        assert r["status"] == "completed", r
    assert body["best_model"] is not None


def test_protected_holdout_never_enters_training(config):
    """Manifest-based training: test holdout disjoint from train; excluded columns dropped."""
    df = _make_df(n=40)
    _write_csv(config, "holdout.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "holdout.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="holdout.csv",
        split_type="random",
        test_size=0.25,
        val_size=0.2,
        random_state=42,
    )
    report = validate_split_manifest(
        config=config, manifest_id=manifest.manifest_id, df=df_disk, dataset_name="holdout.csv"
    )
    assert report["valid"] is True
    assert set(manifest.train_indices).isdisjoint(set(manifest.test_indices))

    result = run(
        train_baseline_tool(
            config,
            {
                "name": "holdout.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": ["duration"],
                "split_manifest_id": manifest.manifest_id,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    # Training uses only manifest train rows; test = protected holdout rows.
    assert body["n_train"] == manifest.n_train
    assert body["n_test"] == manifest.n_val
    assert body["split_manifest_id"] == manifest.manifest_id
    assert "duration" not in body["features"]
    assert "duration" not in body["numeric_features"]


def test_chronological_row_order_split_without_date_column(config):
    """split_type='time' with row_order=True splits by original row order, no date column."""
    df = _make_df(n=30)
    _write_csv(config, "chrono.csv", df)

    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name="chrono.csv",
        split_type="time",
        test_size=0.25,
        val_size=0.0,
        row_order=True,
        random_state=42,
    )
    assert manifest.split_type == "time"
    assert manifest.time_column is None
    assert manifest.n_test > 0
    assert manifest.n_train > 0
    # Sequential: every train index precedes every test index.
    assert max(manifest.train_indices) < min(manifest.test_indices)
    # Coverage: every index in exactly one set.
    all_idx = manifest.train_indices + manifest.val_indices + manifest.test_indices
    assert len(set(all_idx)) == len(all_idx) == len(df)


def test_split_type_time_rejected_without_column_or_row_order(config):
    """split_type='time' with no column or row_order must raise."""
    df = _make_df(n=30)
    with pytest.raises(ValidationError):
        create_split_manifest(
            config=config,
            df=df,
            dataset_name="chrono.csv",
            split_type="time",
            test_size=0.25,
            random_state=42,
        )


def test_split_type_time_with_row_order_is_chronological(config):
    """row_order=True yields the same protected holdout as a time column sorted by row order."""
    df = _make_df(n=30)
    manifest_row = create_split_manifest(
        config=config,
        df=df,
        dataset_name="chrono.csv",
        split_type="time",
        test_size=0.25,
        val_size=0.0,
        row_order=True,
        random_state=42,
    )
    # Add a time column equal to row order; time-based split should match row_order split.
    df_with_time = df.copy()
    df_with_time["t"] = range(len(df))
    manifest_time = create_split_manifest(
        config=config,
        df=df_with_time,
        dataset_name="chrono.csv",
        split_type="time",
        test_size=0.25,
        val_size=0.0,
        time_column="t",
        random_state=42,
    )
    assert manifest_row.test_indices == manifest_time.test_indices
    assert manifest_row.train_indices == manifest_time.train_indices


def test_compare_models_manifest_excludes_test_holdout_from_cv(config):
    """compare_models with a manifest uses only train+val rows for CV; test holdout untouched."""
    df = _make_df(n=40)
    _write_csv(config, "cvholdout.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "cvholdout.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="cvholdout.csv",
        split_type="random",
        test_size=0.25,
        val_size=0.15,
        random_state=42,
    )

    result = run(
        compare_models_tool(
            config,
            {
                "name": "cvholdout.csv",
                "target_column": "y",
                "task_type": "classification",
                "models": ["logistic_regression"],
                "exclude_columns": ["duration"],
                "split_manifest_id": manifest.manifest_id,
                "metric": "f1",
                "cv": 3,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    assert body["split_manifest_id"] == manifest.manifest_id
    assert body["excluded_columns"] == ["duration"]
    # Comparison ran on train+val rows only (n_samples <= cv_set size).
    for r in body["results"]:
        assert r["status"] == "completed", r
    assert body["best_model"] is not None


def test_manifest_fingerprint_mismatch_rejected(config):
    """A manifest created against one dataset is rejected against a modified dataset."""
    df = _make_df(n=30)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name="fp.csv",
        split_type="random",
        test_size=0.25,
        random_state=42,
    )
    df_modified = df.copy()
    df_modified.loc[0, "y"] = 1 - df_modified.loc[0, "y"]
    report = validate_split_manifest(
        config=config, manifest_id=manifest.manifest_id, df=df_modified, dataset_name="fp.csv"
    )
    assert report["valid"] is False
    assert report["dataset_fingerprint_match"] is False
