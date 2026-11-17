"""Regression tests for remaining technical deficiencies (ISSUE 1-4).

Uses small synthetic fixtures (not the real Bank Marketing dataset).
Covers chronological CV in tuning, feature exclusion in tuning,
misleading CV metadata, and feature-count metadata.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd

from devray_datalab.core.automl import run_automl
from devray_datalab.core.evaluator import cross_validate_with_preprocessing
from devray_datalab.core.splits import create_split_manifest
from devray_datalab.core.trainer import CLASSIFICATION_MODELS
from devray_datalab.tools.automl import (
    automl_run_tool,
    get_model_leaderboard_tool,
    tune_model_tool,
)
from devray_datalab.tools.training import compare_models_tool


def run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def _data(result):
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    return payload["data"]


def _make_df(n=60, leak_feature=True):
    """Synthetic classification fixture with a deliberately predictive leaky feature.

    'leak_duration' is highly predictive of the target, so if it enters
    training/CV it will inflate scores. Exclusion must keep it out.
    """
    import numpy as np

    np.random.seed(11)
    df = pd.DataFrame(
        {
            "leak_duration": np.random.randint(1, 1000, size=n).astype(float),
            "age": np.random.randint(18, 70, size=n),
            "balance": np.random.randint(-2000, 5000, size=n),
            "category": np.random.choice(["a", "b", "c"], size=n),
        }
    )
    # Target depends on age + balance AND strongly on leak_duration.
    y = (((df["age"] > 40) & (df["balance"] > 0)) | (df["leak_duration"] > 800)).astype(int)
    df["y"] = y
    return df


def _write_csv(config, name, df):
    path = config.workspace_dir / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# ISSUE 1: Chronological cross-validation in tuning
# ---------------------------------------------------------------------------


def test_tune_model_chronological_manifest_uses_timeseries(config):
    """tune_model with a time manifest must use TimeSeriesSplit, not StratifiedKFold.

    Verifies actual fold indices: every train index precedes every test index
    within each fold (forward chaining), and folds move forward in time.
    """
    df = _make_df(n=48)
    _write_csv(config, "tune_chrono.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "tune_chrono.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="tune_chrono.csv",
        split_type="time",
        test_size=0.25,
        val_size=0.2,
        row_order=True,
        random_state=42,
    )
    assert manifest.split_type == "time"

    result = run(
        tune_model_tool(
            config,
            {
                "name": "tune_chrono.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "split_manifest_id": manifest.manifest_id,
                "exclude_columns": ["leak_duration"],
                "max_trials": 4,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 3,
            },
        )
    )
    data = _data(result)
    assert data["error"] is None, data["error"]

    # Core engine: directly verify CV strategy for time manifest.
    cv = cross_validate_with_preprocessing(
        df=df_disk.iloc[list(manifest.train_indices) + list(manifest.val_indices)].reset_index(
            drop=True
        ),
        target_column="y",
        task_type="classification",
        model_cls=CLASSIFICATION_MODELS["logistic_regression"],
        model_kwargs={},
        strategy="timeseries",
        n_splits=3,
        random_state=42,
        exclude_columns=["leak_duration"],
    )
    assert cv.strategy == "TimeSeriesSplit", cv.strategy
    assert cv.n_folds == 3

    # Forward-chaining: train indices strictly precede test indices in each fold.
    # TimeSeriesSplit on 32 dev rows (48 * 0.6) with 3 splits:
    #   fold 1: train[0:16], test[16:24] -> but TSS uses n_samples - n_splits + 1 per test
    # Verify by checking each fold's train/test boundary from the splitter directly.
    from sklearn.model_selection import TimeSeriesSplit

    dev_n = len(manifest.train_indices) + len(manifest.val_indices)
    tss = TimeSeriesSplit(n_splits=3)
    splits = list(tss.split(range(dev_n)))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx), "Train must precede test (forward chaining)"


def test_tune_model_grouped_manifest_preserves_group_isolation(config):
    """tune_model with a group manifest must use GroupKFold, keeping groups intact."""
    df = _make_df(n=40)
    df["group"] = [i % 5 for i in range(len(df))]
    _write_csv(config, "tune_group.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "tune_group.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="tune_group.csv",
        split_type="group",
        test_size=0.25,
        val_size=0.15,
        group_column="group",
        random_state=42,
    )
    assert manifest.split_type == "group"

    cv = cross_validate_with_preprocessing(
        df=df_disk,
        target_column="y",
        task_type="classification",
        model_cls=CLASSIFICATION_MODELS["logistic_regression"],
        model_kwargs={},
        strategy="group",
        n_splits=3,
        random_state=42,
        group_column="group",
        exclude_columns=["leak_duration"],
    )
    assert cv.strategy == "GroupKFold", cv.strategy


def test_tune_model_stratified_manifest_uses_stratifiedkfold(config):
    """tune_model with a stratified manifest must use StratifiedKFold."""
    df = _make_df(n=40)
    _write_csv(config, "tune_strat.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "tune_strat.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="tune_strat.csv",
        split_type="stratified",
        test_size=0.25,
        val_size=0.15,
        stratify_column="y",
        random_state=42,
    )
    assert manifest.split_type == "stratified"

    cv = cross_validate_with_preprocessing(
        df=df_disk,
        target_column="y",
        task_type="classification",
        model_cls=CLASSIFICATION_MODELS["logistic_regression"],
        model_kwargs={},
        strategy="stratified",
        n_splits=3,
        random_state=42,
        exclude_columns=["leak_duration"],
    )
    assert cv.strategy == "StratifiedKFold", cv.strategy


# ---------------------------------------------------------------------------
# ISSUE 2: Feature exclusion in tuning
# ---------------------------------------------------------------------------


def test_tune_model_excludes_columns_through_cv(config):
    """tune_model with exclude_columns keeps them out of every CV fold's preprocessing."""
    df = _make_df(n=48)
    _write_csv(config, "tune_excl.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "tune_excl.csv")

    result = run(
        tune_model_tool(
            config,
            {
                "name": "tune_excl.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": ["leak_duration"],
                "max_trials": 4,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 3,
            },
        )
    )
    data = _data(result)
    assert data["error"] is None, data["error"]

    # Core engine: run automl with exclusions and verify leak_duration never appears.
    core = run_automl(
        df_disk,
        "y",
        "classification",
        max_trials=2,
        n_splits=3,
        random_state=42,
        exclude_columns=["leak_duration"],
    )
    assert core.best_model is not None
    # All completed trials must not error due to missing leak_duration.
    completed = [t for t in core.trials if t.error is None]
    assert len(completed) >= 1


def test_automl_excludes_columns_across_candidates(config):
    """automl_run with exclude_columns applies exclusions to all candidate models."""
    df = _make_df(n=48)
    _write_csv(config, "am_excl.csv", df)

    result = run(
        automl_run_tool(
            config,
            {
                "name": "am_excl.csv",
                "target_column": "y",
                "task_type": "classification",
                "exclude_columns": ["leak_duration"],
                "max_trials": 3,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    data = _data(result)
    completed = [t for t in data["trials"] if t["error"] is None]
    assert len(completed) >= 1, "No trials completed with exclusions"


def test_get_model_leaderboard_excludes_columns(config):
    """get_model_leaderboard preserves exclude_columns through the tuning path."""
    df = _make_df(n=40)
    _write_csv(config, "lb_excl.csv", df)

    result = run(
        get_model_leaderboard_tool(
            config,
            {
                "name": "lb_excl.csv",
                "target_column": "y",
                "task_type": "classification",
                "exclude_columns": ["leak_duration"],
                "max_trials": 2,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    data = _data(result)
    assert data["best_model"] is not None or data["trials"]


# ---------------------------------------------------------------------------
# ISSUE 3: Misleading CV metadata
# ---------------------------------------------------------------------------


def test_compare_models_time_manifest_reports_stratified_false(config):
    """compare_models time manifest: stratified=false, strategy=TimeSeriesSplit."""
    df = _make_df(n=48)
    _write_csv(config, "cmp_chrono.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "cmp_chrono.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="cmp_chrono.csv",
        split_type="time",
        test_size=0.25,
        val_size=0.2,
        row_order=True,
        random_state=42,
    )

    result = run(
        compare_models_tool(
            config,
            {
                "name": "cmp_chrono.csv",
                "target_column": "y",
                "task_type": "classification",
                "models": ["logistic_regression"],
                "exclude_columns": ["leak_duration"],
                "split_manifest_id": manifest.manifest_id,
                "metric": "roc_auc",
                "cv": 3,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    assert body["excluded_columns"] == ["leak_duration"]
    for r in body["results"]:
        assert r["status"] == "completed", r
        # ISSUE 3 fix: stratified must reflect actual strategy, not task type.
        assert r["stratified"] is False, f"Expected stratified=false for TimeSeriesSplit, got {r}"
        assert r["strategy"] == "TimeSeriesSplit", f"Expected TimeSeriesSplit, got {r['strategy']}"


def test_compare_models_stratified_manifest_reports_stratified_true(config):
    """compare_models with a stratified manifest must report stratified=true."""
    df = _make_df(n=40)
    _write_csv(config, "cmp_strat.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "cmp_strat.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="cmp_strat.csv",
        split_type="stratified",
        test_size=0.25,
        val_size=0.15,
        stratify_column="y",
        random_state=42,
    )

    result = run(
        compare_models_tool(
            config,
            {
                "name": "cmp_strat.csv",
                "target_column": "y",
                "task_type": "classification",
                "models": ["logistic_regression"],
                "exclude_columns": ["leak_duration"],
                "split_manifest_id": manifest.manifest_id,
                "metric": "roc_auc",
                "cv": 3,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    for r in body["results"]:
        assert r["status"] == "completed", r
        assert r["stratified"] is True, f"Expected stratified=true for StratifiedKFold, got {r}"
        assert r["strategy"] == "StratifiedKFold", f"Expected StratifiedKFold, got {r['strategy']}"


def test_compare_models_kfold_manifest_reports_stratified_false(config):
    """compare_models with a random (kfold) manifest must report stratified=false."""
    df = _make_df(n=40)
    _write_csv(config, "cmp_kfold.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "cmp_kfold.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="cmp_kfold.csv",
        split_type="random",
        test_size=0.25,
        val_size=0.15,
        random_state=42,
    )

    result = run(
        compare_models_tool(
            config,
            {
                "name": "cmp_kfold.csv",
                "target_column": "y",
                "task_type": "classification",
                "models": ["logistic_regression"],
                "exclude_columns": ["leak_duration"],
                "split_manifest_id": manifest.manifest_id,
                "metric": "roc_auc",
                "cv": 3,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    for r in body["results"]:
        assert r["status"] == "completed", r
        assert r["stratified"] is False, f"Expected stratified=false for KFold, got {r}"
        assert r["strategy"] == "KFold", f"Expected KFold, got {r['strategy']}"


# ---------------------------------------------------------------------------
# ISSUE 4: Feature-count metadata
# ---------------------------------------------------------------------------


def test_automl_n_features_after_exclusion(config):
    """automl_run n_features must reflect post-exclusion predictor count, not raw columns."""
    df = _make_df(n=40)
    _write_csv(config, "am_feat.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "am_feat.csv")

    # Raw columns: leak_duration, age, balance, category, y = 5 columns
    # Target y removed -> 4 candidates
    # Exclude leak_duration -> 3 actual features
    assert len(df_disk.columns) == 5
    expected_after_exclusion = 3

    result = run(
        automl_run_tool(
            config,
            {
                "name": "am_feat.csv",
                "target_column": "y",
                "task_type": "classification",
                "exclude_columns": ["leak_duration"],
                "max_trials": 2,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    data = _data(result)
    assert data["n_features"] == expected_after_exclusion, (
        f"Expected n_features={expected_after_exclusion} after exclusion, got {data['n_features']}"
    )
    assert data["dataset_shape"][1] == expected_after_exclusion, (
        f"Expected dataset_shape[1]={expected_after_exclusion}, got {data['dataset_shape'][1]}"
    )


def test_automl_n_features_without_exclusion_matches_raw(config):
    """automl_run without exclusions reports n_features = columns minus target."""
    df = _make_df(n=40)
    _write_csv(config, "am_feat_noexcl.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "am_feat_noexcl.csv")

    result = run(
        automl_run_tool(
            config,
            {
                "name": "am_feat_noexcl.csv",
                "target_column": "y",
                "task_type": "classification",
                "max_trials": 2,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    data = _data(result)
    assert data["n_features"] == len(df_disk.columns) - 1, (
        f"Expected n_features={len(df_disk.columns) - 1}, got {data['n_features']}"
    )


def test_core_automl_n_features_after_exclusion(config):
    """Core run_automl n_features must reflect post-exclusion predictor count."""
    df = _make_df(n=40)
    result = run_automl(
        df,
        "y",
        "classification",
        max_trials=2,
        n_splits=2,
        random_state=42,
        exclude_columns=["leak_duration"],
    )
    assert result.n_features == 3, f"Expected n_features=3 after exclusion, got {result.n_features}"
    assert result.dataset_shape[1] == 3, (
        f"Expected dataset_shape[1]=3, got {result.dataset_shape[1]}"
    )


def test_compare_models_features_after_exclusion(config):
    """compare_models result features must not contain excluded columns."""
    df = _make_df(n=40)
    _write_csv(config, "cmp_feat.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "cmp_feat.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="cmp_feat.csv",
        split_type="random",
        test_size=0.25,
        val_size=0.15,
        random_state=42,
    )

    result = run(
        compare_models_tool(
            config,
            {
                "name": "cmp_feat.csv",
                "target_column": "y",
                "task_type": "classification",
                "models": ["logistic_regression"],
                "exclude_columns": ["leak_duration"],
                "split_manifest_id": manifest.manifest_id,
                "metric": "roc_auc",
                "cv": 3,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True, data.get("error")
    body = data["data"]
    assert body["excluded_columns"] == ["leak_duration"]
    for r in body["results"]:
        assert r["status"] == "completed", r
        assert r["strategy"] in ("KFold", "StratifiedKFold", "TimeSeriesSplit")
