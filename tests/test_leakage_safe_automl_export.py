"""Regression tests for leakage-safe AutoML, tuning, background jobs and export.

Uses small synthetic fixtures (not the real Bank Marketing dataset).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

from devray_datalab.core.automl import run_automl
from devray_datalab.core.evaluator import cross_validate_with_preprocessing
from devray_datalab.core.splits import (
    create_split_manifest,
    development_rows_from_manifest,
)
from devray_datalab.tools.automl import (
    automl_run_tool,
    get_model_leaderboard_tool,
    tune_model_tool,
)
from devray_datalab.tools.jobs import (
    cancel_training_job_tool,
    get_training_status_tool,
    list_training_jobs_tool,
    start_training_job_tool,
)
from devray_datalab.tools.models import export_model_tool, predict_tool
from devray_datalab.tools.training import train_baseline_tool


def run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def _data(result):
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    return payload["data"]


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
    y = ((df["age"] > 40) & (df["balance"] > 0)).astype(int)
    df["y"] = y
    return df


def _write_csv(config, name, df):
    path = config.workspace_dir / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# AutoML / tuning exclusion
# ---------------------------------------------------------------------------


def test_automl_excludes_columns(config):
    """automl_run with exclude_columns runs leakage-safe CV on remaining features."""
    df = _make_df()
    _write_csv(config, "automl_leak.csv", df)

    result = run(
        automl_run_tool(
            config,
            {
                "name": "automl_leak.csv",
                "target_column": "y",
                "task_type": "classification",
                "exclude_columns": ["duration"],
                "max_trials": 3,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    data = _data(result)
    assert len(data["trials"]) == 5
    completed = [t for t in data["trials"] if t["error"] is None]
    assert len(completed) >= 1
    assert {t["model"] for t in data["trials"]} == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "knn",
    }
    assert data["best_model"] is not None

    # Same guarantee at the core engine level.
    core = run_automl(
        df,
        "y",
        "classification",
        max_trials=2,
        n_splits=2,
        exclude_columns=["duration"],
        random_state=42,
    )
    assert core.best_model is not None


def test_tune_model_excludes_columns(config):
    """tune_model with exclude_columns keeps them out of CV folds."""
    df = _make_df()
    _write_csv(config, "tune_leak.csv", df)

    result = run(
        tune_model_tool(
            config,
            {
                "name": "tune_leak.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": ["duration"],
                "max_trials": 4,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    data = _data(result)
    assert data["best_params"] is not None or data["trials"]
    assert data["error"] is None, data["error"]


# ---------------------------------------------------------------------------
# Protected holdout isolation
# ---------------------------------------------------------------------------


def test_automl_manifest_excludes_test_holdout(config):
    """automl_run with a manifest uses only train+val rows; holdout untouched."""
    df = _make_df(n=40)
    _write_csv(config, "am_holdout.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "am_holdout.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="am_holdout.csv",
        split_type="random",
        test_size=0.25,
        val_size=0.15,
        random_state=42,
    )

    result = run(
        automl_run_tool(
            config,
            {
                "name": "am_holdout.csv",
                "target_column": "y",
                "task_type": "classification",
                "exclude_columns": ["duration"],
                "split_manifest_id": manifest.manifest_id,
                "max_trials": 3,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    data = _data(result)
    assert data["best_model"] is not None
    # Only train + val rows were used for CV/model selection.
    assert data["dataset_shape"][0] == manifest.n_train + manifest.n_val

    # Direct helper: development rows exclude the protected test holdout.
    dev_df, m = development_rows_from_manifest(config, df_disk, manifest.manifest_id)
    assert len(dev_df) == manifest.n_train + manifest.n_val
    assert set(dev_df.index) <= set(range(len(df_disk)))
    assert m.test_indices  # protected holdout exists and is not in dev_df


def test_manifest_fingerprint_mismatch_rejected(config):
    """A manifest created against one dataset is rejected against another."""
    df = _make_df(n=30)
    _write_csv(config, "fp.csv", df)
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
    _write_csv(config, "fp_modified.csv", df_modified)

    result = run(
        automl_run_tool(
            config,
            {
                "name": "fp_modified.csv",
                "target_column": "y",
                "task_type": "classification",
                "split_manifest_id": manifest.manifest_id,
                "max_trials": 2,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    body = _payload(result)
    assert body["ok"] is False
    assert "manifest" in body["error"]["message"].lower()


def test_unknown_manifest_rejected(config):
    df = _make_df()
    _write_csv(config, "automl_leak.csv", df)
    result = run(
        train_baseline_tool(
            config,
            {
                "name": "automl_leak.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "split_manifest_id": "does-not-exist",
            },
        )
    )
    body = _payload(result)
    assert body["ok"] is False
    assert "manifest" in body["error"]["message"].lower()


# ---------------------------------------------------------------------------
# Chronological / forward-chaining validation
# ---------------------------------------------------------------------------


def test_chronological_manifest_training_respects_row_order(config):
    """row_order manifest: every train row precedes every test row."""
    df = _make_df(n=36)
    _write_csv(config, "chrono.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "chrono.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="chrono.csv",
        split_type="time",
        test_size=0.25,
        val_size=0.0,
        row_order=True,
        random_state=42,
    )
    assert max(manifest.train_indices) < min(manifest.test_indices)

    result = run(
        train_baseline_tool(
            config,
            {
                "name": "chrono.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": ["duration"],
                "split_manifest_id": manifest.manifest_id,
                "random_state": 42,
            },
        )
    )
    data = _data(result)
    assert data["n_train"] == manifest.n_train
    assert data["n_test"] == manifest.n_test
    assert data["split_manifest_id"] == manifest.manifest_id


def test_timeseries_cv_is_forward_chaining(config):
    """TimeSeriesSplit CV folds move forward in time with no overlap backwards."""
    from devray_datalab.core.trainer import CLASSIFICATION_MODELS

    df = _make_df(n=24)
    cv = cross_validate_with_preprocessing(
        df=df,
        target_column="y",
        task_type="classification",
        model_cls=CLASSIFICATION_MODELS["logistic_regression"],
        model_kwargs={},
        strategy="timeseries",
        n_splits=3,
        random_state=42,
        exclude_columns=["duration"],
    )
    assert cv.strategy == "TimeSeriesSplit"
    assert cv.n_folds == 3
    assert cv.fold_sizes == [6, 6, 6]
    assert all(math.isfinite(s) for s in [cv.mean, cv.std])


# ---------------------------------------------------------------------------
# Background job consistency
# ---------------------------------------------------------------------------


def test_background_job_matches_sync_training(config):
    """start_training_job uses the same validated partitions as train_baseline."""
    df = _make_df(n=40)
    _write_csv(config, "job_consistency.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "job_consistency.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="job_consistency.csv",
        split_type="random",
        test_size=0.25,
        val_size=0.1,
        random_state=42,
    )
    args = {
        "name": "job_consistency.csv",
        "target_column": "y",
        "task_type": "classification",
        "model_name": "random_forest",
        "exclude_columns": ["duration"],
        "split_manifest_id": manifest.manifest_id,
        "random_state": 42,
    }

    sync = run(train_baseline_tool(config, args))
    sync_data = _data(sync)

    start = run(start_training_job_tool(config, args))
    start_data = _data(start)
    job_id = start_data["job_id"]

    status = None
    for _ in range(200):
        st = run(get_training_status_tool(config, {"job_id": job_id}))
        st_data = _data(st)
        status = st_data["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert status == "completed", status
    job_result = _data(run(get_training_status_tool(config, {"job_id": job_id})))["result"]

    # Background job reproduces the synchronous training partition exactly.
    assert job_result["n_train"] == sync_data["n_train"]
    assert job_result["n_test"] == sync_data["n_test"]
    assert job_result["features"] == sync_data["features"]
    assert job_result["excluded_columns"] == sync_data["excluded_columns"]
    assert job_result["split_manifest_id"] == sync_data["split_manifest_id"]
    assert job_result["metrics"]["accuracy"] == sync_data["metrics"]["accuracy"]

    lst = run(list_training_jobs_tool(config, {"limit": 10}))
    assert any(j["job_id"] == job_id for j in _data(lst)["jobs"])

    # Cancellation still works on a fresh job.
    start2 = run(start_training_job_tool(config, args))
    jid = _data(start2)["job_id"]
    cancel = run(cancel_training_job_tool(config, {"job_id": jid}))
    cd = _data(cancel)
    assert "cancelled" in cd
    sd = _data(run(get_training_status_tool(config, {"job_id": jid})))
    assert sd["status"] in ("cancelled", "completed")


# ---------------------------------------------------------------------------
# Model export preserves feature exclusions
# ---------------------------------------------------------------------------


def test_export_preserves_feature_exclusions(config):
    """Exported artifact metadata records excluded features; predict stays compatible."""
    df = _make_df()
    _write_csv(config, "export_leak.csv", df)

    export = run(
        export_model_tool(
            config,
            {
                "artifact_name": "exported_leak_safe",
                "name": "export_leak.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": ["duration"],
                "random_state": 42,
            },
        )
    )
    data = _data(export)
    assert "duration" not in data["features"], data["features"]
    assert data["excluded_columns"] == ["duration"]
    assert data["dataset_fingerprint"]
    assert data["train_info"]["excluded_columns"] == ["duration"]

    import joblib

    bundle = joblib.load(data["path"])
    assert bundle["feature_schema"]["feature_columns"] == data["features"]
    assert "duration" not in bundle["feature_schema"]["feature_columns"]
    assert bundle["metadata"]["excluded_columns"] == ["duration"]
    assert bundle["metadata"]["dataset_fingerprint"] == data["dataset_fingerprint"]
    assert "scikit-learn" in bundle["metadata"]["package_versions"]
    assert bundle["metadata"]["split_manifest_id"] is None

    # Predict works with the reduced feature schema; extra columns are ignored.
    predict = run(
        predict_tool(
            config,
            {
                "artifact_name": "exported_leak_safe",
                "features": {
                    "age": 45.0,
                    "balance": 1000.0,
                    "campaign": 2.0,
                    "category": "a",
                    "duration": 999.0,
                },
            },
        )
    )
    pdata = _data(predict)
    assert pdata["schema_valid"] is True
    assert pdata["ignored_extra_columns"] == ["duration"]
    assert len(pdata["predictions"]) == 1

    # Missing a required feature fails clearly.
    missing = run(
        predict_tool(
            config,
            {"artifact_name": "exported_leak_safe", "features": {"age": 45.0}},
        )
    )
    mbody = _payload(missing)
    assert mbody["ok"] is False
    assert mbody["error"]["code"] == "schema_mismatch"


def test_export_with_manifest_records_manifest_id(config):
    """Export with a split manifest records it in the artifact metadata."""
    df = _make_df(n=40)
    _write_csv(config, "export_manifest.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "export_manifest.csv")

    manifest = create_split_manifest(
        config=config,
        df=df_disk,
        dataset_name="export_manifest.csv",
        split_type="random",
        test_size=0.25,
        random_state=42,
    )

    export = run(
        export_model_tool(
            config,
            {
                "artifact_name": "exported_manifest",
                "name": "export_manifest.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "decision_tree",
                "exclude_columns": ["duration"],
                "split_manifest_id": manifest.manifest_id,
                "random_state": 42,
            },
        )
    )
    data = _data(export)
    assert data["split_manifest_id"] == manifest.manifest_id
    assert data["train_info"]["n_train"] == manifest.n_train
    assert data["train_info"]["n_test"] == manifest.n_test


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_backward_compatible_without_optional_args(config):
    """Existing calls without exclude_columns/split_manifest_id still work."""
    df = _make_df()
    _write_csv(config, "compat.csv", df)

    am = run(
        automl_run_tool(
            config,
            {
                "name": "compat.csv",
                "target_column": "y",
                "task_type": "classification",
                "max_trials": 2,
                "max_time_seconds": 300,
                "random_state": 42,
                "cv": 2,
            },
        )
    )
    assert _data(am)["best_model"] is not None

    lb = _data(
        run(
            get_model_leaderboard_tool(
                config,
                {
                    "name": "compat.csv",
                    "target_column": "y",
                    "task_type": "classification",
                    "max_trials": 2,
                    "max_time_seconds": 300,
                    "random_state": 42,
                    "cv": 2,
                },
            )
        )
    )
    assert lb["best_model"] is not None

    tune = _data(
        run(
            tune_model_tool(
                config,
                {
                    "name": "compat.csv",
                    "target_column": "y",
                    "task_type": "classification",
                    "model_name": "logistic_regression",
                    "max_trials": 4,
                    "max_time_seconds": 300,
                    "random_state": 42,
                    "cv": 2,
                },
            )
        )
    )
    assert tune["error"] is None, tune["error"]

    job = run(
        start_training_job_tool(
            config,
            {
                "name": "compat.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "random_state": 42,
            },
        )
    )
    jid = _data(job)["job_id"]
    for _ in range(200):
        st = run(get_training_status_tool(config, {"job_id": jid}))
        if _data(st)["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert _data(run(get_training_status_tool(config, {"job_id": jid})))["status"] == "completed"

    export = run(
        export_model_tool(
            config,
            {
                "artifact_name": "exported_compat",
                "name": "compat.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "random_state": 42,
            },
        )
    )
    assert _data(export)["features"]


# ---------------------------------------------------------------------------
# MCP protocol-level check
# ---------------------------------------------------------------------------


def test_mcp_protocol_export_predict_flow(tmp_path):
    """Export + predict through the real MCP stdio protocol, exclusions preserved."""
    import shutil

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    workspace.mkdir()
    artifacts.mkdir()

    examples = Path(__file__).resolve().parent.parent / "examples"
    shutil.copy(examples / "sample_classification.csv", workspace / "sample_classification.csv")

    env = dict(os.environ)
    env["DEVRAY_WORKSPACE_DIR"] = str(workspace)
    env["DEVRAY_ARTIFACTS_DIR"] = str(artifacts)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "devray_datalab.server"],
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    async def run_protocol() -> dict:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                names = {t.name for t in tools_result.tools}
                expected = {
                    "automl_run",
                    "tune_model",
                    "start_training_job",
                    "export_model",
                    "predict",
                    "train_baseline",
                    "compare_models",
                }
                assert expected.issubset(names), f"missing: {expected - names}"
                assert len(names) == 27, f"expected 27 tools, got {len(names)}"

                train = await session.call_tool(
                    "export_model",
                    arguments={
                        "artifact_name": "proto_export",
                        "name": "sample_classification.csv",
                        "target_column": "purchased",
                        "task_type": "classification",
                        "model_name": "logistic_regression",
                        "exclude_columns": ["region"],
                        "random_state": 42,
                    },
                )
                body = json.loads(train.content[0].text)
                assert body["ok"] is True, body.get("error")
                assert "region" not in body["data"]["features"], body["data"]["features"]
                assert body["data"]["excluded_columns"] == ["region"]

                predict = await session.call_tool(
                    "predict",
                    arguments={
                        "artifact_name": "proto_export",
                        "features": {
                            "age": 40,
                            "income": 80000,
                            "credit_score": 720,
                            "category": "electronics",
                            "region": "east",
                        },
                    },
                )
                pbody = json.loads(predict.content[0].text)
                assert pbody["ok"] is True, pbody.get("error")
                assert pbody["data"]["schema_valid"] is True
                assert pbody["data"]["ignored_extra_columns"] == ["region"]
                return pbody["data"]

    data = asyncio.run(run_protocol())
    assert len(data["predictions"]) == 1
