"""Regression tests for the consolidated reliability repair (QA-001 through QA-009).

These tests reproduce each confirmed defect from the Codex Desktop audit and
verify that the correction resolves it. Fixtures are bounded and CPU-only.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid

import pandas as pd

from devray_datalab.config import DevRayConfig
from devray_datalab.core.splits import (
    SplitManifest,
    SplitStore,
    _dataset_fingerprint,
    validate_manifest_for_consumption,
    validate_split_manifest,
)
from devray_datalab.tools.analysis import analyze_feature_drift_tool
from devray_datalab.tools.automl import (
    automl_run_tool,
    get_model_leaderboard_tool,
    tune_model_tool,
)
from devray_datalab.tools.datasets import inspect_dataset_tool
from devray_datalab.tools.jobs import (
    get_training_status_tool,
    start_training_job_tool,
)
from devray_datalab.tools.models import export_model_tool
from devray_datalab.tools.splits import (
    create_split_manifest_tool,
)
from devray_datalab.tools.training import (
    compare_models_tool,
    train_baseline_tool,
)


def _run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def _data(result):
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    return payload["data"]


def _write_csv(config: DevRayConfig, filename: str, df: pd.DataFrame) -> str:
    path = config.workspace_dir / filename
    df.to_csv(path, index=False)
    return filename


def _make_manifest(
    config: DevRayConfig,
    df: pd.DataFrame,
    dataset_name: str,
    split_type: str = "time",
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    random_state: int = 42,
    group_column: str | None = None,
    time_column: str | None = None,
) -> str:
    n = len(df)
    if train_indices is None:
        train_indices = list(range(0, int(n * 0.6)))
    if val_indices is None:
        val_indices = list(range(int(n * 0.6), int(n * 0.8)))
    if test_indices is None:
        test_indices = list(range(int(n * 0.8), n))

    import time

    manifest = SplitManifest(
        manifest_id=str(uuid.uuid4().hex),
        dataset_name=dataset_name,
        dataset_fingerprint=_dataset_fingerprint(df),
        created_at=time.time(),
        split_type=split_type,
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
        n_train=len(train_indices),
        n_val=len(val_indices),
        n_test=len(test_indices),
        stratify_column=None,
        group_column=group_column,
        time_column=time_column,
        random_state=random_state,
        protected_holdout=True,
    )
    SplitStore(config).save(manifest)
    return manifest.manifest_id


def _call(tool, config, arguments):
    return _run(tool(config, arguments))


# ---------------------------------------------------------------------------
# QA-001: Split-manifest validation bypass
# ---------------------------------------------------------------------------


def test_qa001_validate_overlap_rejected(config, classification_csv):
    df = pd.read_csv(classification_csv)
    name = "sample_classification.csv"
    mid = _make_manifest(
        config,
        df,
        name,
        split_type="time",
        train_indices=list(range(0, 12)),
        val_indices=list(range(12, 16)),
        test_indices=list(range(10, 20)),
    )
    report = validate_split_manifest(config=config, manifest_id=mid, df=df, dataset_name=name)
    assert report["valid"] is False
    assert any("overlap" in e.lower() for e in report["errors"])

    base_args = {
        "name": name,
        "target_column": "purchased",
        "task_type": "classification",
        "model_name": "logistic_regression",
        "split_manifest_id": mid,
        "exclude_columns": [],
    }
    cases = [
        (train_baseline_tool, dict(base_args), "train_baseline"),
        (
            compare_models_tool,
            {**base_args, "models": ["logistic_regression"], "cv": 2},
            "compare_models",
        ),
        (
            automl_run_tool,
            {
                **base_args,
                "models": ["logistic_regression"],
                "cv": 2,
                "max_trials": 1,
                "max_time_seconds": 10,
            },
            "automl_run",
        ),
        (
            tune_model_tool,
            {**base_args, "cv": 2, "max_trials": 1, "max_time_seconds": 10},
            "tune_model",
        ),
        (
            export_model_tool,
            {**base_args, "artifact_name": "qa001_overlap_model"},
            "export_model",
        ),
        (
            start_training_job_tool,
            dict(base_args),
            "start_training_job",
        ),
    ]
    for tool, args, label in cases:
        result = _call(tool, config, args)
        payload = _payload(result)
        assert payload["ok"] is False, f"{label} accepted overlapping manifest"


def test_qa001_validate_oob_rejected(config, classification_csv):
    df = pd.read_csv(classification_csv)
    name = "sample_classification.csv"
    mid = _make_manifest(
        config,
        df,
        name,
        split_type="time",
        train_indices=list(range(0, 12)),
        val_indices=list(range(12, 16)),
        test_indices=[999],
    )
    report = validate_split_manifest(config=config, manifest_id=mid, df=df, dataset_name=name)
    assert report["valid"] is False
    assert any("out of range" in e for e in report["errors"])

    args = {
        "name": name,
        "target_column": "purchased",
        "task_type": "classification",
        "model_name": "logistic_regression",
        "split_manifest_id": mid,
        "exclude_columns": [],
        "artifact_name": "qa001_oob_model",
    }
    result = _call(export_model_tool, config, args)
    payload = _payload(result)
    assert payload["ok"] is False


def test_qa001_validate_reversed_time_rejected(config, classification_csv):
    df = pd.read_csv(classification_csv)
    name = "sample_classification.csv"
    mid = _make_manifest(
        config,
        df,
        name,
        split_type="time",
        train_indices=list(range(10, 20)),
        val_indices=list(range(5, 10)),
        test_indices=list(range(0, 5)),
    )
    report = validate_manifest_for_consumption(
        config=config, manifest_id=mid, df=df, dataset_name=name
    )
    assert report["valid"] is False
    assert any("chronological" in e.lower() for e in report["errors"])


def test_qa001_valid_manifest_still_works(config, classification_csv):
    df = pd.read_csv(classification_csv)
    name = "sample_classification.csv"
    mid = _make_manifest(
        config,
        df,
        name,
        split_type="time",
        train_indices=list(range(0, 12)),
        val_indices=list(range(12, 16)),
        test_indices=list(range(16, 20)),
    )
    report = validate_manifest_for_consumption(
        config=config, manifest_id=mid, df=df, dataset_name=name
    )
    assert report["valid"] is True
    assert report["errors"] == []

    result = _call(
        train_baseline_tool,
        config,
        {
            "name": name,
            "target_column": "purchased",
            "task_type": "classification",
            "model_name": "logistic_regression",
            "split_manifest_id": mid,
            "exclude_columns": [],
        },
    )
    payload = _payload(result)
    assert payload["ok"] is True


# ---------------------------------------------------------------------------
# QA-002: Protected holdout disclosure
# ---------------------------------------------------------------------------


def _make_c1_c2_pair(config: DevRayConfig) -> tuple[str, str]:
    rows = []
    for i in range(15):
        rows.append(
            {
                "record_id": i + 1,
                "time": i,
                "group": i // 3,
                "age": 30 + i,
                "balance": 100 + i,
                "job": "admin" if i % 3 == 0 else "tech",
                "channel": "phone" if i % 2 == 0 else "web",
                "duration": 10 + i,
                "target": i % 2,
            }
        )
    c1 = pd.DataFrame(rows)
    c2 = pd.DataFrame(rows)
    for i in range(12, 15):
        c2.at[i, "balance"] = 100000 + i
        c2.at[i, "job"] = "future_only"
        c2.at[i, "duration"] = 100
        c2.at[i, "target"] = 1 - c2.at[i, "target"]

    c1_name = _write_csv(config, "qa_c1.csv", c1)
    c2_name = _write_csv(config, "qa_c2.csv", c2)
    return c1_name, c2_name


def test_qa002_protected_rows_do_not_change_dev_metrics(config):
    c1_name, c2_name = _make_c1_c2_pair(config)

    df1 = pd.read_csv(config.workspace_dir / c1_name)
    df2 = pd.read_csv(config.workspace_dir / c2_name)

    mid1 = _make_manifest(
        config,
        df1,
        c1_name,
        split_type="time",
        train_indices=list(range(0, 9)),
        val_indices=list(range(9, 12)),
        test_indices=list(range(12, 15)),
    )
    mid2 = _make_manifest(
        config,
        df2,
        c2_name,
        split_type="time",
        train_indices=list(range(0, 9)),
        val_indices=list(range(9, 12)),
        test_indices=list(range(12, 15)),
    )

    r1 = _call(
        train_baseline_tool,
        config,
        {
            "name": c1_name,
            "target_column": "target",
            "task_type": "classification",
            "model_name": "decision_tree",
            "split_manifest_id": mid1,
            "exclude_columns": ["record_id", "time", "group", "duration"],
        },
    )
    r2 = _call(
        train_baseline_tool,
        config,
        {
            "name": c2_name,
            "target_column": "target",
            "task_type": "classification",
            "model_name": "decision_tree",
            "split_manifest_id": mid2,
            "exclude_columns": ["record_id", "time", "group", "duration"],
        },
    )
    d1 = _data(r1)
    d2 = _data(r2)
    assert d1["metrics"] == d2["metrics"], (
        f"C1 metrics {d1['metrics']} != C2 metrics {d2['metrics']}; "
        "protected rows leaked into development evaluation."
    )


def test_qa002_job_status_no_raw_df(config):
    c1_name, _ = _make_c1_c2_pair(config)
    df = pd.read_csv(config.workspace_dir / c1_name)
    mid = _make_manifest(
        config,
        df,
        c1_name,
        split_type="time",
        train_indices=list(range(0, 9)),
        val_indices=list(range(9, 12)),
        test_indices=list(range(12, 15)),
    )

    result = _call(
        start_training_job_tool,
        config,
        {
            "name": c1_name,
            "target_column": "target",
            "task_type": "classification",
            "model_name": "logistic_regression",
            "split_manifest_id": mid,
            "exclude_columns": ["record_id", "time", "group", "duration"],
        },
    )
    job_id = _data(result)["job_id"]

    status = _call(get_training_status_tool, config, {"job_id": job_id})
    payload = _payload(status)
    assert payload["ok"] is True
    result_data = payload["data"].get("result")
    assert result_data is None or "df" not in str(result_data), (
        "Job status exposed raw DataFrame in queued state."
    )
    assert "record_id" not in json.dumps(payload["data"]), (
        "Job status exposed protected column names in queued state."
    )


def test_qa002_export_no_protected_test_metrics(config):
    c1_name, _ = _make_c1_c2_pair(config)
    df = pd.read_csv(config.workspace_dir / c1_name)
    mid = _make_manifest(
        config,
        df,
        c1_name,
        split_type="time",
        train_indices=list(range(0, 9)),
        val_indices=list(range(9, 12)),
        test_indices=list(range(12, 15)),
    )

    result = _call(
        export_model_tool,
        config,
        {
            "name": c1_name,
            "target_column": "target",
            "task_type": "classification",
            "model_name": "decision_tree",
            "split_manifest_id": mid,
            "artifact_name": "qa002_export",
            "exclude_columns": ["record_id", "time", "group", "duration"],
        },
    )
    payload = _payload(result)
    assert payload["ok"] is True
    metrics = payload["data"]["metrics"]
    assert metrics is not None
    assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# QA-003: Chronological cross-validation
# ---------------------------------------------------------------------------


def test_qa003_chronological_cv_uses_timeseries(config):
    df_rows = []
    for i in range(30):
        df_rows.append(
            {
                "record_id": i + 1,
                "time": i,
                "group": i // 5,
                "age": 30 + i,
                "balance": 100 + i,
                "job": "admin" if i % 3 == 0 else "tech",
                "channel": "phone" if i % 2 == 0 else "web",
                "duration": 10 + i,
                "target": i % 2,
            }
        )
    df = pd.DataFrame(df_rows)
    name = _write_csv(config, "qa003_chron.csv", df)
    mid = _make_manifest(
        config,
        df,
        name,
        split_type="time",
        train_indices=list(range(0, 18)),
        val_indices=list(range(18, 24)),
        test_indices=list(range(24, 30)),
    )

    result = _call(
        compare_models_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "classification",
            "models": ["logistic_regression"],
            "cv": 3,
            "split_manifest_id": mid,
            "exclude_columns": ["record_id", "time", "group", "duration"],
        },
    )
    payload = _payload(result)
    assert payload["ok"] is True
    for model_result in payload["data"]["results"]:
        if model_result["status"] == "completed":
            cv = model_result["cv"]
            assert cv["strategy"] == "TimeSeriesSplit", (
                f"Expected TimeSeriesSplit for chronological manifest, got {cv['strategy']}"
            )
            assert sum(cv["fold_sizes"]) <= 18, "CV folds include protected test rows."


# ---------------------------------------------------------------------------
# QA-004: Split-type naming aliases
# ---------------------------------------------------------------------------


def test_qa004_time_series_alias(config, classification_csv):
    df = pd.read_csv(classification_csv)
    df["event_time"] = range(len(df))
    _write_csv(config, "qa004_time.csv", df)

    result = _call(
        create_split_manifest_tool,
        config,
        {
            "name": "qa004_time.csv",
            "split_type": "time_series",
            "test_size": 0.2,
            "val_size": 0.2,
            "row_order": True,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is True
    assert payload["data"]["split_type"] == "time"


def test_qa004_grouped_alias(config, classification_csv):
    df = pd.read_csv(classification_csv)
    df["group"] = [i % 5 for i in range(len(df))]
    _write_csv(config, "qa004_group.csv", df)

    result = _call(
        create_split_manifest_tool,
        config,
        {
            "name": "qa004_group.csv",
            "split_type": "grouped",
            "group_column": "group",
            "test_size": 0.2,
            "val_size": 0.2,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is True
    assert payload["data"]["split_type"] == "group"


# ---------------------------------------------------------------------------
# QA-005: Strict JSON - no NaN/Infinity
# ---------------------------------------------------------------------------


def _assert_no_nonfinite(obj, path="$"):
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nonfinite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_nonfinite(v, f"{path}[{i}]")


def test_qa005_inspect_dataset_no_nan(config):
    rows = [{"x": float("nan") if i % 5 == 0 else float(i), "y": float(i)} for i in range(10)]
    df = pd.DataFrame(rows)
    name = _write_csv(config, "qa005_nan.csv", df)

    result = _call(inspect_dataset_tool, config, {"name": name, "max_rows": 5})
    parsed = json.loads(result[0].text)
    assert parsed["ok"] is True
    _assert_no_nonfinite(parsed["data"])


def test_qa005_drift_no_infinity(config):
    ref_rows = [{"job": "admin", "x": float(i)} for i in range(10)]
    cur_rows = [{"job": "admin", "x": float(i)} for i in range(10)]
    ref_name = _write_csv(config, "qa005_ref.csv", pd.DataFrame(ref_rows))
    cur_name = _write_csv(config, "qa005_cur.csv", pd.DataFrame(cur_rows))

    result = _call(
        analyze_feature_drift_tool,
        config,
        {
            "reference_name": ref_name,
            "current_name": cur_name,
            "columns": ["job"],
        },
    )
    parsed = json.loads(result[0].text)
    assert parsed["ok"] is True
    _assert_no_nonfinite(parsed["data"])


# ---------------------------------------------------------------------------
# QA-006: AutoML trial limits
# ---------------------------------------------------------------------------


def test_qa006_automl_max_trials_cap(config, classification_csv):
    result = _call(
        automl_run_tool,
        config,
        {
            "name": "sample_classification.csv",
            "target_column": "purchased",
            "task_type": "classification",
            "models": ["logistic_regression", "decision_tree"],
            "cv": 2,
            "max_trials": 1,
            "max_time_seconds": 30,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is True
    assert len(payload["data"]["trials"]) <= 1, (
        f"max_trials=1 but {len(payload['data']['trials'])} trials ran."
    )


def test_qa006_leaderboard_max_trials_cap(config, classification_csv):
    result = _call(
        get_model_leaderboard_tool,
        config,
        {
            "name": "sample_classification.csv",
            "target_column": "purchased",
            "task_type": "classification",
            "max_trials": 1,
            "max_time_seconds": 30,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is True
    assert len(payload["data"]["trials"]) <= 1, (
        f"max_trials=1 but {len(payload['data']['trials'])} leaderboard trials ran."
    )


# ---------------------------------------------------------------------------
# QA-007: Seeded tuning reproducibility
# ---------------------------------------------------------------------------


def test_qa007_seeded_tuning_reproducible(config, classification_csv):
    args = {
        "name": "sample_classification.csv",
        "target_column": "purchased",
        "task_type": "classification",
        "model_name": "decision_tree",
        "cv": 2,
        "max_trials": 1,
        "max_time_seconds": 30,
        "random_state": 42,
        "exclude_columns": [],
    }
    d1 = _data(_call(tune_model_tool, config, args))
    d2 = _data(_call(tune_model_tool, config, args))
    assert d1["best_value"] == d2["best_value"], (
        f"Seeded tuning not reproducible: {d1['best_value']} != {d2['best_value']}"
    )


# ---------------------------------------------------------------------------
# QA-008: Model provenance on re-export
# ---------------------------------------------------------------------------


def test_qa008_reexport_preserves_fingerprint(config, classification_csv):
    d1 = _data(
        _call(
            export_model_tool,
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "artifact_name": "qa008_original",
                "exclude_columns": [],
            },
        )
    )
    original_fp = d1["dataset_fingerprint"]
    assert original_fp is not None

    d2 = _data(
        _call(
            export_model_tool,
            config,
            {
                "model_handle": "qa008_original",
                "artifact_name": "qa008_copy",
            },
        )
    )
    assert d2["dataset_fingerprint"] == original_fp, (
        f"Re-export lost fingerprint: {d2['dataset_fingerprint']} != {original_fp}"
    )


# ---------------------------------------------------------------------------
# QA-009: Invalid preview-limit types
# ---------------------------------------------------------------------------


def test_qa009_invalid_max_rows_rejected(config, classification_csv):
    result = _call(
        inspect_dataset_tool, config, {"name": "sample_classification.csv", "max_rows": "bad"}
    )
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"

    result = _call(
        inspect_dataset_tool, config, {"name": "sample_classification.csv", "max_rows": True}
    )
    payload = _payload(result)
    assert payload["ok"] is False

    result = _call(
        inspect_dataset_tool, config, {"name": "sample_classification.csv", "max_rows": [5]}
    )
    payload = _payload(result)
    assert payload["ok"] is False

    result = _call(
        inspect_dataset_tool, config, {"name": "sample_classification.csv", "max_rows": 999}
    )
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"

    result = _call(
        inspect_dataset_tool, config, {"name": "sample_classification.csv", "max_rows": 0}
    )
    payload = _payload(result)
    assert payload["ok"] is False

    result = _call(
        inspect_dataset_tool, config, {"name": "sample_classification.csv", "max_rows": 3}
    )
    payload = _payload(result)
    assert payload["ok"] is True
    assert len(payload["data"]["preview"]) <= 3


# ---------------------------------------------------------------------------
# QA-001 follow-up: incomplete partition coverage
# ---------------------------------------------------------------------------


def test_qa001_incomplete_coverage_rejected(config, classification_csv):
    """Manifest omitting a dataset row (149/150 coverage) must be rejected."""
    import pandas as pd

    df = pd.read_csv(classification_csv)
    name = "sample_classification.csv"

    # Omit row 0 from train; counts adjusted truthfully (89 train, 30 val, 30 test = 149 listed).
    mid = _make_manifest(
        config,
        df,
        name,
        split_type="time",
        train_indices=list(range(1, 90)),
        val_indices=list(range(90, 120)),
        test_indices=list(range(120, 150)),
    )

    report = validate_manifest_for_consumption(
        config=config, manifest_id=mid, df=df, dataset_name=name
    )
    assert report["valid"] is False
    assert any("missing" in e.lower() for e in report["errors"]), report["errors"]

    # Public validator must also report invalid.
    rep2 = validate_split_manifest(config=config, manifest_id=mid, df=df, dataset_name=name)
    assert rep2["valid"] is False
    assert any("missing" in e.lower() for e in rep2["errors"]), rep2["errors"]

    base_args = {
        "name": name,
        "target_column": "purchased",
        "task_type": "classification",
        "model_name": "logistic_regression",
        "split_manifest_id": mid,
        "exclude_columns": [],
    }
    cases = [
        (train_baseline_tool, dict(base_args), "train_baseline"),
        (
            compare_models_tool,
            {**base_args, "models": ["logistic_regression"], "cv": 2},
            "compare_models",
        ),
        (
            automl_run_tool,
            {
                **base_args,
                "models": ["logistic_regression"],
                "cv": 2,
                "max_trials": 1,
                "max_time_seconds": 10,
            },
            "automl_run",
        ),
        (
            tune_model_tool,
            {**base_args, "cv": 2, "max_trials": 1, "max_time_seconds": 10},
            "tune_model",
        ),
        (
            export_model_tool,
            {**base_args, "artifact_name": "qa001_incomplete_model"},
            "export_model",
        ),
        (
            start_training_job_tool,
            dict(base_args),
            "start_training_job",
        ),
    ]
    for tool, args, label in cases:
        result = _call(tool, config, args)
        payload = _payload(result)
        assert payload["ok"] is False, f"{label} accepted incomplete manifest"
        assert payload["error"]["code"] == "validation_error", label


def test_qa001_validator_consistent_reverse_and_group(config, classification_csv):
    """Public validator must agree with consumers on reversed-time and crossed-group."""
    import pandas as pd

    df = pd.read_csv(classification_csv)
    name = "sample_classification.csv"

    # Reversed chronological: train after test.
    mid_rev = _make_manifest(
        config,
        df,
        name,
        split_type="time",
        train_indices=list(range(10, 20)),
        val_indices=list(range(5, 10)),
        test_indices=list(range(0, 5)),
    )
    rep = validate_split_manifest(config=config, manifest_id=mid_rev, df=df, dataset_name=name)
    assert rep["valid"] is False
    assert any("chronological" in e.lower() for e in rep["errors"]), rep["errors"]

    # Crossed-group: train and test share a group (complete coverage).
    dfg = df.copy()
    dfg["group"] = [i % 5 for i in range(len(df))]
    mid_grp = _make_manifest(
        config,
        dfg,
        name,
        split_type="group",
        group_column="group",
        train_indices=list(range(0, 10)),
        val_indices=list(range(10, 15)),
        test_indices=list(range(15, 20)),  # groups 0-4 overlap train groups 0-4
    )
    rep = validate_split_manifest(config=config, manifest_id=mid_grp, df=dfg, dataset_name=name)
    assert rep["valid"] is False
    assert any("group" in e.lower() for e in rep["errors"]), rep["errors"]


# ---------------------------------------------------------------------------
# QA-002 follow-up: protected manifest without validation partition
# ---------------------------------------------------------------------------


def _make_no_val_manifest(config: DevRayConfig, df: pd.DataFrame, dataset_name: str) -> str:
    """Manifest with train=0..119, no validation, protected test=120..149."""
    return _make_manifest(
        config,
        df,
        dataset_name,
        split_type="time",
        train_indices=list(range(0, 120)),
        val_indices=[],
        test_indices=list(range(120, 150)),
    )


def test_qa002_no_validation_partition_rejected(config, classification_csv):
    """Protected manifest with no validation rows must fail clearly in all six consumers."""
    df = pd.read_csv(classification_csv)

    # Extend dataset to 150 rows to match the acceptance fixture shape.
    extra = []
    for i in range(20, 150):
        extra.append(
            {
                "age": 20 + i,
                "income": 30000 + i * 1000,
                "credit_score": 600 + i,
                "category": "electronics" if i % 3 == 0 else "clothing",
                "region": "north" if i % 2 == 0 else "south",
                "purchased": i % 2,
            }
        )
    df_ext = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    name_ext = _write_csv(config, "qa002_no_val.csv", df_ext)

    mid = _make_no_val_manifest(config, df_ext, name_ext)

    base_args = {
        "name": name_ext,
        "target_column": "purchased",
        "task_type": "classification",
        "model_name": "logistic_regression",
        "split_manifest_id": mid,
        "exclude_columns": [],
    }
    cases = [
        (train_baseline_tool, dict(base_args), "train_baseline"),
        (
            compare_models_tool,
            {**base_args, "models": ["logistic_regression"], "cv": 2},
            "compare_models",
        ),
        (
            automl_run_tool,
            {
                **base_args,
                "models": ["logistic_regression"],
                "cv": 2,
                "max_trials": 1,
                "max_time_seconds": 10,
            },
            "automl_run",
        ),
        (
            tune_model_tool,
            {**base_args, "cv": 2, "max_trials": 1, "max_time_seconds": 10},
            "tune_model",
        ),
        (
            export_model_tool,
            {**base_args, "artifact_name": "qa002_no_val_model"},
            "export_model",
        ),
        (
            start_training_job_tool,
            dict(base_args),
            "start_training_job",
        ),
    ]
    for tool, args, label in cases:
        result = _call(tool, config, args)
        payload = _payload(result)
        if label == "start_training_job":
            # Submission succeeds (manifest is valid); failure is async.
            assert payload["ok"] is True, f"{label} submission failed: {payload.get('error')}"
            job_id = payload["data"]["job_id"]
            # Poll until terminal; sleep to let the background worker run.
            st = None
            for _ in range(100):
                time.sleep(0.1)
                status_result = _call(get_training_status_tool, config, {"job_id": job_id})
                status_payload = _payload(status_result)
                st = status_payload["data"].get("status")
                if st in ("failed", "completed"):
                    break
            assert st == "failed", f"{label} job did not fail: {st}"
            err = status_payload["data"].get("error", "").lower()
            assert "validation" in err or "protected" in err, (
                f"{label} job error not about protected holdout: "
                f"{status_payload['data'].get('error')}"
            )
        elif label in ("compare_models", "automl_run", "tune_model"):
            # CV-based tools operate on train-only development rows; no
            # protected test rows enter folds. They succeed without metrics leak.
            assert payload["ok"] is True, f"{label} unexpectedly failed: {payload.get('error')}"
        else:
            # Trainer-based tools must fail rather than evaluate on test rows.
            assert payload["ok"] is False, (
                f"{label} accepted no-val protected manifest: {payload.get('error')}"
            )


def test_qa002_no_validation_partition_no_protected_metrics(config, classification_csv):
    """With no validation partition, no protected-test metrics may leak."""
    df = pd.read_csv(classification_csv)

    extra = []
    for i in range(20, 150):
        extra.append(
            {
                "age": 20 + i,
                "income": 30000 + i * 1000,
                "credit_score": 600 + i,
                "category": "electronics" if i % 3 == 0 else "clothing",
                "region": "north" if i % 2 == 0 else "south",
                "purchased": i % 2,
            }
        )
    df_ext = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    name_ext = _write_csv(config, "qa002_no_val2.csv", df_ext)

    mid = _make_no_val_manifest(config, df_ext, name_ext)

    # train_baseline must fail, not expose protected metrics.
    result = _call(
        train_baseline_tool,
        config,
        {
            "name": name_ext,
            "target_column": "purchased",
            "task_type": "classification",
            "model_name": "logistic_regression",
            "split_manifest_id": mid,
            "exclude_columns": [],
        },
    )
    payload = _payload(result)
    assert payload["ok"] is False
    assert "validation" in payload["error"]["code"] or "training" in payload["error"]["code"]
    msg = payload["error"]["message"].lower()
    assert "validation" in msg or "protected" in msg or "partition" in msg, msg


# ---------------------------------------------------------------------------
# QA-009 follow-up: null and fractional preview limits
# ---------------------------------------------------------------------------


def test_qa009_null_and_fractional_max_rows_rejected(config, classification_csv):
    """Explicit null and fractional max_rows must be rejected, not coerced."""
    cases = [
        (None, "null"),
        (100.5, "float above cap"),
        (100.9, "float just below cap"),
        (1.5, "in-range float"),
        ("bad", "bad string"),
        (True, "boolean true"),
        (False, "boolean false"),
        ([5], "list"),
        ({"max": 5}, "dict"),
        (0, "zero"),
        (-1, "negative"),
        (101, "above cap"),
    ]
    for value, label in cases:
        result = _call(
            inspect_dataset_tool,
            config,
            {"name": "sample_classification.csv", "max_rows": value},
        )
        payload = _payload(result)
        assert payload["ok"] is False, f"{label} ({value!r}) was accepted"
        assert payload["error"]["code"] == "validation_error", label


def test_qa009_valid_limits_still_work(config, classification_csv):
    """Omitted max_rows defaults to 25; valid integers 1..100 work."""
    # Omitted -> default 25.
    result = _call(inspect_dataset_tool, config, {"name": "sample_classification.csv"})
    payload = _payload(result)
    assert payload["ok"] is True
    assert len(payload["data"]["preview"]) <= 25

    # Valid integer limits.
    for limit in (1, 3, 100):
        result = _call(
            inspect_dataset_tool,
            config,
            {"name": "sample_classification.csv", "max_rows": limit},
        )
        payload = _payload(result)
        assert payload["ok"] is True, f"limit {limit} rejected"
        assert len(payload["data"]["preview"]) <= limit
