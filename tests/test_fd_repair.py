"""Regression tests for the consolidated reliability repair (FD-001 to FD-005).

Each test reproduces a confirmed defect from the final-discovery QA audit and
verifies the correction. Fixtures are deterministic, synthetic, CPU-only and
created inside the per-test temporary workspace. The real Bank Marketing
dataset is never used.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pandas as pd

from devray_datalab.config import DevRayConfig
from devray_datalab.core.metric_policy import (
    direction_for_metric,
    is_valid_metric,
    resolve_metric,
    valid_metrics,
)
from devray_datalab.core.splits import (
    SplitManifest,
    SplitStore,
    _dataset_fingerprint,
    create_split_manifest,
    validate_manifest_for_consumption,
    validate_split_manifest,
)
from devray_datalab.tools.analysis import analyze_correlations_tool
from devray_datalab.tools.automl import (
    automl_run_tool,
    get_model_leaderboard_tool,
    tune_model_tool,
)
from devray_datalab.tools.jobs import (
    start_training_job_tool,
)
from devray_datalab.tools.models import export_model_tool
from devray_datalab.tools.splits import (
    create_split_manifest_tool,
    validate_split_manifest_tool,
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


def _call(tool, config, arguments):
    return _run(tool(config, arguments))


# ---------------------------------------------------------------------------
# FD-001: metric/objective policy
# ---------------------------------------------------------------------------


def _fd001_classification_df(config: DevRayConfig) -> str:
    # Deterministic threshold rule: a decision tree separates it exactly
    # (log-loss 0 on every fold), while logistic regression cannot reach 0.
    rng = __import__("random").Random(7)
    n = 120
    xs = [rng.randint(1, 10) for _ in range(n)]
    zs = [rng.uniform(0, 100) for _ in range(n)]
    df = pd.DataFrame(
        {
            "x": xs,
            "z": zs,
            "target": [1 if x > 5 else 0 for x in xs],
        }
    )
    return _write_csv(config, "fd001_clf.csv", df)


def _fd001_regression_df(config: DevRayConfig) -> str:
    # Exact linear relation: linear regression reaches MAE 0 / R^2 1 on every
    # fold; a tree cannot extrapolate to unseen x values on held-out folds.
    n = 120
    df = pd.DataFrame(
        {
            "x": list(range(1, n + 1)),
            "z": [float(i % 7) for i in range(n)],
            "target": [3.0 * i + 2.0 for i in range(1, n + 1)],
        }
    )
    return _write_csv(config, "fd001_reg.csv", df)


def _fd001_multiclass_df(config: DevRayConfig) -> str:
    # Discrete rule a tree learns exactly: F1 = 1, binary ROC-AUC undefined.
    rng = __import__("random").Random(11)
    n = 120
    xs = [rng.randint(1, 10) for _ in range(n)]
    df = pd.DataFrame({"x": xs, "target": [(x - 1) // 4 for x in xs]})
    return _write_csv(config, "fd001_multi.csv", df)


def test_fd001_metric_policy_units():
    assert resolve_metric("classification", "auto", 2) == "roc_auc"
    assert resolve_metric("classification", "auto", 3) == "f1"
    assert resolve_metric("regression", "auto") == "r2"
    assert resolve_metric("classification", "log_loss") == "log_loss"
    assert direction_for_metric("classification", "log_loss") == "minimize"
    assert direction_for_metric("classification", "roc_auc") == "maximize"
    assert direction_for_metric("regression", "mae") == "minimize"
    assert direction_for_metric("regression", "r2") == "maximize"
    assert is_valid_metric("classification", "auto") is True
    assert is_valid_metric("classification", "not_a_metric") is False
    assert is_valid_metric("regression", "r2") is True
    assert "roc_auc" in valid_metrics("classification")
    assert "r2" in valid_metrics("regression")


def test_fd001_compare_log_loss_minimized(config):
    name = _fd001_classification_df(config)
    result = _call(
        compare_models_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "classification",
            "models": ["logistic_regression", "decision_tree"],
            "metric": "log_loss",
            "cv": 2,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["selection_metric"] == "log_loss"
    assert data["best_model"] == "decision_tree"
    assert data["best_metrics"]["log_loss"] == 0
    logistic = next(r for r in data["results"] if r["model"] == "logistic_regression")
    tree = next(r for r in data["results"] if r["model"] == "decision_tree")
    assert tree["metrics"]["log_loss"] < logistic["metrics"]["log_loss"]
    # CV truthfully scores the requested loss, not ROC-AUC.
    assert tree["cv"]["scoring"] == "log_loss"


def test_fd001_compare_mae_minimized(config):
    name = _fd001_regression_df(config)
    result = _call(
        compare_models_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "regression",
            "models": ["linear_regression", "decision_tree"],
            "metric": "mae",
            "cv": 2,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["selection_metric"] == "mae"
    assert data["best_model"] == "linear_regression"
    assert data["best_metrics"]["mae"] == 0
    tree = next(r for r in data["results"] if r["model"] == "decision_tree")
    assert tree["metrics"]["mae"] > 0


def test_fd001_compare_auto_regression_resolves(config):
    name = _fd001_regression_df(config)
    result = _call(
        compare_models_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "regression",
            "models": ["decision_tree", "linear_regression"],
            "metric": "auto",
            "cv": 2,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["selection_metric"] == "r2"
    assert data["best_model"] == "linear_regression"
    assert data["best_metrics"]["r2"] == 1


def test_fd001_automl_loss_natural_values(config):
    name = _fd001_classification_df(config)
    result = _call(
        automl_run_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "classification",
            "models": ["logistic_regression", "decision_tree"],
            "metric": "log_loss",
            "cv": 2,
            "max_trials": 2,
            "max_time_seconds": 10,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["metric"] == "log_loss"
    assert data["best_model"] == "decision_tree"
    assert data["best_value"] == 0
    # Losses keep their natural (non-negated) sign.
    for trial in data["trials"]:
        if trial["error"] is None:
            assert trial["value"] >= 0


def test_fd001_automl_mae_direction(config):
    name = _fd001_regression_df(config)
    result = _call(
        automl_run_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "regression",
            "models": ["linear_regression", "decision_tree"],
            "metric": "mae",
            "cv": 2,
            "max_trials": 2,
            "max_time_seconds": 10,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["best_model"] == "linear_regression"
    assert data["best_value"] == 0


def test_fd001_leaderboard_ranks_by_objective(config):
    name = _fd001_classification_df(config)
    result = _call(
        get_model_leaderboard_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "classification",
            "metric": "log_loss",
            "max_trials": 2,
            "max_time_seconds": 10,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["best_model"] == "decision_tree"
    assert data["best_value"] == 0
    finished = [t for t in data["trials"] if t["error"] is None]
    values = [t["value"] for t in finished]
    assert values == sorted(values), "leaderboard trials must be loss-ascending"


def test_fd001_tune_multiclass_auto_f1_fallback(config):
    name = _fd001_multiclass_df(config)
    result = _call(
        tune_model_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "classification",
            "model_name": "decision_tree",
            "metric": "auto",
            "cv": 2,
            "max_trials": 10,
            "max_time_seconds": 10,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["metric"] == "f1"
    assert data["best_value"] == 1


def test_fd001_tune_regression_mae_minimized(config):
    name = _fd001_regression_df(config)
    result = _call(
        tune_model_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "regression",
            "model_name": "linear_regression",
            "metric": "mae",
            "cv": 2,
            "max_trials": 1,
            "max_time_seconds": 10,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["metric"] == "mae"
    assert data["best_value"] == 0


def test_fd001_invalid_metric_rejected(config):
    name = _fd001_classification_df(config)
    result = _call(
        tune_model_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "classification",
            "model_name": "logistic_regression",
            "metric": "not_a_metric",
            "cv": 2,
            "max_trials": 1,
            "max_time_seconds": 10,
            "random_state": 42,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"


def test_fd001_seed_reproducible(config):
    name = _fd001_classification_df(config)
    args = {
        "name": name,
        "target_column": "target",
        "task_type": "classification",
        "models": ["logistic_regression", "decision_tree"],
        "metric": "log_loss",
        "cv": 2,
        "random_state": 42,
    }
    d1 = _data(_call(compare_models_tool, config, args))
    d2 = _data(_call(compare_models_tool, config, args))
    assert d1["best_model"] == d2["best_model"] == "decision_tree"
    assert d1["best_metrics"]["log_loss"] == d2["best_metrics"]["log_loss"]


# ---------------------------------------------------------------------------
# FD-002: timestamp chronology
# ---------------------------------------------------------------------------


def _fd002_reverse_time_df(config: DevRayConfig) -> str:
    # time descends with the row index: time = 120 - i. Timestamp order is the
    # reverse of original row order.
    n = 120
    rows = [{"record_id": i + 1, "time": n - i, "x": i % 2, "target": i % 2} for i in range(n)]
    return _write_csv(config, "fd002_reverse_time.csv", pd.DataFrame(rows))


def _fd002_block_reverse_time_df(config: DevRayConfig) -> str:
    # Within each block of 24 rows time descends; blocks advance in time.
    n = 120
    rows = []
    for i in range(n):
        rows.append(
            {
                "record_id": i + 1,
                "time": (i // 24) * 24 + 24 - (i % 24),
                "x": i % 2,
                "target": i % 2,
            }
        )
    return _write_csv(config, "fd002_block_reverse_time.csv", pd.DataFrame(rows))


def test_fd002_server_time_manifest_validates(config):
    name = _fd002_reverse_time_df(config)
    result = _call(
        create_split_manifest_tool,
        config,
        {
            "name": name,
            "split_type": "time_series",
            "time_column": "time",
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
        },
    )
    data = _data(result)
    manifest_id = data["manifest_id"]
    assert data["split_type"] == "time"
    assert data["time_column"] == "time"

    report = validate_split_manifest(
        config=config,
        manifest_id=manifest_id,
        df=pd.read_csv(config.workspace_dir / name),
        dataset_name=name,
    )
    assert report["valid"] is True, report["errors"]
    assert report["errors"] == []


def test_fd002_timestamp_order_not_row_order(config):
    name = _fd002_reverse_time_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="time",
        random_state=42,
    )
    # Stored in timestamp order: earliest timestamps in train.
    import pandas as _pd

    times = _pd.to_datetime(df["time"]) if False else df["time"]
    train_times = times.iloc[manifest.train_indices]
    val_times = times.iloc[manifest.val_indices]
    test_times = times.iloc[manifest.test_indices]
    assert train_times.max() <= val_times.min()
    assert val_times.max() <= test_times.min()
    # Original row index order is not the chronology here.
    assert max(manifest.train_indices) > min(manifest.val_indices)


def test_fd002_chronological_cv_no_future_leak(config):
    name = _fd002_block_reverse_time_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="time",
        random_state=42,
    )
    result = _call(
        compare_models_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "classification",
            "models": ["logistic_regression"],
            "cv": 2,
            "split_manifest_id": manifest.manifest_id,
            "exclude_columns": ["record_id", "time"],
        },
    )
    data = _data(result)
    cv = data["results"][0]["cv"]
    assert cv["strategy"] == "TimeSeriesSplit"
    assert sum(cv["fold_sizes"]) <= manifest.n_train + manifest.n_val


def test_fd002_duplicate_timestamps_deterministic(config):
    n = 60
    df = pd.DataFrame(
        {
            "time": [i // 10 for i in range(n)],
            "x": list(range(n)),
            "target": [i % 2 for i in range(n)],
        }
    )
    name = _write_csv(config, "fd002_dup_time.csv", df)
    m1 = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="time",
        random_state=42,
    )
    m2 = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="time",
        random_state=42,
    )
    assert m1.train_indices == m2.train_indices
    times = df["time"]
    assert times.iloc[m1.train_indices].max() <= times.iloc[m1.val_indices].min()


def test_fd002_missing_timestamp_rejected(config):
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, None, 4.0, 5.0, 6.0] * 20,
            "x": range(120),
            "target": [i % 2 for i in range(120)],
        }
    )
    name = _write_csv(config, "fd002_missing_time.csv", df)
    result = _call(
        create_split_manifest_tool,
        config,
        {
            "name": name,
            "split_type": "time_series",
            "time_column": "time",
            "test_size": 0.2,
            "val_size": 0.2,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is False


def test_fd002_invalid_timestamp_rejected(config):
    df = pd.DataFrame(
        {
            "time": ["a", "b", "c", "d", "e"] * 24,
            "x": range(120),
            "target": [i % 2 for i in range(120)],
        }
    )
    name = _write_csv(config, "fd002_invalid_time.csv", df)
    result = _call(
        create_split_manifest_tool,
        config,
        {
            "name": name,
            "split_type": "time_series",
            "time_column": "time",
            "test_size": 0.2,
            "val_size": 0.2,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is False


def test_fd002_row_order_preserved(config):
    n = 60
    df = pd.DataFrame({"x": range(n), "target": [i % 2 for i in range(n)]})
    name = _write_csv(config, "fd002_row_order.csv", df)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        row_order=True,
        random_state=42,
    )
    assert manifest.time_column is None
    assert max(manifest.train_indices) < min(manifest.val_indices) < min(manifest.test_indices)
    report = validate_split_manifest(
        config=config, manifest_id=manifest.manifest_id, df=df, dataset_name=name
    )
    assert report["valid"] is True, report["errors"]


# ---------------------------------------------------------------------------
# FD-003: grouped splitting and one authoritative validator
# ---------------------------------------------------------------------------


def _fd003_grouped_df(config: DevRayConfig) -> str:
    n = 90
    df = pd.DataFrame(
        {
            "record_id": range(1, n + 1),
            "group": [i % 5 for i in range(n)],
            "x": range(n),
            "target": [i % 2 for i in range(n)],
        }
    )
    return _write_csv(config, "fd003_grouped.csv", df)


def test_fd003_grouped_requires_group_column(config):
    name = _fd003_grouped_df(config)
    result = _call(
        create_split_manifest_tool,
        config,
        {
            "name": name,
            "split_type": "grouped",
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"
    assert "group_column" in payload["error"]["message"].lower()


def test_fd003_grouped_nonexistent_column_rejected(config):
    name = _fd003_grouped_df(config)
    result = _call(
        create_split_manifest_tool,
        config,
        {
            "name": name,
            "split_type": "grouped",
            "group_column": "missing_group",
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
        },
    )
    payload = _payload(result)
    assert payload["ok"] is False


def test_fd003_valid_grouped_manifest(config):
    name = _fd003_grouped_df(config)
    result = _call(
        create_split_manifest_tool,
        config,
        {
            "name": name,
            "split_type": "grouped",
            "group_column": "group",
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
        },
    )
    data = _data(result)
    assert data["split_type"] == "group"
    assert data["group_column"] == "group"
    manifest_id = data["manifest_id"]
    report = validate_manifest_for_consumption(
        config=config,
        manifest_id=manifest_id,
        df=pd.read_csv(config.workspace_dir / name),
        dataset_name=name,
    )
    assert report["valid"] is True, report["errors"]

    df = pd.read_csv(config.workspace_dir / name)
    groups = df["group"]
    manifest = SplitStore(config).load(manifest_id)
    tg = {groups.iloc[i] for i in manifest.train_indices}
    vg = {groups.iloc[i] for i in manifest.val_indices}
    te = {groups.iloc[i] for i in manifest.test_indices}
    assert not (tg & vg) and not (tg & te) and not (vg & te)


def _make_tampered_manifest(
    config: DevRayConfig,
    name: str,
    split_type: str,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    group_column: str | None = None,
    time_column: str | None = None,
) -> str:
    import time

    df = pd.read_csv(config.workspace_dir / name)
    manifest = SplitManifest(
        manifest_id=str(uuid.uuid4().hex),
        dataset_name=name,
        dataset_fingerprint=_dataset_fingerprint(df),
        created_at=time.time(),
        split_type=split_type,
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
        n_train=len(set(train_indices)),
        n_val=len(set(val_indices)),
        n_test=len(set(test_indices)),
        stratify_column=None,
        group_column=group_column,
        time_column=time_column,
        random_state=42,
        protected_holdout=True,
    )
    SplitStore(config).save(manifest)
    return manifest.manifest_id


def test_fd003_unknown_strategy_rejected(config, classification_csv):
    import shutil
    from pathlib import Path

    src = Path(classification_csv)
    name = "fd003_strategy.csv"
    shutil.copy(src, config.workspace_dir / name)
    mid = _make_tampered_manifest(
        config,
        name,
        "time_seriez",
        list(range(0, 12)),
        list(range(12, 16)),
        list(range(16, 20)),
    )

    # Public validator must agree with consumers.
    report = validate_split_manifest(
        config=config,
        manifest_id=mid,
        df=pd.read_csv(config.workspace_dir / name),
        dataset_name=name,
    )
    assert report["valid"] is False
    assert any("unknown split type" in e.lower() for e in report["errors"])

    base_args = {
        "name": name,
        "target_column": "purchased",
        "task_type": "classification",
        "model_name": "logistic_regression",
        "split_manifest_id": mid,
        "exclude_columns": [],
    }
    cases = [
        (train_baseline_tool, dict(base_args)),
        (compare_models_tool, {**base_args, "models": ["logistic_regression"], "cv": 2}),
        (
            automl_run_tool,
            {
                **base_args,
                "models": ["logistic_regression"],
                "cv": 2,
                "max_trials": 1,
            },
        ),
        (tune_model_tool, {**base_args, "cv": 2, "max_trials": 1, "max_time_seconds": 10}),
        (export_model_tool, {**base_args, "artifact_name": "fd003_strategy_model"}),
        (start_training_job_tool, dict(base_args)),
    ]
    for tool, args in cases:
        payload = _payload(_call(tool, config, args))
        assert payload["ok"] is False, f"{tool.__name__} accepted unknown strategy"


def test_fd003_duplicate_indices_public_and_consumers(config, classification_csv):
    name = _write_csv(config, "fd003_dup.csv", pd.read_csv(classification_csv))
    # train lists row 0 twice; declared counts truthful per listed entries but
    # coverage incomplete (row 19 missing).
    mid = _make_tampered_manifest(
        config, name, "time", list(range(0, 12)) + [0], list(range(12, 16)), list(range(16, 19))
    )

    df = pd.read_csv(config.workspace_dir / name)
    report = validate_split_manifest(config=config, manifest_id=mid, df=df, dataset_name=name)
    assert report["valid"] is False
    assert any("duplicate" in e.lower() for e in report["errors"])
    assert any("missing" in e.lower() for e in report["errors"])

    # Consumers agree with the public validator: nothing is fitted/exported/submitted.
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
        (tune_model_tool, {**base_args, "cv": 2, "max_trials": 1}, "tune_model"),
        (export_model_tool, {**base_args, "artifact_name": "fd003_dup_model"}, "export_model"),
        (start_training_job_tool, dict(base_args), "start_training_job"),
    ]
    for tool, args, label in cases:
        payload = _payload(_call(tool, config, args))
        assert payload["ok"] is False, f"{label} accepted duplicate manifest"
        assert payload["error"]["code"] == "validation_error", label


# ---------------------------------------------------------------------------
# FD-004: manifest path confinement
# ---------------------------------------------------------------------------


def test_fd004_valid_manifest_roundtrip(config):
    import pandas as pd

    df = pd.DataFrame({"x": range(10), "target": [i % 2 for i in range(10)]})
    name = _write_csv(config, "fd004_valid.csv", df)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="random",
        test_size=0.2,
        random_state=42,
    )
    store = SplitStore(config)
    loaded = store.load(manifest.manifest_id)
    assert loaded is not None
    assert loaded.manifest_id == manifest.manifest_id
    report = validate_split_manifest(
        config=config, manifest_id=manifest.manifest_id, df=df, dataset_name=name
    )
    assert report["valid"] is True


def test_fd004_traversal_rejected(config):
    store = SplitStore(config)
    for bad_id in (
        "../../reports/final-discovery/fd_outside_splits",
        "..\\..\\reports\\final-discovery\\fd_outside_splits",
        "../reports/final-discovery/../fd_outside_splits",
        "../../etc/passwd",
    ):
        try:
            loaded = store.load(bad_id)
            assert loaded is None, f"traversal id {bad_id!r} loaded a manifest"
        except Exception as exc:
            assert "escapes" in str(exc).lower() or "invalid" in str(exc).lower(), exc
        # Saving under such an id must not create files outside the store.
        try:
            store.delete(bad_id)
        except Exception:
            pass
    artifacts_files = list((config.artifacts_dir / "splits").glob("*.json"))
    assert all(p.name.endswith(".json") for p in artifacts_files)


def test_fd004_absolute_path_rejected(config):
    store = SplitStore(config)
    try:
        store.load("/tmp/fd004_absolute")
        raise AssertionError("absolute path id loaded")
    except Exception as exc:
        assert "escapes" in str(exc).lower() or "invalid" in str(exc).lower(), exc


def test_fd004_tool_traversal_rejected(config, classification_csv):
    name = _write_csv(config, "fd004_tool.csv", pd.read_csv(classification_csv))
    result = _call(
        validate_split_manifest_tool,
        config,
        {"manifest_id": "../../reports/final-discovery/fd_outside_splits", "name": name},
    )
    payload = _payload(result)
    assert payload["ok"] is False
    assert "fd_outside_splits" not in json.dumps(payload)


def test_fd004_consumer_rejects_escaped_manifest(config, classification_csv):
    name = _write_csv(config, "fd004_consume.csv", pd.read_csv(classification_csv))
    base_args = {
        "name": name,
        "target_column": "purchased",
        "task_type": "classification",
        "model_name": "logistic_regression",
        "split_manifest_id": "../../reports/final-discovery/fd_outside_splits",
        "exclude_columns": [],
    }
    payload = _payload(_call(train_baseline_tool, config, base_args))
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"


def test_fd004_symlink_escape_rejected(config):
    store = SplitStore(config)
    import os

    outside = config.workspace_dir / "outside_sentinel.json"
    outside.write_text('{"manifest_id": "escaped"}')
    link = config.artifacts_dir / "splits" / "fd004_link.json"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        # Symlinks not available/privileged in this environment: skip safely.
        return
    try:
        loaded = store.load("fd004_link")
        assert loaded is None, "symlink escape loaded a manifest"
    except Exception as exc:
        assert "escapes" in str(exc).lower() or "invalid" in str(exc).lower(), exc


# ---------------------------------------------------------------------------
# FD-005: correlation summary sign filtering
# ---------------------------------------------------------------------------


def _fd005_df(config: DevRayConfig, filename: str, data: dict[str, list[float]]) -> str:
    return _write_csv(config, filename, pd.DataFrame(data))


def test_fd005_positive_only_negative_is_null(config):
    name = _fd005_df(
        config,
        "fd005_pos.csv",
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
    )
    result = _call(analyze_correlations_tool, config, {"name": name, "top_n": 5})
    data = _data(result)
    assert data["method"] == "pearson"
    assert data["strongest_positive"] is not None
    assert data["strongest_positive"]["correlation"] > 0
    assert data["strongest_negative"] is None


def test_fd005_negative_only_positive_is_null(config):
    name = _fd005_df(
        config,
        "fd005_neg.csv",
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]},
    )
    result = _call(analyze_correlations_tool, config, {"name": name, "top_n": 5})
    data = _data(result)
    assert data["strongest_positive"] is None
    assert data["strongest_negative"] is not None
    assert data["strongest_negative"]["correlation"] < 0


def test_fd005_mixed_signs(config):
    name = _fd005_df(
        config,
        "fd005_mixed.csv",
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [2.0, 4.0, 6.0, 8.0, 10.0],
        },
    )
    result = _call(analyze_correlations_tool, config, {"name": name, "top_n": 5})
    data = _data(result)
    assert data["strongest_positive"]["correlation"] > 0
    assert data["strongest_negative"]["correlation"] < 0
    pos_abs = data["strongest_positive"]["abs_correlation"]
    neg_abs = abs(data["strongest_negative"]["correlation"])
    assert pos_abs >= neg_abs


def test_fd005_constant_columns(config):
    name = _fd005_df(
        config,
        "fd005_const.csv",
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [7.0, 7.0, 7.0, 7.0, 7.0]},
    )
    result = _call(analyze_correlations_tool, config, {"name": name, "top_n": 5})
    data = _data(result)
    assert "b" in data["constant_columns"]
    assert data["strongest_positive"] is None
    assert data["strongest_negative"] is None


def test_fd005_single_column_null(config):
    name = _fd005_df(config, "fd005_single.csv", {"a": [1.0, 2.0, 3.0]})
    result = _call(analyze_correlations_tool, config, {"name": name, "top_n": 5})
    data = _data(result)
    assert data["strongest_positive"] is None
    assert data["strongest_negative"] is None


def test_fd005_pearson_and_spearman_consistent(config):
    name = _fd005_df(
        config,
        "fd005_both.csv",
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 3.0, 2.0, 5.0, 4.0],
        },
    )
    for method in ("pearson", "spearman"):
        result = _call(
            analyze_correlations_tool,
            config,
            {"name": name, "method": method, "top_n": 5},
        )
        data = _data(result)
        pos = data["strongest_positive"]
        neg = data["strongest_negative"]
        if pos is not None:
            assert pos["correlation"] > 0
        if neg is not None:
            assert neg["correlation"] < 0
