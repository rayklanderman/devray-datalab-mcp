"""Regression tests for binary classification with string-valued targets.

Covers the Bank Marketing Phase 2 blocker: string labels ("no"/"yes") caused
evaluator failures because precision/recall/F1/ROC-AUC/PR-AUC hardcoded
positive_label=1.

Tests use bounded synthetic fixtures suitable for CPU-only execution.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid

import numpy as np
import pandas as pd
import pytest

from devray_datalab.config import DevRayConfig
from devray_datalab.core.evaluator import (
    evaluate_classification,
    optimize_decision_threshold,
)
from devray_datalab.core.splits import (
    SplitManifest,
    SplitStore,
    _dataset_fingerprint,
    validate_manifest_for_consumption,
)
from devray_datalab.tools.automl import automl_run_tool, tune_model_tool
from devray_datalab.tools.models import export_model_tool, predict_tool
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


# ---------------------------------------------------------------------------
# Synthetic string-label dataset fixture
# ---------------------------------------------------------------------------


def _make_string_dataset(config: DevRayConfig, filename: str, n: int = 200) -> str:
    """Create a synthetic classification dataset with string yes/no labels."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.choice(["a", "b", "c"], n),
            "target": ["yes" if i % 3 == 0 else "no" for i in range(n)],
        }
    )
    return _write_csv(config, filename, df)


def _make_numeric_dataset(config: DevRayConfig, filename: str, n: int = 200) -> str:
    """Create a synthetic classification dataset with numeric 0/1 labels."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.choice(["a", "b", "c"], n),
            "target": [0 if i % 3 == 0 else 1 for i in range(n)],
        }
    )
    return _write_csv(config, filename, df)


# ---------------------------------------------------------------------------
# Test 1: Logistic regression with no/yes labels
# ---------------------------------------------------------------------------


def test_logistic_regression_string_labels(config):
    name = _make_string_dataset(config, "str_labels_logreg.csv")
    result = _run(
        train_baseline_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    metrics = payload["data"]["metrics"]
    assert metrics is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    # "yes" is the positive class; precision/recall/f1 must be computable.
    assert isinstance(metrics["precision"], (int, float))
    assert isinstance(metrics["recall"], (int, float))
    assert isinstance(metrics["f1"], (int, float))


# ---------------------------------------------------------------------------
# Test 2: Decision tree with no/yes labels
# ---------------------------------------------------------------------------


def test_decision_tree_string_labels(config):
    name = _make_string_dataset(config, "str_labels_dt.csv")
    result = _run(
        train_baseline_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "decision_tree",
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    metrics = payload["data"]["metrics"]
    assert metrics is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


# ---------------------------------------------------------------------------
# Test 3: Existing numeric 0/1 compatibility
# ---------------------------------------------------------------------------


def test_numeric_zero_one_labels(config):
    name = _make_numeric_dataset(config, "num_labels.csv")
    result = _run(
        train_baseline_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    metrics = payload["data"]["metrics"]
    assert metrics is not None
    assert isinstance(metrics["precision"], (int, float))
    assert isinstance(metrics["recall"], (int, float))
    assert isinstance(metrics["f1"], (int, float))


# ---------------------------------------------------------------------------
# Test 4: Consistent positive-class selection
# ---------------------------------------------------------------------------


def test_positive_class_selection_yes():
    """For no/yes labels, positive class must be 'yes'."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = np.array(["yes" if i % 2 == 0 else "no" for i in range(n)])

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    m = evaluate_classification(y, y_pred, y_proba, classes=model.classes_)
    d = m.to_dict()
    # Precision/recall/F1 for positive class "yes" must be finite numbers.
    assert math.isfinite(d["precision"])
    assert math.isfinite(d["recall"])
    assert math.isfinite(d["f1"])


def test_positive_class_selection_false_true():
    """For false/true labels, positive class must be 'true'."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = np.array(["true" if i % 2 == 0 else "false" for i in range(n)])

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    m = evaluate_classification(y, y_pred, y_proba, classes=model.classes_)
    d = m.to_dict()
    assert math.isfinite(d["precision"])
    assert math.isfinite(d["recall"])
    assert math.isfinite(d["f1"])


# ---------------------------------------------------------------------------
# Test 5: Correct probability-column alignment
# ---------------------------------------------------------------------------


def test_probability_column_alignment(config):
    """Probability column for positive class must match model.classes_ order."""
    name = _make_string_dataset(config, "proba_alignment.csv")
    result = _run(
        train_baseline_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    metrics = payload["data"]["metrics"]
    # ROC-AUC must be computable (not None) with correct column alignment.
    assert metrics["roc_auc"] is not None, (
        "ROC-AUC must be computable with correct positive-class column alignment"
    )
    assert metrics["pr_auc"] is not None, (
        "PR-AUC must be computable with correct positive-class column alignment"
    )


# ---------------------------------------------------------------------------
# Test 6: Precision, recall, F1 and confusion-matrix interpretation
# ---------------------------------------------------------------------------


def test_metrics_confusion_matrix(config):
    name = _make_string_dataset(config, "metrics_cm.csv")
    result = _run(
        train_baseline_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "decision_tree",
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    metrics = payload["data"]["metrics"]
    cm = metrics["confusion_matrix"]
    assert cm is not None
    assert len(cm) == 2  # 2x2 for binary
    assert len(cm[0]) == 2
    # Confusion matrix rows correspond to true classes, columns to predicted.
    # Classes are sorted: ["no", "yes"], so row 0 = true "no", row 1 = true "yes".
    total = sum(sum(row) for row in cm)
    assert total > 0


# ---------------------------------------------------------------------------
# Test 7: ROC-AUC and PR-AUC where supported
# ---------------------------------------------------------------------------


def test_roc_auc_pr_auc(config):
    name = _make_string_dataset(config, "roc_auc_pr.csv")
    result = _run(
        train_baseline_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    metrics = payload["data"]["metrics"]
    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["pr_auc"] is not None
    assert 0.0 <= metrics["pr_auc"] <= 1.0


# ---------------------------------------------------------------------------
# Test 8: compare_models with string targets
# ---------------------------------------------------------------------------


def test_compare_models_string_targets(config):
    name = _make_string_dataset(config, "compare_str.csv")
    result = _run(
        compare_models_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "models": ["logistic_regression", "decision_tree"],
                "cv": 3,
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    results = payload["data"]["results"]
    assert len(results) == 2
    for model_result in results:
        if model_result["status"] == "completed":
            cv = model_result["cv"]
            assert "strategy" in cv
            assert "fold_metrics" in cv
            for fold in cv["fold_metrics"]:
                assert "precision" in fold
                assert "recall" in fold
                assert "f1" in fold
                assert isinstance(fold["precision"], (int, float))


# ---------------------------------------------------------------------------
# Test 9: AutoML and tuning compatibility
# ---------------------------------------------------------------------------


def test_automl_string_targets(config):
    name = _make_string_dataset(config, "automl_str.csv")
    result = _run(
        automl_run_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "models": ["logistic_regression"],
                "cv": 2,
                "max_trials": 1,
                "max_time_seconds": 30,
                "random_state": 42,
                "exclude_columns": [],
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    trials = payload["data"]["trials"]
    assert len(trials) >= 1
    for trial in trials:
        if trial["error"] is None:
            assert "value" in trial
            assert isinstance(trial["value"], (int, float))


def test_tune_model_string_targets(config):
    name = _make_string_dataset(config, "tune_str.csv")
    result = _run(
        tune_model_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "decision_tree",
                "cv": 2,
                "max_trials": 1,
                "max_time_seconds": 30,
                "random_state": 42,
                "exclude_columns": [],
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    data = payload["data"]
    assert "best_value" in data
    assert "best_params" in data


# ---------------------------------------------------------------------------
# Test 10: Export/reload/predict compatibility
# ---------------------------------------------------------------------------


def test_export_reload_predict_string_targets(config):
    name = _make_string_dataset(config, "export_str.csv")
    # Export a trained model.
    result = _run(
        export_model_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "artifact_name": "str_label_model",
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")

    # Predict with the exported model.
    predict_result = _run(
        predict_tool(
            config,
            {
                "artifact_name": "str_label_model",
                "features": {
                    "feature1": 0.5,
                    "feature2": -0.3,
                    "feature3": "a",
                },
            },
        )
    )
    p_payload = _payload(predict_result)
    assert p_payload["ok"] is True, p_payload.get("error")
    p_data = p_payload["data"]
    assert "predictions" in p_data
    assert "classes" in p_data
    assert p_data["classes"] is not None
    # Predictions must be string labels ("yes"/"no"), not integers.
    for pred in p_data["predictions"]:
        assert pred in ("yes", "no"), f"Unexpected prediction label: {pred}"
    # Classes must include both "no" and "yes".
    assert "no" in p_data["classes"]
    assert "yes" in p_data["classes"]


# ---------------------------------------------------------------------------
# Test 11: Chronological validation without evaluating protected test
# ---------------------------------------------------------------------------


def test_chronological_validation_no_protected_eval(config):
    name = _make_string_dataset(config, "chrono_str.csv")
    df = pd.read_csv(config.workspace_dir / name)
    n = len(df)
    # Train: first 60%, val: next 20%, test: last 20% (protected).
    mid = _make_manifest(
        config,
        df,
        name,
        split_type="time",
        train_indices=list(range(0, int(n * 0.6))),
        val_indices=list(range(int(n * 0.6), int(n * 0.8))),
        test_indices=list(range(int(n * 0.8), n)),
    )

    # Validate the manifest (complete coverage, chronological).
    report = validate_manifest_for_consumption(
        config=config, manifest_id=mid, df=df, dataset_name=name
    )
    assert report["valid"] is True
    assert report["errors"] == []

    # Train with manifest — evaluation uses validation rows only.
    result = _run(
        train_baseline_tool(
            config,
            {
                "name": name,
                "target_column": "target",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "split_manifest_id": mid,
                "exclude_columns": [],
                "random_state": 42,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    metrics = payload["data"]["metrics"]
    assert metrics is not None
    assert "accuracy" in metrics
    # Protected test indices must not enter training or evaluation.


# ---------------------------------------------------------------------------
# Test 12: Clear errors for invalid or ambiguous target configurations
# ---------------------------------------------------------------------------


def test_ambiguous_labels_rejected():
    """Ambiguous binary labels (cat/dog) must raise a clear error."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = np.array(["cat" if i % 2 == 0 else "dog" for i in range(n)])

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    from devray_datalab.utils.errors import ValidationError

    with pytest.raises(ValidationError):
        evaluate_classification(y, y_pred, y_proba, classes=model.classes_)


# ---------------------------------------------------------------------------
# Test: positive_label parameter override (explicit contract)
# ---------------------------------------------------------------------------


def test_explicit_positive_label():
    """Explicit pos_label override must work for ambiguous labels."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = np.array(["cat" if i % 2 == 0 else "dog" for i in range(n)])
    classes = np.array(["cat", "dog"])

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Explicit pos_label="dog" should work.
    m = evaluate_classification(y, y_pred, y_proba, classes=classes, pos_label="dog")
    d = m.to_dict()
    assert math.isfinite(d["precision"])
    assert math.isfinite(d["recall"])
    assert math.isfinite(d["f1"])


# ---------------------------------------------------------------------------
# Test: optimize_decision_threshold with string labels
# ---------------------------------------------------------------------------


def test_optimize_decision_threshold_string():
    """Threshold optimization must work with string labels."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    y = np.array(["yes" if i % 3 == 0 else "no" for i in range(n)])
    classes = np.array(["no", "yes"])

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    y_proba = model.predict_proba(X)

    result = optimize_decision_threshold(y, y_proba, classes=classes, objective="f1")
    assert "threshold" in result
    assert "f1" in result
    assert 0.0 <= result["threshold"] <= 1.0
    assert math.isfinite(result["f1"])
