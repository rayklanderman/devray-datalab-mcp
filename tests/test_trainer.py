"""Tests for training and model comparison."""

from __future__ import annotations

import asyncio
import json

import pandas as pd

from devray_datalab.tools.training import (
    compare_models_tool,
    train_baseline_tool,
)


def run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def test_train_baseline_classification(config, classification_csv):
    result = run(
        train_baseline_tool(
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "test_size": 0.25,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True
    body = data["data"]
    assert body["task_type"] == "classification"
    assert body["model"] == "logistic_regression"
    assert "metrics" in body
    m = body["metrics"]
    assert "accuracy" in m and "precision" in m and "recall" in m and "f1" in m
    assert 0 <= m["accuracy"] <= 1
    assert body["n_train"] + body["n_test"] == 20


def test_train_baseline_regression(config, regression_csv):
    result = run(
        train_baseline_tool(
            config,
            {
                "name": "sample_regression.csv",
                "target_column": "spend",
                "task_type": "regression",
                "model_name": "linear_regression",
                "test_size": 0.25,
                "random_state": 42,
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True
    body = data["data"]
    assert body["task_type"] == "regression"
    m = body["metrics"]
    assert "mae" in m and "rmse" in m and "r2" in m


def test_train_baseline_invalid_model(config, classification_csv):
    result = run(
        train_baseline_tool(
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "model_name": "nonexistent_model",
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is False
    assert data["error"]["code"] == "validation_error"


def test_train_baseline_missing_target(config, classification_csv):
    result = run(
        train_baseline_tool(
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "nonexistent",
                "task_type": "classification",
                "model_name": "logistic_regression",
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is False


def test_train_baseline_single_class_rejected(config, classification_csv):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 0, 0]})
    ws = config.workspace_dir / "single_class.csv"
    df.to_csv(ws, index=False)
    result = run(
        train_baseline_tool(
            config,
            {
                "name": "single_class.csv",
                "target_column": "y",
                "task_type": "classification",
                "model_name": "logistic_regression",
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is False


def test_compare_models(config, classification_csv):
    result = run(
        compare_models_tool(
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "models": ["logistic_regression", "decision_tree", "random_forest"],
                "metric": "f1",
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is True
    body = data["data"]
    assert body["models_compared"] == 3
    assert body["best_model"] is not None
    assert body["selection_metric"] == "f1"
    statuses = {r["status"] for r in body["results"]}
    assert statuses == {"completed"}
