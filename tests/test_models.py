"""Tests for model export and prediction."""

from __future__ import annotations

import asyncio
import json

from devray_datalab.tools.models import export_model_tool, predict_tool


def run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def test_export_and_predict_classification(config, classification_csv):
    export = run(
        export_model_tool(
            config,
            {
                "artifact_name": "exported_clf",
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "model_name": "random_forest",
                "random_state": 42,
            },
        )
    )
    data = _payload(export)
    assert data["ok"] is True
    assert data["data"]["task_type"] == "classification"
    assert data["data"]["model"] == "RandomForestClassifier"
    assert data["data"]["features"] == [
        "age",
        "income",
        "credit_score",
        "category",
        "region",
    ]

    predict = run(
        predict_tool(
            config,
            {
                "artifact_name": "exported_clf",
                "features": {
                    "age": 40,
                    "income": 80000,
                    "credit_score": 720,
                    "category": "electronics",
                    "region": "east",
                },
            },
        )
    )
    pdata = _payload(predict)
    assert pdata["ok"] is True
    assert pdata["data"]["schema_valid"] is True
    assert len(pdata["data"]["predictions"]) == 1
    assert pdata["data"]["probabilities"] is not None


def test_export_and_predict_regression(config, regression_csv):
    export = run(
        export_model_tool(
            config,
            {
                "artifact_name": "exported_reg",
                "name": "sample_regression.csv",
                "target_column": "spend",
                "task_type": "regression",
                "model_name": "linear_regression",
                "random_state": 42,
            },
        )
    )
    data = _payload(export)
    assert data["ok"] is True
    assert data["data"]["task_type"] == "regression"

    predict = run(
        predict_tool(
            config,
            {
                "artifact_name": "exported_reg",
                "features": {
                    "age": 40,
                    "income": 80000,
                    "credit_score": 720,
                    "category": "electronics",
                    "region": "east",
                },
            },
        )
    )
    pdata = _payload(predict)
    assert pdata["ok"] is True
    assert len(pdata["data"]["predictions"]) == 1


def test_predict_missing_artifact(config):
    result = run(predict_tool(config, {"artifact_name": "does_not_exist", "features": {"age": 1}}))
    data = _payload(result)
    assert data["ok"] is False
    assert data["error"]["code"] == "model_error"


def test_predict_schema_mismatch(config, classification_csv):
    run(
        export_model_tool(
            config,
            {
                "artifact_name": "mismatch_clf",
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "random_state": 42,
            },
        )
    )
    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "mismatch_clf",
                "features": {"age": 40, "income": 80000},
            },
        )
    )
    data = _payload(result)
    assert data["ok"] is False
    assert data["error"]["code"] == "schema_mismatch"
