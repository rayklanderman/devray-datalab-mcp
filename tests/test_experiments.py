"""Tests for experiment tracking and calibration/threshold optimization."""

from __future__ import annotations

from devray_datalab.core.evaluator import (
    calibrate_model,
    optimize_decision_threshold,
)
from devray_datalab.tools.experiments import (
    compare_experiments_tool,
    get_experiment_tool,
    list_experiments_tool,
    list_models_tool,
)


def _payload(result):
    import json

    return json.loads(result[0].text)["data"]


def test_calibrate_model():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(42)
    X = rng.rand(80, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X[:60], y[:60])
    calibrated = calibrate_model(model, X[60:], y[60:], method="sigmoid", cv=3)
    assert hasattr(calibrated, "predict_proba")
    proba = calibrated.predict_proba(X[:5])
    assert proba.shape == (5, 2)


def test_calibrate_model_invalid_method():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.random.RandomState(1).rand(20, 2)
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    try:
        calibrate_model(model, X, y, method="bogus")
        raise AssertionError("expected ValidationError")
    except Exception:
        pass


def test_optimize_decision_threshold():
    import numpy as np

    rng = np.random.RandomState(7)
    y = np.array([0] * 40 + [1] * 60)
    proba = np.concatenate([rng.rand(40) * 0.4, rng.rand(60) * 0.6 + 0.4])
    res = optimize_decision_threshold(y, proba, objective="f1")
    assert 0 < res["threshold"] < 1
    assert "f1" in res
    assert "precision" in res
    assert "recall" in res


def test_optimize_decision_threshold_constraints():
    import numpy as np

    rng = np.random.RandomState(3)
    y = np.array([0] * 50 + [1] * 50)
    proba = np.concatenate([rng.rand(50) * 0.5, rng.rand(50) * 0.5 + 0.5])
    res = optimize_decision_threshold(y, proba, objective="f1", precision_min=0.9)
    assert res["precision"] >= 0.9


def test_list_get_compare_experiments(config):
    import asyncio

    async def main():
        r = await list_experiments_tool(config, {"limit": 10})
        data = _payload(r)
        assert data["experiments"] == []

        r = await get_experiment_tool(config, {"experiment_id": "nope"})
        assert not json.loads(r[0].text)["ok"]

        r = await compare_experiments_tool(config, {"experiment_ids": []})
        data = _payload(r)
        assert data["experiments"] == []
        assert data["missing"] == []

        r = await list_models_tool(config, {})
        data = _payload(r)
        assert data["models"] == []

    import json

    asyncio.run(main())


def test_cross_validate_end_to_end(config, classification_csv):
    import asyncio

    from devray_datalab.tools.training import compare_models_tool

    async def main():
        return await compare_models_tool(
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "models": ["logistic_regression", "random_forest"],
                "metric": "f1",
                "cv": 3,
            },
        )

    r = asyncio.run(main())
    data = _payload(r)
    assert data["models_compared"] == 2
    assert data["best_model"] is not None
    for res in data["results"]:
        if res["status"] == "completed":
            assert "cv" in res
            assert res["cv"]["n_folds"] == 3
            assert len(res["cv"]["fold_metrics"]) == 3


def test_experiment_recorded_after_training(config, classification_csv):
    import asyncio

    from devray_datalab.tools.training import train_baseline_tool

    async def main():
        return await train_baseline_tool(
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "random_state": 42,
            },
        )

    r = asyncio.run(main())
    data = _payload(r)
    assert data["metrics"]["accuracy"] >= 0

    lr = asyncio.run(list_experiments_tool(config, {"limit": 10}))
    ld = _payload(lr)
    assert len(ld["experiments"]) >= 1
    exp = ld["experiments"][0]
    assert exp["algorithm"] == "logistic_regression"
    assert exp["task_type"] == "classification"
    assert exp["status"] == "completed"
    assert exp["experiment_id"]

    gr = asyncio.run(get_experiment_tool(config, {"experiment_id": exp["experiment_id"]}))
    gd = _payload(gr)
    assert gd["experiment_id"] == exp["experiment_id"]
