"""Tests for cross-validation and evaluation."""

from __future__ import annotations

from devray_datalab.core.evaluator import (
    cross_validate_classification,
    cross_validate_regression,
    evaluate_classification,
    evaluate_regression,
    permutation_importance,
)


def test_evaluate_classification_binary():
    import numpy as np

    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    m = evaluate_classification(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.f1 == 1.0
    assert m.confusion_matrix == [[3, 0], [0, 3]]


def test_evaluate_classification_multiclass():
    import numpy as np

    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    m = evaluate_classification(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.f1_macro == 1.0


def test_evaluate_regression():
    import numpy as np

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    m = evaluate_regression(y_true, y_pred)
    assert m.r2 > 0.9
    assert m.mae >= 0


def test_cross_validate_classification(classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    r = cross_validate_classification(
        df,
        "purchased",
        __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression,
        {"max_iter": 1000},
        strategy="stratified",
        n_splits=5,
        random_state=42,
    )
    assert r.strategy == "StratifiedKFold"
    assert r.n_folds == 5
    assert len(r.fold_metrics) == 5
    assert len(r.fold_sizes) == 5
    assert 0 <= r.mean <= 1
    assert r.std >= 0
    assert r.explanation
    assert r.fold_metrics[0]["fold"] == 1
    assert "roc_auc" in r.fold_metrics[0]


def test_cross_validate_regression(regression_csv):
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    df = pd.read_csv(regression_csv)
    r = cross_validate_regression(
        df,
        "spend",
        LinearRegression,
        {},
        strategy="kfold",
        n_splits=5,
        random_state=42,
    )
    assert r.strategy == "KFold"
    assert r.n_folds == 5
    assert len(r.fold_metrics) == 5
    assert r.mean <= 1.0  # R^2 <= 1
    assert r.std >= 0


def test_cross_validate_invalid_n_splits(classification_csv):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(classification_csv)
    try:
        cross_validate_classification(
            df, "purchased", LogisticRegression, {"max_iter": 1000}, n_splits=1
        )
        raise AssertionError("expected ValidationError")
    except Exception:
        pass


def test_cross_validate_invalid_strategy(classification_csv):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(classification_csv)
    try:
        cross_validate_classification(
            df,
            "purchased",
            LogisticRegression,
            {"max_iter": 1000},
            strategy="bogus",
        )
        raise AssertionError("expected ValidationError")
    except Exception:
        pass


def test_permutation_importance():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(3)
    X = rng.rand(60, 3)
    # feature 0 is most important
    y = (X[:, 0] * 2 + X[:, 1] * 0.5 > 1).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    imp = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    assert "baseline_score" in imp
    assert "scoring" in imp
    assert "n_repeats" in imp
    assert imp["n_repeats"] == 5
    assert len(imp["importances"]) == 3
    # feature 0 should have the highest mean importance
    means = [f["importance_mean"] for f in imp["importances"]]
    assert means[0] == max(means)
    for f in imp["importances"]:
        assert "feature" in f
        assert "importance_mean" in f
        assert "importance_std" in f
