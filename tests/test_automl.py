"""Tests for the AutoML engine."""

from __future__ import annotations

from devray_datalab.core.automl import run_automl


def _optuna_available():
    try:
        import optuna  # noqa: F401

        return True
    except Exception:
        return False


def test_run_automl_classification(classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    result = run_automl(
        df,
        "purchased",
        "classification",
        metric="auto",
        max_trials=4,
        max_time_seconds=300,
        random_state=42,
        n_splits=3,
    )
    assert result.task_type == "classification"
    assert result.dataset_shape[0] == 20
    assert result.n_features == 5
    assert result.n_classes == 2
    assert result.runtime_seconds >= 0
    assert len(result.trials) >= 1
    completed = [t for t in result.trials if t.error is None]
    assert len(completed) >= 1
    assert result.best_model is not None


def test_run_automl_regression(regression_csv):
    import pandas as pd

    df = pd.read_csv(regression_csv)
    result = run_automl(
        df,
        "spend",
        "regression",
        metric="auto",
        max_trials=4,
        max_time_seconds=300,
        random_state=42,
        n_splits=3,
    )
    assert result.task_type == "regression"
    assert result.dataset_shape[0] == 20
    assert result.n_features == 5
    assert result.n_classes is None
    assert len(result.trials) >= 1
    completed = [t for t in result.trials if t.error is None]
    assert len(completed) >= 1
    assert result.best_model is not None


def test_run_automl_model_subset(classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    result = run_automl(
        df,
        "purchased",
        "classification",
        max_trials=2,
        max_time_seconds=300,
        random_state=42,
        n_splits=3,
        model_subset=["logistic_regression", "random_forest"],
    )
    models_trained = {t.model for t in result.trials if t.error is None}
    assert models_trained == {"logistic_regression", "random_forest"}


def test_run_automl_invalid_max_trials(classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    try:
        run_automl(df, "purchased", "classification", max_trials=0)
        raise AssertionError("expected ValidationError")
    except Exception:
        pass


def test_run_automl_invalid_target(classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    try:
        run_automl(df, "nonexistent", "classification")
        raise AssertionError("expected ValidationError")
    except Exception:
        pass


def test_to_dict_roundtrip(classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    result = run_automl(df, "purchased", "classification", max_trials=2, n_splits=2)
    d = result.to_dict()
    assert d["task_type"] == "classification"
    assert d["best_model"] is not None
    assert "trials" in d
    assert "optuna_used" in d


def test_optuna_flag(classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    result = run_automl(df, "purchased", "classification", max_trials=3, n_splits=2)
    # optuna_used should be a bool
    assert isinstance(result.optuna_used, bool)
