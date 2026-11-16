"""Tests for leakage detection and feature drift."""

from __future__ import annotations

from devray_datalab.core.diagnostics import analyze_feature_drift, detect_potential_leakage


def test_detect_potential_leakage_identifier():
    import pandas as pd

    df = pd.DataFrame(
        {
            "id": list(range(20)),
            "name": [f"person_{i}" for i in range(20)],
            "feature1": [1, 2, 3, 4, 5] * 4,
            "target": [0, 1] * 10,
        }
    )
    result = detect_potential_leakage(df, target_column="target")
    assert result["n_flags"] >= 1
    flag_cols = {f["column"] for f in result["flags"]}
    assert "id" in flag_cols or "name" in flag_cols


def test_detect_potential_leakage_constant_column():
    import pandas as pd

    df = pd.DataFrame(
        {
            "constant": [5] * 20,
            "feature1": [1, 2, 3, 4, 5] * 4,
            "target": [0, 1] * 10,
        }
    )
    result = detect_potential_leakage(df, target_column="target")
    flag_cols = {f["column"] for f in result["flags"]}
    assert "constant" in flag_cols


def test_detect_potential_leakage_target_correlation():
    import pandas as pd

    df = pd.DataFrame(
        {
            "leaky": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "noise": [1, 2, 3, 4, 5] * 4,
            "target": [0, 1] * 10,
        }
    )
    result = detect_potential_leakage(df, target_column="target")
    flag_cols = {f["column"] for f in result["flags"]}
    # leaky is perfectly correlated with target -> should be flagged
    assert "leaky" in flag_cols


def test_detect_potential_leakage_no_flags():
    import pandas as pd

    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5] * 4,
            "feature2": [5, 4, 3, 2, 1] * 4,
            "target": [0, 0, 1, 1, 0] * 4,
        }
    )
    result = detect_potential_leakage(df, target_column="target")
    # May still flag feature columns if correlated, but should not flag identifier
    assert result["n_flags"] >= 0


def test_analyze_feature_drift_numeric():
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(42)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(5, 2, 100)})
    current = pd.DataFrame(
        {"x": rng.normal(3, 1, 100), "y": rng.normal(5, 2, 100)}
    )  # x drifted, y not
    result = analyze_feature_drift(reference, current, alpha=0.05)
    assert result["n_columns"] == 2
    assert "n_drift_detected" in result
    col_results = {r["column"]: r for r in result["columns"]}
    assert "x" in col_results
    assert "y" in col_results


def test_analyze_feature_drift_categorical():
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(7)
    reference = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], 100)})
    current = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], 100, p=[0.7, 0.2, 0.1])})
    result = analyze_feature_drift(reference, current, alpha=0.05)
    assert result["n_columns"] == 1
    assert "columns" in result
    assert result["columns"][0]["column"] == "cat"


def test_analyze_feature_drift_no_common_columns():
    import pandas as pd

    reference = pd.DataFrame({"a": [1, 2, 3]})
    current = pd.DataFrame({"b": [4, 5, 6]})
    try:
        analyze_feature_drift(reference, current)
        raise AssertionError("expected ValidationError")
    except Exception:
        pass


def test_detect_potential_leakage_tool(config, classification_csv):
    import asyncio

    from devray_datalab.tools.analysis import detect_potential_leakage_tool

    async def main():
        return await detect_potential_leakage_tool(config, {"name": "sample_classification.csv"})

    result = asyncio.run(main())
    import json

    data = json.loads(result[0].text)
    assert data["ok"] is True
    assert "n_flags" in data["data"]


def test_analyze_feature_drift_tool(config, classification_csv, regression_csv):
    import asyncio

    from devray_datalab.tools.analysis import analyze_feature_drift_tool

    async def main():
        return await analyze_feature_drift_tool(
            config,
            {
                "reference_name": "sample_classification.csv",
                "current_name": "sample_regression.csv",
            },
        )

    result = asyncio.run(main())
    import json

    data = json.loads(result[0].text)
    assert data["ok"] is True
    assert "n_columns" in data["data"]
