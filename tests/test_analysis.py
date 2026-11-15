"""Tests for correlation and data-quality analysis."""

from __future__ import annotations

import asyncio
import json

from devray_datalab.tools.analysis import (
    analyze_correlations_tool,
    analyze_data_quality_tool,
)


def run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def test_analyze_correlations(config, classification_csv):
    result = run(
        analyze_correlations_tool(
            config, {"name": "sample_classification.csv", "method": "pearson", "top_n": 5}
        )
    )
    data = _payload(result)
    assert data["ok"] is True
    body = data["data"]
    assert body["method"] == "pearson"
    assert "age" in body["numerical_columns"]
    assert "income" in body["numerical_columns"]
    assert "credit_score" in body["numerical_columns"]
    assert "purchased" in body["numerical_columns"]
    assert isinstance(body["pairs"], list)
    assert body["strongest_positive"] is not None
    assert "correlation" in body["strongest_positive"]
    assert "constant_columns" in body


def test_analyze_correlations_spearman(config, classification_csv):
    result = run(
        analyze_correlations_tool(
            config, {"name": "sample_classification.csv", "method": "spearman"}
        )
    )
    data = _payload(result)
    assert data["ok"] is True
    assert data["data"]["method"] == "spearman"


def test_analyze_correlations_invalid_method(config, classification_csv):
    result = run(
        analyze_correlations_tool(
            config, {"name": "sample_classification.csv", "method": "kendall"}
        )
    )
    data = _payload(result)
    assert data["ok"] is False
    assert data["error"]["code"] == "validation_error"


def test_analyze_correlations_missing_file(config):
    result = run(analyze_correlations_tool(config, {"name": "nope.csv"}))
    data = _payload(result)
    assert data["ok"] is False


def test_analyze_data_quality(config, classification_csv):
    result = run(analyze_data_quality_tool(config, {"name": "sample_classification.csv"}))
    data = _payload(result)
    assert data["ok"] is True
    body = data["data"]
    assert body["rows"] == 20
    assert body["columns"] == 6
    assert "missing_values" in body
    assert "duplicate_rows" in body
    assert "constant_columns" in body
    assert "suspected_outliers" in body
    assert "recommendations" in body
    assert isinstance(body["concerns"], list)
