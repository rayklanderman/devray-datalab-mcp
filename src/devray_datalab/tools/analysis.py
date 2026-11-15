"""Analysis tools: analyze_correlations, analyze_data_quality."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.loader import load_dataset
from devray_datalab.utils.errors import DevRayError


def _json(data: Any) -> str:
    import json

    return json.dumps(data, default=str)


def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=_json({"ok": True, "data": data}))]


def _err(message: str, code: str = "error") -> list[TextContent]:

    return [
        TextContent(
            type="text", text=_json({"ok": False, "error": {"message": message, "code": code}})
        )
    ]


async def analyze_correlations_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Compute correlations among numerical variables."""
    name = arguments.get("name")
    if not name:
        return _err("Missing required argument: 'name'.", "validation_error")

    method = str(arguments.get("method", "pearson")).lower()
    if method not in ("pearson", "spearman"):
        return _err("method must be 'pearson' or 'spearman'.", "validation_error")

    top_n = int(arguments.get("top_n", 10))
    try:
        top_n = max(1, min(top_n, 50))
    except (TypeError, ValueError):
        top_n = 10

    try:
        df = load_dataset(config, str(name))
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load dataset: {exc}", "dataset_error")

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return _ok(
            {
                "filename": name,
                "method": method,
                "message": "Fewer than two numerical columns; correlation not applicable.",
                "correlations": [],
                "strongest_positive": None,
                "strongest_negative": None,
                "constant_columns": [],
                "unsuitable_columns": [],
            }
        )

    corr = numeric.corr(method=method)

    # Identify constant columns (zero variance).
    constant_cols = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]

    # Build list of pairwise correlations (upper triangle, excluding diagonal).
    pairs: list[dict[str, Any]] = []
    cols = numeric.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = corr.loc[c1, c2]
            if pd.isna(val):
                continue
            pairs.append(
                {
                    "column_a": c1,
                    "column_b": c2,
                    "correlation": round(float(val), 6),
                    "abs_correlation": round(abs(float(val)), 6),
                }
            )

    pairs.sort(key=lambda p: p["abs_correlation"], reverse=True)
    strongest_positive = pairs[0] if pairs else None
    strongest_negative = min(pairs, key=lambda p: p["correlation"]) if pairs else None

    return _ok(
        {
            "filename": name,
            "method": method,
            "numerical_columns": list(cols),
            "correlation_matrix": {
                c: {k: round(float(v), 6) for k, v in corr.loc[c].items()} for c in cols
            },
            "pairs": pairs[:top_n],
            "strongest_positive": strongest_positive,
            "strongest_negative": strongest_negative,
            "constant_columns": constant_cols,
            "unsuitable_columns": constant_cols,
        }
    )


async def analyze_data_quality_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Report data quality concerns without modifying the dataset."""
    name = arguments.get("name")
    if not name:
        return _err("Missing required argument: 'name'.", "validation_error")

    try:
        df = load_dataset(config, str(name))
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to load dataset: {exc}", "dataset_error")

    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing_cols = {col: int(count) for col, count in missing.items() if count > 0}
    duplicate_rows = int(df.duplicated().sum())

    # Constant columns.
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    # Duplicated columns.
    duplicated_cols = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                duplicated_cols.append(cols[j])

    # Suspected outliers via IQR on numeric columns.
    outliers: dict[str, Any] = {}
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_vals = series[(series < lower) | (series > upper)]
        if len(outlier_vals) > 0:
            outliers[col] = {
                "count": int(len(outlier_vals)),
                "percent": round(float(len(outlier_vals) / len(series) * 100), 2),
                "lower_bound": round(float(lower), 6),
                "upper_bound": round(float(upper), 6),
                "examples": [round(float(v), 6) for v in outlier_vals.head(5).tolist()],
            }

    # Recommendations.
    recommendations: list[str] = []
    if missing_cols:
        recommendations.append(
            "Impute or drop columns/rows with missing values (see missing_values)."
        )
    if duplicate_rows:
        recommendations.append("Remove duplicate rows.")
    if constant_cols:
        recommendations.append("Remove constant columns (they carry no information for modeling).")
    if duplicated_cols:
        recommendations.append("Remove duplicated columns.")
    if outliers:
        recommendations.append("Review suspected outliers; consider winsorizing or removing them.")
    if not recommendations:
        recommendations.append("No major data-quality concerns detected.")

    concerns: list[str] = []
    if missing_cols:
        concerns.append(f"Missing values in {len(missing_cols)} column(s).")
    if duplicate_rows:
        concerns.append(f"{duplicate_rows} duplicate row(s).")
    if constant_cols:
        concerns.append(f"Constant column(s): {', '.join(constant_cols)}.")
    if duplicated_cols:
        concerns.append(f"Duplicated column(s): {', '.join(duplicated_cols)}.")
    if outliers:
        concerns.append(f"Suspected outliers in {len(outliers)} numeric column(s).")

    return _ok(
        {
            "filename": name,
            "rows": n_rows,
            "columns": n_cols,
            "missing_values": missing_cols,
            "missing_values_total": int(sum(missing_cols.values())),
            "duplicate_rows": duplicate_rows,
            "constant_columns": constant_cols,
            "duplicated_columns": duplicated_cols,
            "suspected_outliers": outliers,
            "concerns": concerns,
            "recommendations": recommendations,
        }
    )
