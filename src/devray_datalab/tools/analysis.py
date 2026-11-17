"""Analysis tools: analyze_correlations, analyze_data_quality."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.diagnostics import analyze_feature_drift, detect_potential_leakage
from devray_datalab.core.loader import load_dataset
from devray_datalab.utils.errors import DevRayError, sanitize_for_json


def _json(data: Any) -> str:
    import json

    return json.dumps(sanitize_for_json(data), default=str, allow_nan=False)


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

    delimiter = arguments.get("delimiter")

    try:
        df = load_dataset(config, str(name), delimiter=delimiter)
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
    positives = [p for p in pairs if p["correlation"] > 0]
    negatives = [p for p in pairs if p["correlation"] < 0]
    strongest_positive = max(positives, key=lambda p: p["correlation"]) if positives else None
    strongest_negative = min(negatives, key=lambda p: p["correlation"]) if negatives else None

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

    delimiter = arguments.get("delimiter")

    try:
        df = load_dataset(config, str(name), delimiter=delimiter)
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


async def detect_potential_leakage_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Detect potential data-leakage issues in a dataset.

    Flags identifiers, constant/high-cardinality columns, target-correlated
    columns, duplicated columns, and target-derived column names. Flagged
    columns require investigation; flagging does not confirm leakage.
    """
    name = arguments.get("name")
    if not name:
        return _err("Required argument: 'name' (dataset filename).", "validation_error")

    target_column = arguments.get("target_column")
    delimiter = arguments.get("delimiter")

    try:
        from devray_datalab.core.loader import load_dataset

        try:
            df = load_dataset(config, str(name), delimiter=delimiter)
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Failed to load dataset: {exc}", "dataset_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    try:
        result = detect_potential_leakage(df, target_column=target_column)
    except (DevRayError, ValueError) as exc:
        return _err(str(exc), exc.code if isinstance(exc, DevRayError) else "validation_error")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Leakage detection failed: {exc}", "analysis_error")

    return _ok(result)


async def analyze_feature_drift_tool(
    config: DevRayConfig,
    arguments: dict,
) -> list[TextContent]:
    """Detect feature drift between a reference and current dataset.

    Uses KS test (numeric) and chi-square test (categorical) to compare
    distributions. Returns per-column drift flags and an overall summary.
    """
    reference_name = arguments.get("reference_name")
    current_name = arguments.get("current_name")
    if not reference_name or not current_name:
        return _err(
            "Required arguments: 'reference_name' and 'current_name'.",
            "validation_error",
        )

    alpha = float(arguments.get("alpha", 0.05))
    columns = arguments.get("columns")
    reference_delimiter = arguments.get("reference_delimiter")
    current_delimiter = arguments.get("current_delimiter")

    try:
        from devray_datalab.core.loader import load_dataset

        try:
            reference = load_dataset(config, str(reference_name), delimiter=reference_delimiter)
            current = load_dataset(config, str(current_name), delimiter=current_delimiter)
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Failed to load dataset: {exc}", "dataset_error")
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    try:
        result = analyze_feature_drift(
            reference=reference,
            current=current,
            columns=columns,
            alpha=alpha,
        )
    except (DevRayError, ValueError) as exc:
        return _err(str(exc), exc.code if isinstance(exc, DevRayError) else "validation_error")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Drift analysis failed: {exc}", "analysis_error")

    return _ok(result)
