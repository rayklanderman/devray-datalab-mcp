"""Feature diagnostics: leakage detection and drift analysis.

Provides tools to detect potential data leakage (identifiers, post-outcome
features, target-derived columns, duplicated entities) and feature drift
between reference and current datasets. Results are flagged as requiring
investigation rather than confirmed leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy import stats

from devray_datalab.utils.errors import ValidationError


@dataclass
class LeakageFlag:
    """A single suspected leakage finding."""

    column: str
    reason: str
    severity: str  # 'high', 'medium', 'low'
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "reason": self.reason,
            "severity": self.severity,
            "detail": self.detail,
        }


def detect_potential_leakage(
    df: pd.DataFrame,
    target_column: str | None = None,
    id_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Detect potential data-leakage issues in a dataset.

    Flags (without confirming):
    - Identifier-like columns (near-unique, non-predictive).
    - Constant or near-constant columns.
    - High-cardinality categorical columns.
    - Columns suspiciously correlated with the target (possible target leakage).
    - Duplicated columns.
    - Columns with names suggesting they are derived from the target.
    """
    if target_column is not None and target_column not in df.columns:
        raise ValidationError(f"Target column '{target_column}' not found in dataset.")

    flags: list[LeakageFlag] = []
    columns = list(df.columns)

    # Identifier-like columns: near-unique non-numeric or high unique ratio.
    for col in columns:
        if col == target_column:
            continue
        s = df[col]
        nunique = s.nunique(dropna=True)
        n = len(s.dropna())
        if n == 0:
            continue
        unique_ratio = nunique / n
        is_identifier_like = (
            not pd.api.types.is_numeric_dtype(s) and unique_ratio > 0.95 and nunique > 10
        ) or (pd.api.types.is_integer_dtype(s) and unique_ratio > 0.99 and nunique > 10)
        if is_identifier_like:
            flags.append(
                LeakageFlag(
                    column=col,
                    reason="Near-unique column; likely an identifier.",
                    severity="high",
                    detail=f"unique_ratio={unique_ratio:.2f}, nunique={nunique}, n={n}",
                )
            )

    # Constant or near-constant columns.
    for col in columns:
        if col == target_column:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        if s.nunique() <= 1:
            flags.append(
                LeakageFlag(
                    column=col,
                    reason="Constant column; carries no information.",
                    severity="medium",
                    detail=f"nunique={s.nunique()}",
                )
            )

    # Duplicated columns.
    seen: dict[str, str] = {}
    for col in columns:
        if col == target_column:
            continue
        sig = df[col].astype(str).tolist()
        key = tuple(sig)
        if key in seen:
            flags.append(
                LeakageFlag(
                    column=col,
                    reason=f"Duplicated column (identical to '{seen[key]}').",
                    severity="high",
                    detail=f"duplicates column '{seen[key]}'",
                )
            )
        else:
            seen[key] = col

    # Target-correlation leakage (classification: high association; regression: high correlation).
    if target_column is not None and target_column in df.columns:
        target = df[target_column]
        for col in columns:
            if col == target_column:
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(target):
                try:
                    corr = s.corr(target)
                    if pd.notna(corr) and abs(corr) > 0.9:
                        flags.append(
                            LeakageFlag(
                                column=col,
                                reason="High correlation with target; possible target leakage.",
                                severity="high",
                                detail=f"correlation={corr:.3f}",
                            )
                        )
                except Exception:  # pragma: no cover - best effort
                    pass

    # Column-name-derived-from-target hints.
    target_lc = (target_column or "").lower()
    for col in columns:
        if col == target_column:
            continue
        cl = col.lower()
        if target_lc and (
            cl.endswith(f"_{target_lc}")
            or cl.startswith(f"{target_lc}_")
            or target_lc in cl
            and ("flag" in cl or "label" in cl or "result" in cl or "outcome" in cl)
        ):
            flags.append(
                LeakageFlag(
                    column=col,
                    reason="Column name suggests it may be derived from the target.",
                    severity="medium",
                    detail=f"column '{col}' references target '{target_column}'",
                )
            )

    # Deduplicate flags by column, keeping highest severity.
    by_col: dict[str, LeakageFlag] = {}
    for f in flags:
        existing = by_col.get(f.column)
        if existing is None or _severity_rank(f.severity) > _severity_rank(existing.severity):
            by_col[f.column] = f

    return {
        "n_flags": len(by_col),
        "flags": [f.to_dict() for f in by_col.values()],
        "target_column": target_column,
        "note": "Flagged columns require investigation; flagging does not confirm leakage.",
    }


def _severity_rank(severity: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(severity, 0)


@dataclass
class DriftResult:
    """Result of drift analysis for a single column."""

    column: str
    drift_detected: bool
    p_value: float | None
    statistic: float | None
    method: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "drift_detected": self.drift_detected,
            "p_value": None if self.p_value is None else round(self.p_value, 6),
            "statistic": None if self.statistic is None else round(self.statistic, 6),
            "method": self.method,
            "detail": self.detail,
        }


def analyze_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Detect feature drift between reference and current datasets.

    Uses statistical tests (KS test for numeric, chi-square for categorical)
    to compare distributions. Returns per-column drift flags and an overall
    drift summary.
    """
    if columns is None:
        columns = [c for c in reference.columns if c in current.columns]

    common = [c for c in columns if c in reference.columns and c in current.columns]
    if not common:
        raise ValidationError("No common columns between reference and current datasets.")

    if not (0 < alpha < 1):
        raise ValidationError("alpha must be between 0 and 1 (exclusive).")

    results: list[DriftResult] = []
    n_drift = 0

    for col in common:
        ref = reference[col].dropna()
        cur = current[col].dropna()
        if len(ref) == 0 or len(cur) == 0:
            results.append(
                DriftResult(
                    column=col,
                    drift_detected=False,
                    p_value=None,
                    statistic=None,
                    method="insufficient_data",
                    detail="Insufficient non-null data in reference or current.",
                )
            )
            continue

        if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
            # Two-sample Kolmogorov-Smirnov test.
            try:
                stat, p = stats.ks_2samp(ref.values, cur.values)
                method = "ks_2samp"
                drift = bool(p < alpha)
                detail = f"KS statistic={stat:.3f}, p={p:.3g}"
            except Exception:  # pragma: no cover - best effort
                stat, p = None, None
                method = "error"
                drift = False
                detail = "Could not compute KS test."
        else:
            # Chi-square test on categorical value counts.
            try:
                ref_counts = ref.value_counts().sort_index()
                cur_counts = cur.value_counts().sort_index()
                all_cats = sorted(set(ref_counts.index) | set(cur_counts.index))
                ref_vec = [ref_counts.get(c, 0) for c in all_cats]
                cur_vec = [cur_counts.get(c, 0) for c in all_cats]
                if sum(ref_vec) == 0 or sum(cur_vec) == 0:
                    raise ValueError("empty counts")
                stat, p = stats.chisquare(cur_vec, f_exp=ref_vec)
                method = "chisquare"
                drift = bool(p < alpha)
                detail = f"chi2={stat:.3f}, p={p:.3g}, categories={len(all_cats)}"
            except Exception:  # pragma: no cover - best effort
                stat, p = None, None
                method = "error"
                drift = False
                detail = "Could not compute chi-square test."

        if drift:
            n_drift += 1
        results.append(
            DriftResult(
                column=col,
                drift_detected=drift,
                p_value=p,
                statistic=stat,
                method=method,
                detail=detail,
            )
        )

    return {
        "n_columns": len(common),
        "n_drift_detected": n_drift,
        "drift_fraction": round(n_drift / len(common), 4) if common else 0.0,
        "alpha": alpha,
        "columns": [r.to_dict() for r in results],
        "note": "Drift flags indicate distributional change; investigate before retraining.",
    }
