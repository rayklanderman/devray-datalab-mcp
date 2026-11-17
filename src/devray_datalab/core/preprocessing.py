"""Preprocessing pipelines built from scikit-learn transformers.

Transformations are fitted on training data only and preserved for later
prediction, preventing data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from devray_datalab.utils.errors import ValidationError


@dataclass
class PreprocessingArtifact:
    """A fitted preprocessing pipeline plus the schema it was fitted on."""

    preprocessor: ColumnTransformer
    feature_columns: list[str]
    target_column: str
    task_type: str
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    categorical_categories: dict[str, list[str]] = field(default_factory=dict)


def _infer_feature_types(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical lists."""
    if target_column not in df.columns:
        raise ValidationError(f"Target column '{target_column}' not found in dataset.")
    features = [c for c in df.columns if c != target_column]

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    return numeric_features, categorical_features


def build_preprocessor(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    impute_strategy: str = "mean",
    scale_numeric: bool = True,
    encode_categorical: bool = True,
) -> PreprocessingArtifact:
    """Build and fit a preprocessing pipeline on ``df`` (training data only).

    - Numeric columns: impute (median for robustness) then optionally standard-scale.
    - Categorical columns: one-hot encode with unknown categories ignored.
    """
    numeric_features, categorical_features = _infer_feature_types(df, target_column)

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_features:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=numeric_steps), numeric_features))

    if categorical_features:
        encode_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="most_frequent"))]
        if encode_categorical:
            encode_steps.append(
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            )
        transformers.append(("cat", Pipeline(steps=encode_steps), categorical_features))

    if not transformers:
        raise ValidationError("No feature columns available for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    feature_columns = [c for c in df.columns if c != target_column]
    preprocessor.fit(df[feature_columns])

    # Capture the learned category vocabulary per categorical column so
    # prediction can detect previously unseen values without re-reading the
    # original training dataset.
    categorical_categories: dict[str, list[str]] = {}
    cat_transformer = preprocessor.named_transformers_.get("cat")
    if cat_transformer is not None:
        onehot = getattr(cat_transformer, "named_steps", {}).get("onehot")
        if onehot is not None and getattr(onehot, "categories_", None) is not None:
            for col, cats in zip(categorical_features, onehot.categories_, strict=False):
                categorical_categories[col] = [str(c) for c in cats]

    return PreprocessingArtifact(
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        target_column=target_column,
        task_type=task_type,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_categories=categorical_categories,
    )


def transform_features(artifact: PreprocessingArtifact, df: pd.DataFrame) -> np.ndarray:
    """Transform a dataframe using a fitted preprocessing artifact."""
    missing = [c for c in artifact.feature_columns if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing feature columns: {', '.join(missing)}")
    return artifact.preprocessor.transform(df[artifact.feature_columns])


def get_feature_names_out(artifact: PreprocessingArtifact) -> list[str]:
    """Return the output feature names after preprocessing."""
    try:
        names = artifact.preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:  # pragma: no cover - fallback for older sklearn
        return list(artifact.feature_columns)


def categorical_encoder(artifact: PreprocessingArtifact):
    """Return the fitted OneHotEncoder for categorical features, or None.

    Falls back to inspecting the fitted ColumnTransformer when the artifact
    does not carry cached category metadata (e.g. artifacts serialized before
    this field existed).
    """
    cat_transformer = artifact.preprocessor.named_transformers_.get("cat")
    if cat_transformer is None:
        return None
    return getattr(cat_transformer, "named_steps", {}).get("onehot")


def learned_categories(artifact: PreprocessingArtifact) -> dict[str, list[str]]:
    """Return the learned category vocabulary per categorical column.

    Prefers the cached ``categorical_categories`` metadata on the artifact;
    otherwise extracts it from the fitted OneHotEncoder. Missing values are
    never included in the vocabulary.
    """
    cached = getattr(artifact, "categorical_categories", None) or {}
    if cached:
        return {str(k): [str(c) for c in v] for k, v in cached.items()}

    encoder = categorical_encoder(artifact)
    if encoder is None or getattr(encoder, "categories_", None) is None:
        return {}
    cat_features = list(artifact.categorical_features)
    return {
        str(col): [str(c) for c in cats]
        for col, cats in zip(cat_features, encoder.categories_, strict=False)
    }
