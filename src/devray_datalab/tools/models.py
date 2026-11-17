"""Model tools: export_model, predict."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from mcp.types import TextContent
from sklearn.base import BaseEstimator

from devray_datalab.config import DevRayConfig
from devray_datalab.core.preprocessing import (
    PreprocessingArtifact,
    categorical_encoder,
    learned_categories,
    transform_features,
)
from devray_datalab.core.splits import _dataset_fingerprint, validate_manifest_for_consumption
from devray_datalab.core.trainer import Trainer
from devray_datalab.tools.storage import (
    ARTIFAKIND_MODEL,
    load_artifact,
    save_artifact,
)
from devray_datalab.utils.errors import DevRayError, ModelError, sanitize_for_json


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


def _package_metadata() -> dict[str, str]:
    import importlib.metadata as md

    versions: dict[str, str] = {}
    for pkg in ("scikit-learn", "pandas", "numpy", "scipy", "joblib", "psutil", "mcp"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


# Maximum number of distinct unseen-category warnings returned per prediction
# call. Keeps responses small for large batches or high-cardinality features.
_MAX_UNSEEN_WARNINGS = 25


def _detect_unseen_categories(
    artifact: PreprocessingArtifact, df_input: pd.DataFrame
) -> tuple[list[dict[str, Any]], str]:
    """Detect categorical values in ``df_input`` not seen at fit time.

    Returns ``(warnings, handle_unknown)``. Each warning identifies the type,
    affected feature, unseen value, affected record count, and a plain-language
    message. Missing values and numeric features are skipped; they are handled
    by the fitted imputer/transformer and are not "unseen".

    The encoder's ``handle_unknown`` setting is returned so the caller can
    decide whether prediction may continue ("ignore") or must fail ("error").
    """
    warnings: list[dict[str, Any]] = []
    categorical_features = list(artifact.categorical_features or [])
    if not categorical_features:
        return warnings, "ignore"

    categories = learned_categories(artifact)
    encoder = categorical_encoder(artifact)
    handle_unknown = (
        getattr(encoder, "handle_unknown", "ignore") if encoder is not None else "ignore"
    )

    for col in categorical_features:
        if col not in df_input.columns:
            continue
        learned = set(categories.get(col, []))
        if not learned:
            continue
        value_counts: dict[str, int] = {}
        for value in df_input[col]:
            if pd.isna(value):
                continue
            key = str(value)
            if key not in learned:
                value_counts[key] = value_counts.get(key, 0) + 1
        for value, count in value_counts.items():
            if len(warnings) >= _MAX_UNSEEN_WARNINGS:
                break
            warnings.append(
                {
                    "type": "unseen_category",
                    "column": col,
                    "value": value,
                    "affected_rows": int(count),
                    "message": (
                        f"This value was not observed during model training for feature '{col}'."
                    ),
                }
            )

    return warnings, handle_unknown


async def export_model_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Save a trained model with its preprocessing pipeline and metadata.

    Expects a prior ``train_baseline`` call that returned a ``model_handle``,
    or trains on the fly if ``name``/``target_column`` are provided.
    """
    model_handle = arguments.get("model_handle")
    dataset_name = arguments.get("name")
    target_column = arguments.get("target_column")
    task_type = arguments.get("task_type")
    model_name = arguments.get("model_name")
    artifact_name = arguments.get("artifact_name")

    if not artifact_name:
        return _err("Required argument: 'artifact_name'.", "validation_error")

    exclude_columns = arguments.get("exclude_columns") or []
    split_manifest_id = arguments.get("split_manifest_id")
    dataset_fingerprint = None

    # Determine source: either an already-trained model handle, or train now.
    artifact: PreprocessingArtifact | None = None
    model: BaseEstimator | None = None
    metrics: dict[str, Any] | None = None
    train_info: dict[str, Any] | None = None

    if model_handle:
        try:
            bundle = load_artifact(config, str(model_handle), kind=ARTIFAKIND_MODEL)
        except ModelError as exc:
            return _err(exc.message, exc.code)
        if not isinstance(bundle, dict) or "model" not in bundle:
            return _err("Stored model bundle is missing the trained model.", "model_error")
        model = bundle["model"]
        artifact = bundle.get("preprocessing")
        metrics = bundle.get("metrics")
        train_info = bundle.get("train_info")
        # QA-008: preserve original dataset fingerprint provenance on re-export.
        metadata = bundle.get("metadata") or {}
        dataset_fingerprint = metadata.get("dataset_fingerprint")
    elif dataset_name and target_column and task_type and model_name:
        delimiter = arguments.get("delimiter")
        from devray_datalab.core.loader import load_dataset

        try:
            df = load_dataset(config, str(dataset_name), delimiter=delimiter)
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Failed to load dataset: {exc}", "dataset_error")
        # QA-001: enforce shared mandatory manifest validation before training.
        if split_manifest_id:
            try:
                manifest_validation = validate_manifest_for_consumption(
                    config=config,
                    manifest_id=str(split_manifest_id),
                    df=df,
                    dataset_name=str(dataset_name),
                )
            except DevRayError as exc:
                return _err(exc.message, exc.code)
            except Exception as exc:  # noqa: BLE001
                return _err(f"Manifest validation failed: {exc}", "split_error")
            if not manifest_validation["valid"]:
                return _err(
                    "Invalid split manifest: " + "; ".join(manifest_validation["errors"]),
                    "validation_error",
                )
        try:
            trainer = Trainer(config)
            result, artifact, model = trainer.train_model_with_artifact(
                df=df,
                target_column=str(target_column),
                task_type=str(task_type),
                model_name=str(model_name),
                test_size=float(arguments.get("test_size", 0.2)),
                random_state=int(arguments.get("random_state", 42)),
                stratify=bool(arguments.get("stratify", True)),
                exclude_columns=exclude_columns,
                split_manifest_id=split_manifest_id,
            )
            metrics = result.get("metrics")
            train_info = {
                "n_train": result.get("n_train"),
                "n_test": result.get("n_test"),
                "stratified": result.get("stratified"),
                "excluded_columns": result.get("excluded_columns"),
                "split_manifest_id": result.get("split_manifest_id"),
            }
            dataset_fingerprint = _dataset_fingerprint(df)
        except DevRayError as exc:
            return _err(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001
            return _err(f"Training failed: {exc}", "training_error")
    else:
        return _err(
            "Provide either 'model_handle' (a previously trained model) or "
            "'name', 'target_column', 'task_type', and 'model_name' to train and export.",
            "validation_error",
        )

    if artifact is None or model is None:
        return _err("No trained model available to export.", "model_error")

    bundle = {
        "model": model,
        "preprocessing": artifact,
        "metrics": metrics,
        "train_info": train_info,
        "feature_schema": {
            "feature_columns": artifact.feature_columns,
            "target_column": artifact.target_column,
            "task_type": artifact.task_type,
            "numeric_features": artifact.numeric_features,
            "categorical_features": artifact.categorical_features,
            "preprocessed_feature_names": [],
        },
        "hyperparameters": {},
        "metadata": {
            "created_by": "devray_datalab_mcp",
            "package_versions": _package_metadata(),
            "excluded_columns": train_info.get("excluded_columns", []),
            "split_manifest_id": train_info.get("split_manifest_id"),
            "dataset_fingerprint": dataset_fingerprint,
        },
    }
    # Capture fitted model hyperparameters where available.
    try:
        bundle["hyperparameters"] = model.get_params()
    except Exception:  # pragma: no cover - best effort
        bundle["hyperparameters"] = {}

    try:
        path = save_artifact(config, str(artifact_name), bundle, kind=ARTIFAKIND_MODEL)
    except DevRayError as exc:
        return _err(exc.message, exc.code)

    return _ok(
        {
            "artifact_name": artifact_name,
            "path": str(path),
            "task_type": artifact.task_type,
            "model": type(model).__name__,
            "features": artifact.feature_columns,
            "target": artifact.target_column,
            "excluded_columns": train_info.get("excluded_columns", []),
            "split_manifest_id": train_info.get("split_manifest_id"),
            "dataset_fingerprint": dataset_fingerprint,
            "metrics": metrics,
            "train_info": train_info,
            "note": "Artifact stored locally. Use 'predict' to generate predictions.",
        }
    )


async def predict_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Load a trained model and generate predictions on supplied features."""
    artifact_name = arguments.get("artifact_name")
    features = arguments.get("features")
    if not artifact_name:
        return _err("Required argument: 'artifact_name'.", "validation_error")
    if features is None:
        return _err(
            "Required argument: 'features' (a feature record or list of records).",
            "validation_error",
        )

    try:
        bundle = load_artifact(config, str(artifact_name), kind=ARTIFAKIND_MODEL)
    except ModelError as exc:
        return _err(exc.message, exc.code)

    if not isinstance(bundle, dict) or "model" not in bundle:
        return _err("Stored artifact is not a valid DevRay model bundle.", "model_error")

    model: BaseEstimator = bundle["model"]
    artifact: PreprocessingArtifact | None = bundle.get("preprocessing")

    if artifact is None:
        return _err("Model artifact is missing its preprocessing pipeline.", "model_error")

    import pandas as pd

    # Build input dataframe.
    if isinstance(features, dict):
        df_input = pd.DataFrame([features])
    elif isinstance(features, list) and all(isinstance(r, dict) for r in features):
        df_input = pd.DataFrame(features)
    else:
        return _err("'features' must be an object or a list of objects.", "validation_error")

    expected = artifact.feature_columns
    missing = [c for c in expected if c not in df_input.columns]
    extra = [c for c in df_input.columns if c not in expected]
    if missing:
        return _err(
            f"Missing required feature(s): {', '.join(missing)}.",
            "schema_mismatch",
        )
    if extra:
        # Warn but ignore extra columns rather than failing hard.
        pass

    # Detect previously unseen categorical values before transforming. This
    # does not alter predictions, probabilities, or fitted preprocessing.
    warnings, handle_unknown = _detect_unseen_categories(artifact, df_input)
    if handle_unknown != "ignore" and warnings:
        summary = "; ".join(
            f"{w['column']}={w['value']} ({w['affected_rows']} row(s))"
            for w in warnings[:_MAX_UNSEEN_WARNINGS]
        )
        return _err(
            "Prediction contains unseen categorical values the fitted encoder "
            f"cannot handle: {summary}",
            "unseen_category",
        )

    try:
        X = transform_features(artifact, df_input)
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Feature transformation failed: {exc}", "model_error")

    try:
        predictions = model.predict(X)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Prediction failed: {exc}", "model_error")

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            probabilities = proba.tolist()
        except Exception:  # pragma: no cover - best effort
            probabilities = None
    classes = None
    if hasattr(model, "classes_"):
        classes = [str(c) for c in model.classes_]

    return _ok(
        {
            "artifact_name": artifact_name,
            "model": type(model).__name__,
            "n_predictions": int(len(predictions)),
            "predictions": [
                float(p) if isinstance(p, (np.floating, float)) else p for p in predictions
            ],
            "probabilities": probabilities,
            "classes": classes,
            "schema_valid": True,
            "ignored_extra_columns": extra,
            "warnings": warnings,
        }
    )
