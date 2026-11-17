"""Unseen-category detection tests for the predict tool.

Uses small synthetic fixtures (not the real Bank Marketing dataset).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from devray_datalab.core.preprocessing import (
    PreprocessingArtifact,
    build_preprocessor,
    categorical_encoder,
    learned_categories,
)
from devray_datalab.core.trainer import Trainer
from devray_datalab.tools.models import predict_tool
from devray_datalab.tools.storage import (
    ARTIFAKIND_MODEL,
    load_artifact,
    save_artifact,
)


def run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def _data(result):
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    return payload["data"]


def _make_df(n=60):
    """Synthetic classification fixture with a categorical 'job' feature."""
    np.random.seed(3)
    df = pd.DataFrame(
        {
            "job": np.random.choice(["admin.", "blue-collar", "student"], size=n),
            "edu": np.random.choice(["basic", "secondary"], size=n),
            "age": np.random.randint(18, 70, size=n),
            "balance": np.random.randint(-2000, 5000, size=n),
        }
    )
    df["y"] = ((df["age"] > 40) & (df["balance"] > 0)).astype(int)
    return df


def _write_csv(config, name, df):
    path = config.workspace_dir / name
    df.to_csv(path, index=False)
    return path


def _save_model_artifact(config, artifact_name, preprocessing, model):
    bundle = {
        "model": model,
        "preprocessing": preprocessing,
        "metrics": {},
        "train_info": {},
        "feature_schema": {
            "feature_columns": preprocessing.feature_columns,
            "target_column": preprocessing.target_column,
            "task_type": preprocessing.task_type,
            "numeric_features": preprocessing.numeric_features,
            "categorical_features": preprocessing.categorical_features,
            "preprocessed_feature_names": [],
        },
        "hyperparameters": {},
        "metadata": {"created_by": "test", "package_versions": {}},
    }
    return save_artifact(config, artifact_name, bundle, kind=ARTIFAKIND_MODEL)


def _train_and_export(config, artifact_name, df, model_name="logistic_regression"):
    trainer = Trainer(config)
    result, artifact, model = trainer.train_model_with_artifact(
        df=df,
        target_column="y",
        task_type="classification",
        model_name=model_name,
        random_state=42,
    )
    _save_model_artifact(config, artifact_name, artifact, model)
    return artifact, model


# ---------------------------------------------------------------------------
# Learned-category metadata
# ---------------------------------------------------------------------------


def test_learned_categories_recorded(config):
    """build_preprocessor records per-column learned categories."""
    df = _make_df()
    artifact = build_preprocessor(
        df.assign(y=df["y"]),
        target_column="y",
        task_type="classification",
    )
    cats = learned_categories(artifact)
    assert set(cats) == {"job", "edu"}
    assert "admin." in cats["job"]
    assert "basic" in cats["edu"]
    # Numeric columns have no entry.
    assert "age" not in cats
    assert "balance" not in cats


def test_learned_categories_survive_export_reload(config):
    """Learned categories are available after joblib round-trip."""
    df = _make_df()
    _write_csv(config, "rt.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "rt.csv")
    artifact, model = _train_and_export(config, "rt_art", df_disk)

    loaded = load_artifact(config, "rt_art", kind=ARTIFAKIND_MODEL)
    cats = learned_categories(loaded["preprocessing"])
    assert set(cats) == {"job", "edu"}
    assert "admin." in cats["job"]
    assert categorical_encoder(loaded["preprocessing"]).handle_unknown == "ignore"


# ---------------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------------


def test_predict_known_categories_no_warning(config):
    """Recognized categories produce predictions with no warnings."""
    df = _make_df()
    _write_csv(config, "known.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "known.csv")
    _train_and_export(config, "known_art", df_disk)

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "known_art",
                "features": {
                    "job": "admin.",
                    "edu": "secondary",
                    "age": 40,
                    "balance": 1000,
                },
            },
        )
    )
    data = _data(result)
    assert data["schema_valid"] is True
    assert len(data["predictions"]) == 1
    assert data["warnings"] == []


def test_predict_one_unseen_category_warns(config):
    """An unseen categorical value produces a single warning."""
    df = _make_df()
    _write_csv(config, "one.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "one.csv")
    _train_and_export(config, "one_art", df_disk)

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "one_art",
                "features": {
                    "job": "data_scientist",
                    "edu": "secondary",
                    "age": 40,
                    "balance": 1000,
                },
            },
        )
    )
    data = _data(result)
    assert data["schema_valid"] is True
    assert len(data["predictions"]) == 1
    warnings = data["warnings"]
    assert len(warnings) == 1
    w = warnings[0]
    assert w["type"] == "unseen_category"
    assert w["column"] == "job"
    assert w["value"] == "data_scientist"
    assert w["affected_rows"] == 1
    assert "not observed during model training" in w["message"]


def test_predict_multiple_unseen_categories(config):
    """Multiple distinct unseen values each produce a warning."""
    df = _make_df()
    _write_csv(config, "multi.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "multi.csv")
    _train_and_export(config, "multi_art", df_disk)

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "multi_art",
                "features": {
                    "job": "data_scientist",
                    "edu": "phd",
                    "age": 40,
                    "balance": 1000,
                },
            },
        )
    )
    data = _data(result)
    warnings = data["warnings"]
    columns = {w["column"] for w in warnings}
    assert columns == {"job", "edu"}
    assert any(w["value"] == "data_scientist" for w in warnings)
    assert any(w["value"] == "phd" for w in warnings)


def test_predict_mixed_known_and_unknown_batch(config):
    """Batch with known and unknown values warns only for unknowns."""
    df = _make_df()
    _write_csv(config, "mixed.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "mixed.csv")
    _train_and_export(config, "mixed_art", df_disk)

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "mixed_art",
                "features": [
                    {"job": "admin.", "edu": "secondary", "age": 40, "balance": 1000},
                    {"job": "data_scientist", "edu": "secondary", "age": 30, "balance": 500},
                ],
            },
        )
    )
    data = _data(result)
    assert len(data["predictions"]) == 2
    warnings = data["warnings"]
    assert len(warnings) == 1
    assert warnings[0]["column"] == "job"
    assert warnings[0]["value"] == "data_scientist"
    assert warnings[0]["affected_rows"] == 1


def test_predict_missing_categorical_no_warning(config):
    """Missing categorical values are imputed, not flagged as unseen."""
    df = _make_df()
    _write_csv(config, "miss.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "miss.csv")
    _train_and_export(config, "miss_art", df_disk)

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "miss_art",
                "features": {
                    "job": None,
                    "edu": "secondary",
                    "age": 40,
                    "balance": 1000,
                },
            },
        )
    )
    data = _data(result)
    assert data["schema_valid"] is True
    assert len(data["predictions"]) == 1
    assert data["warnings"] == []


def test_predict_numeric_only_no_warning(config):
    """Numeric-only datasets produce no category warnings."""
    df = pd.DataFrame({"age": [30, 40, 35], "balance": [1000, 2000, 1500], "y": [0, 1, 0]})
    _write_csv(config, "num.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "num.csv")
    artifact, model = _train_and_export(config, "num_art", df_disk)
    assert artifact.categorical_features == []

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "num_art",
                "features": {"age": 40, "balance": 1000},
            },
        )
    )
    data = _data(result)
    assert len(data["predictions"]) == 1
    assert data["warnings"] == []


# ---------------------------------------------------------------------------
# Encoder configuration
# ---------------------------------------------------------------------------


def _build_error_encoder_artifact(config, artifact_name):
    """Artifact with OneHotEncoder(handle_unknown='error')."""
    df = pd.DataFrame(
        {
            "job": ["admin.", "blue-collar", "student"],
            "edu": ["basic", "secondary", "tertiary"],
            "age": [30, 40, 35],
            "y": [0, 1, 0],
        }
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                ["age"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="error", sparse_output=False),
                        ),
                    ]
                ),
                ["job", "edu"],
            ),
        ],
        remainder="drop",
    )
    preprocessor.fit(df[["job", "edu", "age"]])
    artifact = PreprocessingArtifact(
        preprocessor=preprocessor,
        feature_columns=["job", "edu", "age"],
        target_column="y",
        task_type="classification",
        numeric_features=["age"],
        categorical_features=["job", "edu"],
    )
    X = preprocessor.transform(df[["job", "edu", "age"]])
    model = LogisticRegression().fit(X, df["y"].to_numpy())
    _save_model_artifact(config, artifact_name, artifact, model)
    return artifact


def test_encoder_rejecting_unknown_returns_error(config):
    """handle_unknown='error' yields a clear MCP error instead of a result."""
    _build_error_encoder_artifact(config, "err_art")

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "err_art",
                "features": {"job": "data_scientist", "edu": "basic", "age": 40},
            },
        )
    )
    body = _payload(result)
    assert body["ok"] is False
    assert body["error"]["code"] == "unseen_category"
    assert "data_scientist" in body["error"]["message"]


def test_encoder_rejecting_unknown_known_value_ok(config):
    """Known values still predict with handle_unknown='error'."""
    _build_error_encoder_artifact(config, "err2_art")

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "err2_art",
                "features": {"job": "admin.", "edu": "basic", "age": 40},
            },
        )
    )
    data = _data(result)
    assert len(data["predictions"]) == 1
    assert data["warnings"] == []


def test_predict_handle_unknown_ignore_predicts_with_warning(config):
    """handle_unknown='ignore' (default): prediction continues with a warning."""
    df = _make_df()
    _write_csv(config, "ignore.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "ignore.csv")
    artifact, model = _train_and_export(config, "ignore_art", df_disk)
    encoder = categorical_encoder(artifact)
    assert encoder.handle_unknown == "ignore"

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "ignore_art",
                "features": {
                    "job": "never_seen_before",
                    "edu": "secondary",
                    "age": 40,
                    "balance": 1000,
                },
            },
        )
    )
    data = _data(result)
    assert len(data["predictions"]) == 1
    assert len(data["warnings"]) == 1
    assert data["warnings"][0]["value"] == "never_seen_before"


# ---------------------------------------------------------------------------
# Infrequent-category support
# ---------------------------------------------------------------------------


def test_infrequent_category_known_value_no_false_warning(config):
    """A learned (possibly infrequent) category is not flagged as unseen."""
    df = pd.DataFrame(
        {
            "job": ["admin."] * 8 + ["blue-collar"] * 2,
            "edu": ["basic"] * 10,
            "age": [30] * 10,
            "y": [0, 1] * 5,
        }
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                ["age"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=0.2,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                ["job", "edu"],
            ),
        ],
        remainder="drop",
    )
    preprocessor.fit(df[["job", "edu", "age"]])
    artifact = PreprocessingArtifact(
        preprocessor=preprocessor,
        feature_columns=["job", "edu", "age"],
        target_column="y",
        task_type="classification",
        numeric_features=["age"],
        categorical_features=["job", "edu"],
    )
    X = preprocessor.transform(df[["job", "edu", "age"]])
    model = LogisticRegression().fit(X, df["y"].to_numpy())
    _save_model_artifact(config, "infreq_art", artifact, model)

    # A frequently-seen known category must not warn.
    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "infreq_art",
                "features": {"job": "admin.", "edu": "basic", "age": 40},
            },
        )
    )
    data = _data(result)
    assert len(data["predictions"]) == 1
    assert data["warnings"] == []


# ---------------------------------------------------------------------------
# Warning aggregation and limits
# ---------------------------------------------------------------------------


def test_warning_aggregation_counts_affected_rows(config):
    """Repeated identical unseen values aggregate into one warning."""
    df = _make_df()
    _write_csv(config, "agg.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "agg.csv")
    _train_and_export(config, "agg_art", df_disk)

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "agg_art",
                "features": [
                    {"job": "data_scientist", "edu": "secondary", "age": 40, "balance": 1000},
                    {"job": "data_scientist", "edu": "secondary", "age": 30, "balance": 500},
                    {"job": "data_scientist", "edu": "secondary", "age": 25, "balance": 300},
                    {"job": "admin.", "edu": "secondary", "age": 50, "balance": 800},
                ],
            },
        )
    )
    data = _data(result)
    assert len(data["predictions"]) == 4
    warnings = data["warnings"]
    assert len(warnings) == 1
    assert warnings[0]["value"] == "data_scientist"
    assert warnings[0]["affected_rows"] == 3


def test_warning_output_size_limit(config):
    """Many distinct unseen values are capped, not unbounded."""
    df = _make_df()
    _write_csv(config, "cap.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "cap.csv")
    _train_and_export(config, "cap_art", df_disk)

    features = [
        {"job": f"unseen_{i}", "edu": "secondary", "age": 40, "balance": 1000} for i in range(40)
    ]
    result = run(predict_tool(config, {"artifact_name": "cap_art", "features": features}))
    data = _data(result)
    assert len(data["predictions"]) == 40
    assert len(data["warnings"]) <= 25


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_predict_backward_compatible(config):
    """Existing predict calls keep working; warnings key is additive."""
    df = _make_df()
    _write_csv(config, "compat.csv", df)
    df_disk = pd.read_csv(config.workspace_dir / "compat.csv")
    _train_and_export(config, "compat_art", df_disk)

    result = run(
        predict_tool(
            config,
            {
                "artifact_name": "compat_art",
                "features": {"job": "admin.", "edu": "secondary", "age": 40, "balance": 1000},
            },
        )
    )
    data = _data(result)
    assert data["schema_valid"] is True
    assert len(data["predictions"]) == 1
    assert "probabilities" in data
    assert "ignored_extra_columns" in data
    assert "warnings" in data


# ---------------------------------------------------------------------------
# MCP protocol end-to-end
# ---------------------------------------------------------------------------


def test_mcp_protocol_export_predict_unseen(tmp_path):
    """Train, export, reload, and predict via real MCP stdio messages."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    workspace.mkdir()
    artifacts.mkdir()

    examples = Path(__file__).resolve().parent.parent / "examples"
    shutil.copy(examples / "sample_classification.csv", workspace / "sample_classification.csv")

    env = dict(os.environ)
    env["DEVRAY_WORKSPACE_DIR"] = str(workspace)
    env["DEVRAY_ARTIFACTS_DIR"] = str(artifacts)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "devray_datalab.server"],
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    async def run_protocol() -> dict:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                names = {t.name for t in tools_result.tools}
                assert len(names) == 27, f"expected 27 tools, got {len(names)}"
                assert "export_model" in names and "predict" in names

                export = await session.call_tool(
                    "export_model",
                    arguments={
                        "artifact_name": "proto_unseen",
                        "name": "sample_classification.csv",
                        "target_column": "purchased",
                        "task_type": "classification",
                        "model_name": "logistic_regression",
                        "exclude_columns": ["region"],
                        "random_state": 42,
                    },
                )
                body = json.loads(export.content[0].text)
                assert body["ok"] is True, body.get("error")

                known = await session.call_tool(
                    "predict",
                    arguments={
                        "artifact_name": "proto_unseen",
                        "features": {
                            "age": 40,
                            "income": 80000,
                            "credit_score": 720,
                            "category": "electronics",
                        },
                    },
                )
                kbody = json.loads(known.content[0].text)
                assert kbody["ok"] is True
                assert kbody["data"]["warnings"] == []

                unseen = await session.call_tool(
                    "predict",
                    arguments={
                        "artifact_name": "proto_unseen",
                        "features": {
                            "age": 40,
                            "income": 80000,
                            "credit_score": 720,
                            "category": "data_scientist_category",
                        },
                    },
                )
                ubody = json.loads(unseen.content[0].text)
                assert ubody["ok"] is True
                assert len(ubody["data"]["predictions"]) == 1
                assert len(ubody["data"]["warnings"]) == 1
                assert ubody["data"]["warnings"][0]["type"] == "unseen_category"
                assert ubody["data"]["warnings"][0]["column"] == "category"
                assert ubody["data"]["warnings"][0]["value"] == "data_scientist_category"
                return ubody["data"]

    data = asyncio.run(run_protocol())
    assert len(data["predictions"]) == 1
