"""End-to-end integration test: communicates with the MCP server as a client.

Uses the official MCP Python SDK stdio client + ClientSession to launch the
server as a subprocess and exchange real MCP messages (initialize, tools/list,
tools/call).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def test_mcp_server_e2e(tmp_path):
    """Launch the server as a subprocess and exchange MCP messages as a client."""
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    workspace.mkdir()
    artifacts.mkdir()

    examples = Path(__file__).resolve().parent.parent / "examples"
    shutil.copy(examples / "sample_classification.csv", workspace / "sample_classification.csv")

    env = dict(os.environ)
    env["DEVRAY_WORKSPACE_DIR"] = str(workspace)
    env["DEVRAY_ARTIFACTS_DIR"] = str(artifacts)
    env["DEVRAY_EXPERIMENTS_DB_PATH"] = str(tmp_path / "experiments.db")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "devray_datalab.server"],
        env=env,
        cwd=str(tmp_path),
    )

    async def run() -> None:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                initialization = await session.initialize()
                assert initialization.server_info.name == "devray-datalab-mcp"
                assert initialization.server_info.version == "0.3.4"

                # list tools
                tools_result = await session.list_tools()
                tool_names = {t.name for t in tools_result.tools}
                expected = {
                    "list_datasets",
                    "inspect_dataset",
                    "analyze_correlations",
                    "analyze_data_quality",
                    "analyze_feature_drift",
                    "detect_potential_leakage",
                    "prepare_dataset",
                    "train_baseline",
                    "compare_models",
                    "export_model",
                    "predict",
                    "system_info",
                    "list_experiments",
                    "get_experiment",
                    "compare_experiments",
                    "list_models",
                    "load_model_handle",
                    "start_training_job",
                    "get_training_status",
                    "cancel_training_job",
                    "list_training_jobs",
                    "automl_run",
                    "tune_model",
                    "get_model_leaderboard",
                    "create_split_manifest",
                    "validate_split_manifest",
                    "list_split_manifests",
                }
                assert tool_names == expected
                assert len(tools_result.tools) == len(expected)

                # call inspect_dataset
                inspect = await session.call_tool(
                    "inspect_dataset",
                    arguments={"name": "sample_classification.csv", "max_rows": 3},
                )
                body = json.loads(inspect.content[0].text)
                assert body["ok"] is True
                assert body["data"]["rows"] == 20
                assert body["data"]["columns"] == 6

                # call train_baseline
                train = await session.call_tool(
                    "train_baseline",
                    arguments={
                        "name": "sample_classification.csv",
                        "target_column": "purchased",
                        "task_type": "classification",
                        "model_name": "logistic_regression",
                        "random_state": 42,
                    },
                )
                train_body = json.loads(train.content[0].text)
                assert train_body["ok"] is True
                assert train_body["data"]["metrics"]["accuracy"] >= 0.0

                # call system_info
                si = await session.call_tool("system_info", arguments={})
                si_body = json.loads(si.content[0].text)
                assert si_body["ok"] is True
                assert si_body["data"]["training_backend"] == "CPU (scikit-learn)"

    asyncio.run(run())


def test_mcp_leaderboard_model_filter(config, tmp_path):
    dataset = config.workspace_dir / "leaderboard.csv"
    # Both classifiers get an identical perfect score on these two clusters.
    dataset.write_text("feature;target\n" + "0;0\n1;1\n" * 12, encoding="utf-8")
    regression = config.workspace_dir / "leaderboard-regression.csv"
    regression.write_text(
        "feature;target\n" + "".join(f"{i};{i * 2}\n" for i in range(24)), encoding="utf-8"
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "devray_datalab.server"],
        cwd=str(tmp_path),
        env={
            **os.environ,
            "DEVRAY_WORKSPACE_DIR": str(config.workspace_dir),
            "DEVRAY_ARTIFACTS_DIR": str(config.artifacts_dir),
            "DEVRAY_EXPERIMENTS_DB_PATH": str(config.experiments_db_path),
        },
    )
    arguments = {
        "name": dataset.name,
        "target_column": "target",
        "task_type": "classification",
        "max_trials": 2,
        "delimiter": ";",
    }

    async def run() -> None:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            schema = tools["get_model_leaderboard"].input_schema
            models_schema = schema["properties"]["models"]
            assert "models" not in schema["required"]
            assert models_schema == tools["automl_run"].input_schema["properties"]["models"]
            assert models_schema["type"] == "array"
            assert models_schema["minItems"] == 1
            assert models_schema["uniqueItems"] is True
            assert models_schema["items"]["type"] == "string"

            async def call(tool="get_model_leaderboard", **overrides):
                result = await session.call_tool(tool, arguments={**arguments, **overrides})
                return json.loads(result.content[0].text)

            for models in (["decision_tree", "knn"], ["knn", "decision_tree"]):
                body = await call(models=models)
                assert body["ok"], body
                data = body["data"]
                assert [trial["model"] for trial in data["trials"]] == models
                assert [trial["value"] for trial in data["trials"]] == [1.0, 1.0]
                assert data["best_model"] == models[0]
                assert data["best_params"] == data["trials"][0]["params"]
                assert data["best_value"] == data["trials"][0]["value"]

            capped = await call(models=["knn", "decision_tree"], max_trials=1)
            assert capped["data"]["best_model"] == "knn"
            assert len(capped["data"]["trials"]) == 1

            defaults = await call(max_trials=1, delimiter=";")
            assert defaults["ok"], defaults
            assert defaults["data"]["best_model"] == "logistic_regression"
            assert set(defaults["data"]) == {
                "task_type",
                "metric",
                "best_model",
                "best_params",
                "best_value",
                "trials",
                "optuna_used",
            }

            reg = await call(
                name=regression.name,
                task_type="regression",
                models=["ridge_regression", "linear_regression"],
            )
            assert reg["ok"], reg
            assert [trial["model"] for trial in reg["data"]["trials"]] == [
                "linear_regression",
                "ridge_regression",
            ]
            assert reg["data"]["best_model"] == "linear_regression"

            for bad_models in (
                None,
                [],
                "decision_tree",
                {"model": "decision_tree"},
                ["decision_tree", 7],
                [""],
                ["  "],
                ["knn", "knn"],
                ["knn", "made_up"],
                ["linear_regression"],
            ):
                body = await call(models=bad_models)
                assert body["ok"] is False, (bad_models, body)
                assert body["error"]["code"] == "validation_error"

            full_run = await call("automl_run", models=["knn"], max_trials=1)
            assert full_run["data"]["best_model"] == "knn"
            for tool, options in (
                ("get_model_leaderboard", {"models": ["knn"]}),
                ("automl_run", {"models": ["knn"]}),
                ("tune_model", {"model_name": "knn"}),
            ):
                valid = await call(tool, max_trials=1, **options)
                assert valid["ok"], valid
                wrong_delimiter = await call(tool, delimiter=",", max_trials=1, **options)
                assert wrong_delimiter["error"]["code"] == "validation_error"
                assert "Target column 'target' not found" in wrong_delimiter["error"]["message"]

    asyncio.run(run())
