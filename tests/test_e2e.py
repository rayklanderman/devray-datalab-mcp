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

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "devray_datalab.server"],
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    async def run() -> None:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # list tools
                tools_result = await session.list_tools()
                tool_names = {t.name for t in tools_result.tools}
                expected = {
                    "list_datasets",
                    "inspect_dataset",
                    "analyze_correlations",
                    "analyze_data_quality",
                    "prepare_dataset",
                    "train_baseline",
                    "compare_models",
                    "export_model",
                    "predict",
                    "system_info",
                }
                assert expected.issubset(tool_names), f"missing: {expected - tool_names}"

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
