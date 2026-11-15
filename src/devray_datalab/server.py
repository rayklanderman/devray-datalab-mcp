"""MCP server entry point.

Runs a stdio-based MCP server exposing DevRay DataLab tools. Diagnostics and
logging go to stderr; stdout is reserved for MCP protocol messages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from devray_datalab.config import DevRayConfig
from devray_datalab.tools import analysis, datasets, models, system, training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("devray_datalab")


def _json(data: Any) -> str:
    return json.dumps(data, default=str)


def _ok(data: Any) -> str:
    return _json({"ok": True, "data": data})


def _err(message: str, code: str = "error") -> str:
    return _json({"ok": False, "error": {"message": message, "code": code}})


def build_server(config: DevRayConfig | None = None) -> Server:
    """Build and configure the MCP server with all tools registered."""
    if config is None:
        config = DevRayConfig()

    tool_specs: list[Tool] = [
        Tool(
            name="list_datasets",
            description="List datasets in the configured workspace.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        Tool(
            name="inspect_dataset",
            description="Inspect a CSV dataset (types, missing values, stats).",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Dataset filename.",
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Max preview rows (default 25, max 100).",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="analyze_correlations",
            description="Compute numerical correlations and strongest pairs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset filename."},
                    "method": {
                        "type": "string",
                        "description": "'pearson' or 'spearman' (default pearson).",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top pairs to return (default 10).",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="analyze_data_quality",
            description="Report data-quality issues and recommendations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset filename."},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="prepare_dataset",
            description="Define target and task type; fit preprocessing on training data only.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset filename."},
                    "target_column": {
                        "type": "string",
                        "description": "Target variable column.",
                    },
                    "task_type": {
                        "type": "string",
                        "description": "'classification' or 'regression'.",
                    },
                    "test_size": {"type": "number", "description": "Test fraction (default 0.2)."},
                    "random_state": {
                        "type": "integer",
                        "description": "Random seed (default 42).",
                    },
                    "stratify": {
                        "type": "boolean",
                        "description": "Stratify split (classification only, default true).",
                    },
                },
                "required": ["name", "target_column", "task_type"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="train_baseline",
            description="Train one baseline model and report evaluation metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset filename."},
                    "target_column": {
                        "type": "string",
                        "description": "Target variable column.",
                    },
                    "task_type": {
                        "type": "string",
                        "description": "'classification' or 'regression'.",
                    },
                    "model_name": {"type": "string", "description": "Model to train."},
                    "test_size": {"type": "number", "description": "Test fraction (default 0.2)."},
                    "random_state": {
                        "type": "integer",
                        "description": "Random seed (default 42).",
                    },
                    "stratify": {
                        "type": "boolean",
                        "description": "Stratify split (classification only, default true).",
                    },
                },
                "required": ["name", "target_column", "task_type", "model_name"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="compare_models",
            description="Train and compare multiple models; select best by metric.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset filename."},
                    "target_column": {
                        "type": "string",
                        "description": "Target variable column.",
                    },
                    "task_type": {
                        "type": "string",
                        "description": "'classification' or 'regression'.",
                    },
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Models to compare.",
                    },
                    "metric": {"type": "string", "description": "Selection metric (default auto)."},
                    "test_size": {"type": "number", "description": "Test fraction (default 0.2)."},
                    "random_state": {
                        "type": "integer",
                        "description": "Random seed (default 42).",
                    },
                    "cv": {
                        "type": "integer",
                        "description": "Cross-validation folds (default 5).",
                    },
                },
                "required": ["name", "target_column", "task_type"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="export_model",
            description="Save a trained model with preprocessing pipeline and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_name": {
                        "type": "string",
                        "description": "Name for the saved artifact.",
                    },
                    "model_handle": {
                        "type": "string",
                        "description": "Handle of a previously trained model.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Dataset filename (to train and export).",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Target variable column.",
                    },
                    "task_type": {
                        "type": "string",
                        "description": "'classification' or 'regression'.",
                    },
                    "model_name": {"type": "string", "description": "Model to train and export."},
                    "test_size": {"type": "number", "description": "Test fraction (default 0.2)."},
                    "random_state": {
                        "type": "integer",
                        "description": "Random seed (default 42).",
                    },
                    "stratify": {
                        "type": "boolean",
                        "description": "Stratify split (classification only).",
                    },
                },
                "required": ["artifact_name"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="predict",
            description="Generate predictions using a previously trained local model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_name": {
                        "type": "string",
                        "description": "Name of the exported model artifact.",
                    },
                    "features": {"description": "Feature record(s) to predict on."},
                },
                "required": ["artifact_name", "features"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="system_info",
            description="Detect local CPU, RAM, storage, and GPU resources.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    ]

    tool_handlers = {
        "list_datasets": datasets.list_datasets_tool,
        "inspect_dataset": datasets.inspect_dataset_tool,
        "analyze_correlations": analysis.analyze_correlations_tool,
        "analyze_data_quality": analysis.analyze_data_quality_tool,
        "prepare_dataset": training.prepare_dataset_tool,
        "train_baseline": training.train_baseline_tool,
        "compare_models": training.compare_models_tool,
        "export_model": models.export_model_tool,
        "predict": models.predict_tool,
        "system_info": system.system_info_tool,
    }

    async def on_list_tools(context, params) -> ListToolsResult:  # noqa: ARG001
        return ListToolsResult(tools=tool_specs)

    async def on_call_tool(context, params) -> CallToolResult:  # noqa: ARG001
        name = params.name
        arguments = params.arguments or {}
        logger.info("MCP tool called: %s", name)
        try:
            handler = tool_handlers.get(name)
            if handler is None:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=_err(f"Unknown tool: {name}", "unknown_tool"),
                        )
                    ],
                    isError=True,
                )
            content_list = await handler(config, arguments)
            # content_list is already list[TextContent]
            return CallToolResult(content=content_list)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s failed", name)
            return CallToolResult(
                content=[TextContent(type="text", text=_err(str(exc), "unexpected_error"))],
                isError=True,
            )

    server = Server(
        name="devray-datalab-mcp",
        version="0.1.0",
        title="DevRay DataLab MCP",
        instructions=(
            "DevRay DataLab MCP is a local-first data science and machine "
            "learning server. All processing happens on your own computer. "
            "Use list_datasets, inspect_dataset, analyze_correlations, "
            "analyze_data_quality, prepare_dataset, train_baseline, "
            "compare_models, export_model, predict, and system_info."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    return server


async def run_async(config: DevRayConfig | None = None) -> None:
    """Run the MCP server over stdio."""
    server = build_server(config)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DevRay DataLab MCP server")
    parser.add_argument("--workspace", help="Dataset workspace directory")
    parser.add_argument("--artifacts", help="Model artifacts directory")
    args = parser.parse_args()

    import os

    if args.workspace:
        os.environ["DEVRAY_WORKSPACE_DIR"] = args.workspace
    if args.artifacts:
        os.environ["DEVRAY_ARTIFACTS_DIR"] = args.artifacts

    config = DevRayConfig()
    logger.info("Starting DevRay DataLab MCP server (stdio transport)")
    logger.info("Workspace: %s", config.workspace_dir)
    logger.info("Artifacts: %s", config.artifacts_dir)

    import anyio

    anyio.run(run_async, config)


if __name__ == "__main__":
    main()
