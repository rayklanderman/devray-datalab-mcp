"""MCP server entry point.

Runs a stdio-based MCP server exposing DevRay DataLab tools. Diagnostics and
logging go to stderr; stdout is reserved for MCP protocol messages.

Tool modules are imported lazily on first call to keep server startup fast
(no heavy ML dependencies loaded at startup).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)

from devray_datalab.config import DevRayConfig

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


# Tool specs are defined statically (no heavy imports needed).
# Tool handlers are loaded lazily on first call to keep startup fast.
_tool_specs: list[Tool] = [
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
        name="list_split_manifests",
        description="List all stored split manifests, newest first.",
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="create_split_manifest",
        description="Create a reproducible train/validation/test split manifest.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Dataset filename.",
                },
                "task_type": {
                    "type": "string",
                    "description": "'classification' or 'regression' (default classification).",
                },
                "split_type": {
                    "type": "string",
                    "description": "Split strategy: random, stratified, grouped, time_series.",
                },
                "test_size": {
                    "type": "number",
                    "description": "Fraction of data for test holdout (default 0.2).",
                },
                "val_size": {
                    "type": "number",
                    "description": "Fraction of data for validation (default 0.0).",
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (default 42).",
                },
                "stratify_column": {
                    "type": "string",
                    "description": "Column to stratify splits by.",
                },
                "group_column": {
                    "type": "string",
                    "description": "Column to group splits by (groups stay together).",
                },
                "time_column": {
                    "type": "string",
                    "description": "Column for time-based splitting.",
                },
                "row_order": {
                    "type": "boolean",
                    "description": "Split chronologically by original row order.",
                },
                "protected_holdout": {
                    "type": "boolean",
                    "description": "Keep test set as protected holdout (default true).",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semi, tab, pipe). Auto.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="validate_split_manifest",
        description="Validate a stored split manifest against a dataset.",
        inputSchema={
            "type": "object",
            "properties": {
                "manifest_id": {
                    "type": "string",
                    "description": "ID of the split manifest to validate.",
                },
                "name": {
                    "type": "string",
                    "description": "Dataset filename.",
                },
            },
            "required": ["manifest_id", "name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="inspect_dataset",
        description="Inspect a CSV dataset (types, missing values, stats, preview).",
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
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semi, tab, pipe). Auto.",
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
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semi, tab, pipe). Auto.",
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
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semi, tab, pipe). Auto.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="detect_potential_leakage",
        description="Detect potential data-leakage issues in a dataset.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Dataset filename."},
                "target_column": {
                    "type": "string",
                    "description": "Target variable column (optional).",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semi, tab, pipe). Auto.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="analyze_feature_drift",
        description="Detect feature drift between reference/current datasets.",
        inputSchema={
            "type": "object",
            "properties": {
                "reference_name": {
                    "type": "string",
                    "description": "Reference dataset filename.",
                },
                "current_name": {
                    "type": "string",
                    "description": "Current dataset filename.",
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance level (default 0.05).",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to check (default all common).",
                },
                "ref_delimiter": {
                    "type": "string",
                    "description": "Reference CSV delimiter (comma, semi, tab, pipe).",
                },
                "cur_delimiter": {
                    "type": "string",
                    "description": "Current CSV delimiter (comma, semi, tab, pipe).",
                },
            },
            "required": ["reference_name", "current_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="prepare_dataset",
        description="Define target and task type; fit preprocessing on train data only.",
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
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semicolon, tab, pipe). Auto.",
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
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semicolon, tab, pipe). Auto.",
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
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semicolon, tab, pipe). Auto.",
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
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semicolon, tab, pipe). Auto.",
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
    Tool(
        name="list_experiments",
        description="List recorded training experiments, newest first.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max experiments to return (default 50).",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_experiment",
        description="Get a single recorded experiment by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "description": "Experiment ID.",
                },
            },
            "required": ["experiment_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="compare_experiments",
        description="Compare multiple recorded experiments by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of experiment IDs.",
                },
            },
            "required": ["experiment_ids"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_models",
        description="List experiments that produced a model artifact.",
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="load_model_handle",
        description="Load a previously exported model artifact by experiment ID or name.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "description": "Experiment ID.",
                },
                "artifact_name": {
                    "type": "string",
                    "description": "Model artifact name.",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="start_training_job",
        description="Submit a training job and return its id immediately (non-blocking).",
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
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semicolon, tab, pipe). Auto.",
                },
            },
            "required": ["name", "target_column", "task_type", "model_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_training_status",
        description="Return the status and result of a training job.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Training job id."},
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="cancel_training_job",
        description="Request cancellation of a queued or running training job.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Training job id."},
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_training_jobs",
        description="List training jobs, newest first.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max jobs to return (default 50).",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="automl_run",
        description="Run automated ML: train candidate models with CV, rank by objective.",
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
                "metric": {"type": "string", "description": "Selection metric (default auto)."},
                "max_trials": {
                    "type": "integer",
                    "description": "Max models to evaluate (default 10, max 50).",
                },
                "max_time_seconds": {
                    "type": "integer",
                    "description": "Max wall-clock seconds (default 600).",
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed (default 42).",
                },
                "cv": {
                    "type": "integer",
                    "description": "Cross-validation folds (default 3).",
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subset of models to evaluate.",
                },
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semicolon, tab, pipe). Auto.",
                },
            },
            "required": ["name", "target_column", "task_type"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tune_model",
        description="Tune hyperparameters of a single model with cross-validation.",
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
                "model_name": {"type": "string", "description": "Model to tune."},
                "metric": {"type": "string", "description": "Selection metric (default auto)."},
                "max_trials": {
                    "type": "integer",
                    "description": "Max tuning trials (default 20).",
                },
                "max_time_seconds": {
                    "type": "integer",
                    "description": "Max wall-clock seconds (default 600).",
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed (default 42).",
                },
                "cv": {
                    "type": "integer",
                    "description": "Cross-validation folds (default 3).",
                },
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semicolon, tab, pipe). Auto.",
                },
            },
            "required": ["name", "target_column", "task_type", "model_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_model_leaderboard",
        description="Return the ranked leaderboard of an AutoML run.",
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
                "metric": {"type": "string", "description": "Selection metric (default auto)."},
                "max_trials": {
                    "type": "integer",
                    "description": "Max models to evaluate (default 10).",
                },
                "max_time_seconds": {
                    "type": "integer",
                    "description": "Max wall-clock seconds (default 600).",
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed (default 42).",
                },
                "cv": {
                    "type": "integer",
                    "description": "Cross-validation folds (default 3).",
                },
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "CSV delimiter (comma, semicolon, tab, pipe). Auto.",
                },
            },
            "required": ["name", "target_column", "task_type"],
            "additionalProperties": False,
        },
    ),
]


def build_server(config: DevRayConfig | None = None) -> Server:
    """Build and configure the MCP server with all tools registered."""
    if config is None:
        config = DevRayConfig()

    async def list_tools(context: Any, params: Any | None) -> ListToolsResult:
        return ListToolsResult(tools=_tool_specs)

    async def call_tool_handler(context: Any, params: Any | None) -> CallToolResult:
        """Handle a tool call with lazy module loading.

        Tool modules are imported only when first called, avoiding heavy
        imports (pandas, scikit-learn, etc.) at server startup. Each tool
        imports only its own module, so calling system_info does not load
        training/automl/etc. modules.
        """
        arguments = params.arguments if params and params.arguments else {}
        name = params.name if params else None
        logger.info("MCP tool called: %s", name)

        # Import tool module lazily - only the module for this specific tool
        import importlib

        _tool_module_map: dict[str, str] = {
            "list_datasets": "devray_datalab.tools.datasets",
            "inspect_dataset": "devray_datalab.tools.datasets",
            "analyze_correlations": "devray_datalab.tools.analysis",
            "analyze_data_quality": "devray_datalab.tools.analysis",
            "detect_potential_leakage": "devray_datalab.tools.analysis",
            "analyze_feature_drift": "devray_datalab.tools.analysis",
            "prepare_dataset": "devray_datalab.tools.training",
            "train_baseline": "devray_datalab.tools.training",
            "compare_models": "devray_datalab.tools.training",
            "export_model": "devray_datalab.tools.models",
            "predict": "devray_datalab.tools.models",
            "system_info": "devray_datalab.tools.system",
            "list_experiments": "devray_datalab.tools.experiments",
            "get_experiment": "devray_datalab.tools.experiments",
            "compare_experiments": "devray_datalab.tools.experiments",
            "list_models": "devray_datalab.tools.experiments",
            "load_model_handle": "devray_datalab.tools.experiments",
            "start_training_job": "devray_datalab.tools.jobs",
            "get_training_status": "devray_datalab.tools.jobs",
            "cancel_training_job": "devray_datalab.tools.jobs",
            "list_training_jobs": "devray_datalab.tools.jobs",
            "automl_run": "devray_datalab.tools.automl",
            "tune_model": "devray_datalab.tools.automl",
            "get_model_leaderboard": "devray_datalab.tools.automl",
            "create_split_manifest": "devray_datalab.tools.splits",
            "validate_split_manifest": "devray_datalab.tools.splits",
            "list_split_manifests": "devray_datalab.tools.splits",
        }

        module_name = _tool_module_map.get(name)
        if module_name is None:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=_err(f"Unknown tool: {name}", "unknown_tool"),
                    )
                ],
                isError=True,
            )

        module = importlib.import_module(module_name)

        handler_attr_name = f"{name}_tool"
        handler = getattr(module, handler_attr_name, None)
        if handler is None:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=_err(
                            f"Handler {handler_attr_name} not found in {module_name}",
                            "internal_error",
                        ),
                    )
                ],
                isError=True,
            )

        try:
            content_list = await handler(config, arguments)
            return CallToolResult(content=content_list)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s failed", name)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=_err(str(exc), "unexpected_error"),
                    )
                ],
                isError=True,
            )

    server = Server(
        name="devray-datalab-mcp",
        version="0.3.1",
        title="DevRay DataLab MCP",
        instructions=(
            "DevRay DataLab MCP is a local-first data science and machine "
            "learning server. All processing happens on your own computer. "
            "Use list_datasets, inspect_dataset, analyze_correlations, "
            "analyze_data_quality, prepare_dataset, train_baseline, "
            "compare_models, export_model, predict, and system_info."
        ),
        on_list_tools=list_tools,
        on_call_tool=call_tool_handler,
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
