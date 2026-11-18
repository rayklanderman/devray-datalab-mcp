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

from devray_datalab import __version__
from devray_datalab.config import DevRayConfig
from devray_datalab.utils.errors import sanitize_for_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("devray_datalab")


def _json(data: Any) -> str:
    return json.dumps(sanitize_for_json(data), default=str, allow_nan=False)


def _ok(data: Any) -> str:
    return _json({"ok": True, "data": data})


def _err(message: str, code: str = "error") -> str:
    return _json({"ok": False, "error": {"message": message, "code": code}})


_AUTOML_MODEL_SUBSET = {
    "type": "array",
    "items": {"type": "string", "minLength": 1, "pattern": r"\S"},
    "minItems": 1,
    "uniqueItems": True,
    "description": (
        "Ordered task-specific model IDs. Omit for defaults. Both tasks support "
        "decision_tree, random_forest, extra_trees, hist_gradient_boosting, svm, knn. "
        "Classification also has logistic_regression; regression has linear_regression and "
        "ridge_regression. Caps may stop search before all models are scored. Ties retain "
        "evaluation order."
    ),
}

_AUTOML_METRIC = {
    "type": "string",
    "description": (
        "Selection metric. auto (default) uses roc_auc for binary classification, f1 for "
        "multiclass classification or r2 for regression. Loss metrics rank lowest first."
    ),
}

_AUTOML_TIME_BUDGET = {
    "type": "integer",
    "description": (
        "Shared candidate and tuning time budget in seconds (default 600). A trial already "
        "in progress may finish after the deadline."
    ),
}

_AUTOML_DELIMITER = {
    "type": "string",
    "description": "CSV delimiter character: comma (,), semicolon (;), tab (\\t) or pipe (|). "
    "Omit to detect it automatically.",
}


# Tool specs are defined statically (no heavy imports needed).
# Tool handlers are loaded lazily on first call to keep startup fast.
_tool_specs: list[Tool] = [
    Tool(
        name="list_datasets",
        description=(
            "Discover dataset filenames, formats and sizes in the configured workspace without "
            "reading rows. Lists CSV, Parquet and Excel files; use inspect_dataset to profile one."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_split_manifests",
        description=(
            "Find stored split manifest IDs and metadata, newest first. Use "
            "validate_split_manifest to check an ID against the current dataset."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="create_split_manifest",
        description=(
            "Save reproducible row indices and a dataset fingerprint before training. Pass the "
            "returned ID as split_manifest_id to compatible training tools; keep test rows out "
            "of tuning. Use validate_split_manifest to check a saved split without rewriting it."
        ),
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
                    "description": (
                        "random (default), stratified (pass stratify_column), group (requires "
                        "group_column), or time (requires time_column or row_order=true). "
                        "grouped and time_series are aliases."
                    ),
                },
                "test_size": {
                    "type": "number",
                    "description": "Fraction of data for test holdout (default 0.2).",
                },
                "val_size": {
                    "type": "number",
                    "description": "Validation fraction (default 0); test_size + val_size < 1.",
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
        description=(
            "Check a saved manifest against the named workspace dataset before reuse. Returns "
            "valid and any errors for fingerprint or split mismatches; does not repair the file. "
            "Find IDs with list_split_manifests."
        ),
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
        description=(
            "Profile a named workspace dataset: shape, column types, missingness, summaries and "
            "a bounded preview that includes raw values. Use analyze_data_quality for remediation "
            "recommendations. CSV is built in; Parquet and Excel readers need optional packages."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Dataset filename.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Preview row count, 1–100. Omit for the server default (25).",
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
        description=(
            "Rank numeric column pairs by correlation strength within one dataset. Use Pearson "
            "for linear association or Spearman for rank association; neither establishes "
            "causation. For changes between datasets, use analyze_feature_drift."
        ),
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
                    "description": "Number of strongest pairs, clamped to 1–50 (default 10).",
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
        description=(
            "Check a workspace dataset for missing data, duplicates, constants and other quality "
            "issues before modeling. Returns recommendations without modifying the dataset. "
            "Use inspect_dataset for a raw row preview."
        ),
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
        description=(
            "Screen a workspace dataset before fitting for potential leakage, such as ID-like "
            "columns and, when target_column is supplied, predictors suspiciously related to the "
            "label. These flags need domain review; the dataset is not modified."
        ),
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
        description=(
            "Compare feature distributions in a reference (baseline) and current workspace "
            "dataset, using shared columns unless a subset is supplied. Reports statistical drift "
            "flags at alpha without modifying data; use analyze_data_quality for one dataset."
        ),
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
        description=(
            "Split a named dataset for classification or regression and fit preprocessing "
            "on training rows only. Saves a preprocessing artifact and returns its handle and "
            "schema. Training tools fit their own preprocessing."
        ),
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
        description=(
            "Fit one named model and return evaluation metrics and feature metadata. Supply "
            "a dataset, target and task. Use start_training_job for background evaluation or "
            "export_model to fit and save a reusable bundle."
        ),
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
        description=(
            "Fit and compare candidate algorithms on a dataset using cross-validation (CV). "
            "Returns fold metrics and selects a winner by the requested metric. Provide a "
            "split_manifest_id to reserve its test rows. Records experiments when possible. "
            "Use compare_experiments for saved metrics or automl_run for hyperparameter search."
        ),
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
        description=(
            "Write a named local model bundle with fitted preprocessing and feature schema "
            "for predict. Supply name, target_column, task_type and model_name to train, or "
            "pass a trusted existing bundle name as model_handle to copy it. Reusing "
            "artifact_name can overwrite a saved bundle."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "artifact_name": {
                    "type": "string",
                    "description": "Name for the saved artifact.",
                },
                "model_handle": {
                    "type": "string",
                    "description": "Name of a trusted model bundle already saved as an artifact.",
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
        description=(
            "Score a feature object or array of feature objects with a trusted local model "
            "bundle, without fitting. Match feature keys to the stored schema; use "
            "load_model_handle to inspect it. Never load artifacts from untrusted sources."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "artifact_name": {
                    "type": "string",
                    "description": "Name of the exported model artifact.",
                },
                "features": {"description": "One feature-name/value object or a list of them."},
            },
            "required": ["artifact_name", "features"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="system_info",
        description=(
            "Check the server's CPU, RAM, storage and detected GPUs before a costly run. "
            "This does not train or enable GPU acceleration."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_experiments",
        description=(
            "Browse recent persisted experiment records, newest first, to find IDs for "
            "get_experiment or compare_experiments. Use list_training_jobs for in-process jobs."
        ),
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
        description=(
            "Read one saved experiment's metrics and metadata by ID from list_experiments. "
            "Use compare_experiments for several runs; this does not fit or load a model."
        ),
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
        description=(
            "Compare saved metrics for experiment IDs from list_experiments without retraining. "
            "Interpret scores only across comparable datasets, splits and metrics; use "
            "compare_models to run a new comparison on one dataset."
        ),
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
        description=(
            "Find experiment records that reference model artifacts and their identifiers. "
            "Use load_model_handle to confirm a bundle is available and read its schema."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="load_model_handle",
        description=(
            "Inspect feature schema, metrics and metadata from a trusted saved model bundle. "
            "Supply an artifact_name or an experiment_id with a tracked artifact. This loads "
            "the bundle but does not score rows; use predict for inference."
        ),
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
        description=(
            "Queue one named model fit and return a job_id without waiting for training. "
            "Supply a dataset, target and task, then poll get_training_status for results. "
            "Use train_baseline when you want the metrics in the same call."
        ),
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
        description=(
            "Poll a job_id from start_training_job or list_training_jobs for state and any "
            "available result or error. Job IDs are process-local."
        ),
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
        description=(
            "Request cancellation of a queued or running job by job_id. Running work stops "
            "cooperatively, so poll get_training_status to confirm the terminal state. A request "
            "does not undo finished work."
        ),
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
        description=(
            "Discover recent jobs in this server process, newest first. Use get_training_status "
            "for one job's result or list_experiments for persistent run history."
        ),
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
        description=(
            "Run a fresh, synchronous cross-validation search and return ranked trials, winner "
            "and run metadata. Optionally tune the leading candidate when Optuna and budget "
            "remain. Provide a split_manifest_id to reserve its test rows. Use tune_model for "
            "one algorithm. Search returns scores and parameters without saving fitted models."
        ),
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
                "metric": _AUTOML_METRIC,
                "max_trials": {
                    "type": "integer",
                    "description": "Total candidate and tuning trials, 1–50 (default 10).",
                },
                "max_time_seconds": _AUTOML_TIME_BUDGET,
                "random_state": {
                    "type": "integer",
                    "description": "Random seed (default 42).",
                },
                "cv": {
                    "type": "integer",
                    "description": "Cross-validation folds (default 3).",
                },
                "models": _AUTOML_MODEL_SUBSET,
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": _AUTOML_DELIMITER,
            },
            "required": ["name", "target_column", "task_type"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tune_model",
        description=(
            "Score one model by cross-validation, then search its hyperparameters if Optuna "
            "is installed and budget remains. Without Optuna, only the baseline is scored. "
            "Returns scores and parameters. Use automl_run to search "
            "across model families."
        ),
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
                "metric": _AUTOML_METRIC,
                "max_trials": {
                    "type": "integer",
                    "description": "Total trials including the baseline, 1–50 (default 20).",
                },
                "max_time_seconds": _AUTOML_TIME_BUDGET,
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
                "delimiter": _AUTOML_DELIMITER,
            },
            "required": ["name", "target_column", "task_type", "model_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_model_leaderboard",
        description=(
            "Run fresh cross-validation and return ranked trials plus the winner's model "
            "ID, parameters and score. Pass models to restrict candidates. Use automl_run "
            "for full run metadata or list_experiments to find saved training records. "
            "Use export_model to fit and save a model bundle."
        ),
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
                "metric": _AUTOML_METRIC,
                "max_trials": {
                    "type": "integer",
                    "description": "Total candidate and tuning trials, 1–50 (default 10).",
                },
                "max_time_seconds": _AUTOML_TIME_BUDGET,
                "random_state": {
                    "type": "integer",
                    "description": "Random seed (default 42).",
                },
                "cv": {
                    "type": "integer",
                    "description": "Cross-validation folds (default 3).",
                },
                "models": _AUTOML_MODEL_SUBSET,
                "exclude_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature columns to exclude before preprocessing/training.",
                },
                "split_manifest_id": {
                    "type": "string",
                    "description": "ID of a validated split manifest to enforce indices.",
                },
                "delimiter": _AUTOML_DELIMITER,
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
        version=__version__,
        title="DevRay DataLab MCP",
        instructions=(
            "Local-first data science and machine learning. Processing and file access happen "
            "where this server runs. Discover workspace filenames with list_datasets, inspect "
            "the data before fitting, and reserve any test holdout for evaluation. Use only "
            "trusted saved model artifacts."
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
