# DevRay DataLab MCP

**Local-first Data Science & Machine Learning for AI Assistants**

[![PyPI version](https://badge.fury.io/py/devray-datalab-mcp.svg)](https://pypi.org/project/devray-datalab-mcp/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/devray-datalab-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/rayklanderman/devray-datalab-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/rayklanderman/devray-datalab-mcp/actions/workflows/ci.yml)
[![MCP Registry](https://img.shields.io/badge/MCP%20Registry-active-green)](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.rayklanderman%2Fdevray-datalab-mcp)
[![DevRay DataLab MCP MCP server – quality and maintenance score on Glama](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp/badges/card.svg)](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp)
[![Transport: stdio](https://img.shields.io/badge/transport-stdio-purple)](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.rayklanderman%2Fdevray-datalab-mcp)

<!-- mcp-name: io.github.rayklanderman/devray-datalab-mcp -->

DevRay DataLab MCP is a local-first Model Context Protocol (MCP) server that lets AI assistants
such as OpenAI Codex, Cursor, Claude Desktop, Cline, Kilo Code, and others perform **real** data
science and machine learning on your own computer — no cloud services, no API keys, no paid
subscriptions.

Everything runs locally via stdio: dataset loading, exploratory data analysis, preprocessing,
classification/regression training, model comparison, export, and prediction. Your datasets
never leave your machine.

## Installation

### Method A — `uvx` (simplest, no virtualenv)

```powershell
uvx devray-datalab-mcp
```

Requires [uv](https://github.com/astral-sh/uv). Install uv if needed:

```powershell
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Method B — `pip` in a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install devray-datalab-mcp
devray-datalab-mcp
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install devray-datalab-mcp
devray-datalab-mcp
```

### Method C — from GitHub source

```powershell
pip install git+https://github.com/rayklanderman/devray-datalab-mcp.git
devray-datalab-mcp
```

> A stdio MCP server waits for input from an MCP client. Launching `devray-datalab-mcp` directly
> without a connected client will appear to do nothing — this is expected. Connect it through an
> MCP-compatible client (see client configuration below).

Configure where datasets and model artifacts live with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `DEVRAY_WORKSPACE_DIR` | `datasets` | Dataset directory on the machine running the server |
| `DEVRAY_ARTIFACTS_DIR` | `artifacts` | Directory for saved model artifacts |
| `DEVRAY_EXPERIMENTS_DB_PATH` | `experiments.db` | SQLite experiment database path. Its parent directory must be writable. |

## MCP client configuration

Add the server to your MCP-compatible client. Replace `C:\MLData` with your own paths.

### 1. OpenAI Codex (CLI & VS Code)

DevRay DataLab MCP connects to the OpenAI Codex CLI and its VS Code extension through local stdio execution. Codex and its VS Code extension share configuration, so registering the server once makes it available in both.

**Prerequisites**

- Install the [Codex CLI](https://github.com/openai/codex) (`codex` on PATH).
- Install [uv](https://github.com/astral-sh/uv) (provides `uvx`).
- Create the dataset and artifact directories before registering:

```powershell
New-Item -ItemType Directory -Force "C:\MLData\datasets","C:\MLData\artifacts"
Copy-Item "examples\sample_classification.csv" "C:\MLData\datasets\"
```

**Register the server**

```powershell
codex mcp add devray-datalab `
  --env DEVRAY_WORKSPACE_DIR=C:\MLData\datasets `
  --env DEVRAY_ARTIFACTS_DIR=C:\MLData\artifacts `
  -- uvx --from devray-datalab-mcp devray-datalab-mcp
```

- `DEVRAY_WORKSPACE_DIR` — directory containing your CSV datasets.
- `DEVRAY_ARTIFACTS_DIR` — directory where trained model artifacts are saved.
- Replace `C:\MLData` with your own paths.

**Verify registration**

```powershell
codex mcp list
codex mcp get devray-datalab
```

Expected output shows `enabled: true`, `transport: stdio`, `command: uvx`, and the `devray-datalab-mcp` args.

**Manual `config.toml` configuration**

Codex also reads `~/.codex/config.toml` (Windows: `%USERPROFILE%\.codex\config.toml`):

```toml
[mcp_servers.devray-datalab]
command = "uvx"
args = ["--from", "devray-datalab-mcp", "devray-datalab-mcp"]

[mcp_servers.devray-datalab.env]
DEVRAY_WORKSPACE_DIR = "C:/MLData/datasets"
DEVRAY_ARTIFACTS_DIR = "C:/MLData/artifacts"
```

Codex CLI and its VS Code extension share this configuration. Register once, use everywhere.

### 2. Cursor

Cursor uses the standard MCP client configuration format.

Add to your Cursor MCP settings (`settings.json`):

```json
{
  "mcp.servers": {
    "devray-datalab": {
      "command": "devray-datalab-mcp",
      "args": [],
      "env": {
        "DEVRAY_WORKSPACE_DIR": "C:\\MLData\\datasets",
        "DEVRAY_ARTIFACTS_DIR": "C:\\MLData\\artifacts"
      }
    }
  }
}
```

Replace `C:\MLData` with your own paths. Save the file and restart Cursor for changes to take effect.

### 3. Claude Desktop

Add to your Claude Desktop MCP settings (`claude_desktop_config.json`, typically located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "devray-datalab": {
      "command": "devray-datalab-mcp",
      "args": [],
      "env": {
        "DEVRAY_WORKSPACE_DIR": "C:\\MLData\\datasets",
        "DEVRAY_ARTIFACTS_DIR": "C:\\MLData\\artifacts"
      }
    }
  }
}
```

Replace `C:\MLData` with your own paths. Save the file and restart Claude Desktop for changes to take effect.

### 4. Cline

Add to your Cline MCP settings (`settings.json`):

```json
{
  "mcp.servers": {
    "devray-datalab": {
      "command": "devray-datalab-mcp",
      "args": [],
      "env": {
        "DEVRAY_WORKSPACE_DIR": "C:\\MLData\\datasets",
        "DEVRAY_ARTIFACTS_DIR": "C:\\MLData\\artifacts"
      }
    }
  }
}
```

Replace `C:\MLData` with your own paths. Save the file and restart Cline for changes to take effect.

### 5. VS Code / GitHub Copilot MCP

Add to your VS Code MCP settings (`settings.json`):

```json
{
  "mcp.servers": {
    "devray-datalab": {
      "command": "devray-datalab-mcp",
      "args": [],
      "env": {
        "DEVRAY_WORKSPACE_DIR": "${workspaceFolder}/datasets",
        "DEVRAY_ARTIFACTS_DIR": "${workspaceFolder}/artifacts"
      }
    }
  }
}
```

### 6. Other MCP clients

DevRay DataLab MCP works with any MCP-compatible client that supports stdio transport. Use `devray-datalab-mcp` as the command with no arguments, and set `DEVRAY_WORKSPACE_DIR` and `DEVRAY_ARTIFACTS_DIR` as environment variables.

### 7. Kilo Code

```json
{
  "mcpServers": {
    "devray-datalab": {
      "command": "uvx",
      "args": [
        "--from",
        "devray-datalab-mcp",
        "devray-datalab-mcp"
      ],
      "env": {
        "DEVRAY_WORKSPACE_DIR": "C:\\MLData\\datasets",
        "DEVRAY_ARTIFACTS_DIR": "C:\\MLData\\artifacts"
      }
    }
  }
}
```

## Available tools

| Tool | What it does | Category |
|---|---|---|
| `list_datasets` | List CSV datasets in the workspace | Dataset |
| `inspect_dataset` | Load a CSV; report types, missing values, stats, preview | Dataset |
| `analyze_correlations` | Pearson/Spearman correlations, strongest pairs | Analysis |
| `analyze_data_quality` | Missing values, duplicates, constants, outliers, recommendations | Analysis |
| `detect_potential_leakage` | Flag potential data-leakage issues in a dataset | Analysis |
| `analyze_feature_drift` | Detect feature drift between reference and current datasets | Analysis |
| `prepare_dataset` | Define target + task type; fit preprocessing on train data only | Preprocessing |
| `train_baseline` | Train a classification or regression model | Training |
| `start_training_job` | Submit a training job (non-blocking); returns a job id | Training |
| `get_training_status` | Return the status and result of a training job | Training |
| `cancel_training_job` | Request cancellation of a queued or running training job | Training |
| `list_training_jobs` | List training jobs, newest first | Training |
| `compare_models` | Compare multiple algorithms; select best by metric (with cross-validation) | Training |
| `automl_run` | Run automated ML: train candidates with CV, rank by objective | AutoML |
| `tune_model` | Tune hyperparameters of a single model with cross-validation | AutoML |
| `get_model_leaderboard` | Run a fresh AutoML search and return ranked trials and the winner | AutoML |
| `export_model` | Save model with pipeline, metadata, and schema | Models |
| `predict` | Generate predictions from a saved local model | Models |
| `list_experiments` | List recorded training experiments, newest first | Experiments |
| `get_experiment` | Get a single recorded experiment by ID | Experiments |
| `compare_experiments` | Compare multiple recorded experiments by ID | Experiments |
| `list_models` | List experiments that produced a model artifact | Experiments |
| `load_model_handle` | Load a previously exported model artifact by experiment ID or name | Experiments |
| `create_split_manifest` | Create a reproducible train/validation/test split manifest | Data Splits |
| `validate_split_manifest` | Validate a stored split manifest against a dataset | Data Splits |
| `list_split_manifests` | List all stored split manifests, newest first | Data Splits |
| `system_info` | CPU, RAM, storage, GPU detection | System |

## Supported algorithms

Classification: Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, SVM, K-Nearest Neighbors.

Regression: Linear Regression, Ridge Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, SVM, K-Nearest Neighbors.

Additional models (XGBoost, LightGBM) are available via the optional `devray-datalab-mcp[boosting]` extra.

Optional dependencies:
- `devray-datalab-mcp[boosting]` — XGBoost, LightGBM
- `devray-datalab-mcp[explain]` — SHAP explanations
- `devray-datalab-mcp[parquet]` — Parquet file support (pyarrow)
- `devray-datalab-mcp[excel]` — Excel file support (openpyxl)

## What's new in v0.3.4

- All 27 tool descriptions explain when to use them, what they return, and where they read or write data.
- `get_model_leaderboard` now accepts `models`, for example `["decision_tree", "logistic_regression"]` for classification. Omit it for the default candidates. The search evaluates explicit IDs in order within the available budget and ranks completed trials by the selection metric.
- `automl_run`, `tune_model`, and `get_model_leaderboard` honor explicit CSV delimiters. Install `optuna` into the server's Python environment to enable hyperparameter search. The base install scores the selected candidates with cross-validation.
- The reliability repairs cover chronological CV, reproducible tuning with a shared deadline, cancellation, model coverage, and prediction validation.

See the [v0.3.4 release notes draft](docs/release-notes-v0.3.4.md) for details.

## What's new in v0.3.2

- **Faster native MCP initialization**: tool modules load lazily on first call, keeping server startup fast; no heavy ML imports at startup.
- **All 27 MCP tools correctly registered**: every tool listed in the server spec is available via `tools/list`.
- **Leakage-safe training with excluded features**: `exclude_columns` honored across prepare/train/compare/export/jobs/automl; preprocessing fitted on train data only.
- **Chronological splitting and reusable split manifests**: `row_order` time-based splitting plus `create_split_manifest`/`validate_split_manifest`/`list_split_manifests` with dataset fingerprints.
- **Protected test-holdout isolation**: test set kept as protected holdout, isolated from training.
- **AutoML, tuning and background-job safeguards**: cooperative cancellation, trial and time limits.
- **Model export metadata improvements**: schema, fingerprint, and library versions recorded with exported artifacts.
- **Unseen-category warnings in predictions**: warns when categorical features were not seen during training.

## What's new in v0.3.1

- **CSV delimiter detection**: Automatic detection of comma, semicolon, tab, and pipe delimiters, with explicit override support.
- **Package version consistency**: `__version__` now correctly reports `0.3.1`.
- **CSV parsing regression tests**: 7 new tests covering all delimiter types including semicolon-delimited data with 21 columns.
- **Full tool compatibility**: All 27 MCP tools work correctly with the new loader.

## What's new in v0.3.0

- **Non-blocking training jobs**: `start_training_job`, `get_training_status`, `cancel_training_job`, `list_training_jobs` — training runs off the MCP request loop with cooperative cancellation.
- **Automated Machine Learning**: `automl_run`, `tune_model`, `get_model_leaderboard` — train candidates with CV, rank by objective, and optionally tune with Optuna.
- **More dataset formats**: Parquet (`.parquet`) and Excel (`.xlsx`/`.xls`) via optional `[parquet]`/`[excel]` dependencies.
- **Reproducible split manifests**: `create_split_manifest`, `validate_split_manifest`, `list_split_manifests` — stratified/group/time-ordered splits with dataset fingerprints; test set is a protected holdout.
- **Feature diagnostics**: `detect_potential_leakage`, `analyze_feature_drift` — flag potential target leakage and distribution drift.

## What's new in v0.2.0

- **Real cross-validation**: `compare_models` now performs actual cross-validation (the `cv` parameter was previously ignored). Preprocessing is fitted independently inside each fold to prevent data leakage. Strategies: KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit.
- **Richer metrics**: balanced accuracy, macro/weighted F1, PR-AUC, log loss, Brier score, confusion matrix (classification); median absolute error, MAPE (regression).
- **Calibration**: probability calibration with sigmoid or isotonic methods.
- **Threshold optimization**: maximize F1/precision/recall or minimize a cost matrix, with optional precision/recall constraints.
- **Feature importance**: permutation importance with baseline score and per-feature mean/std.
- **Experiment tracking**: SQLite-backed recording of training runs (dataset fingerprint, feature schema, algorithm, hyperparameters, library versions, metrics, duration, artifacts). Tools: `list_experiments`, `get_experiment`, `compare_experiments`, `list_models`, `load_model_handle`.

## Quickstart workflow

Using the included `sample_classification.csv` (target `purchased`):

1. Inspect: `inspect_dataset` with `{"name": "sample_classification.csv"}`
2. Data quality: `analyze_data_quality` with `{"name": "sample_classification.csv"}`
3. Correlations: `analyze_correlations` with `{"name": "sample_classification.csv", "method": "pearson"}`
4. Train & compare: `compare_models` with `{"name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "metric": "f1"}`
5. Export: `export_model` with `{"artifact_name": "churn_model", "name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "model_name": "random_forest"}`
6. Predict: `predict` with `{"artifact_name": "churn_model", "features": {"age": 40, "income": 80000, "credit_score": 720, "category": "electronics", "region": "east"}}`

A full step-by-step tutorial with copy-ready prompts is available in [`docs/quickstart.md`](docs/quickstart.md).

## Links

- [Install from PyPI](https://pypi.org/project/devray-datalab-mcp/)
- [MCP Registry listing](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.rayklanderman%2Fdevray-datalab-mcp)
- [Glama listing](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp)
- [Source code](https://github.com/rayklanderman/devray-datalab-mcp)
- [Issues](https://github.com/rayklanderman/devray-datalab-mcp/issues)
- [Release notes](https://github.com/rayklanderman/devray-datalab-mcp/releases)

## Security and privacy

All processing is local; no data leaves your machine. File access is confined to the
configured workspace/artifacts directories. Path traversal is blocked, file sizes and
extensions are validated, and only artifacts created by this application are loaded for
prediction. No cloud hosting, no paid AI APIs, no external downloads.

## License

MIT License. See `LICENSE`.
