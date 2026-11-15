# DevRay DataLab MCP

**Local-first Data Science & Machine Learning for AI Assistants**

[![PyPI version](https://badge.fury.io/py/devray-datalab-mcp.svg)](https://pypi.org/project/devray-datalab-mcp/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/devray-datalab-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/rayklanderman/devray-datalab-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/rayklanderman/devray-datalab-mcp/actions/workflows/ci.yml)
[![MCP Registry](https://img.shields.io/badge/MCP%20Registry-active-green)](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.rayklanderman%2Fdevray-datalab-mcp)
[![Transport: stdio](https://img.shields.io/badge/transport-stdio-purple)](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.rayklanderman%2Fdevray-datalab-mcp)

<!-- mcp-name: io.github.rayklanderman/devray-datalab-mcp -->

DevRay DataLab MCP is a local-first Model Context Protocol (MCP) server that lets AI assistants
such as Claude, Cursor, Cline, and Kilo Code perform **real** data science and machine learning
on your own computer — no cloud services, no API keys, no paid subscriptions.

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
| `DEVRAY_WORKSPACE_DIR` | `datasets` | Directory containing CSV datasets |
| `DEVRAY_ARTIFACTS_DIR` | `artifacts` | Directory for saved model artifacts |

## MCP client configuration

Add the server to your MCP-compatible client. Replace `C:\MLData` with your own paths.

### Kilo Code

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

### Cline / Cursor / Claude Desktop

These clients use the same `mcpServers` JSON structure:

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

### VS Code / GitHub Copilot MCP

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

## Available tools

| Tool | What it does |
|---|---|
| `list_datasets` | List CSV datasets in the workspace |
| `inspect_dataset` | Load a CSV; report types, missing values, stats, preview |
| `analyze_correlations` | Pearson/Spearman correlations, strongest pairs |
| `analyze_data_quality` | Missing values, duplicates, constants, outliers, recommendations |
| `prepare_dataset` | Define target + task type; fit preprocessing on train data only |
| `train_baseline` | Train a classification or regression model |
| `compare_models` | Compare multiple algorithms; select best by metric |
| `export_model` | Save model with pipeline, metadata, and schema |
| `predict` | Generate predictions from a saved local model |
| `system_info` | CPU, RAM, storage, GPU detection |

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
- [Source code](https://github.com/rayklanderman/devray-datalab-mcp)
- [Issues](https://github.com/rayklanderman/devray-datalab-mcp/issues)
- [Release notes](https://github.com/rayklanderman/devray-datalab-mcp/releases/tag/v0.1.0)

## Security and privacy

All processing is local; no data leaves your machine. File access is confined to the
configured workspace/artifacts directories. Path traversal is blocked, file sizes and
extensions are validated, and only artifacts created by this application are loaded for
prediction. No cloud hosting, no paid AI APIs, no external downloads.

## License

MIT License. See `LICENSE`.
