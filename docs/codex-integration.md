# OpenAI Codex Integration Guide

This guide shows how to connect DevRay DataLab MCP to the OpenAI Codex CLI and the Codex VS Code
extension through local stdio execution.

DevRay DataLab MCP runs entirely on your computer — no cloud, no API keys, no paid subscriptions.

## Prerequisites

- [Codex CLI](https://github.com/openai/codex) installed (`codex` on PATH).
- [uv](https://github.com/astral-sh/uv) installed (provides `uvx`).
- DevRay DataLab MCP installed from PyPI:

```powershell
pip install devray-datalab-mcp
```

- A local directory for datasets and one for model artifacts.

## Quick installation (PowerShell)

Create the directories and copy the included example dataset:

```powershell
New-Item -ItemType Directory -Force "C:\MLData\datasets","C:\MLData\artifacts"
Copy-Item "examples\sample_classification.csv" "C:\MLData\datasets\"
```

Register the MCP server:

```powershell
codex mcp add devray-datalab `
  --env DEVRAY_WORKSPACE_DIR=C:\MLData\datasets `
  --env DEVRAY_ARTIFACTS_DIR=C:\MLData\artifacts `
  -- uvx --from devray-datalab-mcp devray-datalab-mcp
```

## Manual `config.toml` configuration

Codex also reads `~/.codex/config.toml`. On Windows this is
`%USERPROFILE%\.codex\config.toml`:

```toml
[mcp_servers.devray-datalab]
command = "uvx"
args = ["--from", "devray-datalab-mcp", "devray-datalab-mcp"]

[mcp_servers.devray-datalab.env]
DEVRAY_WORKSPACE_DIR = "C:/MLData/datasets"
DEVRAY_ARTIFACTS_DIR = "C:/MLData/artifacts"
```

Codex CLI and its VS Code extension share this configuration — register once, use everywhere.

## Verify the registration

```powershell
codex mcp list
codex mcp get devray-datalab
```

Expected output:

```
devray-datalab
  enabled: true
  transport: stdio
  command: uvx
  args: --from devray-datalab-mcp devray-datalab-mcp
  cwd: -
  env: DEVRAY_ARTIFACTS_DIR=*****, DEVRAY_WORKSPACE_DIR=*****
```

## Codex VS Code extension

1. Install and sign in to the official OpenAI Codex VS Code extension.
2. Register DevRay DataLab MCP using the Codex CLI (`codex mcp add ...`) or `config.toml` above.
3. Reload the VS Code window / restart the Codex session if needed.
4. Confirm the MCP server appears in the Codex MCP panel.
5. Allow Codex to invoke the project's tools.
6. Run an example dataset analysis workflow (see prompts below).

Codex CLI and the Codex VS Code extension use shared configuration, so they do not need separate
registrations. Codex is distinct from GitHub Copilot; registering the MCP with Codex does not
automatically register it with Copilot, and vice versa.

## Example prompts for Codex

Dataset exploration:

> "Use DevRay DataLab MCP to inspect sample_classification.csv, summarize its structure, identify missing values, and report potential data-quality issues."

Machine learning:

> "Use DevRay DataLab MCP to train and evaluate classification models for predicting purchased. Compare appropriate metrics and recommend the best model based on validation performance."

Prediction:

> "Use DevRay DataLab MCP to load my exported churn_model and generate predictions for new records."

## Tool verification

Once connected, you can ask Codex to invoke any of the ten DevRay DataLab MCP tools:

- `list_datasets`
- `inspect_dataset`
- `analyze_correlations`
- `analyze_data_quality`
- `prepare_dataset`
- `train_baseline`
- `compare_models`
- `export_model`
- `predict`
- `system_info`

All data processing and model training run locally on your CPU and RAM.

## Remove or update the registration

```powershell
codex mcp remove devray-datalab
```

Then re-run `codex mcp add` if you need to change paths or environment variables.

## Privacy considerations

- Datasets and model artifacts are stored only in your configured local directories.
- Codex sends your prompts to OpenAI, but the underlying data files and model artifacts never leave
  your machine — the MCP server reads them locally via stdio.
- Review Codex's privacy settings if you are working with sensitive data.

## Linux / macOS equivalents

Replace PowerShell commands with your shell:

```bash
mkdir -p ~/MLData/datasets ~/MLData/artifacts
cp examples/sample_classification.csv ~/MLData/datasets/

codex mcp add devray-datalab \
  --env DEVRAY_WORKSPACE_DIR=~/MLData/datasets \
  --env DEVRAY_ARTIFACTS_DIR=~/MLData/artifacts \
  -- uvx --from devray-datalab-mcp devray-datalab-mcp
```

In `config.toml`, use forward-slash paths (e.g. `/Users/you/MLData/datasets`).
