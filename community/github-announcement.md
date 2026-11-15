# GitHub Announcement — DevRay DataLab MCP v0.1.0

## Title
Launching DevRay DataLab MCP: local-first data science & machine learning for AI assistants

## Body

We're excited to announce the first public release of **DevRay DataLab MCP** (v0.1.0), a
local-first Model Context Protocol (MCP) server for dataset analysis, preprocessing, machine
learning training, model evaluation, and prediction.

## What it is

DevRay DataLab MCP lets AI coding assistants — Claude, Cursor, Cline, Kilo Code, and OpenAI
Codex — perform real data science and machine learning entirely on your computer. No cloud
services, no API keys, no paid subscriptions. Everything runs locally via stdio using your own
CPU and RAM.

## What it does

- Load and inspect CSV datasets
- Exploratory data analysis (missing values, duplicates, correlations, outliers)
- Preprocess data (imputation, encoding, scaling) fit on training data only — no leakage
- Train classification and regression models (logistic regression, decision trees, random forests,
  linear/ridge regression)
- Compare multiple models and select the best by a user-chosen metric
- Export trained models with preprocessing pipeline, metadata, and schema
- Generate predictions from saved local models
- Report local hardware (CPU, RAM, storage, GPU)

## How to install

```powershell
pip install devray-datalab-mcp
devray-datalab-mcp
```

Or with uvx (no install):

```powershell
uvx devray-datalab-mcp
```

## Links

- PyPI: https://pypi.org/project/devray-datalab-mcp/
- GitHub: https://github.com/rayklanderman/devray-datalab-mcp
- MCP Registry: https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.rayklanderman%2Fdevray-datalab-mcp
- Docs: https://github.com/rayklanderman/devray-datalab-mcp#readme

## License

MIT. Open source.
