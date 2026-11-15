# Developer Community Announcement — DevRay DataLab MCP

## Try it in 30 seconds

DevRay DataLab MCP is a local-first MCP server for data science and machine learning. Connect it
to your AI coding assistant (Claude, Cursor, Cline, Kilo Code, or OpenAI Codex) and analyze
datasets, train models, and generate predictions — all on your own computer.

## Install

```powershell
pip install devray-datalab-mcp
```

or with uvx:

```powershell
uvx devray-datalab-mcp
```

## Connect to Codex (example)

```powershell
codex mcp add devray-datalab `
  --env DEVRAY_WORKSPACE_DIR=C:\MLData\datasets `
  --env DEVRAY_ARTIFACTS_DIR=C:\MLData\artifacts `
  -- uvx --from devray-datalab-mcp devray-datalab-mcp
```

Then verify:

```powershell
codex mcp list
```

## Reproducible example

Ask your assistant:

> "Use DevRay DataLab MCP to inspect sample_classification.csv, check data quality, analyze
> correlations, train and compare classification models predicting 'purchased', export the best
> model as 'churn_model', and predict on a new record."

All processing is local. Links:

- PyPI: https://pypi.org/project/devray-datalab-mcp/
- GitHub: https://github.com/rayklanderman/devray-datalab-mcp
- MCP Registry: https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.rayklanderman%2Fdevray-datalab-mcp

MIT licensed. Contributions welcome.
