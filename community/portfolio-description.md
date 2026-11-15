# Portfolio Project Description — DevRay DataLab MCP

## Project
DevRay DataLab MCP — Local-first Data Science & Machine Learning MCP Server

## One-liner
An open-source Model Context Protocol (MCP) server that lets AI assistants perform real dataset
analysis, preprocessing, and machine learning model training entirely on the user's own computer
— no cloud, no API keys, no paid subscriptions.

## Technology stack
- Python 3.11+
- Official MCP Python SDK (stdio transport)
- Pandas / NumPy (data processing)
- Scikit-learn (machine learning)
- SciPy (statistical analysis)
- Joblib (local model persistence)
- Psutil (hardware monitoring)
- Pytest + Ruff (testing / linting)

## Architecture
Local-first stdio MCP server. All dataset processing and model training run locally via
scikit-learn on the user's CPU and RAM. Path-traversal protection, dataset size/extension
validation, preprocessing fit on training data only (no data leakage), and single concurrent
training job by default. No PyTorch, TensorFlow, or CUDA dependencies.

## Features (10 MCP tools)
`list_datasets`, `inspect_dataset`, `analyze_correlations`, `analyze_data_quality`,
`prepare_dataset`, `train_baseline`, `compare_models`, `export_model`, `predict`, `system_info`

## Supported algorithms
Classification: Logistic Regression, Decision Tree, Random Forest
Regression: Linear Regression, Ridge Regression, Decision Tree, Random Forest

## Distribution
- PyPI: devray-datalab-mcp v0.1.0
- GitHub: rayklanderman/devray-datalab-mcp
- Official MCP Registry: io.github.rayklanderman/devray-datalab-mcp (active, isLatest, stdio)

## License
MIT
