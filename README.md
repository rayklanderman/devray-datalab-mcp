# DevRay DataLab MCP

<!-- mcp-name: io.github.rayklanderman/devray-datalab-mcp -->

A local-first, open-source Model Context Protocol (MCP) server for data science
and machine learning. It lets AI assistants (Claude, Cursor, Cline, Kilo Code,
and other MCP-compatible clients) perform real exploratory data analysis and
train/evaluate machine learning models entirely on the user's own computer --
no cloud services, API keys, or paid subscriptions required.

## Status

Working local MCP server over stdio transport. All dataset processing and model
training run locally via scikit-learn, pandas, and numpy.

## Features

**Dataset tools**
- `list_datasets` -- list CSV datasets in the configured workspace (filename,
  format, size).
- `inspect_dataset` -- load a CSV and report rows, columns, data types, missing
  values, duplicate rows, numerical statistics, categorical summaries, and a
  configurable preview.
- `analyze_correlations` -- Pearson/Spearman correlations among numerical
  variables, strongest positive/negative pairs, constant/unsuitable columns.
- `analyze_data_quality` -- missing values, duplicates, constant columns,
  suspected outliers (IQR), and preprocessing recommendations without modifying
  the original data.

**Machine learning tools**
- `prepare_dataset` -- define target + task type; fit scikit-learn preprocessing
  (imputation, encoding, scaling) on training data only; preserve the fitted
  transform for prediction.
- `train_baseline` -- train logistic regression, decision tree, random forest
  (classification) or linear/ridge regression, decision tree, random forest
  (regression); train/test split with optional stratification; evaluation
  metrics.
- `compare_models` -- train and compare multiple algorithms on consistent
  splits; select the best by a user-specified metric (not accuracy by default
  for every dataset).
- `export_model` -- save a fitted model with its preprocessing pipeline,
  metadata, feature schema, hyperparameters, metrics, and package versions.
- `predict` -- load a previously exported local model, validate features,
  generate predictions (+ probabilities where supported), handle schema
  mismatches.

**System tools**
- `system_info` -- OS, CPU cores, RAM, storage, GPU detection, and whether a
  GPU training backend is installed. scikit-learn runs on CPU by default.

## Supported algorithms

Classification: Logistic Regression, Decision Tree, Random Forest.
Regression: Linear Regression, Ridge Regression, Decision Tree, Random Forest.

## Architecture

```
devray-datalab-mcp/
├── src/devray_datalab/
│   ├── server.py          # MCP stdio server, tool registry
│   ├── config.py          # runtime configuration (env overridable)
│   ├── core/
│   │   ├── loader.py      # CSV loading, dataset listing
│   │   ├── preprocessing.py  # scikit-learn ColumnTransformer pipelines
│   │   ├── trainer.py     # model training, job/resource constraints
│   │   ├── evaluator.py   # classification/regression metrics
│   │   └── hardware.py    # CPU/RAM/storage/GPU detection (psutil)
│   ├── tools/
│   │   ├── datasets.py    # list_datasets, inspect_dataset
│   │   ├── analysis.py    # correlations, data quality
│   │   ├── training.py    # prepare_dataset, train_baseline, compare_models
│   │   ├── models.py      # export_model, predict
│   │   ├── storage.py     # joblib artifact persistence
│   │   └── system.py      # system_info
│   └── utils/
│       ├── validation.py  # path traversal protection, input validation
│       └── errors.py      # structured error types
├── datasets/              # configured workspace (datasets)
├── artifacts/             # trained model artifacts
├── examples/              # example datasets
└── tests/                 # pytest suite incl. stdio e2e
```

- **Local only**: stdio transport; no network, no cloud.
- **Security**: all file access is confined to the workspace; path traversal is
  blocked; file extensions/sizes validated; only artifacts created by this app
  are loaded for prediction.
- **Resource management**: configurable CPU thread limits, training memory
  budget, single concurrent training job, training timeouts, cooperative
  cancellation.
- **No GPU deps**: PyTorch/TensorFlow/CUDA are not required or installed.

## Installation

### Prerequisites

Python 3.11+ (tested with 3.15).

### Installation methods

Choose one of the following:

#### Option A: Run directly with `uvx` (no install)

```powershell
uvx devray-datalab-mcp
```

#### Option B: Install from PyPI into a virtual environment (recommended)

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install devray-datalab-mcp
devray-datalab-mcp
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install devray-datalab-mcp
devray-datalab-mcp
```

#### Option C: Install from GitHub source

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install git+https://github.com/rayklanderman/devray-datalab-mcp.git
devray-datalab-mcp
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/rayklanderman/devray-datalab-mcp.git
devray-datalab-mcp
```

#### Option D: Editable install from a local clone (development)

```powershell
# Windows PowerShell
git clone https://github.com/rayklanderman/devray-datalab-mcp.git
cd devray-datalab-mcp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install pytest ruff
devray-datalab-mcp --workspace ./datasets --artifacts ./artifacts
```

```bash
# Linux / macOS
git clone https://github.com/rayklanderman/devray-datalab-mcp.git
cd devray-datalab-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff
devray-datalab-mcp --workspace ./datasets --artifacts ./artifacts
```

### Exact Windows PowerShell installation commands

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install devray-datalab-mcp
devray-datalab-mcp
```

Configuration via environment variables (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `DEVRAY_WORKSPACE_DIR` | `datasets` | Dataset workspace directory |
| `DEVRAY_ARTIFACTS_DIR` | `artifacts` | Model artifacts directory |
| `DEVRAY_MAX_DATASET_SIZE_BYTES` | `209715200` (200 MB) | Max dataset upload size |
| `DEVRAY_MAX_PREVIEW_ROWS` | `25` | Max preview rows |
| `DEVRAY_MAX_CPU_THREADS` | (unset) | CPU thread limit for training |
| `DEVRAY_TRAINING_MEMORY_BUDGET_MB` | (unset) | Training memory budget |
| `DEVRAY_MAX_CONCURRENT_JOBS` | `1` | Max concurrent training jobs |
| `DEVRAY_TRAINING_TIMEOUT_SECONDS` | `1800` | Training timeout |
| `DEVRAY_LOG_LEVEL` | `INFO` | Log level (logs to stderr) |

## MCP client setup

The server speaks MCP over stdio. Logging/diagnostics go to stderr, so stdout
is reserved for MCP protocol messages.

### Example client configuration

The server speaks MCP over stdio. Logging/diagnostics go to stderr, so stdout
is reserved for MCP protocol messages. Replace `C:\path\to\project` with your
actual project directory, or use the published `devray-datalab-mcp` command
when installed from PyPI.

#### Kilo Code

```json
{
  "mcpServers": {
    "devray-datalab": {
      "command": "C:\\path\\to\\project\\.venv\\Scripts\\python.exe",
      "args": ["-m", "devray_datalab.server"],
      "cwd": "C:\\path\\to\\project",
      "env": {
        "DEVRAY_WORKSPACE_DIR": "C:\\path\\to\\project\\datasets",
        "DEVRAY_ARTIFACTS_DIR": "C:\\path\\to\\project\\artifacts"
      }
    }
  }
}
```

#### Cline / Cursor / Claude Desktop

These clients use the same `mcpServers` JSON structure:

```json
{
  "mcpServers": {
    "devray-datalab": {
      "command": "devray-datalab-mcp",
      "args": [],
      "env": {
        "DEVRAY_WORKSPACE_DIR": "C:\\path\\to\\project\\datasets",
        "DEVRAY_ARTIFACTS_DIR": "C:\\path\\to\\project\\artifacts"
      }
    }
  }
}
```

When using the published `devray-datalab-mcp` command (installed from PyPI or
`uvx`), omit `cwd` and let the client resolve the executable from PATH. Set
`DEVRAY_WORKSPACE_DIR`/`DEVRAY_ARTIFACTS_DIR` to absolute paths outside the
project if you prefer dedicated dataset/artifact directories.

#### VS Code MCP-compatible tooling

Add the server to your VS Code MCP settings (`settings.json`):

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

### MCP Inspector

The official MCP Inspector can be used to test the server:

```bash
npx @modelcontextprotocol/inspector
# Connect to: devray-datalab-mcp (stdio)
# Command: devray-datalab-mcp
```

### Dataset workspace configuration

Designate authorized local directories for datasets and model artifacts using
environment variables:

- `DEVRAY_WORKSPACE_DIR` -- directory containing CSV datasets (default: `datasets`).
- `DEVRAY_ARTIFACTS_DIR` -- directory where trained model artifacts are stored (default: `artifacts`).

Both directories are created automatically if missing. All file access is
confined to these directories; path traversal is blocked. Point them at absolute
paths outside your project if you want a shared workspace across clients.

Example (PowerShell):

```powershell
$env:DEVRAY_WORKSPACE_DIR="C:\ml-data\datasets"
$env:DEVRAY_ARTIFACTS_DIR="C:\ml-data\artifacts"
devray-datalab-mcp
```


## Available tools and example arguments

| Tool | Example arguments |
|---|---|
| `list_datasets` | `{}` |
| `inspect_dataset` | `{"name": "sample_classification.csv", "max_rows": 10}` |
| `analyze_correlations` | `{"name": "sample_classification.csv", "method": "pearson", "top_n": 5}` |
| `analyze_data_quality` | `{"name": "sample_classification.csv"}` |
| `prepare_dataset` | `{"name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification"}` |
| `train_baseline` | `{"name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "model_name": "random_forest", "random_state": 42}` |
| `compare_models` | `{"name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "models": ["logistic_regression", "random_forest"], "metric": "f1"}` |
| `export_model` | `{"artifact_name": "my_model", "name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "model_name": "random_forest"}` |
| `predict` | `{"artifact_name": "my_model", "features": {"age": 40, "income": 80000, "credit_score": 720, "category": "electronics", "region": "east"}}` |
| `system_info` | `{}` |

## Example machine learning workflow

Using `sample_classification.csv` (target `purchased`, binary classification):

1. **Inspect** the dataset:
   `inspect_dataset` with `{"name": "sample_classification.csv"}`.

2. **Check data quality**:
   `analyze_data_quality` with `{"name": "sample_classification.csv"}`.

3. **Check correlations**:
   `analyze_correlations` with `{"name": "sample_classification.csv", "method": "pearson"}`.

4. **Train & compare** models:
   `compare_models` with `{"name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "metric": "f1"}`.

5. **Export** the best model:
   `export_model` with `{"artifact_name": "churn_model", "name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "model_name": "random_forest"}`.

6. **Predict** on new data:
   `predict` with `{"artifact_name": "churn_model", "features": {...}}`.

### Example regression workflow

Using `sample_regression.csv` (target `spend`, numeric target):

1. **Inspect** the dataset:
   `inspect_dataset` with `{"name": "sample_regression.csv"}`.
2. **Check data quality**:
   `analyze_data_quality` with `{"name": "sample_regression.csv"}`.
3. **Check correlations**:
   `analyze_correlations` with `{"name": "sample_regression.csv", "method": "pearson"}`.
4. **Train & compare** models:
   `compare_models` with `{"name": "sample_regression.csv", "target_column": "spend", "task_type": "regression", "metric": "r2"}`.
5. **Export** the best model:
   `export_model` with `{"artifact_name": "spend_model", "name": "sample_regression.csv", "target_column": "spend", "task_type": "regression", "model_name": "random_forest"}`.
6. **Predict** spend on new data:
   `predict` with `{"artifact_name": "spend_model", "features": {...}}`.

## Release links

- **GitHub repository:** https://github.com/rayklanderman/devray-datalab-mcp
- **GitHub releases:** https://github.com/rayklanderman/devray-datalab-mcp/releases
- **PyPI package:** https://pypi.org/project/devray-datalab-mcp/


## Example prompts

- "Inspect the sample_classification.csv dataset and tell me about missing values and column types."
- "Analyze correlations in sample_regression.csv between income and spend."
- "Train a random forest to predict 'purchased' from sample_classification.csv and compare it against logistic regression using F1."
- "Export the trained model as 'churn_model' and predict purchased for a 40-year-old in the east region with income 80000 and credit score 720."
- "Show me the CPU, RAM, and storage available for local training."

## Hardware requirements

- Designed for modest hardware (older Windows laptops with 8-16 GB RAM).
- CPU training is the default; scikit-learn does not use GPU acceleration by
  default.
- No PyTorch, TensorFlow, or CUDA required.
- Dataset size limit defaults to 200 MB.

## Security and privacy

- All processing is local; no data leaves your machine.
- Datasets and artifacts are confined to the configured workspace/artifacts
  directories.
- Path traversal and unauthorized filesystem access are blocked.
- Only CSV datasets are accepted; file size and extensions are validated.
- No arbitrary code or shell execution; no external downloads; no uploads to
  third parties.
- Uploaded datasets and model artifacts are treated as untrusted; only artifacts
  created and validated by this application are loaded for prediction.
- Original datasets are never overwritten without explicit permission.

## Limitations

- Single concurrent training job by default.
- scikit-learn models run on CPU; no GPU acceleration in this release.
- Classification stratification requires at least 2 samples per class.
- ROC-AUC is reported only for binary classification.
- CSV is the only supported dataset format.
- No built-in job queue/status polling tool; training runs within the tool call
  with a configurable timeout.
- Cancellation is cooperative (checked during iterative training loops).

## Roadmap

- Optional GPU-accelerated backend plug-in (without redesigning MCP tools).
- Training job status/polling and cancellation via MCP progress mechanisms.
- Additional models (SVM, k-NN, gradient boosting).
- More preprocessing options and feature selection.
- Dataset profiling and automated preprocessing recommendations.

## Versioning and release notes

The project follows [semantic versioning](https://semver.org/). The current
release is `v0.1.0`. Releases are published to PyPI as
`devray-datalab-mcp` and registered in the MCP Registry as
`io.github.rayklanderman/devray-datalab-mcp`.

### v0.1.0 (initial release)

- Ten MCP tools: list_datasets, inspect_dataset, analyze_correlations,
  analyze_data_quality, prepare_dataset, train_baseline, compare_models,
  export_model, predict, system_info.
- Local-first stdio transport; scikit-learn CPU training.
- Path-traversal protection, dataset size/extension validation.
- 31 automated tests including a stdio end-to-end client test.
- GitHub Actions CI/CD and PyPI Trusted Publishing workflows.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/...`).
3. Install dev dependencies: `pip install -e . pytest ruff`.
4. Run tests: `pytest` (or `.venv\Scripts\python.exe -m pytest` on Windows).
5. Run lint: `ruff check src/ tests/` and `ruff format src/ tests/`.
6. Commit with a concise message and open a pull request.

## License

MIT License. See `LICENSE`.
