# Roadmap

DevRay DataLab MCP follows [semantic versioning](https://semver.org/). Releases are published to PyPI as `devray-datalab-mcp` and registered in the official MCP Registry as `io.github.rayklanderman/devray-datalab-mcp`.

## v0.3.0 — Released

The following v0.3.0 items have been implemented and tested:

- Non-blocking training jobs: `start_training_job`, `get_training_status`, `cancel_training_job`, `list_training_jobs` (training runs off the MCP request loop with cooperative cancellation).
- Automated Machine Learning engine: `automl_run`, `tune_model`, `get_model_leaderboard` (Optuna-based tuning when installed, deterministic grid search otherwise).
- Dataset formats: Parquet (`.parquet`) and Excel (`.xlsx`/`.xls`) via optional `[parquet]`/`[excel]` dependencies.
- Reproducible split manifests: `create_split_manifest`, `validate_split_manifest`, `list_split_manifests` (stratified/group/time-ordered splits with dataset fingerprints; protected holdout).
- Feature diagnostics: `detect_potential_leakage`, `analyze_feature_drift` (KS test for numeric, chi-square for categorical).
- Optional dependency extras: `[parquet]` (pyarrow), `[excel]` (openpyxl).

## v0.4.0 (proposed — not implemented)

The following items are **deferred to a future release**:

- Public reproducible case study (synthetic 100k-record churn dataset) and original learning tutorials.
- Optional SHAP explanations (`[explain]` extra).
- Feature selection and automated preprocessing recommendations.
- Batch prediction workflows.
- Optional GPU-accelerated backend plug-in (without redesigning MCP tools).

## Longer-term

- Dataset profiling.
- Experiment tracking UI/visualization.
- Multi-dataset join support.

## Versioning

Releases are published to PyPI as `devray-datalab-mcp` and registered in the MCP Registry as `io.github.rayklanderman/devray-datalab-mcp`. Release notes accompany each GitHub release.

