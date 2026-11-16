# Changelog

## [0.3.0] - Released

### Added
- Non-blocking training jobs: `start_training_job`, `get_training_status`, `cancel_training_job`, `list_training_jobs` (training runs off the MCP request loop with cooperative cancellation).
- Automated Machine Learning engine: `automl_run`, `tune_model`, `get_model_leaderboard` (Optuna-based tuning when installed, deterministic grid search otherwise).
- Dataset formats: Parquet (`.parquet`) and Excel (`.xlsx`/`.xls`) via optional `[parquet]`/`[excel]` dependencies.
- Reproducible split manifests: `create_split_manifest`, `validate_split_manifest`, `list_split_manifests` (stratified/group/time-ordered splits with dataset fingerprints; protected holdout).
- Feature diagnostics: `detect_potential_leakage`, `analyze_feature_drift` (KS test for numeric, chi-square for categorical).
- Optional dependency extras: `[parquet]` (pyarrow), `[excel]` (openpyxl).

### Changed
- Server version bumped to 0.3.0.
- `list_datasets` now lists `.parquet`, `.xlsx`, and `.xls` in addition to `.csv`.

## [0.2.0] - Released

### Added
- Real cross-validation with leakage-safe preprocessing (preprocessing fitted independently inside each fold).
  - Strategies: KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit.
  - Per-fold metrics, mean/std, fold sizes, runtime, dataset fingerprint.
- The `cv` parameter of `compare_models` now actually performs cross-validation (previously ignored).
- Additional classification metrics: balanced accuracy, macro/weighted F1, PR-AUC, log loss, Brier score, confusion matrix.
- Additional regression metrics: median absolute error, MAPE.
- Probability calibration: `CalibratedClassifierCV` with sigmoid and isotonic methods.
- Decision-threshold optimization: maximize F1/precision/recall or minimize a cost matrix, with optional precision/recall constraints.
- Permutation feature importance with baseline score and per-feature mean/std.
- Experiment tracking (SQLite): `list_experiments`, `get_experiment`, `compare_experiments`, `list_models`, `load_model_handle`.
  - Tracks dataset fingerprint, feature schema, algorithm, hyperparameters, library versions, training duration, validation metrics, resource consumption, warnings, failure states, and exported artifacts.
- New models (scikit-learn core, no extra install):
  - Classification: extra_trees, hist_gradient_boosting, svm, knn.
  - Regression: extra_trees, hist_gradient_boosting, svm, knn.
- Optional dependency extras: `[boosting]` (xgboost, lightgbm), `[explain]` (shap).
- Experiments DB path configurable via `DEVRAY_EXPERIMENTS_DB_PATH`.

### Changed
- `compare_models` now reports cross-validation results (mean/std per metric) instead of only a single train/test split.
- `validate_model_name` accepts an optional `task_type` to validate against the appropriate model set.

### Fixed
- `cv` parameter in `compare_models` was accepted but ignored; it now drives real cross-validation.

### Deprecated / Removed
- None. All ten v0.1.0 MCP tools remain available with backward-compatible arguments.

## [0.1.0] - Initial release

- Ten MCP tools: list_datasets, inspect_dataset, analyze_correlations, analyze_data_quality, prepare_dataset, train_baseline, compare_models, export_model, predict, system_info.
- Local-first stdio transport; scikit-learn CPU training.
- Path-traversal protection, dataset size/extension validation.
- 31 automated tests including a stdio end-to-end client test.
- GitHub Actions CI/CD and PyPI Trusted Publishing workflows.
- Registered in the official MCP Registry: `io.github.rayklanderman/devray-datalab-mcp`.
