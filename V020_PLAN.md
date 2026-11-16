# DevRay DataLab MCP v0.2.0 Implementation Plan

## Baseline (Phase 1)
- 31 tests pass, 10 MCP tools, Ruff clean, PyPI/Glama/MCP Registry published.
- Issues: `cv` param ignored in `compare_models`; `cross_validate_classification` unused; training blocks MCP loop; no split manifests; no experiment tracking; limited models.

## Milestone 1 — Correctness (scientific rigor)
1. Expand `evaluator.py`: `cross_validate_classification` + `cross_validate_regression`; support KFold/StratifiedKFold/GroupKFold/StratifiedGroupKFold/TimeSeriesSplit/fixed folds.
2. Wire `cv` into `compare_models_tool` so it actually performs cross-validation (not ignored).
3. Add `cross_validate` MCP tool (real CV with per-fold metrics, mean/std, fold sizes, runtime).
4. Add `create_split_manifest`, `validate_split_manifest`, `list_split_manifests` tools (reproducible train/val/test splits with dataset fingerprints, stratified/group/time-ordered).
5. Ensure preprocessing is fitted inside each CV fold (no leakage).

## Milestone 2 — Advanced ML
6. Add ExtraTrees, HistGradientBoosting, SVM, KNN (sklearn core).
7. Optional `[boosting]` extra: XGBoost, LightGBM.
8. More metrics: balanced accuracy, PR-AUC, log loss, Brier score, median absolute error, confusion matrix.
9. Calibration (sigmoid/isotonic) via `CalibratedClassifierCV`.
10. `optimize_decision_threshold` tool (F1 / precision / recall constraints, cost matrix).

## Milestone 3 — Reliability
11. SQLite experiment tracking (`list_experiments`, `get_experiment`, `compare_experiments`, `list_models`, `load_model_handle`).
12. Non-blocking jobs (`start_training_job`, `get_training_status`, `cancel_training_job`, `list_training_jobs`) via background thread + job store.
13. Cancellation, CPU thread budget, memory monitoring, storage budget.

## Milestone 4 — Advanced analysis
14. `analyze_feature_importance` (permutation importance).
15. `detect_potential_leakage`, `analyze_feature_drift`.
16. Parquet + Excel `.xlsx` dataset support.
17. Optional SHAP explanations (`[explain]` extra).

## Milestone 5 — Release
18. Synthetic 100k churn case study + `generate_data.py`.
19. Tutorials (docs/tutorials/*.md).
20. CHANGELOG.md, ROADMAP.md update, README update.
21. Full test suite, cross-platform CI, release prep.

## Acceptance criteria
- All 10 existing tools still work; no test regressions.
- CV performs configured folds; splits reproducible; leakage protections tested.
- Advanced models train; AutoML respects budgets; models reusable after export/reload.
- Non-blocking jobs; cancellation works; resource controls tested.
- Codex can invoke enhanced tools; synthetic case study reproducible.
- CI passes; packaging complete.
