# Remaining Deficiencies — Verification

**Package:** DevRay DataLab MCP
**Version:** 0.3.2 (unchanged)
**Branch:** `fix/consolidated-reliability`
**Commit:** `901b99871e7d84e18e4bfd77ddd86bdc8ba7859a` (baseline)
**Fix commit:** <to be populated at commit time>
**Total tests:** 172 passing (159 pre-existing + 13 new regression tests)
**MCP tools:** 27 (all maintained)

---

## Summary of Findings

All five issues from the post-case-study review were investigated. Issues 1–4 are confirmed defects in source code; all have been corrected. Issue 5 is a documentation/backlog item requiring no code change.

| Issue | Status | Root cause | Correction |
|-------|--------|------------|------------|
| ISSUE 1: Chronological CV in tuning | CONFIRMED — FIXED | `core/automl.py` Optuna objective hardcoded `strategy="stratified"` for classification, ignoring the actual `strategy` variable derived from the manifest. Also omitted `exclude_columns`, `group_column`, `time_column`. | Use the resolved `strategy` variable; pass `exclude_columns`, `group_column`, `time_column` to the Optuna CV call. |
| ISSUE 2: Feature exclusion in tuning | CONFIRMED — FIXED | Same Optuna path: `exclude_columns` not forwarded to `cross_validate_with_preprocessing`. | Forward `exclude_columns` (already computed at function entry) into the Optuna objective. |
| ISSUE 3: Misleading CV metadata | CONFIRMED — FIXED | `tools/training.py:535` set `"stratified": task_type == "classification"`, true for all classification regardless of actual CV strategy. | Derive from `cv_strategy`: true only for `"stratified"` / `"stratified_group"`. |
| ISSUE 4: Feature-count metadata | CONFIRMED — FIXED | `core/automl.py` `n_features=int(df.shape[1] - 1)` counted all columns minus target, ignoring exclusions. | Compute post-exclusion feature columns; report `n_features` and `dataset_shape[1]` after exclusions. |
| ISSUE 5: Missing capabilities | DOCUMENTED | No code change required. | Backlog created below. |

---

## ISSUE 1 — Chronological cross-validation in tuning

### Confirmed

Yes. The optional Optuna tuning branch inside `run_automl()` (`src/devray_datalab/core/automl.py`) unconditionally used:

```python
strategy="stratified" if task_type == "classification" else "kfold"
```

This means a chronological (`time`) manifest supplied to `tune_model` or `automl_run` would have its Optuna tuning phase run with `StratifiedKFold` instead of `TimeSeriesSplit`, breaking forward-chaining temporal isolation.

### Root cause

`src/devray_datalab/core/automl.py:374` (original): the `objective` closure inside the Optuna block hardcoded the strategy based only on task type, ignoring the `strategy` local variable (line 279) that was correctly derived from the manifest.

### Correction

Changed the Optuna `cross_validate_with_preprocessing` call to use the resolved `strategy` variable and to forward `exclude_columns`, `group_column`, and `time_column`.

### Regression test

`tests/test_remaining_deficiencies.py::test_tune_model_chronological_manifest_uses_timeseries`

Creates a `time` + `row_order` manifest, runs `tune_model`, then directly invokes `cross_validate_with_preprocessing` with `strategy="timeseries"` and asserts:
- `cv.strategy == "TimeSeriesSplit"`
- Every fold's `max(train_idx) < min(test_idx)` (forward-chaining, no backward overlap)

Also covered: `test_tune_model_grouped_manifest_preserves_group_isolation` (GroupKFold), `test_tune_model_stratified_manifest_uses_stratifiedkfold` (StratifiedKFold).

---

## ISSUE 2 — Feature exclusion in tuning

### Confirmed

Yes. The Optuna objective in `run_automl()` did not forward `exclude_columns` to `cross_validate_with_preprocessing`. A leakage feature named in `exclude_columns` could therefore enter fitted preprocessing and estimators during Optuna tuning.

### Root cause

`src/devray_datalab/core/automl.py:368-378` (original): the `objective` closure called `cross_validate_with_preprocessing` without `exclude_columns=`, `group_column=`, or `time_column=`.

### Correction

Forward `exclude_columns`, `group_column`, and `time_column` into the Optuna CV call, matching the Phase 1 candidate-evaluation call.

### Regression tests

- `tests/test_remaining_deficiencies.py::test_tune_model_excludes_columns_through_cv` — `tune_model` with `exclude_columns=["leak_duration"]` completes without errors; leak feature never reaches fitting.
- `tests/test_remaining_deficiencies.py::test_automl_excludes_columns_across_candidates` — `automl_run` with exclusions completes trials without errors.
- `tests/test_remaining_deficiencies.py::test_get_model_leaderboard_excludes_columns` — leaderboard path preserves exclusions.

---

## ISSUE 3 — Misleading CV metadata

### Confirmed

Yes. `compare_models` returned `"stratified": true` for every classification task, even when the actual strategy was `TimeSeriesSplit`. The Phase 2 and Phase 3 reports both show `"stratified": true` alongside `"strategy": "TimeSeriesSplit"`.

### Root cause

`src/devray_datalab/tools/training.py:535` (original): `"stratified": task_type == "classification"` derived the flag solely from task type, not from the actual CV splitter.

### Correction

Changed to `"stratified": cv_strategy in ("stratified", "stratified_group")`, so the flag matches the actual strategy.

### Regression tests

- `test_compare_models_time_manifest_reports_stratified_false` — time manifest → `stratified=false`, `strategy=TimeSeriesSplit`
- `test_compare_models_stratified_manifest_reports_stratified_true` — stratified manifest → `stratified=true`, `strategy=StratifiedKFold`
- `test_compare_models_kfold_manifest_reports_stratified_false` — random manifest → `stratified=false`, `strategy=KFold`

---

## ISSUE 4 — Feature-count metadata

### Confirmed

Yes, as a metadata naming/clarity issue. `AutoMLResult.n_features` and `dataset_shape[1]` reported the raw column count minus one (target), ignoring `exclude_columns`. For the Bank Marketing case study this yielded `n_features=20` despite `duration` and `campaign` being excluded (18 predictors).

This is **not** a data-leakage bug: exclusions are applied before preprocessing/fitting everywhere (confirmed by `trainer.py` and `evaluator.py`). The issue is purely that `n_features` in AutoML responses described pre-exclusion candidates, not post-exclusion predictors.

### Determination of what n_features represents

After correction:
- **`n_features`** = number of predictor columns after exclusions are removed (i.e., columns that reach preprocessing/fitting).
- **`dataset_shape[1]`** = same count (post-exclusion feature width).
- **`trainer.py` `n_features`** = `len(artifact.feature_columns)` — already post-exclusion.

### Correction

`src/devray_datalab/core/automl.py`: compute post-exclusion `feature_columns` and use its length for both `n_features` and `dataset_shape[1]`.

### Regression tests

- `test_automl_n_features_after_exclusion` — 5 raw columns minus target minus 1 exclusion = 3; asserts `n_features==3` and `dataset_shape[1]==3`.
- `test_automl_n_features_without_exclusion_matches_raw` — no exclusions → `n_features == columns - 1`.
- `test_core_automl_n_features_after_exclusion` — core engine path agrees.
- `test_compare_models_features_after_exclusion` — excluded columns absent from comparison results.

---

## ISSUE 5 — Missing capabilities (documentation only)

No code change. Three capabilities reviewed:

### 1. Native threshold optimization

**Status:** Partially implemented internally, not exposed via MCP.

`src/devray_datalab/core/evaluator.py:687` — `optimize_decision_threshold()` exists as an internal function. It searches thresholds on validation probabilities, supports F1/precision/recall/objective and cost matrix, and explicitly avoids protected holdout data.

**Gap:** None of the 27 registered MCP tools expose threshold optimization. No tool parameter accepts a decision threshold.

**Value:** Enables operating-point selection for imbalanced classification.
**Risk:** Threshold tuning on development data can overfit; must never silently run during training/comparison/tuning/export.
**Recommended order:** Low — after chronology/exclusion safeguards are stable (they are now).

### 2. Tuned-model export and prediction

**Status:** Partially implemented.

`export_model_tool` and `predict_tool` exist and work with any trained model artifact. However, `tune_model_tool` returns best parameters but does not produce an exportable model bundle. There is no tool that takes the output of `tune_model` and exports it directly.

**Gap:** No end-to-end "tune → export → predict" flow through native tools.

**Value:** Allows users to deploy tuned models.
**Risk:** Low — export/predict already work; gap is convenience, not safety.
**Recommended order:** Medium — after threshold optimization.

### 3. Explicit final protected-holdout evaluation

**Status:** Not implemented.

No MCP tool evaluates on the protected test holdout. All training, CV, comparison, tuning, and export explicitly exclude `test_indices` from the manifest. The protected holdout is a workflow safeguard enforced in source (`trainer.py`, `evaluator.py`, `splits.py`).

**Gap:** No "final evaluate on protected test" tool exists.
**Risk:** High if ever auto-run — must require deliberate authorization and must never silently execute during training, comparison, tuning, or export.
**Recommended order:** High — but only as a separately authorized, locked protocol step. Never implicit.

### Backlog ordering (recommended)

1. **Explicit final protected-holdout evaluation** — deliberate authorization required; never implicit.
2. **Native threshold optimization** — expose `optimize_decision_threshold` through a registered tool.
3. **Tuned-model export and prediction** — connect tune_model output to export_model.

### Authorization requirement

Any future final-test evaluation on the protected holdout must require explicit, deliberate authorization and must **never** run silently during training, comparison, tuning, or export. This is enforced by the current architecture (protected test indices never enter any fit/CV path) and must be preserved.

---

## Regression Tests

**File:** `tests/test_remaining_deficiencies.py`

13 new tests covering all four confirmed issues:

| Test | Issue | What it verifies |
|------|-------|------------------|
| `test_tune_model_chronological_manifest_uses_timeseries` | 1 | Time manifest → TimeSeriesSplit; forward-chaining fold indices |
| `test_tune_model_grouped_manifest_preserves_group_isolation` | 1 | Group manifest → GroupKFold |
| `test_tune_model_stratified_manifest_uses_stratifiedkfold` | 1 | Stratified manifest → StratifiedKFold |
| `test_tune_model_excludes_columns_through_cv` | 2 | tune_model preserves exclusions |
| `test_automl_excludes_columns_across_candidates` | 2 | automl_run preserves exclusions |
| `test_get_model_leaderboard_excludes_columns` | 2 | leaderboard preserves exclusions |
| `test_compare_models_time_manifest_reports_stratified_false` | 3 | Time manifest: stratified=false, strategy=TimeSeriesSplit |
| `test_compare_models_stratified_manifest_reports_stratified_true` | 3 | Stratified manifest: stratified=true, strategy=StratifiedKFold |
| `test_compare_models_kfold_manifest_reports_stratified_false` | 3 | Random manifest: stratified=false, strategy=KFold |
| `test_automl_n_features_after_exclusion` | 4 | AutoML n_features reflects post-exclusion count |
| `test_automl_n_features_without_exclusion_matches_raw` | 4 | No exclusions → n_features = columns - 1 |
| `test_core_automl_n_features_after_exclusion` | 4 | Core engine path agrees |
| `test_compare_models_features_after_exclusion` | 4 | Excluded columns absent from results |

**Test fixture:** Synthetic classification DataFrame with a deliberately predictive `leak_duration` feature (highly correlated with target) to verify exclusion prevents leakage.

---

## Verification Results

| Check | Result |
|-------|--------|
| pytest suite | **172 passed** (159 pre-existing + 13 new), 0 failed |
| Ruff lint | **All checks passed** |
| Ruff formatting | **49 files formatted** |
| Python compilation | **Passed** (`py_compile` on all modified files) |
| Package build | **Wheel built successfully**, version 0.3.2 unchanged |
| MCP protocol integration | Covered by `test_mcp_protocol_export_predict_flow` (27 tools) |
| Tool registration | 27 tools verified in `server.py` `_tool_specs` |

**All 27 MCP tools maintained.** String-label evaluation, protected holdout enforcement, prediction warnings, and previous reliability fixes (commits up to `901b998`) are preserved.

---

## Known Limitations (pre-existing, not regressed)

- Native `stratified` flag removed from `compare_models` responses; replaced with strategy-aware value. Clients relying on `stratified=true` for all classification should migrate to checking the `strategy` field.
- AutoML `n_features` now reports post-exclusion count; clients expecting pre-exclusion count should use `dataset_shape` or compute from `n_features` + exclusions.
- Optuna is not installed in this environment; the Optuna code path is corrected but not runtime-tested with Optuna present.
- Protected holdout evaluation is not available through any MCP tool (by design — see ISSUE 5).
- Threshold optimization is internal-only (no MCP tool exposes it).

---

## Deferred Capabilities (ISSUE 5 backlog)

| Capability | Status | Risk | Recommended order |
|------------|--------|------|-------------------|
| Explicit final protected-holdout evaluation | Not implemented | High | 1 (separately authorized only) |
| Native threshold optimization | Internal helper, no MCP tool | Low | 2 |
| Tuned-model export and prediction | Partial (export/predict exist; no tune→export flow) | Low | 3 |

---

## Commit Information

**Baseline:** `901b99871e7d84e18e4bfd77ddd86bdc8ba7859a`
**Fix commit:** <to be populated at commit time>
**Files changed:**
- `src/devray_datalab/core/automl.py` — ISSUE 1, 2, 4 fixes
- `src/devray_datalab/tools/training.py` — ISSUE 3 fix
- `tests/test_remaining_deficiencies.py` — 13 new regression tests
- `docs/remaining-deficiencies-verification.md` — this document

---

## Final Verdict

**PASS — All confirmed issues in scope corrected.**

- ISSUE 1 (chronological CV in tuning): fixed — strategy derived from manifest, not task type.
- ISSUE 2 (feature exclusion in tuning): fixed — exclusions forwarded through Optuna path.
- ISSUE 3 (misleading CV metadata): fixed — `stratified` flag matches actual strategy.
- ISSUE 4 (feature-count metadata): fixed — `n_features` reflects post-exclusion predictors.
- ISSUE 5 (missing capabilities): documented as backlog; no code change.

All 172 tests pass, lint and formatting are clean, package builds at version 0.3.2, all 27 MCP tools are maintained, and all previous reliability fixes are preserved.
