# Post-Repair Follow-up Verification

**Project:** DevRay DataLab MCP
**Branch:** `fix/consolidated-reliability`
**Commit:** `adbffcb`
**Base version:** `0.3.2` (unchanged)
**Audit source:** Codex Desktop MCP Reliability QA reports — post-repair directory (read-only)

---

## Summary

Three confirmed defects (QA-001 incomplete coverage, QA-002 no-validation-partition,
QA-009 null/fractional preview limits) discovered by independent Codex Desktop
post-repair acceptance testing have been corrected. Two previously inconclusive
areas (QA-003 fold ordering, QA-007 tuned-model predictions) have been
investigated and documented. No version bump, no release, no push.

---

## Failures Corrected

### QA-001 — Incomplete manifests accepted

**Root cause:** `validate_manifest_for_consumption()` checked coverage by
comparing `len(covered) == len(all_idx)`, which passes when the manifest lists
N unique indices in N slots — even if some dataset rows (0..n-1) are entirely
absent. A 149/150-row partition passed as "valid". The public
`validate_split_manifest()` had the same bug and also omitted chronological
and group-separation checks, so it reported `valid=true` for reversed-time and
crossed-group manifests that consumers rejected.

**Corrections:**
- `src/devray_datalab/core/splits.py` — `validate_manifest_for_consumption()` now
  requires `covered == set(range(n))`: every dataset row index 0..n-1 must
  appear in exactly one partition. Missing rows are reported explicitly.
- `src/devray_datalab/core/splits.py` — `validate_split_manifest()` now uses the
  same complete-coverage check, plus chronological ordering (train < val < test
  for time splits) and group separation (no shared groups across partitions for
  group splits), so the public validator agrees with all six consumers.

**Applied to all six consumers:** `train_baseline`, `compare_models`,
`automl_run`, `tune_model`, `export_model`, `start_training_job` — each calls
`validate_manifest_for_consumption()` before any dataset indexing, preprocessing,
fitting, tuning, or model selection.

**Tests:** `tests/test_reliability_repair.py::test_qa001_incomplete_coverage_rejected`
and `test_qa001_validator_consistent_reverse_and_group`.

**Status:** PASS — incomplete 149/150 manifests rejected by validator and all six
consumers; reversed-time and crossed-group manifests rejected by public validator.

---

### QA-002 — Protected metrics exposed without validation rows

**Root cause:** Both `Trainer._run_training()` and `Trainer.train_model_with_artifact()`
fell back to `test_indices` (protected holdout) when `val_indices` was empty.
A manifest with `val_size=0` and `protected_holdout=true` caused development
metrics to be computed on protected test rows, and those scores were exposed in
baseline, export, and completed-job responses.

**Corrections:**
- `src/devray_datalab/core/trainer.py` — When `manifest.protected_holdout` is
  true and no validation rows exist, raise `TrainingError`:
  `"Protected holdout requires a validation partition; no validation rows
  available for development-stage evaluation."` Protected test rows are never
  substituted for validation rows. Applied to both trainer code paths.

**Behavior by tool:**
- `train_baseline`, `export_model`: fail clearly before fitting/evaluation.
- `start_training_job`: submission succeeds (manifest is structurally valid),
  job fails asynchronously when the worker runs training.
- `compare_models`, `automl_run`, `tune_model`: use cross-validation on
  development (train+val) rows only; with no val rows they CV on train data
  only. Protected test rows never enter folds. No protected metrics exposed.

**Tests:** `tests/test_reliability_repair.py::test_qa002_no_validation_partition_rejected`
and `test_qa002_no_validation_partition_no_protected_metrics`.

**Status:** PASS — no-protected-validation manifests fail or succeed without
protected-test metrics across all six consumers.

---

### QA-009 — Invalid preview limits still accepted

**Root cause:** `inspect_dataset_tool()` used `arguments.get("max_rows", default)`,
which cannot distinguish "parameter omitted" from "parameter explicitly null".
Floats were accepted and silently truncated via `int()`. Values like `null`,
`100.5`, `100.9`, `1.5` were coerced instead of rejected.

**Corrections:**
- `src/devray_datalab/tools/datasets.py` — Default applies only when `"max_rows"`
  key is absent from arguments. Explicit `null`, booleans, floats, strings,
  lists, dicts, zero, negatives, and values >100 are all rejected with clear
  `validation_error` messages. No coercion or truncation.

**Tests:** `tests/test_reliability_repair.py::test_qa009_null_and_fractional_max_rows_rejected`
and `test_qa009_valid_limits_still_work`.

**Status:** PASS — null, floats (in-range and above-cap), booleans, strings,
containers, zero, negative, and above-cap values all rejected; valid integers
1..100 and omitted parameter (default 25) work.

---

## Previously Inconclusive Areas

### QA-003 — Fold ordering

**Investigation:** Focused internal tests verify actual fold indices, not just
strategy labels. For chronological manifests, `compare_models` derives
`cv_strategy="timeseries"` and passes `time_column` from the manifest to
`cross_validate_with_preprocessing`, which uses `TimeSeriesSplit`. Every
training fold precedes its validation fold; protected test indices never enter
folds (df is restricted to train+val rows before CV). Preprocessing is fitted
independently within each training fold via `_fit_preprocess_and_split`.

For grouped manifests, `cv_strategy="group"` and `group_column` is passed;
`GroupKFold` preserves group isolation (no group appears in both train and
test portions of any fold).

**Finding:** The implementation correctly produces forward-chaining temporal
CV and group-aware CV. The native acceptance test could not observe actual
fold indices because `compare_models` returns only strategy labels and
aggregate scores, not per-fold row indices. This is an observability
limitation, not a correctness defect.

**Status:** Resolved by code inspection and focused tests; no public API change
needed. No new diagnostic tool added (per scope constraint).

---

### QA-007 — Tuned-model predictions

**Investigation:** Three identical seeded `tune_model` runs return identical
`best_value`, `std`, and `params`. `_model_init_kwargs(model_cls, random_state)`
is propagated through all AutoML/tuning CV calls, making estimator fitting
deterministic for a given seed. Repeated runs and fresh-process runs agree.

**Limitation:** The public interface does not return a fitted model handle or
predictions from tuned models. `tune_model` returns best parameters and CV
scores only; there is no native path to export a tuned model for prediction
comparison. Two separate seeded exports produce equal metrics and probe
probabilities, which supports (but does not prove) tuning reproducibility.

**Status:** Score reproducibility confirmed. Prediction-level reproducibility
cannot be verified through the existing public interface (no tuned-model export
for prediction). Documented as a limitation; no public API expansion per scope
constraint.

---

## Verification Results

| Check | Result |
|-------|--------|
| Total tests passed | **143** (126 existing + 17 new) |
| Ruff lint | PASS (0 errors) |
| Ruff format | PASS (0 files unformatted) |
| Python compilation | PASS (all modified files compile) |
| Package build | PASS (wheel builds at version 0.3.2) |
| MCP e2e protocol | PASS (`tests/test_e2e.py`) |
| MCP tool count | 27 (all preserved) |
| Version | 0.3.2 (unchanged) |

---

## Exact Source Corrections

| File | Change |
|------|--------|
| `src/devray_datalab/core/splits.py` | Complete-coverage check (`covered == set(range(n))`) in both `validate_manifest_for_consumption` and `validate_split_manifest`; chronological ordering and group separation added to `validate_split_manifest` |
| `src/devray_datalab/core/trainer.py` | `TrainingError` raised when protected manifest has no validation rows, in both `_run_training` and `train_model_with_artifact` |
| `src/devray_datalab/tools/datasets.py` | `max_rows` rejects null/bool/float/out-of-range; default only when key absent |

---

## Regression Tests Added

`tests/test_reliability_repair.py` — 6 new test functions:

- `test_qa001_incomplete_coverage_rejected` — 149/150 partition rejected by validator and all 6 consumers
- `test_qa001_validator_consistent_reverse_and_group` — public validator rejects reversed-time and crossed-group
- `test_qa002_no_validation_partition_rejected` — no-val protected manifest fails in trainer-based tools; async job fails; CV-based tools succeed without protected metrics
- `test_qa002_no_validation_partition_no_protected_metrics` — train_baseline with no-val manifest fails, no protected metrics leak
- `test_qa009_null_and_fractional_max_rows_rejected` — null, floats, booleans, containers, 0, negative, >100 all rejected
- `test_qa009_valid_limits_still_work` — omitted default (25) and valid integers 1..100 work

---

## Remaining Limitations

1. **No native final protected-holdout evaluation tool:** The interface does not
   expose a separate final evaluation on protected test data. Development
   metrics use validation data only.
2. **Tuned-model predictions not observable through native MCP:** `tune_model`
   returns parameters and CV scores but no fitted model handle or predictions.
   Prediction-level reproducibility cannot be natively verified.
3. **Fold indices not exposed:** `compare_models` returns strategy labels and
   aggregate scores but not per-fold row indices, so native clients cannot
   independently verify forward chaining or fold-local preprocessing.
4. **Group-aware CV tested for classification path only:** Grouped regression
   follows the same code path but lacks a dedicated fixture.

---

## Local Commit Hash

`adbffcb` — fix: QA-001/QA-002/QA-009 post-repair acceptance failures

Parent commits: `dd89006`, `1fc1bc2` (all on `fix/consolidated-reliability`)

No tags modified, no version bumped, no push/merge/release performed.
