# Consolidated Reliability Repair — Verification

**Project:** DevRay DataLab MCP
**Branch:** `fix/consolidated-reliability`
**Commit:** `1fc1bc2`
**Base version:** `0.3.2` (unchanged)
**Audit source:** Codex Desktop MCP Reliability QA reports (read-only)

---

## Summary

All nine confirmed defect families (QA-001 through QA-009) from the Codex Desktop
audit of DevRay DataLab MCP v0.3.2 have been corrected. No version bump, no
release, no push. All repairs are on branch `fix/consolidated-reliability`.

---

## Defects Fixed

| ID | Defect | Root Cause | Correction | Test Reference | Status |
|----|--------|-----------|------------|----------------|--------|
| QA-001 | Split-manifest validation bypass | Six training/export/job pathways accepted invalid manifests (overlap, OOB, duplicate, reversed-time, crossed-group) without structural validation | Added `validate_manifest_for_consumption()` in `core/splits.py` covering fingerprint, index types/bounds, duplicates, overlaps, declared counts, coverage, chronological ordering, group separation, protected-holdout constraints. Enforced in `train_baseline`, `compare_models`, `automl_run`, `tune_model`, `export_model`, `start_training_job` | `tests/test_reliability_repair.py::test_qa001_*` | PASS |
| QA-002 | Protected holdout disclosure | Development metrics computed on protected test rows; raw DataFrame exposed via job status `result.df` | (a) Training uses validation rows for development metrics; (b) Job params stored separately from public result (`JobInfo._params`), raw df never serialized in status | `tests/test_reliability_repair.py::test_qa002_*` | PASS |
| QA-003 | Chronological CV misreported | `compare_models` hardcoded `stratified`/`kfold` regardless of manifest split type | Strategy derived from manifest: `time`→`TimeSeriesSplit`, `group`→group-aware, `stratified`/`random`→existing. `time_column`/`group_column` passed to CV | `tests/test_reliability_repair.py::test_qa003_chronological_cv_uses_timeseries` | PASS |
| QA-004 | Split-type naming mismatch | Schema advertises `time_series`/`grouped`, handlers accept `time`/`group` | Alias map `time_series`→`time`, `grouped`→`group` in `create_split_manifest`; canonical names stored in manifests | `tests/test_reliability_repair.py::test_qa004_*` | PASS |
| QA-005 | Invalid JSON numbers | Bare NaN/Infinity in `inspect_dataset` and `analyze_feature_drift` responses | `sanitize_for_json()` recursively converts non-finite floats to `null`; all `_json()` serializers use `allow_nan=False` | `tests/test_reliability_repair.py::test_qa005_*` | PASS |
| QA-006 | AutoML trial limits ignored | `max_trials=1` evaluated all candidates in fallback Phase 1 | Cap enforced in Phase 1 loop (`trials_run >= max_trials` break) and Optuna phase; applies to `automl_run`, `tune_model`, `get_model_leaderboard` | `tests/test_reliability_repair.py::test_qa006_*` | PASS |
| QA-007 | Seeded tuning non-reproducible | `model_kwargs={}` passed to estimators, dropping `random_state`; splitters unseeded | `_model_init_kwargs(model_cls, random_state)` propagated to all AutoML/tuning CV calls; effective seeds recorded | `tests/test_reliability_repair.py::test_qa007_seeded_tuning_reproducible` | PASS |
| QA-008 | Model provenance loss | Re-export via `model_handle` set `dataset_fingerprint = None` | Fingerprint preserved from bundle metadata (`metadata.dataset_fingerprint`) on re-export | `tests/test_reliability_repair.py::test_qa008_reexport_preserves_fingerprint` | PASS |
| QA-009 | Invalid preview-limit types | `max_rows="bad"` silently coerced to default | Reject booleans, containers, non-numeric strings, and out-of-range values with clear `validation_error`; default only when omitted | `tests/test_reliability_repair.py::test_qa009_invalid_max_rows_rejected` | PASS |

---

## Exact Files Modified

### Source (14 files)

1. `src/devray_datalab/core/splits.py` — `validate_manifest_for_consumption()`, QA-004 alias map, chronological/group validation
2. `src/devray_datalab/core/trainer.py` — QA-002 validation metrics on val rows (both `train_baseline` and `train_model_with_artifact`)
3. `src/devray_datalab/core/automl.py` — QA-006 max_trials cap, QA-007 random_state in model_kwargs, QA-003 group/time column passthrough
4. `src/devray_datalab/core/jobs.py` — QA-002 `JobInfo._params` separate from `result`, raw df never in public status
5. `src/devray_datalab/server.py` — QA-005 `sanitize_for_json` import, `allow_nan=False` in `_json()`
6. `src/devray_datalab/utils/errors.py` — QA-005 `sanitize_for_json()` function
7. `src/devray_datalab/tools/training.py` — QA-001 validation in `train_baseline_tool`, `compare_models_tool`; QA-003 strategy/time_column/group_column
8. `src/devray_datalab/tools/automl.py` — QA-001 validation in all 3 tools; QA-003 group/time columns; QA-006/QA-007 passthrough
9. `src/devray_datalab/tools/jobs.py` — QA-001 validation in `start_training_job_tool`
10. `src/devray_datalab/tools/models.py` — QA-001 validation in `export_model_tool`; QA-008 fingerprint preservation on re-export
11. `src/devray_datalab/tools/datasets.py` — QA-005 strict JSON; QA-009 max_rows validation
12. `src/devray_datalab/tools/analysis.py` — QA-005 strict JSON
13. `src/devray_datalab/tools/system.py` — QA-005 strict JSON
14. `src/devray_datalab/tools/splits.py` — QA-005 strict JSON

### Tests (2 files)

15. `tests/test_reliability_repair.py` — NEW: 17 regression tests covering all nine defects
16. `tests/test_leakage_safe_automl_export.py` — Updated `test_automl_excludes_columns` to assert `max_trials=3` yields 3 trials (corrected from old buggy expectation of 5)

---

## Verification Results

| Check | Result |
|-------|--------|
| Total tests passed | **137** (120 pre-existing + 17 new regression) |
| Ruff lint | PASS (0 errors) |
| Ruff format | PASS (0 files unformatted) |
| Python compilation | PASS (all modified files compile) |
| Package build | PASS (wheel builds at version 0.3.2) |
| MCP e2e protocol | PASS (`tests/test_e2e.py`) |
| MCP tool count | 27 (all preserved) |
| Version | 0.3.2 (unchanged) |

---

## Unresolved Limitations

1. **No implicit final-test evaluation:** QA-002 states that if an explicit final evaluation cannot be safely supported through the existing interface, the limitation should be documented. The current interface does not expose a separate protected-holdout final evaluation step; development metrics use validation data only. Callers needing a final holdout evaluation must request it through a future interface addition.

2. **Group-aware CV tested for classification only:** QA-003 group-aware CV path is exercised through `compare_models` with a group manifest; regression-grouped CV follows the same code path but is not separately tested with a dedicated fixture.

3. **Background job cancelled/failed states:** Job status serialization was verified for queued and completed states. Cancelled and failed states follow the same serialization path (no `df` in `result`) but are not individually exercised with protected-data assertions beyond the queued test.

---

## Exact Tool Behavior Changes

| Tool | Before | After |
|------|--------|-------|
| `train_baseline` | Accepted invalid manifests; evaluated on protected test rows | Rejects invalid manifests; evaluates on validation rows |
| `compare_models` | Always reported StratifiedKFold/StratifiedKFold | Reports TimeSeriesSplit for time manifests, group-aware for group manifests |
| `automl_run` | Ignored `max_trials` in fallback | Enforces `max_trials` cap |
| `tune_model` | Non-deterministic despite same seed | Deterministic with propagated `random_state` |
| `export_model` | Lost `dataset_fingerprint` on re-export | Preserves fingerprint from bundle metadata |
| `start_training_job` | Accepted invalid manifests; raw df in status | Rejects invalid manifests before queuing; no raw df in status |
| `inspect_dataset` | Silently defaulted invalid `max_rows` | Rejects non-numeric, boolean, container, out-of-range values |
| `create_split_manifest` | Rejected `time_series`/`grouped` | Accepts as aliases for `time`/`group` |
| All JSON responses | Could contain bare NaN/Infinity | Non-finite values converted to `null` |

---

## Reproduction Instructions for Codex Desktop

1. Checkout branch `fix/consolidated-reliability` at commit `1fc1bc2`.
2. Run `python -m pytest tests/` — 137 tests pass.
3. Run `ruff check` and `ruff format --check` — clean.
4. Run `python -m build --wheel` — builds at v0.3.2.
5. Run `python -m pytest tests/test_e2e.py` — MCP protocol e2e passes.
6. To reproduce QA-002 paired-fixture behavior: train on `qa_c1.csv` and `qa_c2.csv` (differing only in protected test rows 12–14) with identical manifests; development metrics must be identical.
7. To reproduce QA-007: call `tune_model` twice with identical `random_state`; `best_value` must match.
8. To reproduce QA-008: export a model, re-export via `model_handle`; `dataset_fingerprint` must match.

---

## Machine-Readable Regression Summary

```json
{
  "branch": "fix/consolidated-reliability",
  "commit": "1fc1bc2",
  "base_version": "0.3.2",
  "defects_fixed": ["QA-001","QA-002","QA-003","QA-004","QA-005","QA-006","QA-007","QA-008","QA-009"],
  "tests_total": 137,
  "tests_new": 17,
  "mcp_tool_count": 27,
  "ruff_lint": "pass",
  "ruff_format": "pass",
  "build": "pass",
  "e2e_mcp": "pass",
  "version_bumped": false,
  "release_created": false
}
```
