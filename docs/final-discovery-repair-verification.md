# Final Discovery Consolidated Defect Repair — Verification

**Branch:** `fix/consolidated-reliability`
**Baseline commit:** `877ca97`
**Package version:** `0.3.2` (unchanged)
**MCP tools:** 27 (unchanged)
**Pre-existing tests:** 172 passing (all preserved)

## Summary

All five confirmed defect families from the final-discovery QA audit (FD-001 to FD-005) are corrected, verified, and committed on `fix/consolidated-reliability`. No new features, no architecture changes, no version change, no release.

## Defects corrected

### FD-001 — Incorrect model selection and objective handling

**Root cause.** Model comparison, AutoML, and tuning applied `max` to every metric including losses (log-loss, MAE), so higher-loss models could win; regression `metric="auto"` stayed unresolved and fell through to first-model selection; the evaluator scored the wrong objective (e.g. `log_loss` requests were scored with F1/ROC-AUC); reported best values negated losses (double negation of the loss convention).

**Source changes.**
- New module `src/devray_datalab/core/metric_policy.py`: task-aware metric registry (maximize vs minimize), `resolve_metric()` (`auto` → ROC-AUC binary / F1 multiclass / R² regression), `direction_for_metric()`, `is_valid_metric()`.
- `src/devray_datalab/core/evaluator.py`: validates the scoring metric before fitting; scores folds under the requested objective (log-loss, MAE, R², ROC-AUC, F1, …) instead of substituting F1/ROC-AUC/R².
- `src/devray_datalab/core/automl.py`: resolves `auto` up front; ranks trials by natural-signed value in the correct direction (losses minimized); reports natural best values (no negation); Optuna study direction set accordingly; unknown metrics rejected before fitting; leaderboard trials sorted by objective.
- `src/devray_datalab/tools/training.py` (`compare_models`): resolves `auto`, validates unknown metrics, passes the resolved scoring to CV, selects best by direction, reports resolved `selection_metric`.

**Regression tests added (tests/test_fd_repair.py):** metric-policy units; compare log-loss (lowest wins; decision tree 0 vs logistic > 0); compare MAE (lowest wins; linear 0 vs tree > 0); compare regression `auto` resolves to R² and picks R²=1; AutoML log-loss best value natural 0; AutoML MAE direction; leaderboard trials loss-ascending; tune multiclass `auto` → F1 best value 1; tune regression MAE → 0; invalid metric rejected with validation_error; seed reproducibility. All 12 FD-001 tests fail on the unfixed baseline and pass after the fix.

### FD-002 — Timestamp chronology is incorrect

**Root cause.** `create_split_manifest` sorted by timestamp but stored indices re-sorted numerically, and validation compared original row indices rather than timestamp values — so out-of-order timestamps (e.g. descending `time`) produced chronologically wrong manifests that failed validation, and CV consumed rows in original order instead of timestamp order.

**Source changes.** `src/devray_datalab/core/splits.py`: time splits are sorted by actual timestamp values (stable merge sort; ties keep original-row order) and stored in timestamp order; `_parse_time_column()` coerces datetime-like strings while keeping numerics; missing/unparseable timestamps are rejected at creation and validation; chronology validated on timestamp values (train ≤ val ≤ test, documented deterministic tie policy); row-order (`row_order=True`) keeps original-index chronology.

**Regression tests added:** reverse-time manifest validates (timestamp chronology, not row index); stored order is timestamp order; TimeSeriesSplit CV consumes manifest in timestamp order; duplicate timestamps deterministic; missing timestamps rejected; invalid timestamps rejected; row-order split preserved. All fail pre-fix, pass post-fix.

### FD-003 — Grouped splitting and manifest validation inconsistencies

**Root cause.** Grouped splitting with a missing/invalid `group_column` silently fell through to a random split labeled "grouped"; unknown split types were not rejected in manifests; the public validator (`validate_split_manifest`) and the consumer validator (`validate_manifest_for_consumption`) checked different things (consumer checked index types, duplicates, declared counts; public did not).

**Source changes.** `src/devray_datalab/core/splits.py`: grouped creation requires a valid group column (clear validation error, never a silent random split); one authoritative `_validate_manifest_core()` backs both public and consumer entry points so they can never disagree; checks: allowed split type, fingerprint, integer/non-boolean in-range indices, duplicates, overlaps, non-empty, complete coverage, declared counts, time-column chronology by timestamps, row-order chronology, group-column presence and actual group disjointness.

**Regression tests added:** grouped without group_column rejected; grouped with nonexistent column rejected; valid grouped manifest validates and isolates groups; unknown split-type manifest rejected by public validator and all six consumers (baseline, compare, automl, tune, export, background job); duplicate/incomplete manifest rejected by public validator and all six consumers. All fail pre-fix, pass post-fix.

### FD-004 — Manifest path traversal and filesystem confinement

**Root cause.** `SplitStore._path` joined a raw manifest id with the artifacts dir and never resolved/checked containment, so `../../…` ids escaped the splits directory; the embedded manifest id was not checked against the locating id.

**Source changes.** `src/devray_datalab/core/splits.py`: manifest ids validated as safe portable identifiers (`A-Za-z0-9._-`, no separators, no `.`/`..`); `_path()` resolves and confirms `relative_to()` the store root; `save`/`load`/`delete` all confined; `load` verifies the embedded `manifest_id` matches the locating id; errors are generic and never expose file contents; `list_all` scoped to the store directory.

**Regression tests added:** valid manifest roundtrip; traversal ids (`../../…`, `..\\..`, `../…/…`, `/etc/passwd` style) rejected at store and tool level with safe messages; absolute paths rejected; consumers (train_baseline) reject escaped manifest ids; symlink/junction escape rejected where the OS permits symlink creation. All fail pre-fix, pass post-fix.

### FD-005 — Incorrect correlation summary

**Root cause.** `strongest_positive` took the largest-|correlation| pair regardless of sign, and `strongest_negative` took the minimum over all pairs (positive when all correlations were positive), confusing magnitude with sign.

**Source changes.** `src/devray_datalab/tools/analysis.py`: filter pairs by sign; `strongest_positive` = max over positive correlations only; `strongest_negative` = min over negative correlations only; `None` when a category is empty; constant columns and undefined correlations excluded; Pearson and Spearman consistent.

**Regression tests added:** positive-only → negative is None; negative-only → positive is None; mixed signs → correct signs and magnitude ordering; constant columns handled; single column → None/None; Pearson and Spearman. All fail pre-fix, pass post-fix.

## Regression tests

New file: `tests/test_fd_repair.py` — 35 tests, deterministic synthetic fixtures inside the per-test workspace (no Bank Marketing data, no case-store fixtures). Each FD-001..FD-005 test was verified to fail against the unfixed baseline (23 failed / 12 invariant-passing pre-fix) and pass after the fix.

## Previously passing behavior preserved

- Full suite: 207 passed (172 pre-existing + 35 new), 0 failed.
- All 27 MCP tools registered (verified via `tools/list`).
- Protected holdout isolation, chronological/group-aware CV, fold-local preprocessing, feature exclusion, string-valued binary classification, positive-class probability alignment, strict JSON serialization, AutoML budgets, seed reproducibility, model export/reload, dataset fingerprints, unseen-category warnings, background-job privacy — intact (covered by pre-existing suites).

## Verification results

| Check | Result |
|---|---|
| `pytest tests` | 207 passed |
| `ruff check src tests` | All checks passed |
| `ruff format --check src tests` | 51 files formatted |
| `compileall src tests` | OK |
| `pyproject-build --no-isolation` | sdist + wheel built, version 0.3.2 (not published) |
| MCP tools registered | 27 |
| MCP protocol integration tests | Passed (part of suite) |
| Native MCP reproduction (fresh stdio server process) | All five defect families confirmed corrected |

## Remaining uncertainties

- **Optuna runtime pathway:** Optuna was not installed in this environment and was not installed for this repair (per instructions, no unnecessary dependencies). The Optuna code path follows the same resolved-metric/direction logic and is executed only when Optuna is importable; a safe isolated Optuna runtime test is left as a separate optional verification task.
- **Symlink/junction escape:** verified where the operating system permits symlink creation; skipped otherwise.
- Windows UNC/drive-path traversal is rejected by the identifier validator (separators and `:` are not legal id characters).

## Commit hashes

- Baseline (verified): `877ca97` on `fix/consolidated-reliability`
- Repair commit: created locally after verification (not pushed, not tagged, not merged).

## Read-only evidence respected

All reports under the local QA evidence directory were treated as read-only; Bank Marketing datasets, model results, previous QA reports, and version numbers were not modified.
