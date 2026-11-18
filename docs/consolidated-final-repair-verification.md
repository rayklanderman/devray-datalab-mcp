# Consolidated Reliability Repair — Verification Report

**Branch:** `fix/consolidated-reliability` (local, not pushed)
**Starting HEAD:** `db65625` (original local baseline, clean working tree). Sanitized equivalent: `18f828a41cfe`.
**Package version:** `0.3.2` (unchanged, not bumped)
**MCP interface:** 27 tools (preserved)
**Optuna version executed:** `5.1.0` (QA env); fallback verified without Optuna
**Total automated tests:** 239 passed with Optuna 5.1.0 / 234 passed + 5 skipped without Optuna (baseline: 207)
**Total native MCP verification calls:** ~31 (optuna env) + ~9 (fallback env)
**Final status:** PASS (all seven defects; all four high-severity blockers pass native checks)

## Defect matrix

| ID | Severity | Root cause | Affected implementation | Original failing native requests | Smallest necessary change | Status |
|---|---|---|---|---|---|---|
| OPT-001 | HIGH / blocker | `create_split_manifest` numerically sorted train/val/test row indices for timestamp splits, destroying chronology; `TimeSeriesSplit` then consumed physical CSV order; `compare_models_tool` duplicated the loader logic instead of using the shared consumer. | `core/splits.py` (`create_split_manifest`, `development_rows_from_manifest`), `tools/training.py` (`compare_models_tool`) | N014, N024, N025, N026, N135, N136 | Store timestamp-split indices in genuine timestamp order at creation; re-sort legacy numerically-stored time manifests by timestamp at consumption; route `compare_models_tool` through `development_rows_from_manifest`. | PASS |
| OPT-002 | HIGH / blocker | Candidate evaluation checked elapsed time, but the Optuna phase received no deadline/admission control and ran `n_trials` regardless of budget. | `core/automl.py` (`run_automl`) | N087, N088, N089 | One monotonic deadline across both phases; compute remaining budget before each evaluation/admission; skip Optuna when `remaining <= 0`; pass `timeout=remaining` to `study.optimize`; document that in-flight fits may complete cooperatively. | PASS |
| OPT-003 | HIGH / blocker | Optuna study used an unseeded default sampler and objective estimators did not receive the init seed, so trials and scores differed run to run. | `core/automl.py` (`run_automl`) | N050, N051, N052, N119 | Seed the sampler with the requested random state (`TPESampler(seed=random_state)`); merge seeded estimator init kwargs with suggested hyperparameters (keeping optimized params distinct from effective params); stable CV folds; no unnecessary global RNG mutation. | PASS |
| AUD-004 | HIGH / blocker | Optuna's best trial replaced an already-superior candidate without cross-phase comparison; optimized evaluations were omitted from the returned trial collection. | `core/automl.py` (`run_automl`) | N026, N033, N052, N090–N092, C003 | One coherent ledger of successful evaluated configurations across candidate + optimization phases; select the winner across all using the existing metric-direction policy; never let an inferior Optuna trial displace a superior candidate; report trials/leaderboard consistent with evaluations performed; retain a valid candidate if Optuna yields no successful trial; handle non-finite/failed trials safely. | PASS |
| AUD-005 | MEDIUM | Cooperative cancellation raised `TrainingError`, classified by the job manager as `FAILED`. | `utils/errors.py`, `core/trainer.py`, `core/jobs.py` | N065–N069, N110–N114 | Introduce `CancellationError` (code `cancelled`); trainer raises it on cancellation and marks the job `CANCELLED`; job manager maps `CancellationError` -> `CANCELLED` and genuine exceptions -> `FAILED`; completed jobs win cancellation races. | PASS |
| AUD-006 | MEDIUM | Explicit tuning requests were filtered through the smaller default AutoML candidate set instead of the full supported model registry. | `core/automl.py` (`run_automl`) | N124, N125 | Separate default AutoML candidate selection from explicit tuning models; resolve explicit model names (`svm`, `hist_gradient_boosting`, classification counterparts) against the full task-specific registry; use existing search-space definitions; reject unknown models with a precise `validation_error` before optimization. | PASS |
| AUD-007 | LOW | `predict` caught only `ModelError`, while an invalid artifact identifier raises `ValidationError`, which the generic server wrapper reported as `unexpected_error`. | `tools/models.py` (`predict_tool`) | N131, N132 | Catch the `DevRayError` family around `load_artifact` and preserve its code (`validation_error` for invalid artifact identifiers); keep unexpected exceptions distinguishable; preserve filesystem confinement. | PASS |

## Before-fix failing test evidence (reproduced on `db65625`)

All tests below are in `tests/test_consolidated_repair.py` and fail on `db65625` for the expected reason:

- OPT-001: `test_opt001_manifest_stores_timestamp_order_not_numeric`, `test_opt001_development_rows_consumed_in_timestamp_order`, `test_opt001_legacy_numeric_manifest_reordered_on_consumption`, `test_opt001_datetime_timestamps_chronological` (manifest numerically sorts indices; development rows not timestamp-ordered).
- OPT-002: `test_opt002_deadline_stops_new_candidates_and_optuna` and `test_opt002_optuna_admits_no_trial_after_deadline` (Optuna admits trials at/after the deadline) — with Optuna installed.
- OPT-003: `test_opt003_seeded_optuna_deterministic` (trial parameter sequences / objective values differ across identical runs) — with Optuna installed.
- AUD-004: `test_aud004_inferior_optuna_does_not_displace_candidate_maximize/minimize` and `test_aud004_optuna_failure_retains_candidate` (inferior Optuna trial displaces candidate; NaN trial selected) — with Optuna installed.
- AUD-005: `test_aud005_queued_cancellation_reaches_cancelled`, `test_aud005_active_cancellation_is_cancelled_not_failed` (cancelled jobs reach `FAILED`).
- AUD-006: `test_aud006_svm_regression_baseline_and_tuning`, `test_aud006_hist_gradient_boosting_tuning`, `test_aud006_classification_svm_tuning` (`tune_model(model_name="svm")` → "No candidate models selected").
- AUD-007: `test_aud007_predict_rejects_unsafe_paths_with_validation_error` (predict returns `unexpected_error`/raises instead of `validation_error`).

## Changed implementation

- `src/devray_datalab/core/splits.py` — timestamp-split indices stored in timestamp order; legacy numerically-stored time manifests re-ordered by timestamp at consumption.
- `src/devray_datalab/tools/training.py` — `compare_models_tool` uses `development_rows_from_manifest` (shared loader).
- `src/devray_datalab/core/automl.py` — single monotonic deadline; remaining-budget admission; `timeout=remaining` for Optuna; seeded `TPESampler`; seeded init kwargs merged with suggested hyperparameters; unified candidate+Optuna ledger with direction-based winner selection; explicit models resolve against the full task registry with a precise unknown-model error.
- `src/devray_datalab/core/trainer.py` — raises `CancellationError` on cancellation; marks job `CANCELLED`; genuine exceptions remain `FAILED`.
- `src/devray_datalab/core/jobs.py` — maps `CancellationError` to `CANCELLED`, other exceptions to `FAILED`.
- `src/devray_datalab/utils/errors.py` — added `CancellationError` (code `cancelled`).
- `src/devray_datalab/tools/models.py` — `predict_tool` catches `DevRayError` and preserves `validation_error` for invalid artifact identifiers.

## After-fix automated test evidence

- Standard env (no Optuna): 234 passed, 5 skipped (Optuna-gated), 0 failed.
- QA env (Optuna 5.1.0): 239 passed, 0 failed.
- All pre-existing FD-001…FD-005 regression checks, protected-holdout exclusion, string-label classification, feature exclusions, strict JSON, model export/reload/predict, metadata/provenance, and all 27 tool schemas remain green.

## Fresh native MCP evidence (fresh stdio worker; real JSON-RPC, not in-process calls)

All seven scenarios replayed against a freshly started `devray_datalab.server` worker:

- Interface: `tools/list` returns exactly 27 tools; all expected names present.
- OPT-001: time manifest created; `compare_models` with the manifest reports `TimeSeriesSplit`, fold sizes within train+val (protected holdout excluded).
- OPT-002: `tune_model` with `max_time_seconds=1` completes bounded (~0.7s); no runaway search.
- OPT-003: two `automl_run` runs with the same seed return identical trial parameter sequences and objective values.
- AUD-004: reported `best_value`/`best_model` equal the best actual evaluated trial; no unevaluated winner.
- AUD-005: cancelled job reaches `CANCELLED`; `get_training_status` and `list_training_jobs` agree; no raw dataframe leaked.
- AUD-006: `tune_model(model_name="svm")` accepted for regression and classification; unknown model rejected with `validation_error` before optimization.
- AUD-007: traversal/absolute/Windows-drive/mixed-separator artifact names return `validation_error`, strict JSON, no sentinel contents leaked.
- Baseline: `export_model` → `load_model_handle` → `predict` round trip functions.

## Fallback verification (without Optuna)

Standard-env native script and unit suite confirm `tune_model`/`automl_run`/`train_baseline` work without Optuna (Optuna phase gracefully skipped), `svm` baseline works, unsafe paths still return `validation_error`, and export/load/predict round trip works.

## Remaining limitations (explicitly excluded, not implemented)

- OPT-004 tuned-configuration refit/export handoff; GAP-002 fallback grid search; GAP-003 preprocessing-handle handoff; resource enforcement/concurrency; optional format enhancements; new MCP tools. These are separate contract/feature/unverified items and were not touched.
- Active (mid-run) cancellation through the live MCP worker is timing-sensitive and covered at the JobManager/unit level rather than a deterministic native polling test.

## 27-tool interface

Preserved: exactly 27 tools, no additions or removals; all schemas intact.

## Git status

- Working tree clean except the 7 modified source files and 1 new test file (see below).
- Branch: `fix/consolidated-reliability`. No push, merge, tag, publish, or deploy performed.

## Files modified

1. `src/devray_datalab/core/automl.py`
2. `src/devray_datalab/core/jobs.py`
3. `src/devray_datalab/core/splits.py`
4. `src/devray_datalab/core/trainer.py`
5. `src/devray_datalab/tools/models.py`
6. `src/devray_datalab/tools/training.py`
7. `src/devray_datalab/utils/errors.py`
8. `tests/test_consolidated_repair.py` (new)

## Completion verdict

All seven defects repaired; all four high-severity blockers (OPT-001, OPT-002, OPT-003, AUD-004) pass the specified native checks; full suite green; ruff/format/compile clean; 27-tool interface preserved; Optuna 5.1.0 executed; fallback without Optuna verified. Overall: **PASS**.
