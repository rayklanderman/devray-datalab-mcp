"""Regression tests for the consolidated reliability repair.

Covers the seven confirmed defect families:
  OPT-001 Timestamp CV leakage
  OPT-002 Optuna ignores remaining time budget
  OPT-003 Seeded Optuna is nondeterministic
  AUD-004 Inferior configurations can replace better candidates
  AUD-005 Accepted cancellation becomes failure
  AUD-006 Supported model rejected by explicit tuning
  AUD-007 Unsafe path receives incorrect error classification

Fixtures are deterministic, synthetic, CPU-only and created inside the
per-test temporary workspace. The real Bank Marketing dataset is never used.
A test that exists to reproduce a confirmed defect must fail on db65625.
"""

from __future__ import annotations

import asyncio
import json
import random
import time as _time

import pandas as pd

from devray_datalab.config import DevRayConfig
from devray_datalab.core.automl import run_automl
from devray_datalab.core.evaluator import CVResult
from devray_datalab.core.jobs import JobManager, JobStatus
from devray_datalab.core.splits import (
    SplitManifest,
    SplitStore,
    _dataset_fingerprint,
    create_split_manifest,
    development_rows_from_manifest,
)
from devray_datalab.core.trainer import Trainer
from devray_datalab.tools.automl import (
    automl_run_tool,
    get_model_leaderboard_tool,
    tune_model_tool,
)
from devray_datalab.tools.jobs import (
    cancel_training_job_tool,
    get_training_status_tool,
    list_training_jobs_tool,
    start_training_job_tool,
)
from devray_datalab.tools.models import export_model_tool, predict_tool
from devray_datalab.tools.training import compare_models_tool, train_baseline_tool

try:  # optuna is optional; gate Optuna-specific assertions accordingly
    import optuna  # noqa: F401
    from optuna.trial import TrialState

    _OPTUNA_INSTALLED = True
except Exception:  # pragma: no cover - exercised without optuna
    _OPTUNA_INSTALLED = False


def _run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def _data(result):
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    return payload["data"]


def _write_csv(config: DevRayConfig, filename: str, df: pd.DataFrame) -> str:
    path = config.workspace_dir / filename
    df.to_csv(path, index=False)
    return filename


def _call(tool, config, arguments):
    return _run(tool(config, arguments))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _opt001_df(config: DevRayConfig) -> str:
    """Shuffled numeric timestamps: time is NOT monotonic with row index."""
    rng = random.Random(7)
    n = 60
    ticks = list(range(n))
    rng.shuffle(ticks)
    df = pd.DataFrame(
        {
            "row_id": range(n),
            "tick": ticks,
            "group": [i % 3 for i in range(n)],
            "x": [float(i) for i in range(n)],
            "z": [float(i % 5) for i in range(n)],
            "cat": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "target": [float(i) for i in range(n)],
        }
    )
    return _write_csv(config, "opt001_shuffled_tick.csv", df)


def _regression_perfect_df(config: DevRayConfig) -> str:
    """Exact linear relation: linear regression reaches R^2 1 / MAE 0."""
    n = 60
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(1, n + 1)],
            "z": [float(i % 7) for i in range(n)],
            "target": [3.0 * i + 2.0 for i in range(1, n + 1)],
        }
    )
    return _write_csv(config, "aud004_perfect_reg.csv", df)


def _small_regression_df() -> pd.DataFrame:
    n = 40
    return pd.DataFrame(
        {
            "x": [float(i) for i in range(n)],
            "z": [float(i % 5) for i in range(n)],
            "target": [3.0 * i + 1.0 for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# OPT-001: timestamp CV leakage
# ---------------------------------------------------------------------------


def test_opt001_manifest_stores_timestamp_order_not_numeric(config):
    name = _opt001_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="tick",
        random_state=42,
    )
    times = df["tick"]
    assert times.iloc[manifest.train_indices].max() <= times.iloc[manifest.val_indices].min()
    assert times.iloc[manifest.val_indices].max() <= times.iloc[manifest.test_indices].min()
    # Stored in timestamp order; must NOT be numerically sorted.
    assert list(manifest.train_indices) != sorted(manifest.train_indices)


def test_opt001_development_rows_consumed_in_timestamp_order(config):
    name = _opt001_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="tick",
        random_state=42,
    )
    dev_df, m = development_rows_from_manifest(config, df, manifest.manifest_id)
    # Protected holdout excluded; development rows in timestamp order.
    assert len(dev_df) == manifest.n_train + manifest.n_val
    assert dev_df["tick"].is_monotonic_increasing


def test_opt001_legacy_numeric_manifest_reordered_on_consumption(config):
    name = _opt001_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="tick",
        random_state=42,
    )
    # Tamper the stored manifest so train/val are numerically sorted (legacy
    # shape produced by the buggy creator). Consumption must still re-order by
    # timestamp rather than trust physical CSV order.
    import time as _time

    tampered = SplitManifest(
        manifest_id=manifest.manifest_id,
        dataset_name=name,
        dataset_fingerprint=_dataset_fingerprint(df),
        created_at=_time.time(),
        split_type="time",
        train_indices=sorted(manifest.train_indices),
        val_indices=sorted(manifest.val_indices),
        test_indices=sorted(manifest.test_indices),
        n_train=manifest.n_train,
        n_val=manifest.n_val,
        n_test=manifest.n_test,
        stratify_column=None,
        group_column=None,
        time_column="tick",
        random_state=42,
        protected_holdout=True,
    )
    SplitStore(config).save(tampered)
    dev_df, m = development_rows_from_manifest(config, df, manifest.manifest_id)
    assert dev_df["tick"].is_monotonic_increasing


def test_opt001_datetime_timestamps_chronological(config):
    n = 60
    base = list(pd.date_range("2020-01-01", periods=n, freq="h"))
    rng = random.Random(7)
    rng.shuffle(base)  # chronology is the timestamp value, not row order
    df = pd.DataFrame(
        {
            "t": base,
            "x": [float(i) for i in range(n)],
            "target": [float(i % 2) for i in range(n)],
        }
    )
    name = _write_csv(config, "opt001_datetime.csv", df)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="t",
        random_state=42,
    )
    times = pd.to_datetime(df["t"])
    assert times.iloc[manifest.train_indices].max() <= times.iloc[manifest.val_indices].min()
    assert times.iloc[manifest.val_indices].max() <= times.iloc[manifest.test_indices].min()
    # Stored in timestamp order; must NOT be numerically sorted.
    assert list(manifest.train_indices) != sorted(manifest.train_indices)


def test_opt001_ties_deterministic_boundary(config):
    # Shuffled timestamps with ties: chronology != row order; boundary policy
    # must be deterministic and never place a later training time before val.
    n = 60
    base = [pd.Timestamp("2020-01-01") + pd.Timedelta(hours=(i // 10)) for i in range(n)]
    rng = random.Random(7)
    rng.shuffle(base)
    df = pd.DataFrame(
        {
            "time": base,
            "x": [float(i) for i in range(n)],
            "target": [i % 2 for i in range(n)],
        }
    )
    name = _write_csv(config, "opt001_ties.csv", df)
    m1 = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="time",
        random_state=42,
    )
    m2 = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="time",
        random_state=42,
    )
    assert m1.train_indices == m2.train_indices
    times = pd.to_datetime(df["time"])
    assert times.iloc[m1.train_indices].max() <= times.iloc[m1.val_indices].min()


def test_opt001_row_order_preserved(config):
    n = 60
    df = pd.DataFrame({"x": [float(i) for i in range(n)], "target": [i % 2 for i in range(n)]})
    name = _write_csv(config, "opt001_row_order.csv", df)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        row_order=True,
        random_state=42,
    )
    assert manifest.time_column is None
    assert max(manifest.train_indices) < min(manifest.val_indices) < min(manifest.test_indices)
    from devray_datalab.core.splits import validate_split_manifest

    rep = validate_split_manifest(
        config=config, manifest_id=manifest.manifest_id, df=df, dataset_name=name
    )
    assert rep["valid"] is True, rep["errors"]


def test_opt001_group_isolation(config):
    n = 90
    df = pd.DataFrame(
        {
            "group": [i % 5 for i in range(n)],
            "x": [float(i) for i in range(n)],
            "target": [i % 2 for i in range(n)],
        }
    )
    name = _write_csv(config, "opt001_groups.csv", df)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="group",
        test_size=0.2,
        val_size=0.2,
        group_column="group",
        random_state=42,
    )
    groups = df["group"]
    tg = {groups.iloc[i] for i in manifest.train_indices}
    vg = {groups.iloc[i] for i in manifest.val_indices}
    te = {groups.iloc[i] for i in manifest.test_indices}
    assert not (tg & vg) and not (tg & te) and not (vg & te)


def test_opt001_compare_models_time_manifest_chronological(config):
    name = _opt001_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="tick",
        random_state=42,
    )
    result = _call(
        compare_models_tool,
        config,
        {
            "name": name,
            "target_column": "target",
            "task_type": "regression",
            "models": ["linear_regression", "decision_tree"],
            "cv": 3,
            "split_manifest_id": manifest.manifest_id,
            "exclude_columns": ["row_id", "tick", "group", "cat"],
        },
    )
    data = _data(result)
    for model_result in data["results"]:
        if model_result["status"] == "completed":
            cv = model_result["cv"]
            assert cv["strategy"] == "TimeSeriesSplit", cv["strategy"]
            assert sum(cv["fold_sizes"]) <= manifest.n_train + manifest.n_val


def test_opt001_tune_and_automl_time_manifest(config):
    name = _opt001_df(config)
    base = {
        "name": name,
        "target_column": "target",
        "task_type": "regression",
        "split_manifest_id": None,  # set below
        "exclude_columns": ["row_id", "tick", "group", "cat"],
    }
    df = pd.read_csv(config.workspace_dir / name)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name=name,
        split_type="time",
        test_size=0.2,
        val_size=0.2,
        time_column="tick",
        random_state=42,
    )
    base["split_manifest_id"] = manifest.manifest_id

    tune = _call(
        tune_model_tool,
        config,
        {**base, "model_name": "decision_tree", "cv": 3, "max_trials": 3, "max_time_seconds": 30},
    )
    assert _payload(tune)["ok"] is True, _payload(tune).get("error")

    auto = _call(
        automl_run_tool,
        config,
        {**base, "models": ["decision_tree"], "cv": 3, "max_trials": 3, "max_time_seconds": 30},
    )
    assert _payload(auto)["ok"] is True, _payload(auto).get("error")

    board = _call(
        get_model_leaderboard_tool,
        config,
        {**base, "models": ["decision_tree"], "cv": 3, "max_trials": 3, "max_time_seconds": 30},
    )
    assert _payload(board)["ok"] is True, _payload(board).get("error")


# ---------------------------------------------------------------------------
# OPT-002: Optuna ignores remaining time budget
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def monotonic(self) -> float:
        return self.t


def _make_fake_cv(clock: _FakeClock, cost: float, eval_starts: list, score: float = 0.5):
    def fake_cv(
        df,
        target_column,
        task_type,
        model_cls,
        model_kwargs,
        strategy,
        n_splits,
        random_state,
        group_column=None,
        time_column=None,
        scoring="r2",
        exclude_columns=None,
    ):
        eval_starts.append(clock.monotonic())
        clock.t += cost
        return CVResult(
            strategy=strategy,
            n_folds=n_splits,
            scoring=scoring,
            fold_metrics=[{scoring: score}] * n_splits,
            mean=score,
            std=0.0,
            fold_sizes=[10] * n_splits,
            runtime_seconds=0.0,
            explanation="x",
        )

    return fake_cv


def test_opt002_deadline_stops_new_candidates_and_optuna(monkeypatch):
    import devray_datalab.core.automl as am

    clock = _FakeClock(0.0)
    eval_starts: list = []
    fake_cv = _make_fake_cv(clock, cost=0.4, eval_starts=eval_starts)
    monkeypatch.setattr(am, "cross_validate_with_preprocessing", fake_cv)
    monkeypatch.setattr(am, "monotonic", clock.monotonic)
    if _OPTUNA_INSTALLED:

        def unexpected_study(**kwargs):
            raise AssertionError("candidate evaluations exhausted the time budget")

        monkeypatch.setattr(am.optuna, "create_study", unexpected_study)

    result = run_automl(
        _small_regression_df(),
        "target",
        "regression",
        metric="r2",
        max_trials=10,
        max_time_seconds=1,
        random_state=42,
        n_splits=2,
    )

    deadline = 1.0
    assert all(t < deadline for t in eval_starts), (
        f"evaluation admitted at/after deadline: {eval_starts}"
    )
    assert result.optuna_used is False
    assert len(result.trials) == 3
    assert all(t.params == {} for t in result.trials)
    assert result.best_value == 0.5
    assert result.error is None


def test_opt002_candidate_exhaustion_before_phase_two(monkeypatch):
    import devray_datalab.core.automl as am

    clock = _FakeClock(0.0)
    eval_starts: list = []
    fake_cv = _make_fake_cv(clock, cost=0.05, eval_starts=eval_starts)
    monkeypatch.setattr(am, "cross_validate_with_preprocessing", fake_cv)
    monkeypatch.setattr(am, "monotonic", clock.monotonic)

    # max_trials=2: phase 1 consumes the cap; Optuna must not add trials.
    result = run_automl(
        _small_regression_df(),
        "target",
        "regression",
        metric="r2",
        max_trials=2,
        max_time_seconds=60,
        random_state=42,
        n_splits=2,
    )

    assert all(t < 60.0 for t in eval_starts)
    assert result.optuna_used is False
    assert len(result.trials) == 2


def test_opt002_without_optuna_respects_deadline(monkeypatch):
    import devray_datalab.core.automl as am

    monkeypatch.setattr(am, "_OPTUNA_AVAILABLE", False)
    clock = _FakeClock(0.0)
    eval_starts: list = []
    fake_cv = _make_fake_cv(clock, cost=0.5, eval_starts=eval_starts)
    monkeypatch.setattr(am, "cross_validate_with_preprocessing", fake_cv)
    monkeypatch.setattr(am, "monotonic", clock.monotonic)

    result = run_automl(
        _small_regression_df(),
        "target",
        "regression",
        metric="r2",
        max_trials=10,
        max_time_seconds=1,
        random_state=42,
        n_splits=2,
    )

    assert result.optuna_used is False
    assert len(result.trials) == 2
    assert eval_starts == [0.0, 0.5]
    assert result.best_value == 0.5
    assert result.error is None


def _trace_optuna_studies(monkeypatch, clock, setup_cost, optimize_delay=0.0):
    if not _OPTUNA_INSTALLED:
        import pytest

        pytest.skip("optuna not installed")

    import devray_datalab.core.automl as am

    studies = []
    timeouts = []
    create_study = am.optuna.create_study

    def delayed_create_study(**kwargs):
        clock.t += setup_cost
        study = create_study(**kwargs)
        studies.append(study)
        optimize = study.optimize

        def delayed_optimize(objective, *, n_trials, timeout):
            timeouts.append(timeout)
            clock.t += optimize_delay
            # Let the real Optuna scheduler run without its relative timeout
            # so only DataLab can reject a late objective or stop the study.
            optimize(objective, n_trials=n_trials)

        monkeypatch.setattr(study, "optimize", delayed_optimize)
        return study

    monkeypatch.setattr(am.optuna, "create_study", delayed_create_study)
    monkeypatch.setattr(am, "monotonic", clock.monotonic)
    return studies, timeouts


def _run_deadline_search():
    return run_automl(
        _small_regression_df(),
        "target",
        "regression",
        metric="r2",
        max_trials=5,
        max_time_seconds=1,
        random_state=42,
        model_subset=["decision_tree"],
        n_splits=2,
    )


def test_opt002_study_setup_can_exhaust_deadline(monkeypatch):
    import devray_datalab.core.automl as am

    clock = _FakeClock(100.0)
    eval_starts = []
    studies, timeouts = _trace_optuna_studies(monkeypatch, clock, setup_cost=0.75)
    monkeypatch.setattr(
        am, "cross_validate_with_preprocessing", _make_fake_cv(clock, 0.25, eval_starts)
    )

    result = _run_deadline_search()

    assert eval_starts == [100.0]
    assert timeouts == []
    assert studies[0].trials == []
    assert result.optuna_used is False
    assert result.best_value == 0.5
    assert len(result.trials) == 1
    assert result.error is None


def test_opt002_optuna_rejects_objective_at_deadline(monkeypatch):
    import devray_datalab.core.automl as am

    clock = _FakeClock(100.0)
    eval_starts = []
    studies, timeouts = _trace_optuna_studies(
        monkeypatch, clock, setup_cost=0.25, optimize_delay=0.5
    )
    monkeypatch.setattr(
        am, "cross_validate_with_preprocessing", _make_fake_cv(clock, 0.25, eval_starts)
    )

    result = _run_deadline_search()

    assert timeouts == [0.5]
    assert eval_starts == [100.0]
    [rejected_trial] = studies[0].trials
    assert rejected_trial.state == TrialState.PRUNED
    assert rejected_trial.params == {}
    assert result.best_model == "decision_tree"
    assert result.best_params == {}
    assert result.best_value == 0.5
    assert len(result.trials) == 1
    assert result.error is None


def test_opt002_in_flight_trial_can_finish_after_deadline(monkeypatch):
    import devray_datalab.core.automl as am

    clock = _FakeClock(100.0)
    eval_starts = []
    studies, timeouts = _trace_optuna_studies(monkeypatch, clock, setup_cost=0.125)
    candidate_cv = _make_fake_cv(clock, 0.25, eval_starts)
    tuned_cv = _make_fake_cv(clock, 0.75, eval_starts, score=0.75)

    def evaluate(**kwargs):
        cv = tuned_cv if "max_depth" in kwargs["model_kwargs"] else candidate_cv
        return cv(**kwargs)

    monkeypatch.setattr(am, "cross_validate_with_preprocessing", evaluate)

    result = _run_deadline_search()

    assert timeouts == [0.625]
    assert eval_starts == [100.0, 100.375]
    assert clock.monotonic() == 101.125
    [completed_trial] = studies[0].trials
    assert completed_trial.state == TrialState.COMPLETE
    assert result.optuna_used is True
    assert [t.value for t in result.trials] == [0.75, 0.5]
    assert result.best_model == "decision_tree"
    assert result.best_params == completed_trial.params
    assert result.best_value == 0.75
    assert result.error is None


# ---------------------------------------------------------------------------
# OPT-003: seeded Optuna is nondeterministic
# ---------------------------------------------------------------------------


def test_opt003_seeded_optuna_deterministic(config, regression_csv):
    if not _OPTUNA_INSTALLED:
        import pytest

        pytest.skip("optuna not installed")
    # Nonlinear relation with no constant folds: a tunable tree model wins
    # phase 1 and is then tuned by Optuna, so the unseeded sampler's randomness
    # would otherwise show across runs. Seeding the sampler (and estimator init)
    # must make trials identical.
    n = 60
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(n)],
            "z": [float(i % 5) for i in range(n)],
            "target": [float(i * i) for i in range(n)],
        }
    )
    base = dict(
        df=df,
        target_column="target",
        task_type="regression",
        metric="r2",
        max_trials=10,
        max_time_seconds=60,
        random_state=73,
        n_splits=3,
    )
    runs = [run_automl(**base) for _ in range(3)]
    assert all(r.optuna_used for r in runs)
    assert all(len(r.trials) == len(runs[0].trials) for r in runs)
    # At least one tuned trial carries suggested hyperparameters (sampler
    # randomness would otherwise change them across runs).
    tuned = [t for t in runs[0].trials if t.params]
    assert tuned, "expected Optuna to suggest hyperparameters for a tunable model"
    for r1, r2 in zip(runs, runs[1:], strict=False):
        assert [t.params for t in r1.trials] == [t.params for t in r2.trials], (
            f"trial parameter sequences differ:\n{r1.trials}\n{r2.trials}"
        )
        assert [t.value for t in r1.trials] == [t.value for t in r2.trials], (
            f"objective values differ:\n{r1.trials}\n{r2.trials}"
        )


# ---------------------------------------------------------------------------
# AUD-004: inferior configurations can replace better candidates
# ---------------------------------------------------------------------------


class _FakeTrial:
    def __init__(self, value, params, model="decision_tree"):
        self.value = value
        self.params = params
        self.model = model
        self.state = TrialState.COMPLETE


class _FakeStudy:
    def __init__(self, best_value, best_params, model="decision_tree"):
        self.best_trial = _FakeTrial(best_value, best_params, model)
        self.trials = [self.best_trial]

    def optimize(self, objective, n_trials=None, timeout=None):
        return None


def _patch_optuna_study(monkeypatch, best_value, best_params, model="decision_tree"):
    import devray_datalab.core.automl as am

    monkeypatch.setattr(am, "_OPTUNA_AVAILABLE", True)
    monkeypatch.setattr(
        am.optuna,
        "create_study",
        lambda direction, **kw: _FakeStudy(best_value, best_params, model),
    )


def test_aud004_inferior_optuna_does_not_displace_candidate_maximize(monkeypatch, config):
    if not _OPTUNA_INSTALLED:
        import pytest

        pytest.skip("optuna not installed")
    _patch_optuna_study(monkeypatch, best_value=0.5, best_params={"max_depth": 999})
    name = _regression_perfect_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    result = run_automl(
        df,
        "target",
        "regression",
        metric="r2",
        max_trials=10,
        max_time_seconds=60,
        random_state=42,
        n_splits=3,
    )
    # linear_regression scores R^2 == 1; Optuna's best trial (0.5) is inferior.
    assert result.best_value == 1.0, result.trials
    assert result.best_model == "linear_regression", result.trials
    assert result.best_params == {}, result.trials
    assert result.optuna_used is True
    optuna_trials = [t for t in result.trials if t.params == {"max_depth": 999}]
    assert len(optuna_trials) == 1
    assert optuna_trials[0].value == 0.5
    best_trials = [t for t in result.trials if t.value == result.best_value]
    assert len(best_trials) == 1
    assert best_trials[0].model == "linear_regression"


def test_aud004_inferior_optuna_does_not_displace_candidate_minimize(monkeypatch, config):
    if not _OPTUNA_INSTALLED:
        import pytest

        pytest.skip("optuna not installed")
    _patch_optuna_study(monkeypatch, best_value=5.0, best_params={"C": 999.0})
    name = _regression_perfect_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    result = run_automl(
        df,
        "target",
        "regression",
        metric="mae",
        max_trials=10,
        max_time_seconds=60,
        random_state=42,
        n_splits=3,
    )
    # linear_regression scores MAE == 0 (minimize); Optuna's best (5.0) is inferior.
    assert result.best_value == 0.0, result.trials
    assert result.best_model == "linear_regression", result.trials
    assert result.best_params == {}, result.trials
    assert result.optuna_used is True
    optuna_trials = [t for t in result.trials if t.params == {"C": 999.0}]
    assert len(optuna_trials) == 1
    assert optuna_trials[0].value == 5.0


def test_aud004_optuna_failure_retains_candidate(monkeypatch, config):
    if not _OPTUNA_INSTALLED:
        import pytest

        pytest.skip("optuna not installed")
    _patch_optuna_study(monkeypatch, best_value=float("nan"), best_params={"max_depth": 999})
    name = _regression_perfect_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    result = run_automl(
        df,
        "target",
        "regression",
        metric="r2",
        max_trials=10,
        max_time_seconds=60,
        random_state=42,
        n_splits=3,
    )
    assert result.best_value == 1.0
    assert result.best_model == "linear_regression"
    assert result.best_params == {}


def test_aud004_leaderboard_agrees_with_best(config):
    name = _regression_perfect_df(config)
    df = pd.read_csv(config.workspace_dir / name)
    result = run_automl(
        df,
        "target",
        "regression",
        metric="r2",
        max_trials=10,
        max_time_seconds=60,
        random_state=42,
        n_splits=3,
    )
    valid = [t for t in result.trials if t.error is None and t.value == t.value]
    assert valid
    expected_best = max(valid, key=lambda t: t.value)
    assert result.best_value == expected_best.value
    assert result.best_model == expected_best.model
    # leaderboard trials are ranked consistently
    values = [t.value for t in valid]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# AUD-005: accepted cancellation becomes failure
# ---------------------------------------------------------------------------


def test_aud005_queued_cancellation_reaches_cancelled(config, classification_csv):
    df = pd.read_csv(classification_csv)
    manager = JobManager(config, Trainer(config))
    job_id = manager.submit(
        df=df,
        target_column="purchased",
        task_type="classification",
        model_name="logistic_regression",
        random_state=42,
    )
    assert manager.cancel_job(job_id) is True
    manager._process_job(job_id)
    job = manager.get_job(job_id)
    assert job.status == JobStatus.CANCELLED, job.status


def test_aud005_active_cancellation_is_cancelled_not_failed(config, classification_csv):
    df = pd.read_csv(classification_csv)
    manager = JobManager(config, Trainer(config))
    job_id = manager.submit(
        df=df,
        target_column="purchased",
        task_type="classification",
        model_name="logistic_regression",
        random_state=42,
    )
    # Request cancellation before the worker starts training.
    manager.get_job(job_id).cancel_requested = True
    manager._process_job(job_id)
    job = manager.get_job(job_id)
    assert job.status == JobStatus.CANCELLED, job.status
    assert job.error == "Training cancelled."


def test_aud005_genuine_training_failure_remains_failed(config, classification_csv):
    df = pd.read_csv(classification_csv)
    manager = JobManager(config, Trainer(config))
    job_id = manager.submit(
        df=df,
        target_column="does_not_exist",
        task_type="classification",
        model_name="logistic_regression",
        random_state=42,
    )
    manager._process_job(job_id)
    job = manager.get_job(job_id)
    assert job.status == JobStatus.FAILED, job.status


def test_aud005_completed_job_wins_cancellation_race(config, classification_csv):
    df = pd.read_csv(classification_csv)
    manager = JobManager(config, Trainer(config))
    job_id = manager.submit(
        df=df,
        target_column="purchased",
        task_type="classification",
        model_name="logistic_regression",
        random_state=42,
    )
    manager._process_job(job_id)
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED, job.status
    # Cancelling an already-terminal job must not change its state.
    assert manager.cancel_job(job_id) is False
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED


def test_aud005_subsequent_job_succeeds_after_cancellation(config, classification_csv):
    df = pd.read_csv(classification_csv)
    manager = JobManager(config, Trainer(config))
    bad = manager.submit(
        df=df,
        target_column="does_not_exist",
        task_type="classification",
        model_name="logistic_regression",
        random_state=42,
    )
    manager._process_job(bad)
    assert manager.get_job(bad).status == JobStatus.FAILED

    good = manager.submit(
        df=df,
        target_column="purchased",
        task_type="classification",
        model_name="logistic_regression",
        random_state=42,
    )
    manager._process_job(good)
    assert manager.get_job(good).status == JobStatus.COMPLETED


def test_aud005_tool_queued_cancellation_status_and_list_agree(config, classification_csv):
    start = _call(
        start_training_job_tool,
        config,
        {
            "name": "sample_classification.csv",
            "target_column": "purchased",
            "task_type": "classification",
            "model_name": "logistic_regression",
            "random_state": 42,
        },
    )
    start_data = _data(start)
    job_id = start_data["job_id"]
    cancel = _call(cancel_training_job_tool, config, {"job_id": job_id})
    assert _data(cancel).get("cancelled") is True

    status = None
    for _ in range(100):
        st = _call(get_training_status_tool, config, {"job_id": job_id})
        status = _data(st)["status"]
        if status in ("cancelled", "completed", "failed"):
            break
        _time.sleep(0.05)
    assert status == "cancelled", status

    # Listing must agree with status and contain no raw dataframe.
    lst = _call(list_training_jobs_tool, config, {"limit": 10})
    lst_data = _data(lst)
    ids = [j["job_id"] for j in lst_data["jobs"]]
    assert job_id in ids
    listed = next(j for j in lst_data["jobs"] if j["job_id"] == job_id)
    assert listed["status"] == "cancelled"
    assert "df" not in json.dumps(lst_data)


# ---------------------------------------------------------------------------
# AUD-006: supported model rejected by explicit tuning
# ---------------------------------------------------------------------------


def test_aud006_svm_regression_baseline_and_tuning(config, regression_csv):
    # train_baseline accepts regression SVM.
    r1 = _call(
        train_baseline_tool,
        config,
        {
            "name": "sample_regression.csv",
            "target_column": "spend",
            "task_type": "regression",
            "model_name": "svm",
            "exclude_columns": [],
        },
    )
    assert _payload(r1)["ok"] is True, _payload(r1).get("error")

    # tune_model(model_name="svm") must resolve against the full registry.
    r2 = _call(
        tune_model_tool,
        config,
        {
            "name": "sample_regression.csv",
            "target_column": "spend",
            "task_type": "regression",
            "model_name": "svm",
            "max_trials": 2,
            "max_time_seconds": 30,
            "cv": 2,
            "exclude_columns": [],
        },
    )
    payload = _payload(r2)
    assert payload["ok"] is True, payload.get("error")
    data = payload["data"]
    assert data["model"] == "svm"


def test_aud006_hist_gradient_boosting_tuning(config, regression_csv):
    r = _call(
        tune_model_tool,
        config,
        {
            "name": "sample_regression.csv",
            "target_column": "spend",
            "task_type": "regression",
            "model_name": "hist_gradient_boosting",
            "max_trials": 2,
            "max_time_seconds": 30,
            "cv": 2,
            "exclude_columns": [],
        },
    )
    payload = _payload(r)
    assert payload["ok"] is True, payload.get("error")
    assert payload["data"]["model"] == "hist_gradient_boosting"


def test_aud006_classification_svm_tuning(config, classification_csv):
    r = _call(
        tune_model_tool,
        config,
        {
            "name": "sample_classification.csv",
            "target_column": "purchased",
            "task_type": "classification",
            "model_name": "svm",
            "max_trials": 2,
            "max_time_seconds": 30,
            "cv": 2,
            "exclude_columns": [],
        },
    )
    payload = _payload(r)
    assert payload["ok"] is True, payload.get("error")
    assert payload["data"]["model"] == "svm"


def test_aud006_unknown_tuning_model_rejected_before_optimization(config, regression_csv):
    r = _call(
        tune_model_tool,
        config,
        {
            "name": "sample_regression.csv",
            "target_column": "spend",
            "task_type": "regression",
            "model_name": "not_a_real_model",
            "max_trials": 2,
            "max_time_seconds": 30,
            "cv": 2,
            "exclude_columns": [],
        },
    )
    payload = _payload(r)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"


# ---------------------------------------------------------------------------
# AUD-007: unsafe path receives incorrect error classification
# ---------------------------------------------------------------------------


_AUD007_NAMES = [
    "/synthetic/workspace/sentinel",
    "/tmp/sentinel",
    "..\\..\\reports\\sentinel",
    "../reports/sentinel",
    "C:/Windows/sentinel",
    "sentinel/../escape",
]


def test_aud007_predict_rejects_unsafe_paths_with_validation_error(config):
    for name in _AUD007_NAMES:
        r = _call(
            predict_tool,
            config,
            {
                "artifact_name": name,
                "features": [{"x": 0, "z": 0}],
            },
        )
        payload = _payload(r)
        assert payload["ok"] is False, f"{name!r} was accepted"
        assert payload["error"]["code"] == "validation_error", f"{name!r}: {payload}"
        # Strict JSON: must parse and leak no sentinel contents.
        text = r[0].text
        json.loads(text)
        assert "sentinel" not in text.lower()


def test_aud007_load_model_handle_rejects_unsafe_paths(config):
    from devray_datalab.tools.experiments import load_model_handle_tool

    for name in _AUD007_NAMES:
        r = _call(load_model_handle_tool, config, {"artifact_name": name})
        payload = _payload(r)
        assert payload["ok"] is False, f"{name!r} was accepted"
        assert payload["error"]["code"] == "validation_error", f"{name!r}: {payload}"
        text = r[0].text
        json.loads(text)
        assert "sentinel" not in text.lower()


def test_aud007_export_rejects_unsafe_paths(config, classification_csv):
    from devray_datalab.tools.models import export_model_tool

    _write_csv(config, "sample_classification.csv", pd.read_csv(classification_csv))

    for name in _AUD007_NAMES:
        r = _call(
            export_model_tool,
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "artifact_name": name,
                "exclude_columns": [],
            },
        )
        payload = _payload(r)
        assert payload["ok"] is False, f"{name!r} was accepted"
        assert payload["error"]["code"] == "validation_error", f"{name!r}: {payload}"
        text = r[0].text
        json.loads(text)
        assert "sentinel" not in text.lower()


def test_aud007_valid_artifact_predict_works(config, classification_csv):
    # A real export -> load -> predict round trip still functions.
    export = _call(
        export_model_tool,
        config,
        {
            "name": "sample_classification.csv",
            "target_column": "purchased",
            "task_type": "classification",
            "model_name": "logistic_regression",
            "artifact_name": "aud007_valid_model",
            "exclude_columns": [],
        },
    )
    assert _payload(export)["ok"] is True, _payload(export).get("error")

    from devray_datalab.tools.experiments import load_model_handle_tool

    load = _call(load_model_handle_tool, config, {"artifact_name": "aud007_valid_model"})
    assert _payload(load)["ok"] is True, _payload(load).get("error")

    predict = _call(
        predict_tool,
        config,
        {
            "artifact_name": "aud007_valid_model",
            "features": [
                {
                    "age": 30,
                    "income": 50000,
                    "credit_score": 700,
                    "category": "electronics",
                    "region": "north",
                }
            ],
        },
    )
    assert _payload(predict)["ok"] is True, _payload(predict).get("error")
