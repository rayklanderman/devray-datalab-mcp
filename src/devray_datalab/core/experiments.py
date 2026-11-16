"""Local experiment tracking using SQLite.

Tracks each training run: dataset fingerprint, feature schema, algorithm,
hyperparameters, library versions, training duration, validation metrics,
resource consumption, warnings, and exported artifacts. No cloud experiment
tracking service is required.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    """A recorded training experiment."""

    experiment_id: str
    created_at: float
    dataset_name: str
    dataset_fingerprint: str
    task_type: str
    algorithm: str
    hyperparameters: dict[str, Any]
    library_versions: dict[str, str]
    metrics: dict[str, Any]
    cv_result: dict[str, Any] | None
    n_train: int
    n_test: int
    feature_schema: dict[str, Any]
    training_duration_seconds: float
    artifact_name: str | None
    status: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentStore:
    """SQLite-backed store for training experiments."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_fingerprint TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    library_versions TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    cv_result TEXT,
                    n_train INTEGER NOT NULL,
                    n_test INTEGER NOT NULL,
                    feature_schema TEXT NOT NULL,
                    training_duration_seconds REAL NOT NULL,
                    artifact_name TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC)"
            )
            conn.commit()

    def _fingerprint_dataset(self, df_bytes: bytes) -> str:
        return hashlib.sha256(df_bytes).hexdigest()

    def record(
        self,
        experiment_id: str,
        dataset_name: str,
        df_bytes: bytes,
        task_type: str,
        algorithm: str,
        hyperparameters: dict[str, Any],
        library_versions: dict[str, str],
        metrics: dict[str, Any],
        cv_result: dict[str, Any] | None,
        n_train: int,
        n_test: int,
        feature_schema: dict[str, Any],
        training_duration_seconds: float,
        artifact_name: str | None,
        status: str,
        error: str | None,
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            experiment_id=experiment_id,
            created_at=time.time(),
            dataset_name=dataset_name,
            dataset_fingerprint=self._fingerprint_dataset(df_bytes),
            task_type=task_type,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            library_versions=library_versions,
            metrics=metrics,
            cv_result=cv_result,
            n_train=n_train,
            n_test=n_test,
            feature_schema=feature_schema,
            training_duration_seconds=training_duration_seconds,
            artifact_name=artifact_name,
            status=status,
            error=error,
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, created_at, dataset_name, dataset_fingerprint,
                    task_type, algorithm, hyperparameters, library_versions,
                    metrics, cv_result, n_train, n_test, feature_schema,
                    training_duration_seconds, artifact_name, status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.created_at,
                    record.dataset_name,
                    record.dataset_fingerprint,
                    record.task_type,
                    record.algorithm,
                    json.dumps(record.hyperparameters, default=str),
                    json.dumps(record.library_versions, default=str),
                    json.dumps(record.metrics, default=str),
                    json.dumps(record.cv_result, default=str) if record.cv_result else None,
                    record.n_train,
                    record.n_test,
                    json.dumps(record.feature_schema, default=str),
                    record.training_duration_seconds,
                    record.artifact_name,
                    record.status,
                    record.error,
                ),
            )
            conn.commit()
        return record

    def list_experiments(self, limit: int = 50) -> list[ExperimentRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def compare_experiments(self, experiment_ids: list[str]) -> list[ExperimentRecord]:
        records: list[ExperimentRecord] = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for eid in experiment_ids:
                row = conn.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (eid,),
                ).fetchone()
                if row:
                    records.append(self._row_to_record(row))
        return records

    def list_models(self) -> list[ExperimentRecord]:
        """Return experiments that produced a model artifact, latest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM experiments
                WHERE artifact_name IS NOT NULL AND status = 'completed'
                ORDER BY created_at DESC
                """,
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            created_at=row["created_at"],
            dataset_name=row["dataset_name"],
            dataset_fingerprint=row["dataset_fingerprint"],
            task_type=row["task_type"],
            algorithm=row["algorithm"],
            hyperparameters=json.loads(row["hyperparameters"]),
            library_versions=json.loads(row["library_versions"]),
            metrics=json.loads(row["metrics"]),
            cv_result=(json.loads(row["cv_result"]) if row["cv_result"] else None),
            n_train=row["n_train"],
            n_test=row["n_test"],
            feature_schema=json.loads(row["feature_schema"]),
            training_duration_seconds=row["training_duration_seconds"],
            artifact_name=row["artifact_name"],
            status=row["status"],
            error=row["error"],
        )
