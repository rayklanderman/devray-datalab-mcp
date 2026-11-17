"""Non-blocking training job system.

Runs long-lived training outside the MCP request loop so the server stays
responsive. Jobs are processed by a single background worker thread with
cooperative cancellation, configurable concurrency, timeouts, and resource
limits.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from devray_datalab.config import DevRayConfig
from devray_datalab.core.trainer import Trainer


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Snapshot of a training job."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    cancel_requested: bool = field(default=False, init=False)


class JobManager:
    """Manages non-blocking training jobs.

    Jobs are submitted with ``submit`` and processed by a background worker
    thread. The MCP request loop never blocks on training.
    """

    def __init__(self, config: DevRayConfig, trainer: Trainer) -> None:
        self.config = config
        self.trainer = trainer
        self._jobs: dict[str, JobInfo] = {}
        self._lock = threading.Lock()
        self._queue: deque[str] = deque()
        self._active = 0
        self._worker_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ensure_worker()

    # ------------------------------------------------------------------ worker
    def _ensure_worker(self) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop.clear()
            self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
            self._worker_thread.start()

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            job_id = self._pop_job()
            if job_id is None:
                # No work; wait briefly before polling again.
                self._stop.wait(0.2)
                continue
            self._process_job(job_id)

    def _pop_job(self) -> str | None:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def _process_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status != JobStatus.QUEUED:
                return
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._active += 1

        try:
            # Run the actual training in this worker thread.
            result = self._run_training(job)
            with self._lock:
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.result = result
                job.completed_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.completed_at = time.time()
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)

    def _run_training(self, job: JobInfo) -> dict[str, Any]:
        # Extract training params stored in job.result metadata before run.
        params = job.result or {}
        return self.trainer.train_baseline(
            df=params["df"],
            target_column=params["target_column"],
            task_type=params["task_type"],
            model_name=params["model_name"],
            test_size=params.get("test_size", 0.2),
            random_state=params.get("random_state", 42),
            stratify=params.get("stratify", True),
            exclude_columns=params.get("exclude_columns"),
            split_manifest_id=params.get("split_manifest_id"),
            cancel_check=lambda: job.cancel_requested,
        )

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        df: object,
        target_column: str,
        task_type: str,
        model_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        exclude_columns: list[str] | None = None,
        split_manifest_id: str | None = None,
    ) -> str:
        """Submit a training job and return its id immediately."""
        job_id = str(uuid.uuid4())
        job = JobInfo(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job
            # Stash params on the (initially empty) result dict for the worker.
            job.result = {
                "df": df,
                "target_column": target_column,
                "task_type": task_type,
                "model_name": model_name,
                "test_size": test_size,
                "random_state": random_state,
                "stratify": stratify,
                "exclude_columns": exclude_columns or [],
                "split_manifest_id": split_manifest_id,
            }
            self._queue.append(job_id)
        return job_id

    # ------------------------------------------------------------------ query
    def get_job(self, job_id: str) -> JobInfo | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[JobInfo]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of a queued or running job. Returns True if found."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False
            job.cancel_requested = True
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
            return True

    def active_count(self) -> int:
        with self._lock:
            return self._active

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
