"""Reproducible dataset split manifests.

Creates, validates, and lists train/validation/test split manifests with
dataset fingerprints. Splits are stored as JSON files so they can be reused
across tool calls. The protected test/holdout partition is never exposed to
training or hyperparameter tuning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from devray_datalab.config import DevRayConfig
from devray_datalab.utils.errors import DatasetError, ValidationError

SPLITS_SUBDIR = "splits"


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of dataset contents (values + dtypes + columns)."""
    h = hashlib.sha256()
    h.update(json.dumps(list(df.columns.astype(str))).encode("utf-8"))
    h.update(json.dumps(list(df.dtypes.astype(str))).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


@dataclass
class SplitManifest:
    """A stored train/validation/test split."""

    manifest_id: str
    dataset_name: str
    dataset_fingerprint: str
    created_at: float
    split_type: str  # 'random', 'stratified', 'group', 'time'
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    n_train: int
    n_val: int
    n_test: int
    stratify_column: str | None
    group_column: str | None
    time_column: str | None
    random_state: int
    protected_holdout: bool  # True if test set is a protected holdout

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def overlaps(self, other: "SplitManifest") -> bool:
        """True if any index appears in both manifests' train/val/test sets."""
        a = set(self.train_indices + self.val_indices + self.test_indices)
        b = set(other.train_indices + other.val_indices + other.test_indices)
        return bool(a & b)

    def is_disjoint(self, other: "SplitManifest") -> bool:
        return not self.overlaps(other)


class SplitStore:
    """JSON-file-backed store for split manifests."""

    def __init__(self, config: DevRayConfig) -> None:
        self.config = config
        self.dir = config.artifacts_dir / SPLITS_SUBDIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, manifest_id: str) -> Path:
        return self.dir / f"{manifest_id}.json"

    def save(self, manifest: SplitManifest) -> None:
        with open(self._path(manifest.manifest_id), "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def load(self, manifest_id: str) -> SplitManifest | None:
        path = self._path(manifest_id)
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return SplitManifest(**data)

    def list_all(self) -> list[SplitManifest]:
        manifests: list[SplitManifest] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                manifests.append(SplitManifest(**data))
            except Exception:  # pragma: no cover - best effort
                continue
        return sorted(manifests, key=lambda m: m.created_at, reverse=True)

    def delete(self, manifest_id: str) -> bool:
        path = self._path(manifest_id)
        if path.is_file():
            path.unlink()
            return True
        return False


def create_split_manifest(
    config: DevRayConfig,
    df: pd.DataFrame,
    dataset_name: str,
    split_type: str = "random",
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
    stratify_column: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    row_order: bool = False,
    protected_holdout: bool = True,
) -> SplitManifest:
    """Create a reproducible train/validation/test split manifest.

    Returns a SplitManifest and persists it. The test set is a protected
    holdout that should not be used for training or hyperparameter tuning.
    """

    if split_type not in ("random", "stratified", "group", "time"):
        raise ValidationError(
            f"Invalid split_type '{split_type}'. Use 'random', 'stratified', 'group', or 'time'."
        )

    n = len(df)
    if n < 3:
        raise ValidationError("Dataset must have at least 3 rows to split.")

    if test_size <= 0 or test_size >= 1:
        raise ValidationError("test_size must be between 0 and 1 (exclusive).")
    if val_size < 0 or val_size >= 1:
        raise ValidationError("val_size must be between 0 and 1 (exclusive).")
    if test_size + val_size >= 1:
        raise ValidationError("test_size + val_size must be less than 1.")

    # Validate optional columns exist.
    for col in (stratify_column, group_column, time_column):
        if col is not None and col not in df.columns:
            raise ValidationError(f"Column '{col}' not found in dataset.")

    indices = list(range(n))

    if split_type == "time" and time_column is not None:
        # Time-ordered: sort by time column, then split sequentially
        # (earliest = train, latest = protected test holdout).
        order = df[time_column].argsort(kind="mergesort").tolist()
        indices = [int(i) for i in order]
        n_test = max(1, int(math.ceil(n * test_size)))
        n_val = max(0, int(math.ceil(n * val_size)))
        test_indices = indices[-n_test:]
        val_indices = indices[-n_test - n_val : -n_test] if n_val > 0 else []
        train_indices = indices[: -n_test - n_val] if n_val > 0 else indices[:-n_test]
        stratify_column = None  # stratification not applicable with time split
    elif split_type == "time" and row_order:
        # Chronological split by original row order (no date column required):
        # earliest rows = train, latest rows = protected test holdout.
        indices = list(range(n))
        n_test = max(1, int(math.ceil(n * test_size)))
        n_val = max(0, int(math.ceil(n * val_size)))
        test_indices = indices[-n_test:]
        val_indices = indices[-n_test - n_val : -n_test] if n_val > 0 else []
        train_indices = indices[: -n_test - n_val] if n_val > 0 else indices[:-n_test]
        stratify_column = None
    elif split_type == "time":
        raise ValidationError(
            "split_type='time' requires a valid 'time_column' or 'row_order=True'. "
            "A random split is never silently substituted."
        )
    elif split_type == "group" and group_column is not None:
        # Group-aware: split by unique groups.
        groups = df[group_column]
        unique_groups = list(groups.unique())
        rng = random.Random(random_state)
        rng.shuffle(unique_groups)
        n_test_groups = max(1, int(math.ceil(len(unique_groups) * test_size)))
        n_val_groups = max(0, int(math.ceil(len(unique_groups) * val_size)))
        test_groups = set(unique_groups[:n_test_groups])
        val_groups = set(unique_groups[n_test_groups : n_test_groups + n_val_groups])
        train_groups = set(unique_groups[n_test_groups + n_val_groups :])
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]
        val_indices = [i for i, g in enumerate(groups) if g in val_groups]
        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
    elif split_type == "stratified" and stratify_column is not None:
        # Stratified split preserving class proportions.
        from sklearn.model_selection import train_test_split

        stratify_vals = df[stratify_column]
        # First split off test.
        idx_temp, idx_test = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_vals,
        )
        stratify_temp = stratify_vals.iloc[idx_temp]
        if val_size > 0:
            idx_train, idx_val = train_test_split(
                idx_temp,
                test_size=val_size / (1 - test_size),
                random_state=random_state,
                stratify=stratify_temp,
            )
        else:
            idx_train, idx_val = list(idx_temp), []
        train_indices = [int(i) for i in idx_train]
        val_indices = [int(i) for i in idx_val]
        test_indices = [int(i) for i in idx_test]
    else:
        # Random split.
        rng = random.Random(random_state)
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(math.ceil(n * test_size)))
        n_val = max(0, int(math.ceil(n * val_size)))
        test_indices = shuffled[:n_test]
        val_indices = shuffled[n_test : n_test + n_val]
        train_indices = shuffled[n_test + n_val :]

    if len(test_indices) == 0:
        raise ValidationError("Test set is empty; dataset too small or test_size too small.")
    if len(train_indices) == 0:
        raise ValidationError("Train set is empty; dataset too small.")

    manifest = SplitManifest(
        manifest_id=os.urandom(8).hex(),
        dataset_name=dataset_name,
        dataset_fingerprint=_dataset_fingerprint(df),
        created_at=time.time(),
        split_type=split_type,
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
        n_train=len(train_indices),
        n_val=len(val_indices),
        n_test=len(test_indices),
        stratify_column=stratify_column,
        group_column=group_column,
        time_column=time_column,
        random_state=random_state,
        protected_holdout=protected_holdout,
    )

    store = SplitStore(config)
    store.save(manifest)
    return manifest


def validate_split_manifest(
    config: DevRayConfig,
    manifest_id: str,
    df: pd.DataFrame,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Validate a stored split manifest against a dataset.

    Checks: manifest exists, dataset fingerprint matches, splits are disjoint
    and non-empty, indices are within range.
    """
    store = SplitStore(config)
    manifest = store.load(manifest_id)
    if manifest is None:
        raise DatasetError(f"Split manifest not found: {manifest_id}")

    errors: list[str] = []

    # Fingerprint match.
    current_fp = _dataset_fingerprint(df)
    if manifest.dataset_fingerprint != current_fp:
        errors.append(
            "Dataset fingerprint mismatch: the dataset has changed since the manifest was created."
        )

    # Indices within range.
    n = len(df)
    all_idx = manifest.train_indices + manifest.val_indices + manifest.test_indices
    out_of_range = [i for i in all_idx if i < 0 or i >= n]
    if out_of_range:
        errors.append(f"Index(es) out of range for dataset with {n} rows: {out_of_range[:10]}")

    # Disjointness within manifest.
    train_set = set(manifest.train_indices)
    val_set = set(manifest.val_indices)
    test_set = set(manifest.test_indices)
    if train_set & val_set:
        errors.append("Train and validation sets overlap.")
    if train_set & test_set:
        errors.append("Train and test sets overlap (protected holdout violated).")
    if val_set & test_set:
        errors.append("Validation and test sets overlap (protected holdout violated).")

    # Non-empty.
    if not train_set:
        errors.append("Train set is empty.")
    if not test_set:
        errors.append("Test set is empty.")

    # Coverage: every index should appear in exactly one set.
    covered = train_set | val_set | test_set
    if len(covered) != len(all_idx):
        errors.append("Some indices appear in multiple splits or are missing.")

    valid = len(errors) == 0

    return {
        "manifest_id": manifest_id,
        "valid": valid,
        "errors": errors,
        "dataset_fingerprint_match": manifest.dataset_fingerprint == current_fp,
        "n_train": manifest.n_train,
        "n_val": manifest.n_val,
        "n_test": manifest.n_test,
        "split_type": manifest.split_type,
        "protected_holdout": manifest.protected_holdout,
    }


def development_rows_from_manifest(
    config: DevRayConfig, df: pd.DataFrame, manifest_id: str
) -> tuple[pd.DataFrame, SplitManifest]:
    """Load and validate a manifest, returning df restricted to train+val rows.

    The protected test holdout is excluded from the returned dataframe. The
    dataset fingerprint is validated against the supplied dataframe so a
    changed dataset cannot silently reuse stale indices.
    """
    store = SplitStore(config)
    manifest = store.load(manifest_id)
    if manifest is None:
        raise DatasetError(f"Split manifest not found: {manifest_id}")

    if manifest.dataset_fingerprint != _dataset_fingerprint(df):
        raise ValidationError(
            "Dataset fingerprint mismatch: dataset changed since the manifest was created."
        )

    cv_indices = list(manifest.train_indices) + list(manifest.val_indices)
    if not cv_indices:
        raise ValidationError("Manifest has no train/validation rows.")

    return df.iloc[cv_indices].reset_index(drop=True), manifest


def list_split_manifests(config: DevRayConfig) -> list[SplitManifest]:
    """List all stored split manifests, newest first."""
    store = SplitStore(config)
    return store.list_all()
