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
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from devray_datalab.config import DevRayConfig
from devray_datalab.utils.errors import DatasetError, ValidationError

SPLITS_SUBDIR = "splits"

# Manifest identifiers are untrusted input: only safe, portable filenames are
# accepted so every resolved store path stays inside the configured splits
# directory (FD-004).
_ID_PATTERN = re.compile(r"\A[A-Za-z0-9._-]+\Z")
_RESERVED_IDS = {".", ".."}
_ALLOWED_SPLIT_TYPES = frozenset({"random", "stratified", "group", "time"})


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of dataset contents (values + dtypes + columns)."""
    h = hashlib.sha256()
    h.update(json.dumps(list(df.columns.astype(str))).encode("utf-8"))
    h.update(json.dumps(list(df.dtypes.astype(str))).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def _check_manifest_id(manifest_id: Any) -> str:
    """Validate a manifest identifier as untrusted, path-safe input.

    Rejects empty ids, path separators, parent-directory segments, absolute
    paths, drive/UNC prefixes and mixed separators. Returns the id when safe.
    """
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValidationError("Invalid manifest identifier.")
    if manifest_id in _RESERVED_IDS:
        raise ValidationError("Invalid manifest identifier.")
    if "/" in manifest_id or "\\" in manifest_id or "\x00" in manifest_id:
        raise ValidationError("Invalid manifest identifier: path separators are not allowed.")
    if not _ID_PATTERN.match(manifest_id):
        raise ValidationError(
            "Invalid manifest identifier: only letters, digits, dots, "
            "underscores and hyphens are allowed."
        )
    return manifest_id


def _parse_time_column(series: pd.Series) -> pd.Series:
    """Coerce a time column into comparable values without silencing problems.

    Numeric columns are kept as-is (their order is the chronology).
    Datetime-like strings are parsed. Missing or unparseable values surface
    as NaT/NaN so callers can reject them explicitly.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="coerce")
    except (ValueError, TypeError):
        return series


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
    """JSON-file-backed store for split manifests.

    All resolved paths are confined to the configured splits directory;
    manifest identifiers are treated as untrusted input (FD-004).
    """

    def __init__(self, config: DevRayConfig) -> None:
        self.config = config
        self.dir = config.artifacts_dir / SPLITS_SUBDIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._base = self.dir.resolve()

    def _path(self, manifest_id: str) -> Path:
        """Resolve a manifest path and confirm it stays inside the store."""
        manifest_id = _check_manifest_id(manifest_id)
        path = (self._base / f"{manifest_id}.json").resolve()
        try:
            path.relative_to(self._base)
        except ValueError:
            raise ValidationError(
                "Manifest path escapes the configured artifacts directory."
            ) from None
        return path

    def save(self, manifest: SplitManifest) -> None:
        with open(self._path(manifest.manifest_id), "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def load(self, manifest_id: str) -> SplitManifest | None:
        path = self._path(manifest_id)
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        manifest = SplitManifest(**data)
        # Never trust an identifier embedded in the file: it must match the
        # validated identifier used to locate the file (FD-004).
        if manifest.manifest_id != manifest_id:
            raise ValidationError("Manifest identifier mismatch.")
        return manifest

    def list_all(self) -> list[SplitManifest]:
        manifests: list[SplitManifest] = []
        for path in sorted(self._base.glob("*.json")):
            try:
                manifest = SplitManifest(**json.loads(path.read_text(encoding="utf-8")))
                if manifest.manifest_id == path.stem:
                    manifests.append(manifest)
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

    # QA-004: accept canonical names with backward-compatible aliases.
    # Advertised: time_series, grouped. Handler canonical: time, group.
    _SPLIT_TYPE_ALIASES = {
        "time_series": "time",
        "grouped": "group",
    }
    split_type = _SPLIT_TYPE_ALIASES.get(split_type, split_type)

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
        # Time-ordered: sort by actual timestamp values (stable: ties keep
        # their original row order), then split sequentially so the earliest
        # observations are training and the latest are the protected holdout.
        # The stored indices preserve the mapping from timestamp-sorted
        # observations back to their original dataset row positions.
        if time_column not in df.columns:
            raise ValidationError(f"Column '{time_column}' not found in dataset.")
        times = _parse_time_column(df[time_column])
        if times.isna().any():
            bad = [int(i) for i in times.index[times.isna()].tolist()]
            raise ValidationError(
                f"Time column '{time_column}' contains missing or unparseable "
                f"values at row index(es): {bad[:10]}"
            )
        ordered = [int(i) for i in times.argsort(kind="mergesort").tolist()]
        n_test = max(1, int(math.ceil(n * test_size)))
        n_val = max(0, int(math.ceil(n * val_size)))
        test_indices = ordered[-n_test:]
        val_indices = ordered[-n_test - n_val : -n_test] if n_val > 0 else []
        train_indices = ordered[: -n_test - n_val] if n_val > 0 else ordered[:-n_test]
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
    elif split_type == "group":
        # Group-aware splitting always requires a valid group column. A random
        # split is never silently substituted while labeled as grouped.
        if group_column is None:
            raise ValidationError(
                "split_type='group' requires a valid 'group_column'. "
                "A random split is never silently labeled as grouped."
            )
        if group_column not in df.columns:
            raise ValidationError(f"Column '{group_column}' not found in dataset.")
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
        # Preserve genuine chronological ordering for timestamp-based splits
        # (train earliest, validation middle, test latest). Do not re-sort into
        # original row-index order, which would destroy timestamp chronology and
        # leak future observations into training folds downstream. Other split
        # types keep their numerically sorted index lists for backward compat.
        train_indices=(
            train_indices
            if (split_type == "time" and time_column is not None)
            else sorted(train_indices)
        ),
        val_indices=(
            val_indices
            if (split_type == "time" and time_column is not None)
            else sorted(val_indices)
        ),
        test_indices=(
            test_indices
            if (split_type == "time" and time_column is not None)
            else sorted(test_indices)
        ),
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


def _validate_manifest_core(
    config: DevRayConfig,
    manifest_id: str,
    df: pd.DataFrame,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Authoritative manifest validation contract.

    Used identically by the public validator and every consuming pathway so
    public validation and consumers can never disagree (FD-003).
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

    # Strategy metadata.
    if manifest.split_type not in _ALLOWED_SPLIT_TYPES:
        errors.append(f"Unknown split type '{manifest.split_type}'.")

    n = len(df)
    all_idx = (
        list(manifest.train_indices) + list(manifest.val_indices) + list(manifest.test_indices)
    )

    # Index types: integers only, never booleans.
    non_int = [i for i in all_idx if not isinstance(i, int) or isinstance(i, bool)]
    if non_int:
        errors.append(f"Non-integer index(es): {non_int[:10]}")

    # Index bounds.
    out_of_range = [i for i in all_idx if i < 0 or i >= n]
    if out_of_range:
        errors.append(f"Index(es) out of range for dataset with {n} rows: {out_of_range[:10]}")

    train_set = set(manifest.train_indices)
    val_set = set(manifest.val_indices)
    test_set = set(manifest.test_indices)

    # Duplicate indices within each partition.
    for part_name, indices in (
        ("train", manifest.train_indices),
        ("val", manifest.val_indices),
        ("test", manifest.test_indices),
    ):
        if len(indices) != len(set(indices)):
            errors.append(f"Duplicate indices in {part_name} split.")

    # Overlapping partitions.
    if train_set & val_set:
        errors.append("Train and validation sets overlap.")
    if train_set & test_set:
        errors.append("Train and test sets overlap (protected holdout violated).")
    if val_set & test_set:
        errors.append("Validation and test sets overlap (protected holdout violated).")

    # Non-empty partitions.
    if not train_set:
        errors.append("Train set is empty.")
    if not test_set:
        errors.append("Test set is empty.")

    # Coverage: every dataset row must appear in exactly one partition.
    covered = train_set | val_set | test_set
    expected = set(range(n))
    if covered != expected:
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        msg_parts: list[str] = []
        if missing:
            msg_parts.append(f"missing row index(es): {missing[:10]}")
        if extra:
            msg_parts.append(f"unexpected row index(es): {extra[:10]}")
        errors.append("Incomplete partition coverage: " + "; ".join(msg_parts))

    # Declared partition counts match actual counts.
    if manifest.n_train != len(train_set):
        errors.append(
            f"Declared n_train={manifest.n_train} does not match actual "
            f"train count {len(train_set)}."
        )
    if manifest.n_val != len(val_set):
        errors.append(
            f"Declared n_val={manifest.n_val} does not match actual val count {len(val_set)}."
        )
    if manifest.n_test != len(test_set):
        errors.append(
            f"Declared n_test={manifest.n_test} does not match actual test count {len(test_set)}."
        )

    # Chronological ordering for time-based splits: train before val before
    # test. With a time column, chronology is judged on actual timestamp
    # values (FD-002), not on original row indices. Ties at a boundary are
    # ordered deterministically by original row index at creation time, so a
    # non-decreasing boundary is the documented tie policy.
    if manifest.split_type == "time":
        time_column = manifest.time_column
        if time_column and time_column in df.columns:
            times = _parse_time_column(df[time_column])
            na_rows = [int(i) for i in all_idx if times.isna().iloc[i]]
            if na_rows:
                errors.append(
                    "Unparseable or missing timestamp(s) at row index(es): "
                    f"{na_rows[:10]}; chronological correctness cannot be guaranteed."
                )
            else:

                def _ts_max(idxs: set[int]):
                    return max(times.iloc[list(idxs)])

                def _ts_min(idxs: set[int]):
                    return min(times.iloc[list(idxs)])

                if train_set and val_set and _ts_max(train_set) > _ts_min(val_set):
                    errors.append(
                        "Chronological split violated: train before validation timestamps."
                    )
                if train_set and test_set and _ts_max(train_set) > _ts_min(test_set):
                    errors.append("Chronological split violated: train before test timestamps.")
                if val_set and test_set and _ts_max(val_set) > _ts_min(test_set):
                    errors.append(
                        "Chronological split violated: validation before test timestamps."
                    )
        elif time_column:
            errors.append(
                f"Time column '{time_column}' not found in dataset; "
                "chronological correctness cannot be guaranteed."
            )
        else:
            # row_order-only time split: chronology is original row order.
            ti = list(manifest.train_indices)
            vi = list(manifest.val_indices)
            tei = list(manifest.test_indices)
            if train_set and val_set and max(ti) >= min(vi):
                errors.append("Chronological split violated: train precedes validation indices.")
            if train_set and test_set and max(ti) >= min(tei):
                errors.append("Chronological split violated: train precedes test indices.")
            if val_set and test_set and max(vi) >= min(tei):
                errors.append("Chronological split violated: validation precedes test indices.")

    # Group separation for group-based splits: validate actual group
    # membership, not just metadata labels. A group manifest requires a valid
    # group column present in the dataset.
    if manifest.split_type == "group":
        if not manifest.group_column:
            errors.append("Grouped split requires a group column.")
        elif manifest.group_column not in df.columns:
            errors.append(f"Group column '{manifest.group_column}' not found in dataset.")
        else:
            groups = df[manifest.group_column]
            train_groups = {groups.iloc[i] for i in train_set} if train_set else set()
            val_groups = {groups.iloc[i] for i in val_set} if val_set else set()
            test_groups = {groups.iloc[i] for i in test_set} if test_set else set()
            if train_groups & test_groups:
                errors.append("Train and test groups overlap (protected holdout violated).")
            if train_groups & val_groups:
                errors.append("Train and validation groups overlap.")
            if val_groups & test_groups:
                errors.append("Validation and test groups overlap.")

    return {
        "manifest_id": manifest_id,
        "valid": len(errors) == 0,
        "errors": errors,
        "dataset_fingerprint_match": manifest.dataset_fingerprint == current_fp,
        "n_train": manifest.n_train,
        "n_val": manifest.n_val,
        "n_test": manifest.n_test,
        "split_type": manifest.split_type,
        "protected_holdout": manifest.protected_holdout,
    }


def validate_split_manifest(
    config: DevRayConfig,
    manifest_id: str,
    df: pd.DataFrame,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Validate a stored split manifest against a dataset.

    Public entry point to the authoritative manifest validation contract
    (see _validate_manifest_core).
    """
    return _validate_manifest_core(config, manifest_id, df, dataset_name)


def validate_manifest_for_consumption(
    config: DevRayConfig,
    manifest_id: str,
    df: pd.DataFrame,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Run complete structural validation before any consumer accesses rows.

    Shares the authoritative contract with validate_split_manifest so public
    validation and all consumers always agree (FD-003).
    """
    return _validate_manifest_core(config, manifest_id, df, dataset_name)


def development_rows_from_manifest(
    config: DevRayConfig, df: pd.DataFrame, manifest_id: str
) -> tuple[pd.DataFrame, SplitManifest]:
    """Load and validate a manifest, returning df restricted to train+val rows.

    The protected test holdout is excluded from the returned dataframe. The
    dataset fingerprint is validated against the supplied dataframe so a
    changed dataset cannot silently reuse stale indices. Development rows are
    returned in manifest order, which for time-based splits is timestamp
    order so forward-chaining CV consumes the chronology correctly.
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

    # For timestamp-based splits, development rows must be consumed in genuine
    # timestamp order regardless of how the manifest's index lists were stored
    # (older manifests may hold numerically sorted indices). This keeps forward
    # chaining chronological and protects the holdout from leakage.
    if (
        manifest.split_type == "time"
        and manifest.time_column
        and manifest.time_column in df.columns
    ):
        times = _parse_time_column(df[manifest.time_column])
        # Stable sort preserves manifest order among equal timestamps.
        cv_indices = sorted(cv_indices, key=lambda i: times.iloc[i])

    return df.iloc[cv_indices].reset_index(drop=True), manifest


def list_split_manifests(config: DevRayConfig) -> list[SplitManifest]:
    """List all stored split manifests, newest first."""
    store = SplitStore(config)
    return store.list_all()
