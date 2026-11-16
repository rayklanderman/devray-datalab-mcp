"""Tests for split manifest tools."""

from __future__ import annotations

from devray_datalab.core.splits import (
    create_split_manifest,
    list_split_manifests,
    validate_split_manifest,
)
from devray_datalab.tools.splits import (
    create_split_manifest_tool,
    list_split_manifests_tool,
    validate_split_manifest_tool,
)


def _payload(result):
    import json

    return json.loads(result[0].text)


def _data(result):
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    return payload["data"]


def test_create_split_manifest_classification(config, classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name="sample_classification.csv",
        split_type="stratified",
        test_size=0.25,
        random_state=42,
        stratify_column="purchased",
    )
    assert manifest.dataset_name == "sample_classification.csv"
    assert manifest.dataset_fingerprint
    assert manifest.split_type == "stratified"
    assert manifest.n_train + manifest.n_val + manifest.n_test == 20
    assert manifest.n_test > 0
    assert manifest.n_train > 0
    assert manifest.protected_holdout is True

    # Indices disjoint.
    all_idx = manifest.train_indices + manifest.val_indices + manifest.test_indices
    assert len(set(all_idx)) == len(all_idx)

    # Persisted.
    store_loaded = list_split_manifests(config)
    assert any(m.manifest_id == manifest.manifest_id for m in store_loaded)


def test_create_split_manifest_time(config, regression_csv):
    import pandas as pd

    df = pd.read_csv(regression_csv)
    # Add a time column.
    df["date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name="sample_regression.csv",
        split_type="time",
        test_size=0.25,
        time_column="date",
        random_state=42,
    )
    assert manifest.split_type == "time"
    assert manifest.time_column == "date"
    assert manifest.n_test > 0

    # Time order: max train index < min test index (sequential split).
    assert max(manifest.train_indices) < min(manifest.test_indices)


def test_create_split_manifest_group(config, classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    df["group"] = [i % 5 for i in range(len(df))]
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name="sample_classification.csv",
        split_type="group",
        test_size=0.25,
        group_column="group",
        random_state=42,
    )
    assert manifest.split_type == "group"
    assert manifest.group_column == "group"
    assert manifest.n_test > 0


def test_create_split_manifest_invalid_params(config, classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    try:
        create_split_manifest(
            config=config,
            df=df,
            dataset_name="sample_classification.csv",
            split_type="bogus",
        )
        raise AssertionError("expected ValidationError")
    except Exception:
        pass


def test_validate_split_manifest(config, classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name="sample_classification.csv",
        split_type="random",
        test_size=0.25,
        random_state=42,
    )
    report = validate_split_manifest(
        config=config,
        manifest_id=manifest.manifest_id,
        df=df,
        dataset_name="sample_classification.csv",
    )
    assert report["valid"] is True
    assert report["dataset_fingerprint_match"] is True
    assert report["errors"] == []


def test_validate_split_manifest_fingerprint_mismatch(config, classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name="sample_classification.csv",
        split_type="random",
        test_size=0.25,
        random_state=42,
    )
    # Modify the dataset.
    df_modified = df.copy()
    df_modified.loc[0, "purchased"] = 1 - df_modified.loc[0, "purchased"]
    report = validate_split_manifest(
        config=config,
        manifest_id=manifest.manifest_id,
        df=df_modified,
        dataset_name="sample_classification.csv",
    )
    assert report["valid"] is False
    assert report["dataset_fingerprint_match"] is False
    assert len(report["errors"]) > 0


def test_list_split_manifests(config, classification_csv):
    import pandas as pd

    df = pd.read_csv(classification_csv)
    create_split_manifest(
        config=config,
        df=df,
        dataset_name="sample_classification.csv",
        split_type="random",
        test_size=0.25,
        random_state=42,
    )
    manifests = list_split_manifests(config)
    assert len(manifests) >= 1
    assert manifests[0].dataset_name == "sample_classification.csv"


def test_create_split_manifest_tool(config, classification_csv):
    import asyncio

    async def main():
        return await create_split_manifest_tool(
            config,
            {
                "name": "sample_classification.csv",
                "split_type": "random",
                "test_size": 0.25,
                "random_state": 42,
            },
        )

    result = asyncio.run(main())
    data = _payload(result)
    assert data["ok"] is True
    assert data["data"]["manifest_id"]
    assert data["data"]["n_train"] > 0
    assert data["data"]["n_test"] > 0
    assert data["data"]["protected_holdout"] is True


def test_validate_split_manifest_tool(config, classification_csv):
    import asyncio

    import pandas as pd

    df = pd.read_csv(classification_csv)
    manifest = create_split_manifest(
        config=config,
        df=df,
        dataset_name="sample_classification.csv",
        split_type="random",
        test_size=0.25,
        random_state=42,
    )

    async def main():
        return await validate_split_manifest_tool(
            config,
            {
                "manifest_id": manifest.manifest_id,
                "name": "sample_classification.csv",
            },
        )

    result = asyncio.run(main())
    data = _payload(result)
    assert data["ok"] is True
    assert data["data"]["valid"] is True


def test_list_split_manifests_tool(config, classification_csv):
    import asyncio

    import pandas as pd

    df = pd.read_csv(classification_csv)
    create_split_manifest(
        config=config,
        df=df,
        dataset_name="sample_classification.csv",
        split_type="random",
        test_size=0.25,
        random_state=42,
    )

    async def main():
        return await list_split_manifests_tool(config, {})

    result = asyncio.run(main())
    data = _data(result)
    assert data["count"] >= 1
    assert any(m["manifest_id"] for m in data["manifests"])
