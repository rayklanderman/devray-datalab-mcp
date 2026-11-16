"""Tests for the non-blocking training job system."""

from __future__ import annotations

import asyncio

from devray_datalab.tools.jobs import (
    cancel_training_job_tool,
    get_training_status_tool,
    list_training_jobs_tool,
    start_training_job_tool,
)


def _payload(result):
    import json

    return json.loads(result[0].text)


def _data(result):
    payload = _payload(result)
    assert payload["ok"] is True, payload.get("error")
    return payload["data"]


def test_job_submit_get_cancel_list(config, classification_csv):
    async def main():
        # Submit a job.
        start = await start_training_job_tool(
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
        assert start_data["status"] == "queued"
        job_id = start_data["job_id"]

        # Immediately poll status - should complete quickly.
        status = None
        for _ in range(100):
            st = await get_training_status_tool(config, {"job_id": job_id})
            st_data = _data(st)
            status = st_data["status"]
            if status in ("completed", "failed"):
                break
            await asyncio.sleep(0.1)

        assert status == "completed", f"unexpected status {status}"

        st_data = _data(await get_training_status_tool(config, {"job_id": job_id}))
        assert "result" in st_data
        assert st_data["result"]["metrics"]["accuracy"] >= 0

        # List jobs should include ours.
        lst = await list_training_jobs_tool(config, {"limit": 10})
        lst_data = _data(lst)
        job_ids = [j["job_id"] for j in lst_data["jobs"]]
        assert job_id in job_ids

        return job_id

    asyncio.run(main())

    # Cancel a fresh job quickly (queued).
    async def cancel_test():
        start2 = await start_training_job_tool(
            config,
            {
                "name": "sample_classification.csv",
                "target_column": "purchased",
                "task_type": "classification",
                "model_name": "logistic_regression",
                "random_state": 42,
            },
        )
        jid = _data(start2)["job_id"]
        cancel = await cancel_training_job_tool(config, {"job_id": jid})
        cd = _data(cancel)
        assert "cancelled" in cd

        # Status should be cancelled (if job was queued) or completed (tiny dataset trains fast).
        st = await get_training_status_tool(config, {"job_id": jid})
        sd = _data(st)
        assert sd["status"] in ("cancelled", "completed")
        if cd["cancelled"]:
            assert sd["status"] == "cancelled"

    asyncio.run(cancel_test())

    # Get nonexistent job.
    async def nonexistent_test():
        nr = await get_training_status_tool(config, {"job_id": "does-not-exist"})
        nd = _payload(nr)
        assert nd["ok"] is False
        assert nd["error"]["code"] == "not_found"

    asyncio.run(nonexistent_test())
