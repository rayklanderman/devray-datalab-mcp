"""Tests for system info and resource handling."""

from __future__ import annotations

import asyncio
import json

from devray_datalab.tools.system import system_info_tool


def run(coro):
    return asyncio.run(coro)


def _payload(result):
    return json.loads(result[0].text)


def test_system_info(config):
    result = run(system_info_tool(config, {}))
    data = _payload(result)
    assert data["ok"] is True
    body = data["data"]
    assert body["operating_system"]
    assert body["cpu_cores_logical"] >= 1
    assert body["ram_total_gb"] > 0
    assert body["storage_total_gb"] > 0
    assert "gpu" in body
    assert body["sklearn_uses_gpu_by_default"] is False
    assert body["training_backend"] == "CPU (scikit-learn)"
    assert "config" in body
