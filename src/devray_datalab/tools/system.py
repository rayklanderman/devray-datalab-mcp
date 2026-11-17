"""System tool: system_info."""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from devray_datalab.config import DevRayConfig
from devray_datalab.core.hardware import get_system_info
from devray_datalab.utils.errors import DevRayError, sanitize_for_json


def _json(data: Any) -> str:
    import json

    return json.dumps(sanitize_for_json(data), default=str, allow_nan=False)


def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=_json({"ok": True, "data": data}))]


def _err(message: str, code: str = "error") -> list[TextContent]:

    return [
        TextContent(
            type="text", text=_json({"ok": False, "error": {"message": message, "code": code}})
        )
    ]


async def system_info_tool(config: DevRayConfig, arguments: dict) -> list[TextContent]:
    """Report local hardware and resource information."""
    try:
        info = get_system_info()
    except DevRayError as exc:
        return _err(exc.message, exc.code)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to gather system info: {exc}", "hardware_error")

    gpu = info.gpu
    # Determine whether a GPU training backend is installed. scikit-learn itself
    # does not use GPUs, so report availability of optional GPU libs only.
    gpu_backend_installed = False
    try:
        import py3nvml  # noqa: F401

        gpu_backend_installed = True
    except ImportError:
        pass

    return _ok(
        {
            "operating_system": info.operating_system,
            "os_version": info.os_version,
            "processor": info.processor,
            "cpu_cores_logical": info.cpu_cores_logical,
            "cpu_cores_physical": info.cpu_cores_physical,
            "cpu_percent": info.cpu_percent,
            "ram_total_gb": info.ram_total_gb,
            "ram_available_gb": info.ram_available_gb,
            "ram_percent": info.ram_percent,
            "storage_total_gb": info.storage_total_gb,
            "storage_free_gb": info.storage_free_gb,
            "storage_used_percent": info.storage_used_percent,
            "gpu": {
                "detected": gpu.detected,
                "detail": gpu.detail,
                "backend_available": gpu_backend_installed,
            },
            "sklearn_uses_gpu_by_default": False,
            "training_backend": "CPU (scikit-learn)",
            "config": {
                "workspace_dir": str(config.workspace_dir),
                "artifacts_dir": str(config.artifacts_dir),
                "max_cpu_threads": config.max_cpu_threads,
                "training_memory_budget_mb": config.training_memory_budget_mb,
                "max_concurrent_jobs": config.max_concurrent_jobs,
                "training_timeout_seconds": config.training_timeout_seconds,
            },
        }
    )
