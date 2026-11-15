"""Hardware and resource detection.

Reports CPU, memory, storage and GPU availability using only psutil and the
standard library. No PyTorch, TensorFlow, or CUDA dependencies are required.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field

import psutil

from devray_datalab.utils.errors import HardwareError


@dataclass
class GpuInfo:
    """Detected GPU information."""

    detected: bool = False
    detail: str = "No supported GPU detected"
    backend_available: bool = False


@dataclass
class SystemInfo:
    operating_system: str
    os_version: str
    processor: str
    cpu_cores_logical: int
    cpu_cores_physical: int
    cpu_percent: float
    ram_total_gb: float
    ram_available_gb: float
    ram_percent: float
    storage_total_gb: float
    storage_free_gb: float
    storage_used_percent: float
    gpu: GpuInfo = field(default_factory=GpuInfo)
    sklearn_cpu_backend: bool = True


def _detect_gpu() -> GpuInfo:
    """Best-effort GPU detection without GPU dependencies.

    Checks common GPU indicators exposed by psutil/the OS and reports whether
    a GPU-accelerated sklearn backend appears to be installed.
    """
    detail_parts: list[str] = []
    detected = False

    # NVIDIA GPU hint via nvidia-smi availability.
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        detected = True
        detail_parts.append("NVIDIA GPU (nvidia-smi available)")

    # Try to detect GPU via py3nvml if present (optional, no hard dep).
    try:
        import py3nvml  # type: ignore

        try:
            py3nvml.nvmlInit()
            count = py3nvml.nvmlDeviceGetCount()
            if count > 0:
                detected = True
                detail_parts.append(f"NVIDIA GPU(s) via NVML: {count}")
        except Exception:
            pass
    except ImportError:
        pass

    detail = "; ".join(detail_parts) if detail_parts else "No supported GPU detected"
    return GpuInfo(detected=detected, detail=detail)


def get_system_info() -> SystemInfo:
    """Gather system hardware information."""
    try:
        vm = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk = shutil.disk_usage(os.getcwd())
        gpu = _detect_gpu()

        return SystemInfo(
            operating_system=platform.system(),
            os_version=platform.version(),
            processor=platform.processor() or platform.machine(),
            cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
            cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
            cpu_percent=cpu_percent,
            ram_total_gb=round(vm.total / (1024**3), 2),
            ram_available_gb=round(vm.available / (1024**3), 2),
            ram_percent=round(vm.percent, 1),
            storage_total_gb=round(disk.total / (1024**3), 2),
            storage_free_gb=round(disk.free / (1024**3), 2),
            storage_used_percent=round(disk.used / disk.total * 100, 1),
            gpu=gpu,
        )
    except Exception as exc:  # noqa: BLE001
        raise HardwareError(f"Failed to gather system info: {exc}") from exc
