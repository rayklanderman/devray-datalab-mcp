"""Custom exceptions for DevRay DataLab MCP."""

from __future__ import annotations

import math
from typing import Any


def sanitize_for_json(data: Any) -> Any:
    """Recursively convert non-finite numeric values (NaN, Infinity) to None.

    Ensures the output serializes as strict, standards-compliant JSON with no
    bare NaN or Infinity literals. Finite numbers, strings, booleans, None,
    lists, and dicts are preserved.
    """
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    if isinstance(data, tuple):
        return [sanitize_for_json(v) for v in data]
    return data


class DevRayError(Exception):
    """Base exception for all DevRay DataLab errors."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "devray_error"


class ValidationError(DevRayError):
    """Raised when input validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="validation_error")


class DatasetError(DevRayError):
    """Raised when a dataset operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="dataset_error")


class SecurityError(DevRayError):
    """Raised when a security policy is violated (e.g. path traversal)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="security_error")


class TrainingError(DevRayError):
    """Raised when model training fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="training_error")


class CancellationError(DevRayError):
    """Raised when a job is cooperatively cancelled and stops successfully."""

    def __init__(self, message: str = "Training cancelled.") -> None:
        super().__init__(message, code="cancelled")


class ModelError(DevRayError):
    """Raised when model export/load/prediction fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="model_error")


class ResourceError(DevRayError):
    """Raised when resource limits are exceeded."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="resource_error")


class HardwareError(DevRayError):
    """Raised when hardware detection fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="hardware_error")
