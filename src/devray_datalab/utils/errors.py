"""Custom exceptions for DevRay DataLab MCP."""


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
