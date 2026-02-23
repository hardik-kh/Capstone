# Domain-specific exceptions for ingestion with structured error info

class IngestionError(Exception):
    """Base class for all ingestion-related errors with structured payload."""

    def __init__(self, message: str, code: str = "INGESTION_ERROR", hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint

    def to_dict(self) -> dict:
        """Converts the exception into a JSON-serializable dict."""
        return {
            "error_code": self.code,
            "message": self.message,
            "hint": self.hint,
        }


class UnsupportedFormatError(IngestionError):
    """Raised when a file has an unsupported extension."""

    def __init__(self, ext: str, supported: set[str]):
        super().__init__(
            message=f"Unsupported file format: '{ext}'.",
            code="UNSUPPORTED_FORMAT",
            hint=f"Allowed formats are: {', '.join(sorted(supported))}",
        )


class FileTooLargeError(IngestionError):
    """Raised when a file exceeds the configured size limit."""

    def __init__(self, filename: str, size: int, max_size: int):
        super().__init__(
            message=f"File '{filename}' exceeds the maximum allowed size.",
            code="FILE_TOO_LARGE",
            hint=f"File size: {size} bytes. Max allowed: {max_size} bytes.",
        )


class FileReadError(IngestionError):
    """Raised when a file cannot be read into a DataFrame."""

    def __init__(self, filename: str, details: str):
        super().__init__(
            message=f"Failed to read file '{filename}'.",
            code="FILE_READ_ERROR",
            hint=f"Details: {details}",
        )


class ValidationError(IngestionError):
    """Raised when a validation rule fails."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            hint=hint,
        )
