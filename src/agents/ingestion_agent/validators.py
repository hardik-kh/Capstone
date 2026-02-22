import os
from src.core.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from src.core.exceptions import UnsupportedFormatError, FileTooLargeError


def validate_file(file):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported format: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )

    if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise FileTooLargeError(
            f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"
        )

    return ext
