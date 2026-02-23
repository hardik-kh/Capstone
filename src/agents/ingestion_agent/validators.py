# Validation utilities for uploaded files

import os
from fastapi import UploadFile

from src.core.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from src.core.exceptions import UnsupportedFormatError, FileTooLargeError


def validate_file(file: UploadFile) -> str:
    """Validates file extension and size, returns extension if valid."""
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(ext, SUPPORTED_EXTENSIONS)

    # UploadFile may not always have .size; guard with getattr
    size = getattr(file, "size", None)
    if size is not None and size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise FileTooLargeError(
            filename=file.filename,
            size=size,
            max_size=MAX_FILE_SIZE_MB * 1024 * 1024,
        )

    return ext
