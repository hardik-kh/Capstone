# Validation utilities for uploaded files
import pandas as pd
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

#ADDED BY PREKSHA


def validate_dataframe_rows(df: pd.DataFrame) -> dict:
    """
    Performs basic row-level validation and returns validation metrics.
    """

    total_rows = len(df)

    if total_rows == 0:
        return {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "validation_coverage_percent": 0.0,
        }

    # Basic validation rules
    rules = []

    # Rule 1: No completely null rows
    rules.append(~df.isnull().all(axis=1))

    # Rule 2: No negative numeric values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        rules.append((df[numeric_cols] >= 0).all(axis=1))

    # Combine all rules
    valid_mask = pd.Series(True, index=df.index)
    for rule in rules:
        valid_mask &= rule

    valid_rows = valid_mask.sum()
    invalid_rows = total_rows - valid_rows

    validation_coverage = (valid_rows / total_rows) * 100

    return {
        "total_rows": int(total_rows),
        "valid_rows": int(valid_rows),
        "invalid_rows": int(invalid_rows),
        "validation_coverage_percent": round(validation_coverage, 2),
    }
#ENDED