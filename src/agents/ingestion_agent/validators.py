# Validation utilities for uploaded files
import os

import pandas as pd
from fastapi import UploadFile

from src.core.config import SUPPORTED_EXTENSIONS
from src.core.exceptions import UnsupportedFormatError


def validate_file(file: UploadFile) -> str:
    """Validates file extension and returns it if valid."""
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(ext, SUPPORTED_EXTENSIONS)

    return ext


def validate_dataframe_rows(df: pd.DataFrame) -> dict:
    """Performs generalized row-level validation and returns validation metrics.

    Rules are domain-agnostic and safe for any business dataset:
    - Completely null rows are always invalid
    - Duplicate rows are flagged (counted, not removed here)
    - Fully null columns are identified
    Negative values are intentionally NOT flagged — they are valid in financial,
    temperature, coordinate, and many other real-world datasets.
    """
    total_rows = len(df)

    if total_rows == 0:
        return {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "duplicate_rows": 0,
            "fully_null_columns": [],
            "validation_coverage_percent": 0.0,
        }

    # Rule: No completely null rows
    valid_mask = ~df.isnull().all(axis=1)

    valid_rows = int(valid_mask.sum())
    invalid_rows = total_rows - valid_rows
    duplicate_rows = int(df.duplicated().sum())
    fully_null_columns = [col for col in df.columns if df[col].isnull().all()]
    validation_coverage = (valid_rows / total_rows) * 100

    return {
        "total_rows": int(total_rows),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicate_rows,
        "fully_null_columns": fully_null_columns,
        "validation_coverage_percent": round(validation_coverage, 2),
    }