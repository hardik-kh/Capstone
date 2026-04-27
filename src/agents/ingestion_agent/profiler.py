# Data cleaning + profiling utilities

import numpy as np
import pandas as pd

from src.core.config import MAX_PREVIEW_ROWS, MAX_CATEGORY_VALUES


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names to snake_case and strips whitespace."""
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempts to convert columns that look like dates to datetime."""
    df = df.copy()
    date_tokens = ["date", "time", "timestamp", "created", "updated", "at"]
    for col in df.columns:
        col_lower = col.lower()
        if any(token in col_lower for token in date_tokens):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values using context-aware rules.

    - ID-like columns (high uniqueness, integer): left as-is — filling would fabricate keys
    - Numeric columns with all non-negative values: fill with median
    - Numeric columns with negative values (financial, temp, etc): fill with median
    - Columns that are entirely null: left as-is
    - Categorical/object columns: fill with mode if available, else 'unknown'
    """
    df = df.copy()
    total_rows = len(df)

    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        if null_count == total_rows:
            # Entirely null column — don't fabricate values
            continue

        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            # Skip ID-like columns: integer, high uniqueness, no negatives
            non_null = df[col].dropna()
            uniqueness = non_null.nunique() / len(non_null) if len(non_null) > 0 else 0
            is_id_like = (
                uniqueness > 0.95
                and (non_null % 1 == 0).all()
                and non_null.min() >= 0
            )
            if is_id_like:
                continue
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            mode = df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill_value)

    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes exact duplicate rows from the DataFrame."""
    return df.drop_duplicates()


def _flag_outliers_iqr(df: pd.DataFrame) -> dict:
    """Detects outliers using IQR rule and returns a report — does NOT modify data.

    Outlier clipping is intentionally removed. A sales spike, a high-value transaction,
    or an extreme temperature reading is real signal, not noise. We report outliers
    so downstream agents can decide what to do.
    """
    report = {}
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        if outlier_count > 0:
            report[col] = {
                "outlier_count": outlier_count,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }
    return report


def clean_and_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleans a DataFrame and returns it with profiling metadata."""
    original_df = df.copy()

    df = _normalize_columns(df)
    df = _coerce_dates(df)
    df = _handle_missing(df)
    df = _remove_duplicates(df)

    # Outlier report only — data is NOT modified
    outlier_report = _flag_outliers_iqr(df)

    numeric_summary = {}
    if not df.select_dtypes(include=["float", "int"]).empty:
        numeric_summary = df.select_dtypes(include=["float", "int"]).describe().to_dict()

    categorical_summary = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        categorical_summary[col] = (
            df[col].value_counts().head(MAX_CATEGORY_VALUES).to_dict()
        )

    # Serialize datetime columns to ISO strings for JSON safety
    preview_df = df.head(MAX_PREVIEW_ROWS).copy()
    for col in preview_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        preview_df[col] = preview_df[col].astype(str)

    profiling = {
        "shape_before": list(original_df.shape),
        "shape_after": list(df.shape),
        "missing_values_before": original_df.isna().sum().to_dict(),
        "missing_values_after": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "outlier_report": outlier_report,
        "preview": preview_df.to_dict(orient="records"),
    }

    return df, profiling