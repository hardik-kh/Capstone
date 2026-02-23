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
    """Attempts to convert columns containing 'date' to datetime."""
    df = df.copy()
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing numeric with median and categorical with mode/unknown."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            mode = df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill_value)
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate rows from the DataFrame."""
    return df.drop_duplicates()


def _clip_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Clips numeric outliers using the IQR rule."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df


def clean_and_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleans a DataFrame and returns it with profiling metadata."""
    original_df = df.copy()

    df = _normalize_columns(df)
    df = _coerce_dates(df)
    df = _handle_missing(df)
    df = _remove_duplicates(df)
    df = _clip_outliers_iqr(df)

    numeric_summary = {}
    if not df.select_dtypes(include=["float", "int"]).empty:
        numeric_summary = df.select_dtypes(include=["float", "int"]).describe().to_dict()

    categorical_summary = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        categorical_summary[col] = (
            df[col].value_counts().head(MAX_CATEGORY_VALUES).to_dict()
        )

    profiling = {
        "shape_before": list(original_df.shape),
        "shape_after": list(df.shape),
        "missing_values_before": original_df.isna().sum().to_dict(),
        "missing_values_after": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "preview": df.head(MAX_PREVIEW_ROWS).to_dict(orient="records"),
    }

    return df, profiling
