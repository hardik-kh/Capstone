# CSV loading with encoding and delimiter detection

import csv
import os
import shutil
from datetime import datetime
import pandas as pd

from core.config import (
    BRONZE_OUTPUT_DIR,
    CSV_DELIMITERS,
    ENCODING_FALLBACKS,
    LARGE_CSV_SAMPLE_ROWS,
)
from core.exceptions import FileReadError


def detect_encoding(file_path: str) -> str:
    """Detects text encoding by trying a list of common encodings."""
    for enc in ENCODING_FALLBACKS:
        try:
            with open(file_path, encoding=enc) as f:
                f.read(2048)
            return enc
        except Exception:
            continue
    return "latin-1"


def detect_delimiter(file_path: str, encoding: str) -> str:
    """Detects CSV delimiter using csv.Sniffer with a set of candidates."""
    try:
        with open(file_path, encoding=encoding) as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS)
        return dialect.delimiter
    except Exception:
        return ","


def load_csv(file_path: str) -> tuple[pd.DataFrame, dict]:
    """Loads a CSV file into a DataFrame with detected encoding and delimiter."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)

    try:
        df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)
    except Exception as e:
        raise FileReadError(file_path, str(e))

    return df, {"encoding": encoding, "delimiter": delimiter}


def load_csv_sample(file_path: str, sample_rows: int = LARGE_CSV_SAMPLE_ROWS) -> tuple[pd.DataFrame, dict]:
    """Loads a bounded CSV sample for profiling large files safely."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)

    try:
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            nrows=sample_rows,
            low_memory=False,
        )
    except Exception as e:
        raise FileReadError(file_path, str(e))

    return df, {"encoding": encoding, "delimiter": delimiter, "sample_rows_loaded": len(df)}


def copy_csv_to_bronze(file_path: str, original_file: str) -> dict:
    """Copies a CSV into the bronze layer without materializing it in memory."""
    os.makedirs(BRONZE_OUTPUT_DIR, exist_ok=True)

    file_name = os.path.basename(original_file).replace(".csv", "")
    csv_path = os.path.join(BRONZE_OUTPUT_DIR, f"{file_name}_bronze.csv")
    shutil.copyfile(file_path, csv_path)

    return {
        "output_path": csv_path,
        "storage_format": "csv",
        "warning": "Large CSV was streamed directly to bronze as CSV to avoid excessive memory usage.",
        "timestamp": datetime.now().isoformat(),
    }


def save_to_bronze(df: pd.DataFrame, original_file: str) -> dict:
    """Saves raw dataframe to Bronze layer, preferring parquet with CSV fallback."""
    os.makedirs(BRONZE_OUTPUT_DIR, exist_ok=True)

    file_name = os.path.basename(original_file).replace(".csv", "")
    parquet_path = os.path.join(BRONZE_OUTPUT_DIR, f"{file_name}_bronze.parquet")
    csv_path = os.path.join(BRONZE_OUTPUT_DIR, f"{file_name}_bronze.csv")

    storage_format = "parquet"
    warning = None

    try:
        df.to_parquet(parquet_path, index=False)
        output_path = parquet_path
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        # Parquet requires an optional engine such as pyarrow or fastparquet.
        df.to_csv(csv_path, index=False)
        output_path = csv_path
        storage_format = "csv"
        warning = (
            "Parquet engine unavailable; saved bronze output as CSV instead. "
            f"Details: {exc}"
        )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "output_path": output_path,
        "storage_format": storage_format,
        "warning": warning,
        "timestamp": datetime.now().isoformat(),
    }
