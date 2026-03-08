# CSV loading with encoding and delimiter detection

import csv
import pandas as pd
import os
from datetime import datetime
from src.core.config import CSV_DELIMITERS, ENCODING_FALLBACKS
from src.core.exceptions import FileReadError


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

#ADDED BY PREKSHA

BRONZE_PATH = "data/bronze"


def save_to_bronze(df: pd.DataFrame, original_file: str) -> dict:
    """Saves raw dataframe to Bronze layer as parquet."""
    os.makedirs(BRONZE_PATH, exist_ok=True)

    file_name = os.path.basename(original_file).replace(".csv", "")
    output_path = os.path.join(
        BRONZE_PATH,
        f"{file_name}_bronze.parquet"
    )

    df.to_parquet(output_path, index=False)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "output_path": output_path,
        "timestamp": datetime.now()
    }