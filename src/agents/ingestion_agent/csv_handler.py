# CSV loading with encoding and delimiter detection

import csv
import pandas as pd

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
