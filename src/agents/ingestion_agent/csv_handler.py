import csv
import pandas as pd
from src.core.config import CSV_DELIMITERS, ENCODING_FALLBACKS
from src.core.exceptions import FileReadError


def detect_encoding(file_path):
    for enc in ENCODING_FALLBACKS:
        try:
            with open(file_path, encoding=enc) as f:
                f.read(2048)
            return enc
        except Exception:
            continue
    return "latin-1"


def detect_delimiter(file_path, encoding):
    with open(file_path, encoding=encoding) as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS).delimiter
    except Exception:
        return ","


def load_csv(file_path):
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)

    try:
        df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)
    except Exception as e:
        raise FileReadError(str(e))

    return df, {"encoding": encoding, "delimiter": delimiter}
