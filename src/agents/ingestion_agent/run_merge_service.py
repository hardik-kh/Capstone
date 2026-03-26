import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.ingestion_agent.csv_handler import detect_delimiter, detect_encoding
from agents.ingestion_agent.merge_service import merge_csv_files


def _normalize_columns(columns: list[str]) -> list[str]:
    return [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in columns
    ]


def _read_column_names(file_path: Path) -> list[str]:
    encoding = detect_encoding(str(file_path))
    delimiter = detect_delimiter(str(file_path), encoding)
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=delimiter,
        nrows=0,
        low_memory=False,
    )
    return _normalize_columns(list(df.columns))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ingestion-agent merge service on two CSV files.")
    parser.add_argument("left_file", help="First CSV filename under the data directory")
    parser.add_argument("right_file", help="Second CSV filename under the data directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    left_path = PROJECT_ROOT / "data" / args.left_file
    right_path = PROJECT_ROOT / "data" / args.right_file

    if not left_path.exists():
        raise FileNotFoundError(f"Left CSV not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right CSV not found: {right_path}")

    left_columns = _read_column_names(left_path)
    right_columns = _read_column_names(right_path)

    print(f"Left file: {left_path.name}")
    print(f"Left columns: {left_columns}")
    print(f"Right file: {right_path.name}")
    print(f"Right columns: {right_columns}")

    _, merge_log = merge_csv_files(
        left_name=left_path.name,
        left_path=str(left_path),
        right_name=right_path.name,
        right_path=str(right_path),
    )

    merge_decision = merge_log["ollama_inference"]["decision"]
    print(f"Ollama raw decision: {json.dumps(merge_decision, indent=2, default=str)}")
    print(
        "Merge columns selected by Ollama: "
        f"{merge_decision.get('merge_columns') or [merge_decision['left_column']]}"
    )
    print(f"Merged file path: {merge_log['output_path']}")
    if merge_log["merge_summary"]["merged_rows"] == 0:
        print("Warning: merged file has zero rows. The selected merge keys have no overlapping values across the two CSV files.")
    print(json.dumps(merge_log, indent=2, default=str))


if __name__ == "__main__":
    main()
