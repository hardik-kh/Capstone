# Excel loading with multi-sheet support and bronze persistence

import os
from datetime import datetime

import pandas as pd

from core.config import BRONZE_OUTPUT_DIR
from core.exceptions import FileReadError


def load_excel(file_path: str) -> dict[str, pd.DataFrame]:
    """Loads all sheets from an Excel file into a dict of DataFrames."""
    try:
        xls = pd.ExcelFile(file_path)
        return {sheet: pd.read_excel(xls, sheet) for sheet in xls.sheet_names}
    except Exception as e:
        raise FileReadError(file_path, str(e))


def save_excel_sheet_to_bronze(df: pd.DataFrame, original_file: str, sheet_name: str) -> dict:
    """Saves an Excel sheet to the bronze layer as a CSV file."""
    os.makedirs(BRONZE_OUTPUT_DIR, exist_ok=True)

    file_stem = os.path.splitext(os.path.basename(original_file))[0]
    safe_sheet = str(sheet_name).strip().lower().replace(" ", "_").replace("-", "_")
    output_filename = f"{file_stem}__{safe_sheet}_bronze.csv"
    output_path = os.path.join(BRONZE_OUTPUT_DIR, output_filename)

    df.to_csv(output_path, index=False)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "output_path": output_path,
        "storage_format": "csv",
        "sheet_name": sheet_name,
        "timestamp": datetime.now().isoformat(),
    }