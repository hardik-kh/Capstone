# Excel loading with multi-sheet support

import pandas as pd

from src.core.exceptions import FileReadError


def load_excel(file_path: str) -> dict[str, pd.DataFrame]:
    """Loads all sheets from an Excel file into a dict of DataFrames."""
    try:
        xls = pd.ExcelFile(file_path)
        return {sheet: pd.read_excel(xls, sheet) for sheet in xls.sheet_names}
    except Exception as e:
        raise FileReadError(file_path, str(e))
