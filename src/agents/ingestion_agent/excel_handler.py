import pandas as pd
from src.core.exceptions import FileReadError


def load_excel(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        return {
            sheet: pd.read_excel(xls, sheet)
            for sheet in xls.sheet_names
        }
    except Exception as e:
        raise FileReadError(str(e))
