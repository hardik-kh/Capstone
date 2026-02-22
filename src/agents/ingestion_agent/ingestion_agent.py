import os
import tempfile
from fastapi import UploadFile
from typing import List

from src.agents.ingestion_agent.validators import validate_file
from src.agents.ingestion_agent.csv_handler import load_csv
from src.agents.ingestion_agent.excel_handler import load_excel
from src.agents.ingestion_agent.profiler import profile_dataframe


async def ingest_files(files: List[UploadFile]):
    results = []
    errors = []

    for file in files:
        try:
            ext = validate_file(file)

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            if ext == ".csv":
                df, meta = load_csv(tmp_path)
                results.append({
                    "table_name": file.filename,
                    "source_type": "csv",
                    "profiling": profile_dataframe(df),
                    "ingestion_meta": meta
                })

            else:
                sheets = load_excel(tmp_path)
                for sheet, df in sheets.items():
                    results.append({
                        "table_name": f"{file.filename}__{sheet}",
                        "source_type": "excel",
                        "sheet_name": sheet,
                        "profiling": profile_dataframe(df)
                    })

            os.remove(tmp_path)

        except Exception as e:
            errors.append({
                "file": file.filename,
                "error": str(e)
            })

    return {
        "tables": results,
        "errors": errors
    }
