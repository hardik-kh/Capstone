# Core ingestion orchestration: validation, loading, cleaning, profiling

import os
import tempfile
from typing import Any

from fastapi import UploadFile

from src.core.exceptions import IngestionError
from src.core.logger import get_logger
from src.agents.ingestion_agent.validators import validate_file
from src.agents.ingestion_agent.csv_handler import load_csv
from src.agents.ingestion_agent.excel_handler import load_excel
from src.agents.ingestion_agent.profiler import clean_and_profile

logger = get_logger("DataIngestionAgent")


def _build_table_entry(
    table_name: str,
    df_clean: Any,
    profiling: dict,
    meta: dict,
    source_type: str,
    sheet_name: str | None = None,
) -> dict:
    """Builds a JSON-serializable table entry for the ingestion response."""
    return {
        "table_name": table_name,
        "source_type": source_type,
        "sheet_name": sheet_name,
        "profiling": profiling,
        "ingestion_meta": meta,
    }


async def ingest_files(files: list[UploadFile]) -> dict:
    """Ingests multiple uploaded files and returns tables + errors."""
    results: list[dict] = []
    errors: list[dict] = []

    for file in files:
        logger.info(f"Processing file: {file.filename}")
        tmp_path = None

        try:
            ext = validate_file(file)

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            if ext == ".csv":
                df, meta = load_csv(tmp_path)
                df_clean, profiling = clean_and_profile(df)
                results.append(
                    _build_table_entry(
                        table_name=file.filename,
                        df_clean=df_clean,
                        profiling=profiling,
                        meta=meta,
                        source_type="csv",
                    )
                )
            else:
                sheets = load_excel(tmp_path)
                for sheet_name, df in sheets.items():
                    df_clean, profiling = clean_and_profile(df)
                    results.append(
                        _build_table_entry(
                            table_name=f"{file.filename}__{sheet_name}",
                            df_clean=df_clean,
                            profiling=profiling,
                            meta={},
                            source_type="excel",
                            sheet_name=sheet_name,
                        )
                    )

        except IngestionError as e:
            logger.error(f"Ingestion error for file {file.filename}: {e}")
            err = e.to_dict()
            err["file"] = file.filename
            errors.append(err)

        except Exception as e:
            logger.exception(f"Unexpected error for file {file.filename}")
            errors.append(
                {
                    "file": file.filename,
                    "error_code": "UNEXPECTED_ERROR",
                    "message": "An unexpected error occurred during ingestion.",
                    "hint": str(e),
                }
            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {"tables": results, "errors": errors}
