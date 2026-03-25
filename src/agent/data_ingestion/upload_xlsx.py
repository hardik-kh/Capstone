import pandas as pd
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from agent.data_ingestion.common import (
    DEFAULT_SCHEMA,
    IngestionError,
    get_engine,
    read_file_to_bytes_buffer,
    schema_table_name,
    table_name_from_filename,
)

router = APIRouter()

async def upload_single_xlsx(file: UploadFile, schema: str = "dbo"):
    table_name = table_name_from_filename(file.filename, ".xlsx")
    full_table_name = schema_table_name(schema, table_name)
    excel_bytes = await read_file_to_bytes_buffer(file)
    engine = get_engine()

    try:
        df = pd.read_excel(excel_bytes, sheet_name=0)
    except ValueError as exc:
        raise IngestionError(f"XLSX parsing failed for '{file.filename}': {exc}") from exc
    except Exception as exc:
        raise IngestionError(f"XLSX ingestion failed for '{file.filename}': {exc}") from exc

    chunk_size = 50_000
    total_rows = 0
    write_mode = "replace"

    for start in range(0, len(df), chunk_size):
        df_chunk = df.iloc[start:start + chunk_size]
        df_chunk.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists=write_mode,
            index=False,
            chunksize=1000,
        )
        write_mode = "append"
        total_rows += len(df_chunk)

    if len(df) == 0:
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=1000,
        )

    return {"table": full_table_name, "rows": total_rows}


@router.post("/UploadXLSX")
async def upload_multiple_xlsx(files: list[UploadFile] = File(...)):
    schema = DEFAULT_SCHEMA
    results = []

    try:
        if not files:
            raise IngestionError("No files were uploaded.")
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".xlsx"):
                raise IngestionError(f"File '{file.filename}' must have a .xlsx extension.")
            result = await upload_single_xlsx(file, schema)
            results.append(result)

        return JSONResponse(
            status_code=200,
            content={
                "detail": "All XLSX files ingested successfully.",
                "results": results
            }
        )

    except IngestionError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected XLSX ingestion error: {exc}"}
        )
