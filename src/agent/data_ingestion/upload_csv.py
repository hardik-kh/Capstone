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

async def upload_single_csv(file: UploadFile, schema: str = "dbo"):
    table_name = table_name_from_filename(file.filename, ".csv")
    full_table_name = schema_table_name(schema, table_name)
    stream = await read_file_to_bytes_buffer(file)
    chunk_size = 50_000
    total_rows = 0
    write_mode = "replace"
    has_chunk = False
    engine = get_engine()

    try:
        for df_chunk in pd.read_csv(stream, chunksize=chunk_size):
            has_chunk = True
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
    except pd.errors.EmptyDataError as exc:
        raise IngestionError(f"CSV parsing failed for '{file.filename}': no columns found.") from exc
    except Exception as exc:
        raise IngestionError(f"CSV ingestion failed for '{file.filename}': {exc}") from exc

    if not has_chunk:
        raise IngestionError(f"CSV file '{file.filename}' has no readable content.")

    return {"table": full_table_name, "rows": total_rows}


@router.post("/UploadCSVs")
async def upload_multiple_csvs(files: list[UploadFile] = File(...)):
    schema = DEFAULT_SCHEMA
    results = []

    try:
        if not files:
            raise IngestionError("No files were uploaded.")
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".csv"):
                raise IngestionError(f"File '{file.filename}' must have a .csv extension.")
            result = await upload_single_csv(file, schema)
            results.append(result)

        return JSONResponse(
            status_code=200,
            content={
                "detail": "All CSV files ingested successfully.",
                "results": results
            }
        )

    except IngestionError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected CSV ingestion error: {exc}"}
        )
