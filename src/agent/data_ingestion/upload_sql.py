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

def _is_query_statement(sql_text: str) -> bool:
    first_token = sql_text.strip().split(maxsplit=1)[0].lower()
    return first_token in {"select", "with"}


async def upload_single_sql(file: UploadFile, schema: str = "dbo"):
    table_name = table_name_from_filename(file.filename, ".sql")
    full_table_name = schema_table_name(schema, table_name)
    engine = get_engine()
    buffer = await read_file_to_bytes_buffer(file)

    try:
        sql_text = buffer.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"SQL file '{file.filename}' must be UTF-8 encoded.") from exc

    sql_text = sql_text.strip()
    if not sql_text:
        raise IngestionError(f"SQL file '{file.filename}' has no executable content.")

    if _is_query_statement(sql_text):
        try:
            df = pd.read_sql_query(sql_text, engine)
            df.to_sql(
                table_name,
                engine,
                schema=schema,
                if_exists="replace",
                index=False,
                chunksize=1000,
            )
            return {"table": full_table_name, "rows": len(df)}
        except Exception as exc:
            raise IngestionError(f"Failed to run SQL query from '{file.filename}': {exc}") from exc

    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(sql_text)
    except Exception as exc:
        raise IngestionError(f"Failed to execute SQL script from '{file.filename}': {exc}") from exc

    return {"executed": True, "file": file.filename}


@router.post("/UploadSQL")
async def upload_multiple_sql(files: list[UploadFile] = File(...)):
    schema = DEFAULT_SCHEMA
    results = []

    try:
        if not files:
            raise IngestionError("No files were uploaded.")
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".sql"):
                raise IngestionError(f"File '{file.filename}' must have a .sql extension.")
            result = await upload_single_sql(file, schema)
            results.append(result)

        return JSONResponse(
            status_code=200,
            content={
                "detail": "All SQL files processed successfully.",
                "results": results
            }
        )

    except IngestionError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected SQL ingestion error: {exc}"}
        )
