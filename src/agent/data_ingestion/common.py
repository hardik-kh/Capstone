import io
import os
import re
import urllib.parse
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import UploadFile
from sqlalchemy import create_engine

DEFAULT_SCHEMA = "dbo"
DEFAULT_MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024


def get_max_upload_size_bytes() -> int:
    load_dotenv()
    raw_value = os.getenv("MAX_UPLOAD_SIZE_BYTES")
    if raw_value is None:
        return DEFAULT_MAX_UPLOAD_SIZE_BYTES

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise IngestionError(
            "MAX_UPLOAD_SIZE_BYTES must be a valid integer.",
            status_code=500,
        ) from exc

    if value <= 0:
        raise IngestionError(
            "MAX_UPLOAD_SIZE_BYTES must be greater than zero.",
            status_code=500,
        )

    return value


class IngestionError(Exception):
    def __init__(self, detail: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _validate_extension(filename: str | None, extension: str) -> str:
    if not filename:
        raise IngestionError("Uploaded file is missing a filename.")
    if not filename.lower().endswith(extension):
        raise IngestionError(f"File '{filename}' is not a {extension} file.")
    return filename


def table_name_from_filename(filename: str | None, extension: str) -> str:
    safe_filename = _validate_extension(filename, extension)
    base_name = os.path.basename(safe_filename)[: -len(extension)].strip()
    normalized = re.sub(r"\s+", "_", base_name)
    normalized = re.sub(r"[^0-9a-zA-Z_]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    if not normalized:
        raise IngestionError(f"File '{safe_filename}' does not contain a valid table name.")
    if normalized[0].isdigit():
        normalized = f"t_{normalized}"
    return normalized[:128]


def schema_table_name(schema: str, table_name: str) -> str:
    return f"{schema}.{table_name}"


async def read_file_to_bytes_buffer(
    file: UploadFile,
    max_size_bytes: int | None = None,
) -> io.BytesIO:
    if max_size_bytes is None:
        max_size_bytes = get_max_upload_size_bytes()

    total_size = 0
    buffer = io.BytesIO()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size_bytes:
            raise IngestionError(
                f"File '{file.filename}' exceeds maximum allowed size of {max_size_bytes} bytes."
            )
        buffer.write(chunk)

    if total_size == 0:
        raise IngestionError(f"File '{file.filename}' is empty.")

    buffer.seek(0)
    return buffer


def _get_db_env() -> dict[str, str | None]:
    load_dotenv()
    return {
        "AZURE_SQL_SERVER": os.getenv("AZURE_SQL_SERVER"),
        "AZURE_SQL_BRONZE_DATABASE": os.getenv("AZURE_SQL_BRONZE_DATABASE"),
        "AZURE_SQL_SILVER_DATABASE": os.getenv("AZURE_SQL_SILVER_DATABASE"),
        "AZURE_SQL_USERNAME": os.getenv("AZURE_SQL_USERNAME"),
        "AZURE_SQL_PASSWORD": os.getenv("AZURE_SQL_PASSWORD"),
    }


def _build_engine(database_name: str):
    env = _get_db_env()
    missing = [
        key
        for key in ["AZURE_SQL_SERVER", "AZURE_SQL_USERNAME", "AZURE_SQL_PASSWORD"]
        if not env.get(key)
    ]
    if missing:
        raise IngestionError(
            "Missing required database environment variables: " + ", ".join(missing),
            status_code=500,
        )

    if not database_name:
        raise IngestionError("Target database name is empty.", status_code=500)

    driver = os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")
    params = urllib.parse.quote_plus(
        f"DRIVER={{{driver}}};"
        f"SERVER={env['AZURE_SQL_SERVER']};"
        f"DATABASE={database_name};"
        f"UID={env['AZURE_SQL_USERNAME']};"
        f"PWD={env['AZURE_SQL_PASSWORD']};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        fast_executemany=True,
    )


def get_database_name(tier: str) -> str:
    env = _get_db_env()
    key_by_tier = {
        "bronze": "AZURE_SQL_BRONZE_DATABASE",
        "silver": "AZURE_SQL_SILVER_DATABASE",
    }
    key = key_by_tier.get(tier.lower())
    if not key:
        raise IngestionError(
            f"Unsupported database tier '{tier}'. Use one of: bronze, silver.",
            status_code=500,
        )

    db_name = env.get(key)
    if not db_name:
        raise IngestionError(
            f"Missing required database environment variable: {key}",
            status_code=500,
        )
    return db_name


@lru_cache(maxsize=1)
def get_engine():
    return _build_engine(get_database_name("bronze"))


@lru_cache(maxsize=1)
def get_silver_engine():
    return _build_engine(get_database_name("silver"))


def get_engine_for_tier(tier: str):
    if tier.lower() == "bronze":
        return get_engine()
    if tier.lower() == "silver":
        return get_silver_engine()
    raise IngestionError(
        f"Unsupported database tier '{tier}'. Use one of: bronze, silver.",
        status_code=500,
    )
