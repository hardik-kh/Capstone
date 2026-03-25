import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from agent.data_ingestion.common import (
    DEFAULT_SCHEMA,
    IngestionError,
    get_database_name,
    get_engine,
    get_engine_for_tier,
)

router = APIRouter()
INSTRUCTION_FILE = Path(__file__).with_name("validation_instructions.json")
GENERATED_SQL_FILE = Path(__file__).with_name("generated_validation_sql.json")
NUMERIC_TYPES = {
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "decimal",
    "numeric",
    "float",
    "real",
    "money",
    "smallmoney",
    "bit",
}
DATE_TYPES = {"date", "datetime", "datetime2", "smalldatetime", "datetimeoffset", "time"}
STRING_TYPES = {"char", "varchar", "nchar", "nvarchar", "text", "ntext"}


def _quote_ident(identifier: str) -> str:
    return f"[{identifier.replace(']', ']]')}]"


def _quote_3part(db: str, schema: str, table: str) -> str:
    return f"{_quote_ident(db)}.{_quote_ident(schema)}.{_quote_ident(table)}"


def _quote_2part(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def _load_instructions() -> dict:
    with INSTRUCTION_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_generated_sql_file(payload: dict) -> str:
    GENERATED_SQL_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(GENERATED_SQL_FILE)


def _load_generated_sql_file() -> dict:
    if not GENERATED_SQL_FILE.exists():
        raise IngestionError(
            "No generated validation SQL file found. Run /GenerateValidationSQL first.",
            status_code=404,
        )

    with GENERATED_SQL_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_bronze_metadata() -> list[dict]:
    engine = get_engine()
    column_query = """
        SELECT
            c.TABLE_SCHEMA,
            c.TABLE_NAME,
            c.COLUMN_NAME,
            c.DATA_TYPE,
            c.IS_NULLABLE,
            c.ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_SCHEMA = :schema
        ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION;
    """
    pk_query = """
        SELECT
            ku.TABLE_SCHEMA,
            ku.TABLE_NAME,
            ku.COLUMN_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
          ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
         AND tc.TABLE_SCHEMA = ku.TABLE_SCHEMA
         AND tc.TABLE_NAME = ku.TABLE_NAME
        WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
          AND ku.TABLE_SCHEMA = :schema
        ORDER BY ku.TABLE_SCHEMA, ku.TABLE_NAME, ku.ORDINAL_POSITION;
    """

    with engine.connect() as conn:
        column_rows = conn.execute(text(column_query), {"schema": DEFAULT_SCHEMA}).fetchall()
        pk_rows = conn.execute(text(pk_query), {"schema": DEFAULT_SCHEMA}).fetchall()

    pk_map: dict[tuple[str, str], list[str]] = {}
    for schema_name, table_name, column_name in pk_rows:
        pk_map.setdefault((schema_name, table_name), []).append(column_name)

    tables: dict[tuple[str, str], dict] = {}
    for schema_name, table_name, col, dtype, is_nullable, ordinal in column_rows:
        key = (schema_name, table_name)
        if key not in tables:
            tables[key] = {
                "schema": schema_name,
                "name": table_name,
                "columns": [],
                "primary_keys": pk_map.get(key, []),
            }
        tables[key]["columns"].append(
            {
                "name": col,
                "type": str(dtype).lower(),
                "nullable": is_nullable == "YES",
                "ordinal": ordinal,
            }
        )

    return list(tables.values())


def _is_numeric(data_type: str) -> bool:
    return data_type in NUMERIC_TYPES


def _is_date(data_type: str) -> bool:
    return data_type in DATE_TYPES


def _is_string(data_type: str) -> bool:
    return data_type in STRING_TYPES


def _negative_not_allowed(column_name: str) -> bool:
    markers = ("amount", "price", "cost", "qty", "quantity", "count", "total", "sales")
    lowered = column_name.lower()
    return any(marker in lowered for marker in markers)


def _enabled(items: list[str], check_name: str) -> bool:
    lowered = {item.lower() for item in items}
    return check_name.lower() in lowered


def _build_validation_sql(table: dict, bronze_db: str, checks: list[str]) -> list[str]:
    schema_name = table["schema"]
    table_name = table["name"]
    table_ref = _quote_2part(schema_name, table_name)
    sql = []

    if _enabled(checks, "Row count check"):
        sql.append(f"SELECT COUNT(*) AS total_rows FROM {table_ref};")

    for col in table["columns"]:
        col_name = col["name"]
        col_ref = _quote_ident(col_name)

        if _enabled(checks, "Null percentage check for each column"):
            sql.append(
                "\n".join(
                    [
                        "SELECT",
                        f"    '{col_name}' AS column_name,",
                        "    CAST(SUM(CASE WHEN " + col_ref + " IS NULL THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0) AS null_pct",
                        f"FROM {table_ref};",
                    ]
                )
            )
        if _enabled(checks, "Distinct count check"):
            sql.append(f"SELECT COUNT(DISTINCT {col_ref}) AS distinct_count FROM {table_ref};")

        if _is_numeric(col["type"]) and _enabled(checks, "Min/Max/Avg for numeric columns"):
            sql.append(
                f"SELECT MIN(TRY_CAST({col_ref} AS FLOAT)) AS min_val, "
                f"MAX(TRY_CAST({col_ref} AS FLOAT)) AS max_val, "
                f"AVG(TRY_CAST({col_ref} AS FLOAT)) AS avg_val "
                f"FROM {table_ref};"
            )
        if _is_numeric(col["type"]) and _enabled(checks, "Schema drift detection"):
            sql.append(
                f"SELECT COUNT(*) AS non_numeric_rows FROM {table_ref} "
                f"WHERE {col_ref} IS NOT NULL AND TRY_CAST({col_ref} AS FLOAT) IS NULL;"
            )
        if (
            _is_numeric(col["type"])
            and _enabled(checks, "Unexpected negative values check")
            and _negative_not_allowed(col_name)
        ):
                sql.append(
                    f"SELECT COUNT(*) AS negative_value_rows FROM {table_ref} "
                    f"WHERE TRY_CAST({col_ref} AS FLOAT) < 0;"
                )

        if _is_date(col["type"]) and _enabled(checks, "Date range check for date/datetime columns"):
            sql.append(
                f"SELECT MIN(TRY_CAST({col_ref} AS DATETIME2)) AS min_date, "
                f"MAX(TRY_CAST({col_ref} AS DATETIME2)) AS max_date FROM {table_ref};"
            )
        if _is_date(col["type"]) and _enabled(checks, "Schema drift detection"):
            sql.append(
                f"SELECT COUNT(*) AS invalid_date_rows FROM {table_ref} "
                f"WHERE {col_ref} IS NOT NULL AND TRY_CAST({col_ref} AS DATETIME2) IS NULL;"
            )

        if _is_string(col["type"]) and _enabled(checks, "String length anomalies"):
            sql.append(
                f"SELECT MAX(LEN({col_ref})) AS max_length, AVG(CAST(LEN({col_ref}) AS FLOAT)) AS avg_length "
                f"FROM {table_ref};"
            )

    pk_cols = table["primary_keys"]
    if pk_cols and _enabled(checks, "Duplicate primary key check (if key columns exist)"):
        pk_expr = ", ".join(_quote_ident(c) for c in pk_cols)
        sql.append(
            f"SELECT {pk_expr}, COUNT(*) AS duplicate_count FROM {table_ref} "
            f"GROUP BY {pk_expr} HAVING COUNT(*) > 1;"
        )

    return sql


def _clean_expr(column: dict, cleaning_rules: list[str]) -> str:
    col_name = column["name"]
    col_ref = _quote_ident(col_name)
    data_type = column["type"]

    if _is_numeric(data_type):
        numeric_expr = f"TRY_CAST({col_ref} AS FLOAT)"
        if (
            _enabled(cleaning_rules, "Remove negative values where not allowed")
            and _negative_not_allowed(col_name)
        ):
            numeric_expr = f"ABS({numeric_expr})"
        if _enabled(cleaning_rules, "Replace NULL numeric values with 0 or a safe default"):
            return f"COALESCE({numeric_expr}, 0) AS {col_ref}"
        return f"{numeric_expr} AS {col_ref}"

    if _is_date(data_type):
        if _enabled(cleaning_rules, "Cast date strings to DATE or DATETIME") or _enabled(
            cleaning_rules, "Standardize date formats"
        ):
            return f"TRY_CAST({col_ref} AS DATETIME2) AS {col_ref}"
        return f"{col_ref} AS {col_ref}"

    if _is_string(data_type):
        base_expr = f"CAST({col_ref} AS NVARCHAR(4000))"
        if _enabled(cleaning_rules, "Trim whitespace from string columns"):
            base_expr = f"LTRIM(RTRIM({base_expr}))"
        if _enabled(cleaning_rules, "Replace NULL strings with '' (empty string) or 'UNKNOWN'"):
            return f"COALESCE({base_expr}, '') AS {col_ref}"
        return f"{base_expr} AS {col_ref}"

    return f"{col_ref} AS {col_ref}"


def _build_silver_create_sql(table: dict, bronze_db: str, cleaning_rules: list[str]) -> str:
    schema_name = table["schema"]
    table_name = table["name"]
    bronze_ref = _quote_3part(bronze_db, schema_name, table_name)
    silver_ref = _quote_2part(schema_name, table_name)

    select_list = ",\n            ".join(_clean_expr(col, cleaning_rules) for col in table["columns"])
    pk_cols = table["primary_keys"]
    if pk_cols:
        partition_by = ", ".join(_quote_ident(c) for c in pk_cols)
    else:
        partition_by = ", ".join(_quote_ident(col["name"]) for col in table["columns"])

    return "\n".join(
        [
            f"IF OBJECT_ID(N'{silver_ref}', N'U') IS NOT NULL DROP TABLE {silver_ref};",
            "WITH source_data AS (",
            "    SELECT",
            f"            {select_list}",
            f"    FROM {bronze_ref}",
            "), deduped AS (",
            "    SELECT *,",
            f"           ROW_NUMBER() OVER (PARTITION BY {partition_by} ORDER BY (SELECT 1)) AS rn",
            "    FROM source_data",
            ")",
            f"SELECT * INTO {silver_ref}",
            "FROM deduped",
            "WHERE rn = 1;",
        ]
    )


def _build_silver_select_sql(table: dict, bronze_db: str, cleaning_rules: list[str]) -> str:
    schema_name = table["schema"]
    table_name = table["name"]
    bronze_ref = _quote_2part(schema_name, table_name)
    select_list = ",\n            ".join(_clean_expr(col, cleaning_rules) for col in table["columns"])
    pk_cols = table["primary_keys"]
    if pk_cols:
        partition_by = ", ".join(_quote_ident(c) for c in pk_cols)
    else:
        partition_by = ", ".join(_quote_ident(col["name"]) for col in table["columns"])

    return "\n".join(
        [
            "WITH source_data AS (",
            "    SELECT",
            f"            {select_list}",
            f"    FROM {bronze_ref}",
            "), deduped AS (",
            "    SELECT *,",
            f"           ROW_NUMBER() OVER (PARTITION BY {partition_by} ORDER BY (SELECT 1)) AS rn",
            "    FROM source_data",
            ")",
            "SELECT *",
            "FROM deduped",
            "WHERE rn = 1;",
        ]
    )


def _execute_silver_sql(table_scripts: list[dict]) -> list[dict]:
    bronze_engine = get_engine_for_tier("bronze")
    silver_engine = get_engine_for_tier("silver")
    execution_results = []
    for table_script in table_scripts:
        try:
            select_sql = table_script["load_sql"]
            df = pd.read_sql_query(select_sql, bronze_engine)
            df.to_sql(
                table_script["table_name"],
                silver_engine,
                schema=table_script["schema"],
                if_exists="replace",
                index=False,
                chunksize=1000,
            )
            execution_results.append(
                {
                    "schema": table_script["schema"],
                    "table_name": table_script["table_name"],
                    "status": "executed",
                    "rows_loaded": len(df),
                }
            )
        except Exception as exc:
            raise IngestionError(
                f"Silver table load failed for {table_script['schema']}.{table_script['table_name']}: {exc}",
                status_code=500,
            ) from exc
    return execution_results


def _build_generated_payload() -> dict:
    instructions = _load_instructions()
    bronze_db = get_database_name("bronze")
    silver_db = get_database_name("silver")
    tables = fetch_bronze_metadata()

    if not tables:
        raise IngestionError(
            f"No tables found in Bronze schema '{DEFAULT_SCHEMA}'.",
            status_code=404,
        )

    result = []
    checks = instructions.get("validation_checks", [])
    cleaning_rules = instructions.get("etl_cleaning_rules", [])
    for table in tables:
        validation_sql = _build_validation_sql(table, bronze_db, checks)
        silver_sql = _build_silver_create_sql(table, bronze_db, cleaning_rules)
        load_sql = _build_silver_select_sql(table, bronze_db, cleaning_rules)
        result.append(
            {
                "table_name": table["name"],
                "schema": table["schema"],
                "primary_keys": table["primary_keys"],
                "validation_sql": validation_sql,
                "cleaning_sql": [silver_sql],
                "load_sql": load_sql,
            }
        )

    return {
        "detail": "Validation and Silver transformation SQL generated successfully.",
        "instruction_purpose": instructions.get("purpose"),
        "bronze_database": bronze_db,
        "silver_database": silver_db,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed": False,
        "tables": result,
    }


@router.get("/GenerateValidationSQL")
def generate_validation_sql():
    try:
        generated_payload = _build_generated_payload()
        output_file = _write_generated_sql_file(generated_payload)

        return JSONResponse(
            status_code=200,
            content={
                **generated_payload,
                "generated_sql_file": output_file,
            },
        )

    except IngestionError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected validation SQL error: {exc}"},
        )

