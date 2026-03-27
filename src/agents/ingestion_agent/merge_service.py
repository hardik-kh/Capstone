# Merge inference and CSV merge utilities powered by Ollama

import json
import os
import re
import time
from datetime import datetime
from typing import Any

import pandas as pd

from agents.ingestion_agent.csv_handler import detect_delimiter, detect_encoding
from core.config import (
    MERGED_OUTPUT_DIR,
    MERGE_SAMPLE_ROWS,
    OLLAMA_MODEL,
)
from core.exceptions import MergeInferenceError
from core.logger import get_logger

logger = get_logger("MergeService")


def _sample_rows(df: pd.DataFrame) -> list[dict]:
    """Builds a compact sample payload safe for JSON serialization."""
    return df.head(MERGE_SAMPLE_ROWS).astype("string").where(df.notna(), None).to_dict(orient="records")


def _column_profile(df: pd.DataFrame) -> list[dict]:
    """Summarizes each column to help the model pick a join key."""
    profile: list[dict] = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        unique_non_null = int(series.nunique(dropna=True))
        uniqueness_ratio = round(unique_non_null / non_null, 4) if non_null else 0.0
        profile.append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "non_null_rows": non_null,
                "null_rows": int(total_rows - non_null),
                "unique_non_null_values": unique_non_null,
                "uniqueness_ratio": uniqueness_ratio,
                "sample_values": series.dropna().astype("string").head(5).tolist(),
            }
        )
    return profile


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes CSV column names to match the ingestion cleaning step."""
    df = df.copy()
    df.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in df.columns
    ]
    return df


def _normalize_column_name(column: str) -> str:
    """Normalizes a single column name to snake_case."""
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_column_names(columns: list[str]) -> list[str]:
    """Normalizes a list of columns while keeping names unique."""
    normalized_names: list[str] = []
    seen: dict[str, int] = {}
    for column in columns:
        normalized = _normalize_column_name(column)
        count = seen.get(normalized, 0)
        if count:
            normalized = f"{normalized}_{count + 1}"
        seen[_normalize_column_name(column)] = count + 1
        normalized_names.append(normalized)
    return normalized_names


def _load_csv_sample(file_path: str, sample_rows: int = 5_000) -> tuple[pd.DataFrame, dict]:
    """Loads a bounded CSV sample for schema inference without reading the whole file."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=delimiter,
        nrows=sample_rows,
        low_memory=False,
    )
    return _normalize_columns(df), {"encoding": encoding, "delimiter": delimiter}


def _read_csv_columns(file_path: str) -> tuple[list[str], dict]:
    """Reads only the CSV header and returns normalized column names."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=delimiter,
        nrows=0,
        low_memory=False,
    )
    return _normalize_column_names(list(df.columns)), {"encoding": encoding, "delimiter": delimiter}


def _extract_json_object(text: str) -> dict:
    """Extracts a JSON object from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _shared_columns(left_df: pd.DataFrame, right_df: pd.DataFrame) -> list[str]:
    """Returns columns that exist with the same name in both dataframes."""
    return [column for column in left_df.columns if column in right_df.columns]


def _is_date_like_column(column_name: str) -> bool:
    """Returns True when a column name looks like a date/time field."""
    normalized = column_name.strip().lower()
    return any(token in normalized for token in ["date", "time", "timestamp", "year", "month", "day"])


def _business_priority(column_name: str) -> int:
    """Ranks business key columns above generic or date-like columns."""
    normalized = column_name.strip().lower()
    priority_order = [
        "store_nbr",
        "store_id",
        "item_nbr",
        "item_id",
        "id",
        "nbr",
    ]
    for index, preferred_name in enumerate(priority_order):
        if normalized == preferred_name:
            return len(priority_order) - index
    if _is_date_like_column(normalized):
        return 0
    return 1


def _normalized_overlap_ratio(left_series: pd.Series, right_series: pd.Series) -> float:
    """Estimates how suitable two columns are for joining based on value overlap."""
    left_values = {
        str(value).strip().lower()
        for value in left_series.dropna().astype("string")
        if str(value).strip()
    }
    right_values = {
        str(value).strip().lower()
        for value in right_series.dropna().astype("string")
        if str(value).strip()
    }
    if not left_values or not right_values:
        return 0.0
    intersection_size = len(left_values & right_values)
    baseline = min(len(left_values), len(right_values))
    return round(intersection_size / baseline, 4) if baseline else 0.0


def _fallback_merge_decision(left_df: pd.DataFrame, right_df: pd.DataFrame, common_columns: list[str]) -> dict:
    """Chooses the best shared merge key deterministically when Ollama output is unusable."""
    scored_columns: list[tuple[int, float, int, str]] = []
    for column in common_columns:
        overlap_ratio = _normalized_overlap_ratio(left_df[column], right_df[column])
        left_unique = int(left_df[column].nunique(dropna=True))
        right_unique = int(right_df[column].nunique(dropna=True))
        business_priority = _business_priority(column)
        scored_columns.append(
            (
                business_priority,
                overlap_ratio,
                min(left_unique, right_unique),
                column,
            )
        )

    scored_columns.sort(reverse=True)
    best_priority, best_overlap, _, best_column = scored_columns[0]
    selected_columns = [best_column]
    second_reason = ""
    if len(scored_columns) > 1:
        second_priority, second_overlap, _, second_column = scored_columns[1]
        second_is_relevant = (
            second_column != best_column
            and not _is_date_like_column(second_column)
            and (
                second_priority > 1
                or second_overlap >= 0.5
                or (best_overlap and second_overlap >= best_overlap * 0.75)
            )
        )
        if second_is_relevant:
            selected_columns.append(second_column)
            second_reason = f" Added {second_column} to form a more reliable composite key."

    return {
        "merge_columns": selected_columns,
        "left_column": selected_columns[0],
        "right_column": selected_columns[0],
        "confidence": best_overlap,
        "join_type": "left",
        "reason": (
            "Fallback heuristic selected the shared columns with the strongest business relevance and normalized "
            f"value overlap: {selected_columns}.{second_reason}"
        ),
    }


def _normalize_merge_columns(decision: dict, common_columns: list[str]) -> list[str]:
    """Normalizes the selected merge columns to one or two valid shared columns."""
    merge_columns = decision.get("merge_columns") or []
    if isinstance(merge_columns, str):
        merge_columns = [merge_columns]

    valid_columns: list[str] = []
    for column in merge_columns:
        if column in common_columns and column not in valid_columns:
            valid_columns.append(column)
    if valid_columns:
        prioritized = sorted(
            valid_columns,
            key=lambda column: (_business_priority(column), not _is_date_like_column(column), column),
            reverse=True,
        )
        return prioritized[:2]

    left_column = decision.get("left_column")
    if left_column in common_columns:
        return [left_column]
    return []


def _sanitize_dataset_name(name: str) -> str:
    """Builds a safe output filename stem."""
    stem = os.path.splitext(os.path.basename(name))[0].strip().lower()
    sanitized = re.sub(r"[^a-z0-9_]+", "_", stem.replace(" ", "_").replace("-", "_"))
    return sanitized.strip("_") or "dataset"


def _get_duckdb_connection() -> Any:
    """Returns a DuckDB connection used for large-file-safe CSV operations."""
    try:
        import duckdb
    except ImportError as exc:
        raise MergeInferenceError(
            "DuckDB is not installed in the runtime environment.",
            hint="Install it with: python3 -m pip install duckdb",
        ) from exc

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    database_path = os.path.join(MERGED_OUTPUT_DIR, "merge_service.duckdb")
    connection = duckdb.connect(database_path)
    connection.execute("PRAGMA temp_directory='/tmp';")
    return connection


def _csv_select_with_normalized_columns(file_path: str, table_alias: str) -> tuple[str, list[str]]:
    """Builds a SELECT query that aliases CSV columns to normalized names."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    original_df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=delimiter,
        nrows=0,
        low_memory=False,
    )
    original_columns = list(original_df.columns)
    normalized_columns = _normalize_column_names(original_columns)
    projections = ", ".join(
        [
            f'{table_alias}."{str(original).replace(chr(34), chr(34) * 2)}" AS "{normalized}"'
            for original, normalized in zip(original_columns, normalized_columns)
        ]
    )
    csv_path_sql = file_path.replace("'", "''")
    select_sql = (
        f"SELECT {projections} "
        f"FROM read_csv_auto('{csv_path_sql}', HEADER=TRUE) AS {table_alias}"
    )
    return select_sql, normalized_columns


def _write_processed_table_from_csv(file_path: str, original_name: str) -> dict:
    """Writes a normalized processed CSV copy without loading the full file into pandas."""
    output_filename = f"{_sanitize_dataset_name(original_name)}__processed.csv"
    output_path = os.path.join(MERGED_OUTPUT_DIR, output_filename)
    connection = _get_duckdb_connection()
    try:
        select_sql, normalized_columns = _csv_select_with_normalized_columns(file_path, "source_table")
        output_path_sql = output_path.replace("'", "''")
        connection.execute(
            f"""
            COPY (
                {select_sql}
            ) TO '{output_path_sql}' (HEADER, DELIMITER ',');
            """
        )
        row_count = connection.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{output_path_sql}', HEADER=TRUE)"
        ).fetchone()[0]
    finally:
        connection.close()

    return {
        "source_name": original_name,
        "output_filename": output_filename,
        "output_path": output_path,
        "rows": int(row_count),
        "columns": int(len(normalized_columns)),
    }


def infer_merge_columns_with_ollama(
    left_name: str,
    left_df: pd.DataFrame,
    right_name: str,
    right_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Uses Ollama to choose the best shared merge key for two tables."""
    common_columns = _shared_columns(left_df, right_df)
    if not common_columns:
        decision = {
            "merge_columns": [],
            "left_column": None,
            "right_column": None,
            "confidence": 0.0,
            "join_type": None,
            "reason": "No common columns exist across the two tables, so merge inference was skipped.",
        }
        inference_log = {
            "provider": "ollama",
            "model": OLLAMA_MODEL,
            "duration_seconds": 0.0,
            "common_columns_considered": [],
            "used_fallback": False,
            "raw_response": None,
            "decision": decision,
        }
        return decision, inference_log
    non_date_common_columns = [column for column in common_columns if not _is_date_like_column(column)]
    preferred_business_columns = [
        column for column in common_columns
        if _business_priority(column) > 1
    ]

    try:
        from ollama import generate
    except ImportError as exc:
        raise MergeInferenceError(
            "The Python Ollama client is not installed.",
            hint="Install it in the API environment with: python3 -m pip install ollama",
        ) from exc

    prompt_payload = {
        "left_dataset": {
            "name": left_name,
            "columns": _column_profile(left_df),
            "sample_rows": _sample_rows(left_df),
        },
        "right_dataset": {
            "name": right_name,
            "columns": _column_profile(right_df),
            "sample_rows": _sample_rows(right_df),
        },
        "constraints": {
            "shared_column_names_only": True,
            "allowed_shared_columns": common_columns,
            "preferred_non_date_columns": non_date_common_columns,
            "preferred_business_columns": preferred_business_columns,
            "left_column_must_equal_right_column": True,
            "avoid_date_columns_when_other_shared_columns_exist": bool(non_date_common_columns),
        },
        "task": (
            "Choose the safest and most accurate way to merge these two tables. Select one shared column when a "
            "single column is sufficient. Select two shared columns as a composite key when using both together "
            "would produce a more reliable match than any single column alone. Prefer a composite key whenever "
            "two shared business columns together clearly reduce ambiguity compared with a single shared column. "
            "Prefer business identifiers and "
            "high-overlap columns. Avoid date-only joins when stronger shared columns exist. Use only columns from "
            "allowed_shared_columns and keep left_column and right_column the same shared name when a single-column "
            "merge is chosen. Respond as strict JSON with keys: merge_columns, left_column, right_column, "
            "confidence, join_type, reason. merge_columns must contain one or two shared columns."
        ),
    }

    started_at = time.time()
    try:
        payload = generate(
            OLLAMA_MODEL,
            json.dumps(prompt_payload),
            format="json",
        )
    except Exception as exc:
        raise MergeInferenceError(
            "Failed while requesting merge-key inference from Ollama.",
            hint=str(exc),
        ) from exc

    raw_response = payload.get("response", "")
    used_fallback = False
    try:
        decision = _extract_json_object(raw_response)
    except Exception:
        decision = _fallback_merge_decision(left_df, right_df, common_columns)
        used_fallback = True

    merge_columns = _normalize_merge_columns(decision, common_columns)

    invalid_shared_choice = (
        len(merge_columns) not in {1, 2}
        or any(column not in common_columns for column in merge_columns)
    )
    if invalid_shared_choice:
        decision = _fallback_merge_decision(left_df, right_df, common_columns)
        merge_columns = _normalize_merge_columns(decision, common_columns)
        used_fallback = True

    left_column = merge_columns[0] if merge_columns else None
    right_column = merge_columns[0] if merge_columns else None

    inference_log = {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "duration_seconds": round(time.time() - started_at, 4),
        "common_columns_considered": common_columns,
        "used_fallback": used_fallback,
        "raw_response": raw_response,
        "decision": {
            "merge_columns": merge_columns,
            "left_column": left_column,
            "right_column": right_column,
            "confidence": decision.get("confidence"),
            "join_type": decision.get("join_type", "inner"),
            "reason": decision.get("reason"),
        },
    }
    return decision, inference_log


def _merge_with_duckdb(
    left_path: str,
    right_path: str,
    output_path: str,
    merge_columns: list[str],
    join_type: str,
    left_name: str,
    right_name: str,
) -> tuple[int, int]:
    """Merges CSV files with DuckDB and writes the result directly to CSV."""
    if not merge_columns:
        raise MergeInferenceError(
            "DuckDB merge requires at least one merge column.",
            hint=f"Received merge columns: {merge_columns}",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    connection = _get_duckdb_connection()
    try:
        logger.info("DuckDB merge progress: 20%")
        join_keyword = join_type.upper()
        output_path_sql = output_path.replace("'", "''")
        left_select_sql, left_columns = _csv_select_with_normalized_columns(left_path, "left_source")
        right_select_sql, right_columns = _csv_select_with_normalized_columns(right_path, "right_source")
        join_conditions = " AND ".join(
            [
                f"""CAST(left_table."{column.replace('"', '""')}" AS VARCHAR) = CAST(right_table."{column.replace('"', '""')}" AS VARCHAR)"""
                for column in merge_columns
            ]
        )
        left_projection = ", ".join(
            [
                f'left_table."{column.replace(chr(34), chr(34) * 2)}" AS "{_sanitize_dataset_name(left_name)}__{column}"'
                for column in left_columns
            ]
        )
        right_projection = ", ".join(
            [
                f'right_table."{column.replace(chr(34), chr(34) * 2)}" AS "{_sanitize_dataset_name(right_name)}__{column}"'
                for column in right_columns
            ]
        )
        connection.execute(
            f"""
            COPY (
                SELECT
                    {left_projection},
                    {right_projection}
                FROM ({left_select_sql}) AS left_table
                {join_keyword} JOIN ({right_select_sql}) AS right_table
                ON {join_conditions}
            ) TO '{output_path_sql}' (HEADER, DELIMITER ',');
            """
        )
        logger.info("DuckDB merge progress: 80%")
        row_count = connection.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{output_path_sql}', HEADER=TRUE)"
        ).fetchone()[0]
        column_count = len(
            connection.execute(
                f"SELECT * FROM read_csv_auto('{output_path_sql}', HEADER=TRUE) LIMIT 0"
            ).df().columns
        )
        logger.info("DuckDB merge progress: 100%")
        return int(row_count), int(column_count)
    finally:
        connection.close()


def merge_csv_files(
    left_name: str,
    left_path: str,
    right_name: str,
    right_path: str,
) -> tuple[pd.DataFrame, dict]:
    """Merges two CSV files using header/sample inference and DuckDB-backed execution."""
    left_sample_df, _ = _load_csv_sample(left_path)
    right_sample_df, _ = _load_csv_sample(right_path)
    left_columns, _ = _read_csv_columns(left_path)
    right_columns, _ = _read_csv_columns(right_path)

    logger.info(
        "Pre-merge column counts | %s: %s columns | %s: %s columns",
        left_name,
        len(left_columns),
        right_name,
        len(right_columns),
    )

    decision, inference_log = infer_merge_columns_with_ollama(left_name, left_sample_df, right_name, right_sample_df)
    merged_dataset_name = (
        f"{os.path.splitext(os.path.basename(left_name))[0]}"
        f"__{os.path.splitext(os.path.basename(right_name))[0]}"
    )
    common_columns = [column for column in left_columns if column in right_columns]

    merge_columns = _normalize_merge_columns(decision, common_columns)
    left_column = merge_columns[0] if merge_columns else None
    right_column = merge_columns[0] if merge_columns else None
    join_type = decision.get("join_type", "inner")
    if join_type not in {"inner", "left", "right", "outer"}:
        join_type = "inner"

    merge_started_at = time.time()
    output_filename = f"{merged_dataset_name}__merged.csv"
    output_path = os.path.join(MERGED_OUTPUT_DIR, output_filename)
    retry_used = False
    used_chunked_merge = False
    match_breakdown = {}
    logger.info(
        "Merge decision for %s: common_columns=%s selected_merge_columns=%s join_type=%s",
        merged_dataset_name,
        common_columns,
        merge_columns,
        join_type,
    )

    left_processed_info = _write_processed_table_from_csv(left_path, left_name)
    right_processed_info = _write_processed_table_from_csv(right_path, right_name)

    if not common_columns or not merge_columns:
        logger.info(
            "No common columns found for %s. Saved processed copies only at %s and %s.",
            merged_dataset_name,
            left_processed_info["output_path"],
            right_processed_info["output_path"],
        )
        logger.info(
            "Final row counts | %s: %s rows | %s: %s rows | processed outputs: %s=%s rows, %s=%s rows",
            left_name,
            left_processed_info["rows"],
            right_name,
            right_processed_info["rows"],
            left_processed_info["output_filename"],
            left_processed_info["rows"],
            right_processed_info["output_filename"],
            right_processed_info["rows"],
        )
        merge_log = {
            "status": "copied_without_merge",
            "save_status": "copied",
            "merged_dataset_name": merged_dataset_name,
            "output_path": None,
            "output_filename": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "warning": "No common columns were found, so the two tables were copied to data/pro without merging.",
            "copied_tables": [left_processed_info, right_processed_info],
            "merge_summary": {
                "left_dataset": left_name,
                "right_dataset": right_name,
                "merged_dataset_name": merged_dataset_name,
                "left_rows": left_processed_info["rows"],
                "right_rows": right_processed_info["rows"],
                "left_columns": left_processed_info["columns"],
                "right_columns": right_processed_info["columns"],
                "merged_rows": None,
                "merged_columns": None,
                "left_column": None,
                "right_column": None,
                "merge_columns": [],
                "merge_key_column": None,
                "join_type": None,
                "retry_used_after_empty_merge": retry_used,
                "used_chunked_merge": used_chunked_merge,
                "match_breakdown": match_breakdown,
                "common_columns": [],
            },
            "ollama_inference": inference_log,
            "duration_seconds": round(time.time() - merge_started_at, 4),
        }
        return pd.DataFrame(), merge_log

    logger.info(
        "Ollama selected merge columns for %s: %s",
        merged_dataset_name,
        merge_columns,
    )
    logger.info(f"Merge progress for {merged_dataset_name}: 10%")
    logger.info(f"Merge progress for {merged_dataset_name}: 20%")
    merged_rows, merged_columns = _merge_with_duckdb(
        left_path=left_path,
        right_path=right_path,
        output_path=output_path,
        merge_columns=merge_columns,
        join_type=join_type,
        left_name=left_name,
        right_name=right_name,
    )
    merged_df = pd.read_csv(output_path, nrows=0)
    logger.info(
        "Post-merge column count | %s: %s columns",
        output_filename,
        merged_columns,
    )
    logger.info(
        "Final row counts | %s: %s rows | %s: %s rows | %s: %s rows",
        left_name,
        left_processed_info["rows"],
        right_name,
        right_processed_info["rows"],
        output_filename,
        merged_rows,
    )

    merge_log = {
        "status": "completed",
        "save_status": "saved_empty" if merged_rows == 0 else "saved",
        "merged_dataset_name": merged_dataset_name,
        "output_path": output_path,
        "output_filename": output_filename,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "warning": (
            "The merged output contains zero rows because the selected merge keys did not overlap across the two files."
            if merged_rows == 0
            else None
        ),
        "copied_tables": [left_processed_info, right_processed_info],
        "merge_summary": {
            "left_dataset": left_name,
            "right_dataset": right_name,
            "merged_dataset_name": merged_dataset_name,
            "left_rows": left_processed_info["rows"],
            "right_rows": right_processed_info["rows"],
            "left_columns": left_processed_info["columns"],
            "right_columns": right_processed_info["columns"],
            "merged_rows": merged_rows,
            "merged_columns": merged_columns,
            "left_column": left_column,
            "right_column": right_column,
            "merge_columns": merge_columns,
            "merge_key_column": "composite_merge_key" if len(merge_columns) > 1 else merge_columns[0],
            "join_type": join_type,
            "retry_used_after_empty_merge": retry_used,
            "used_chunked_merge": used_chunked_merge,
            "match_breakdown": match_breakdown,
            "common_columns": common_columns,
        },
        "ollama_inference": inference_log,
        "duration_seconds": round(time.time() - merge_started_at, 4),
    }
    logger.info(f"Merge progress for {merged_dataset_name}: 100%")
    return merged_df, merge_log
