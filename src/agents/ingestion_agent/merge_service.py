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


def _log_merge_progress(task_name: str, percent: int, last_logged_percent: int) -> int:
    """Logs merge progress in 10 percent increments."""
    normalized_percent = max(0, min(100, percent))
    if normalized_percent >= last_logged_percent + 10 or normalized_percent == 100:
        logger.info(f"{task_name} progress: {normalized_percent}%")
        return normalized_percent
    return last_logged_percent


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


def _read_csv_for_merge(
    file_path: str,
    *,
    chunksize: int | None = None,
) -> tuple[pd.DataFrame | None, dict] | tuple[Any, dict]:
    """Reads a CSV for merging, optionally as an iterator of normalized chunks."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    reader = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=delimiter,
        chunksize=chunksize,
        low_memory=False,
    )
    if chunksize is None:
        return _normalize_columns(reader), {"encoding": encoding, "delimiter": delimiter}

    def _generator() -> Any:
        for chunk in reader:
            yield _normalize_columns(chunk)

    return _generator(), {"encoding": encoding, "delimiter": delimiter}


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
    """Chooses the best shared column deterministically when Ollama output is unusable."""
    if "store_nbr" in common_columns:
        return {
            "merge_columns": ["store_nbr"],
            "left_column": "store_nbr",
            "right_column": "store_nbr",
            "confidence": 1.0,
            "join_type": "left",
            "reason": (
                "Fallback heuristic selected store_nbr because it is the preferred shared business key."
            ),
        }
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
    if best_priority > 1:
        reason_prefix = "preferred business key column"
    elif not _is_date_like_column(best_column):
        reason_prefix = "non-date shared column"
    else:
        reason_prefix = "shared date-like column"
    return {
        "merge_columns": [best_column],
        "left_column": best_column,
        "right_column": best_column,
        "confidence": best_overlap,
        "join_type": "left",
        "reason": (
            "Fallback heuristic selected the "
            f"{reason_prefix} with the strongest normalized value overlap among common columns: {best_column}."
        ),
    }


def _normalize_single_merge_column(decision: dict, common_columns: list[str]) -> list[str]:
    """Forces the merge decision down to exactly one shared column."""
    merge_columns = decision.get("merge_columns") or []
    if isinstance(merge_columns, str):
        merge_columns = [merge_columns]

    valid_columns = [column for column in merge_columns if column in common_columns]
    if "store_nbr" in common_columns:
        return ["store_nbr"]
    if valid_columns:
        prioritized = sorted(
            valid_columns,
            key=lambda column: (_business_priority(column), not _is_date_like_column(column), column),
            reverse=True,
        )
        return [prioritized[0]]

    left_column = decision.get("left_column")
    if left_column in common_columns:
        return [left_column]
    return []


def infer_merge_columns_with_ollama(
    left_name: str,
    left_df: pd.DataFrame,
    right_name: str,
    right_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Uses Ollama to choose the best pair of columns for merging two CSVs."""
    common_columns = _shared_columns(left_df, right_df)
    if not common_columns:
        raise MergeInferenceError(
            "The uploaded CSV files do not share any column names, so merge inference cannot enforce same-name keys.",
            hint="Add or rename a common join column in both CSV files before uploading them together.",
        )
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
            "Choose the best merge key for both datasets. If store_nbr is available in both datasets, choose "
            "store_nbr. Otherwise choose exactly one shared column name from allowed_shared_columns, preferring a "
            "non-date business column over a date column. Respond as strict JSON with keys: merge_columns, "
            "left_column, right_column, confidence, join_type, reason. merge_columns must be an array containing "
            "exactly one shared column."
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

    merge_columns = _normalize_single_merge_column(decision, common_columns)

    invalid_shared_choice = (
        len(merge_columns) != 1
        or any(column not in common_columns for column in merge_columns)
    )
    if invalid_shared_choice:
        decision = _fallback_merge_decision(left_df, right_df, common_columns)
        merge_columns = _normalize_single_merge_column(decision, common_columns)
        used_fallback = True

    left_column = merge_columns[0]
    right_column = merge_columns[0]

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
) -> tuple[int, int]:
    """Merges CSV files with DuckDB and writes the result directly to CSV."""
    try:
        import duckdb
    except ImportError as exc:
        raise MergeInferenceError(
            "DuckDB is not installed in the runtime environment.",
            hint="Install it with: python3 -m pip install duckdb",
        ) from exc

    if len(merge_columns) != 1:
        raise MergeInferenceError(
            "DuckDB merge is configured to use exactly one merge column.",
            hint=f"Received merge columns: {merge_columns}",
        )

    merge_column = merge_columns[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    database_path = os.path.join(MERGED_OUTPUT_DIR, "merge_service.duckdb")
    connection = duckdb.connect(database_path)
    try:
        connection.execute("PRAGMA temp_directory='/tmp';")
        logger.info("DuckDB merge progress: 20%")
        join_keyword = join_type.upper()
        output_path_sql = output_path.replace("'", "''")
        left_path_sql = left_path.replace("'", "''")
        right_path_sql = right_path.replace("'", "''")
        merge_column_sql = merge_column.replace('"', '""')
        connection.execute(
            f"""
            COPY (
                SELECT
                    left_table.*,
                    right_table.*,
                    left_table."{merge_column_sql}" AS merge_key
                FROM read_csv_auto('{left_path_sql}', HEADER=TRUE) AS left_table
                {join_keyword} JOIN read_csv_auto('{right_path_sql}', HEADER=TRUE) AS right_table
                ON CAST(left_table."{merge_column_sql}" AS VARCHAR) = CAST(right_table."{merge_column_sql}" AS VARCHAR)
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
    """Merges two CSV files, using chunked processing for large inputs."""
    left_sample_df, _ = _load_csv_sample(left_path)
    right_sample_df, _ = _load_csv_sample(right_path)
    decision, inference_log = infer_merge_columns_with_ollama(left_name, left_sample_df, right_name, right_sample_df)
    logger.info(
        "Ollama selected merge columns for %s__%s: %s",
        os.path.splitext(os.path.basename(left_name))[0],
        os.path.splitext(os.path.basename(right_name))[0],
        decision.get("merge_columns") or [decision["left_column"]],
    )
    merged_dataset_name = (
        f"{os.path.splitext(os.path.basename(left_name))[0]}"
        f"__{os.path.splitext(os.path.basename(right_name))[0]}"
    )
    common_columns = _shared_columns(left_sample_df, right_sample_df)

    merge_columns = decision.get("merge_columns") or [decision["left_column"]]
    left_column = merge_columns[0]
    right_column = merge_columns[0]
    join_type = decision.get("join_type", "inner")
    if join_type not in {"inner", "left", "right", "outer"}:
        join_type = "inner"

    merge_started_at = time.time()
    output_filename = f"{merged_dataset_name}__merged.csv"
    output_path = os.path.join(MERGED_OUTPUT_DIR, output_filename)
    retry_used = False
    used_chunked_merge = False
    logger.info(f"Merge progress for {merged_dataset_name}: 10%")
    logger.info(f"Merge progress for {merged_dataset_name}: 20%")
    merged_rows, merged_columns = _merge_with_duckdb(
        left_path=left_path,
        right_path=right_path,
        output_path=output_path,
        merge_columns=merge_columns,
        join_type=join_type,
    )
    merged_df = pd.read_csv(output_path, nrows=0)
    match_breakdown = {}

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
        "merge_summary": {
            "left_dataset": left_name,
            "right_dataset": right_name,
            "merged_dataset_name": merged_dataset_name,
            "left_rows": None,
            "right_rows": None,
            "merged_rows": merged_rows,
            "merged_columns": merged_columns,
            "left_column": left_column,
            "right_column": right_column,
            "merge_columns": merge_columns,
            "merge_key_column": "merge_key",
            "join_type": join_type,
            "retry_used_after_empty_merge": retry_used,
            "used_chunked_merge": used_chunked_merge,
            "match_breakdown": match_breakdown,
        },
        "ollama_inference": inference_log,
        "duration_seconds": round(time.time() - merge_started_at, 4),
    }
    logger.info(f"Merge progress for {merged_dataset_name}: 100%")
    return merged_df, merge_log
