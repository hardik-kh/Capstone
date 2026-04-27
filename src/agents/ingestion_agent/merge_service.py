# Merge inference and execution utilities powered by Azure OpenAI + DuckDB
# Supports merging any number of CSV files pairwise when common columns exist.
# Files with no common columns are stored as cleaned processed copies.

import json
import os
import re
import time
from datetime import datetime
from typing import Any

import pandas as pd

from src.agents.ingestion_agent.csv_handler import detect_delimiter, detect_encoding
from src.core.config import (
    MERGED_OUTPUT_DIR,
    MERGE_SAMPLE_ROWS,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
)
from src.core.exceptions import MergeInferenceError
from src.core.logger import get_logger

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
    """Normalizes CSV column names to snake_case."""
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
    """Loads a bounded CSV sample for schema inference."""
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
    # Generic ID patterns get highest priority
    high_priority_patterns = ["_id", "_nbr", "_key", "_code", "_num", "_no"]
    if any(normalized.endswith(p) or normalized == p.lstrip("_") for p in high_priority_patterns):
        return 3
    if normalized in {"id", "key", "code"}:
        return 2
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
    """Chooses the best shared merge key deterministically when AI output is unusable."""
    scored_columns: list[tuple[int, float, int, str]] = []
    for column in common_columns:
        overlap_ratio = _normalized_overlap_ratio(left_df[column], right_df[column])
        left_unique = int(left_df[column].nunique(dropna=True))
        right_unique = int(right_df[column].nunique(dropna=True))
        priority = _business_priority(column)
        scored_columns.append((priority, overlap_ratio, min(left_unique, right_unique), column))

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
            "Fallback heuristic selected the shared columns with the strongest business relevance and "
            f"normalized value overlap: {selected_columns}.{second_reason}"
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
            key=lambda c: (_business_priority(c), not _is_date_like_column(c), c),
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
    """Returns a tuned DuckDB in-memory connection for fast CSV merges."""
    try:
        import duckdb
    except ImportError as exc:
        raise MergeInferenceError(
            "DuckDB is not installed in the runtime environment.",
            hint="Install it with: pip install duckdb",
        ) from exc

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

    # In-memory is faster than file-based for single-session merges
    connection = duckdb.connect(":memory:")

    # Use system temp dir — works on both Windows and Linux
    import tempfile
    tmp_dir = tempfile.gettempdir().replace("\\", "/").replace("'", "''")
    connection.execute(f"SET temp_directory='{tmp_dir}';")
    connection.execute("SET memory_limit='4GB';")
    connection.execute("SET threads TO 4;")

    return connection


def _csv_select_with_normalized_columns(file_path: str, table_alias: str) -> tuple[str, list[str]]:
    """Builds a SELECT query that aliases CSV columns to normalized names."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    original_df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, nrows=0, low_memory=False)
    original_columns = list(original_df.columns)
    normalized_columns = _normalize_column_names(original_columns)
    projections = ", ".join(
        [
            f'{table_alias}."{str(orig).replace(chr(34), chr(34)*2)}" AS "{norm}"'
            for orig, norm in zip(original_columns, normalized_columns)
        ]
    )
    csv_path_sql = file_path.replace("'", "''")
    select_sql = (
        f"SELECT {projections} "
        f"FROM read_csv_auto('{csv_path_sql}', HEADER=TRUE) AS {table_alias}"
    )
    return select_sql, normalized_columns


def _write_processed_table_from_csv(file_path: str, original_name: str) -> dict:
    """Writes a normalized processed CSV copy using DuckDB without loading full file into pandas."""
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


def _infer_merge_columns(
    left_name: str,
    left_df: pd.DataFrame,
    right_name: str,
    right_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Uses Azure OpenAI to choose the best shared merge key for two tables.
    Falls back to deterministic heuristics if AI is unavailable or returns unusable output.
    """
    common_columns = _shared_columns(left_df, right_df)
    if not common_columns:
        decision = {
            "merge_columns": [],
            "left_column": None,
            "right_column": None,
            "confidence": 0.0,
            "join_type": None,
            "reason": "No common columns exist across the two tables.",
        }
        inference_log = {
            "provider": "azure_openai",
            "model": AZURE_OPENAI_DEPLOYMENT_NAME,
            "duration_seconds": 0.0,
            "common_columns_considered": [],
            "used_fallback": False,
            "raw_response": None,
            "decision": decision,
        }
        return decision, inference_log

    non_date_common_columns = [c for c in common_columns if not _is_date_like_column(c)]
    preferred_business_columns = [c for c in common_columns if _business_priority(c) > 1]

    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise MergeInferenceError(
            "The openai Python package is not installed.",
            hint="Install it with: pip install openai",
        ) from exc

    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise MergeInferenceError(
            "Azure OpenAI credentials are not configured.",
            hint="Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file.",
        )

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
            "Choose the safest and most accurate way to merge these two tables from any business domain. "
            "Select one shared column when a single column uniquely identifies rows. "
            "Select two shared columns as a composite key when both together produce a more reliable match. "
            "Prefer business identifiers (IDs, codes, keys) and high-overlap columns. "
            "Avoid date-only joins when stronger shared columns exist. "
            "Use only columns from allowed_shared_columns. "
            "Respond as strict JSON with keys: merge_columns, left_column, right_column, confidence, join_type, reason. "
            "merge_columns must be a list of one or two shared column names."
        ),
    }

    started_at = time.time()
    raw_response = ""
    used_fallback = False

    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data engineering assistant. Always respond with valid JSON only, no explanation.",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw_response = response.choices[0].message.content or ""
    except Exception as exc:
        raise MergeInferenceError(
            "Failed while requesting merge-key inference from Azure OpenAI.",
            hint=str(exc),
        ) from exc

    try:
        decision = _extract_json_object(raw_response)
    except Exception:
        decision = _fallback_merge_decision(left_df, right_df, common_columns)
        used_fallback = True

    merge_columns = _normalize_merge_columns(decision, common_columns)
    invalid = len(merge_columns) not in {1, 2} or any(c not in common_columns for c in merge_columns)
    if invalid:
        decision = _fallback_merge_decision(left_df, right_df, common_columns)
        merge_columns = _normalize_merge_columns(decision, common_columns)
        used_fallback = True

    left_column = merge_columns[0] if merge_columns else None

    inference_log = {
        "provider": "azure_openai",
        "model": AZURE_OPENAI_DEPLOYMENT_NAME,
        "duration_seconds": round(time.time() - started_at, 4),
        "common_columns_considered": common_columns,
        "used_fallback": used_fallback,
        "raw_response": raw_response,
        "decision": {
            "merge_columns": merge_columns,
            "left_column": left_column,
            "right_column": left_column,
            "confidence": decision.get("confidence"),
            "join_type": decision.get("join_type", "left"),
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
    """Merges two CSV files using DuckDB streaming — never loads full files into RAM.

    DuckDB reads both CSVs lazily via read_csv_auto, executes a hash join on disk
    when memory is tight, and streams the result directly to the output CSV.
    This is why it matches MySQL speed: the large file is never materialized in Python.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lp = _sanitize_dataset_name(left_name)
    rp = _sanitize_dataset_name(right_name)

    join_map = {"inner": "INNER", "left": "LEFT", "right": "RIGHT", "outer": "FULL OUTER"}
    sql_join = join_map.get(join_type.lower(), "INNER")

    conn = _get_duckdb_connection()
    try:
        # Build column-aliasing SELECT for both sides so names are normalized + prefixed
        left_select, left_cols   = _csv_select_with_normalized_columns(left_path,  "l")
        right_select, right_cols = _csv_select_with_normalized_columns(right_path, "r")

        # Prefix every column with the dataset stem to avoid name collisions
        left_aliases  = ", ".join(f'l2."{c}" AS "{lp}__{c}"'  for c in left_cols)
        right_aliases = ", ".join(
            f'r2."{c}" AS "{rp}__{c}"'
            for c in right_cols
            if c not in merge_columns          # drop duplicate key cols from right side
        )
        all_aliases = left_aliases + (", " + right_aliases if right_aliases else "")

        # Join conditions cast both sides to VARCHAR so int/string mismatches don't fail
        join_conditions = " AND ".join(
            f'CAST(l2."{c}" AS VARCHAR) = CAST(r2."{c}" AS VARCHAR)'
            for c in merge_columns
        )

        output_sql = output_path.replace("'", "''")

        query = f"""
            COPY (
                SELECT {all_aliases}
                FROM ({left_select})  AS l2
                {sql_join} JOIN ({right_select}) AS r2
                ON {join_conditions}
            ) TO '{output_sql}' (HEADER, DELIMITER ',');
        """

        logger.info("DuckDB %s JOIN on %s — streaming directly to disk...", sql_join, merge_columns)
        conn.execute(query)

        # Count rows in the written file without loading it into Python
        row_count = conn.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{output_sql}', HEADER=TRUE)"
        ).fetchone()[0]

        # Count columns from the header
        col_count = len(conn.execute(
            f"SELECT * FROM read_csv_auto('{output_sql}', HEADER=TRUE) LIMIT 0"
        ).description)

        logger.info("Merge complete: %d rows, %d cols written to %s", row_count, col_count, output_path)
        return int(row_count), int(col_count)

    finally:
        conn.close()


FAN_OUT_RATIO_THRESHOLD = 3.0       # skip merge if output > 3x the larger file
FAN_OUT_MAX_ROWS        = 5_000_000  # hard cap — skip if estimated output exceeds 5M rows


def _estimate_fan_out(left_path: str, right_path: str, merge_columns: list[str]) -> dict:
    """Uses DuckDB to cheaply count rows and unique key values on both files.
    Returns estimated output size and whether the merge should be skipped.
    All queries run in milliseconds — no full table scan of values, just COUNT/COUNT DISTINCT.
    """
    conn = _get_duckdb_connection()
    try:
        left_sql  = left_path.replace("'", "''")
        right_sql = right_path.replace("'", "''")

        # Build CONCAT key expression for composite keys
        if len(merge_columns) == 1:
            key_expr = f'"{merge_columns[0]}"'
        else:
            key_expr = " || '||' || ".join(f'CAST("{c}" AS VARCHAR)' for c in merge_columns)

        left_rows, left_unique = conn.execute(f"""
            SELECT COUNT(*), COUNT(DISTINCT {key_expr})
            FROM read_csv_auto('{left_sql}', HEADER=TRUE)
        """).fetchone()

        right_rows, right_unique = conn.execute(f"""
            SELECT COUNT(*), COUNT(DISTINCT {key_expr})
            FROM read_csv_auto('{right_sql}', HEADER=TRUE)
        """).fetchone()

        # Average rows per key on each side
        left_avg  = left_rows  / left_unique  if left_unique  else left_rows
        right_avg = right_rows / right_unique if right_unique else right_rows

        # Estimate matching keys (upper bound = smaller unique set)
        matching_keys    = min(left_unique, right_unique)
        estimated_output = left_avg * right_avg * matching_keys
        max_input_rows   = max(left_rows, right_rows)
        fan_out_ratio    = estimated_output / max_input_rows if max_input_rows else 0

        should_skip = fan_out_ratio > FAN_OUT_RATIO_THRESHOLD or estimated_output > FAN_OUT_MAX_ROWS

        logger.info(
            "Fan-out check — left: %d rows / %d unique keys, right: %d rows / %d unique keys, "
            "estimated output: %d rows, ratio: %.2f — %s",
            left_rows, left_unique, right_rows, right_unique,
            int(estimated_output), fan_out_ratio,
            "SKIP merge" if should_skip else "proceed with merge",
        )

        return {
            "left_rows": int(left_rows),
            "left_unique_keys": int(left_unique),
            "right_rows": int(right_rows),
            "right_unique_keys": int(right_unique),
            "estimated_output_rows": int(estimated_output),
            "fan_out_ratio": round(fan_out_ratio, 4),
            "should_skip": should_skip,
        }
    finally:
        conn.close()


def merge_csv_files(
    left_name: str,
    left_path: str,
    right_name: str,
    right_path: str,
) -> tuple[pd.DataFrame, dict]:
    """Merges two CSV files using AI inference and DuckDB streaming execution.
    DuckDB never loads the full file into Python RAM — it streams both sides and
    spills to disk automatically, making 120 MB+ merges as fast as small ones.
    If no common columns exist, saves processed copies separately instead.
    """
    left_sample_df, _  = _load_csv_sample(left_path)
    right_sample_df, _ = _load_csv_sample(right_path)
    left_columns, _    = _read_csv_columns(left_path)
    right_columns, _   = _read_csv_columns(right_path)

    decision, inference_log = _infer_merge_columns(left_name, left_sample_df, right_name, right_sample_df)

    merged_dataset_name = (
        f"{os.path.splitext(os.path.basename(left_name))[0]}"
        f"__{os.path.splitext(os.path.basename(right_name))[0]}"
    )
    common_columns = [c for c in left_columns if c in right_columns]
    merge_columns  = _normalize_merge_columns(decision, common_columns)
    left_column    = merge_columns[0] if merge_columns else None
    join_type      = decision.get("join_type", "inner")
    if join_type not in {"inner", "left", "right", "outer"}:
        join_type = "inner"

    merge_started_at = time.time()

    if not common_columns or not merge_columns:
        logger.info("No common columns for %s — storing processed copies separately.", merged_dataset_name)
        left_processed_info  = _write_processed_table_from_csv(left_path, left_name)
        right_processed_info = _write_processed_table_from_csv(right_path, right_name)
        merge_log = {
            "status": "stored_separately",
            "merged_dataset_name": merged_dataset_name,
            "output_path": None,
            "output_filename": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "warning": "No common columns were found. Files were cleaned and stored separately in data/pro.",
            "copied_tables": [left_processed_info, right_processed_info],
            "merge_summary": {
                "left_dataset": left_name, "right_dataset": right_name,
                "merged_dataset_name": merged_dataset_name,
                "left_rows": left_processed_info["rows"], "right_rows": right_processed_info["rows"],
                "left_columns": left_processed_info["columns"], "right_columns": right_processed_info["columns"],
                "merged_rows": None, "merged_columns": None,
                "merge_columns": [], "join_type": None, "common_columns": [],
            },
            "ai_inference": inference_log,
            "duration_seconds": round(time.time() - merge_started_at, 4),
        }
        return pd.DataFrame(), merge_log

    # ── Fan-out guard ─────────────────────────────────────────────────────────
    # Run a cheap COUNT/COUNT DISTINCT via DuckDB before touching the full files.
    # If the join would explode row count (low-cardinality key), store separately.
    fan_out = _estimate_fan_out(left_path, right_path, merge_columns)
    if fan_out["should_skip"]:
        logger.warning(
            "Fan-out guard triggered for %s — ratio %.2f exceeds threshold %.1f "
            "(estimated %d output rows). Storing files separately.",
            merged_dataset_name, fan_out["fan_out_ratio"], FAN_OUT_RATIO_THRESHOLD,
            fan_out["estimated_output_rows"],
        )
        left_processed_info  = _write_processed_table_from_csv(left_path, left_name)
        right_processed_info = _write_processed_table_from_csv(right_path, right_name)
        merge_log = {
            "status": "stored_separately",
            "merged_dataset_name": merged_dataset_name,
            "output_path": None,
            "output_filename": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "fan_out_check": fan_out,
            "copied_tables": [left_processed_info, right_processed_info],
            "merge_summary": {
                "left_dataset": left_name, "right_dataset": right_name,
                "merged_dataset_name": merged_dataset_name,
                "left_rows": fan_out["left_rows"], "right_rows": fan_out["right_rows"],
                "left_columns": left_processed_info["columns"], "right_columns": right_processed_info["columns"],
                "merged_rows": None, "merged_columns": None,
                "merge_columns": merge_columns, "join_type": join_type, "common_columns": common_columns,
            },
            "ai_inference": inference_log,
            "duration_seconds": round(time.time() - merge_started_at, 4),
        }
        return pd.DataFrame(), merge_log

    output_filename = f"{merged_dataset_name}__merged.csv"
    output_path     = os.path.join(MERGED_OUTPUT_DIR, output_filename)

    logger.info("Merging %s + %s on %s using %s join", left_name, right_name, merge_columns, join_type)

    # DuckDB streams both files lazily — never materialises a 120 MB CSV into Python RAM.
    # It builds a hash table on the smaller (build) side and streams the larger (probe) side,
    # spilling to disk automatically when memory is tight — same strategy MySQL uses.
    merged_rows, merged_columns = _merge_with_duckdb(
        left_path=left_path,
        right_path=right_path,
        output_path=output_path,
        merge_columns=merge_columns,
        join_type=join_type,
        left_name=left_name,
        right_name=right_name,
    )

    # Write processed copies after the merge so each file is only opened by DuckDB once
    left_processed_info  = _write_processed_table_from_csv(left_path, left_name)
    right_processed_info = _write_processed_table_from_csv(right_path, right_name)

    merge_log = {
        "status": "completed",
        "save_status": "saved_empty" if merged_rows == 0 else "saved",
        "merged_dataset_name": merged_dataset_name,
        "output_path": output_path,
        "output_filename": output_filename,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "warning": (
            "The merged output contains zero rows — the join keys had no overlapping values."
            if merged_rows == 0 else None
        ),
        "fan_out_check": fan_out,
        "copied_tables": [left_processed_info, right_processed_info],
        "merge_summary": {
            "left_dataset": left_name, "right_dataset": right_name,
            "merged_dataset_name": merged_dataset_name,
            "left_rows": left_processed_info["rows"], "right_rows": right_processed_info["rows"],
            "left_columns": left_processed_info["columns"], "right_columns": right_processed_info["columns"],
            "merged_rows": merged_rows, "merged_columns": merged_columns,
            "left_column": left_column, "right_column": left_column,
            "merge_columns": merge_columns,
            "merge_key_column": "composite_merge_key" if len(merge_columns) > 1 else merge_columns[0],
            "join_type": join_type,
            "common_columns": common_columns,
        },
        "ai_inference": inference_log,
        "duration_seconds": round(time.time() - merge_started_at, 4),
    }
    return pd.DataFrame(), merge_log

def merge_all_csv_files(csv_artifacts: list[dict]) -> list[dict]:
    """Merges all uploaded CSV files pairwise where common columns exist.

    Strategy:
    - For 2 files: attempt one merge
    - For N files: attempt merges for every unique pair
    - Files with no common columns are stored separately as processed copies
    - Returns a list of merge result dicts (one per pair attempted)
    """
    results = []
    n = len(csv_artifacts)

    if n == 0:
        return results

    if n == 1:
        # Single file — just write processed copy
        artifact = csv_artifacts[0]
        processed_info = _write_processed_table_from_csv(artifact["tmp_path"], artifact["filename"])
        results.append({
            "status": "stored_separately",
            "files": [artifact["filename"]],
            "output_path": processed_info["output_path"],
            "output_filename": processed_info["output_filename"],
            "rows": processed_info["rows"],
            "columns": processed_info["columns"],
            "warning": "Only one CSV file uploaded — no merge attempted.",
            "copied_tables": [processed_info],
            "merge_summary": None,
        })
        return results

    # N >= 2: try all unique pairs
    paired: set[frozenset] = set()
    for i in range(n):
        for j in range(i + 1, n):
            pair_key = frozenset([csv_artifacts[i]["filename"], csv_artifacts[j]["filename"]])
            if pair_key in paired:
                continue
            paired.add(pair_key)

            left = csv_artifacts[i]
            right = csv_artifacts[j]
            logger.info("Attempting merge: %s <-> %s", left["filename"], right["filename"])

            try:
                _, merge_log = merge_csv_files(
                    left_name=left["filename"],
                    left_path=left["tmp_path"],
                    right_name=right["filename"],
                    right_path=right["tmp_path"],
                )
                result = {
                    "files": [left["filename"], right["filename"]],
                    "status": merge_log["status"],
                    "output_path": merge_log["output_path"],
                    "output_filename": merge_log["output_filename"],
                    "rows": merge_log["merge_summary"].get("merged_rows"),
                    "columns": merge_log["merge_summary"].get("merged_columns"),
                    "warning": merge_log.get("warning"),
                    "copied_tables": merge_log.get("copied_tables", []),
                    "merge_summary": merge_log["merge_summary"],
                    "ai_inference": merge_log.get("ai_inference"),
                }
                results.append(result)
            except Exception as exc:
                logger.error("Merge failed for %s <-> %s: %s", left["filename"], right["filename"], exc)
                results.append({
                    "files": [left["filename"], right["filename"]],
                    "status": "failed",
                    "error": str(exc),
                    "output_path": None,
                    "output_filename": None,
                    "rows": None,
                    "columns": None,
                    "merge_summary": None,
                })

    return results