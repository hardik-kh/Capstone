# Core ingestion orchestration: validation, loading, cleaning, profiling

import os
import time
import tempfile
from typing import Any
from datetime import datetime

from fastapi import UploadFile

from core.config import UPLOAD_CHUNK_SIZE_BYTES
from core.config import LARGE_CSV_THRESHOLD_BYTES
from core.exceptions import IngestionError
from core.logger import get_logger
from agents.ingestion_agent.validators import validate_file
from agents.ingestion_agent.validators import validate_dataframe_rows
from agents.ingestion_agent.csv_handler import load_csv, load_csv_sample, save_to_bronze, copy_csv_to_bronze
from agents.ingestion_agent.excel_handler import load_excel
from agents.ingestion_agent.merge_service import merge_csv_files
from agents.ingestion_agent.profiler import clean_and_profile

logger = get_logger("DataIngestionAgent")


def _log_percent_progress(task_name: str, percent: int, last_logged_percent: int) -> int:
    """Logs progress updates in 10 percent increments."""
    normalized_percent = max(0, min(100, percent))
    if normalized_percent >= last_logged_percent + 10 or normalized_percent == 100:
        logger.info(f"{task_name} progress: {normalized_percent}%")
        return normalized_percent
    return last_logged_percent


async def _persist_upload_to_temp(file: UploadFile) -> str:
    """Streams an uploaded file to a temp file without loading it all into memory."""
    total_size = getattr(file, "size", None)
    bytes_written = 0
    last_logged_percent = -10
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            tmp.write(chunk)
            bytes_written += len(chunk)
            if total_size:
                percent = int((bytes_written / total_size) * 100)
                last_logged_percent = _log_percent_progress(
                    f"Upload stream for {file.filename}",
                    percent,
                    last_logged_percent,
                )
        if total_size:
            _log_percent_progress(f"Upload stream for {file.filename}", 100, last_logged_percent)
        else:
            logger.info(f"Upload stream for {file.filename}: wrote {bytes_written} bytes")
        return tmp.name


def _build_table_entry(
    table_name: str,
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
    processing_log: dict[str, Any] = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "files_received": [file.filename for file in files],
        "events": [],
    }
    csv_artifacts: list[dict[str, Any]] = []
    deferred_cleanup_paths: list[str] = []

    for file in files:
        logger.info(f"Processing file: {file.filename}")
        tmp_path = None
        file_log: dict[str, Any] = {
            "file": file.filename,
            "status": "started",
            "steps": [],
        }
        processing_log["events"].append(file_log)

        try:
            ext = validate_file(file)
            file_log["extension"] = ext
            file_log["steps"].append({"step": "validate_file", "status": "completed"})
            start_time = time.time()
            tmp_path = await _persist_upload_to_temp(file)
            file_log["tmp_path"] = tmp_path
            file_log["steps"].append({"step": "persist_temp_file", "status": "completed"})
            logger.info(f"File processing progress for {file.filename}: 20%")

            if ext == ".csv":
                file_size_bytes = os.path.getsize(tmp_path)
                is_large_csv = file_size_bytes >= LARGE_CSV_THRESHOLD_BYTES
                if is_large_csv:
                    df, meta = load_csv_sample(tmp_path)
                else:
                    df, meta = load_csv(tmp_path)
                file_log["steps"].append(
                    {
                        "step": "load_csv_sample" if is_large_csv else "load_csv",
                        "status": "completed",
                        "details": {
                            "rows": int(len(df)),
                            "columns": int(len(df.columns)),
                            "file_size_bytes": file_size_bytes,
                            "sample_based": is_large_csv,
                            **meta,
                        },
                    }
                )
                logger.info(f"File processing progress for {file.filename}: 40%")
                bronze_info = (
                    copy_csv_to_bronze(tmp_path, file.filename)
                    if is_large_csv
                    else save_to_bronze(df, file.filename)
                )
                file_log["steps"].append(
                    {
                        "step": "save_to_bronze",
                        "status": "completed",
                        "details": bronze_info,
                    }
                )
                logger.info(f"File processing progress for {file.filename}: 60%")

                validation_metrics = validate_dataframe_rows(df)
                file_log["steps"].append(
                    {
                        "step": "validate_rows",
                        "status": "completed",
                        "details": validation_metrics,
                    }
                )
                logger.info(f"File processing progress for {file.filename}: 80%")

                _, profiling = clean_and_profile(df)
                file_log["steps"].append(
                    {
                        "step": "clean_and_profile",
                        "status": "completed",
                        "details": {
                            "shape_before": profiling["shape_before"],
                            "shape_after": profiling["shape_after"],
                        },
                    }
                )

                end_time = time.time()
                latency_seconds = round(end_time - start_time, 4)
                file_log["status"] = "completed"
                file_log["latency_seconds"] = latency_seconds
                logger.info(f"File processing progress for {file.filename}: 100%")

                csv_artifacts.append(
                    {
                        "filename": file.filename,
                        "tmp_path": tmp_path,
                    }
                )
                deferred_cleanup_paths.append(tmp_path)
                tmp_path = None

                results.append(
                    _build_table_entry(
                        table_name=file.filename,
                        profiling={
                            **profiling,
                            "validation_metrics": validation_metrics,
                            "latency_seconds": latency_seconds,
                        },
                        meta={**meta, "bronze_info": bronze_info},
                        source_type="csv",
                    )
                )
            else:
                sheets = load_excel(tmp_path)
                file_log["steps"].append(
                    {
                        "step": "load_excel",
                        "status": "completed",
                        "details": {"sheet_count": len(sheets)},
                    }
                )
                logger.info(f"File processing progress for {file.filename}: 50%")
                for sheet_name, df in sheets.items():
                    _, profiling = clean_and_profile(df)
                    file_log["steps"].append(
                        {
                            "step": "clean_and_profile_sheet",
                            "status": "completed",
                            "details": {
                                "sheet_name": sheet_name,
                                "shape_before": profiling["shape_before"],
                                "shape_after": profiling["shape_after"],
                            },
                        }
                    )
                    results.append(
                        _build_table_entry(
                            table_name=f"{file.filename}__{sheet_name}",
                            profiling=profiling,
                            meta={},
                            source_type="excel",
                            sheet_name=sheet_name,
                        )
                    )
                file_log["status"] = "completed"
                file_log["latency_seconds"] = round(time.time() - start_time, 4)
                logger.info(f"File processing progress for {file.filename}: 100%")

        except IngestionError as e:
            logger.error(f"Ingestion error for file {file.filename}: {e}")
            err = e.to_dict()
            err["file"] = file.filename
            errors.append(err)
            file_log["status"] = "failed"
            file_log["error"] = err

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
            file_log["status"] = "failed"
            file_log["error"] = errors[-1]

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                file_log["steps"].append({"step": "cleanup_temp_file", "status": "completed"})

    merge_result: dict[str, Any] | None = None
    if len(csv_artifacts) == 2:
        merge_log_entry: dict[str, Any] = {
            "step": "merge_csv_pair",
            "status": "started",
            "files": [artifact["filename"] for artifact in csv_artifacts],
        }
        processing_log["events"].append(merge_log_entry)
        try:
            _, merge_log = merge_csv_files(
                left_name=csv_artifacts[0]["filename"],
                left_path=csv_artifacts[0]["tmp_path"],
                right_name=csv_artifacts[1]["filename"],
                right_path=csv_artifacts[1]["tmp_path"],
            )
            merge_result = {
                "output_path": merge_log["output_path"],
                "output_filename": merge_log["output_filename"],
                "rows": merge_log["merge_summary"]["merged_rows"],
                "columns": merge_log["merge_summary"]["merged_columns"],
                "copied_tables": merge_log.get("copied_tables", []),
                "merge_summary": merge_log["merge_summary"],
            }
            merge_log_entry.update(merge_log)
        except IngestionError as e:
            err = e.to_dict()
            err["file"] = ",".join(artifact["filename"] for artifact in csv_artifacts)
            errors.append(err)
            merge_log_entry["status"] = "failed"
            merge_log_entry["error"] = err
        except Exception as e:
            err = {
                "file": ",".join(artifact["filename"] for artifact in csv_artifacts),
                "error_code": "UNEXPECTED_MERGE_ERROR",
                "message": "An unexpected error occurred during CSV merge.",
                "hint": str(e),
            }
            errors.append(err)
            merge_log_entry["status"] = "failed"
            merge_log_entry["error"] = err
    else:
        processing_log["events"].append(
            {
                "step": "merge_csv_pair",
                "status": "skipped",
                "reason": "Merge runs only when exactly two CSV files are uploaded successfully.",
                "csv_files_available": len(csv_artifacts),
            }
        )

    processing_log["completed_at"] = datetime.utcnow().isoformat() + "Z"
    processing_log["status"] = "completed_with_errors" if errors else "completed"

    for path in deferred_cleanup_paths:
        if os.path.exists(path):
            os.remove(path)

    return {
        "tables": results,
        "errors": errors,
        "merge_result": merge_result,
        "processing_log": processing_log,
    }
