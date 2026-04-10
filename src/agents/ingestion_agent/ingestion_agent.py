# Core ingestion orchestration: validation, loading, cleaning, profiling

import os
import time
import tempfile
from typing import Any, Optional
from datetime import datetime

import pandas as pd
from fastapi import UploadFile

from core.config import UPLOAD_CHUNK_SIZE_BYTES, LARGE_CSV_THRESHOLD_BYTES
from core.exceptions import IngestionError
from core.logger import get_logger
from agents.ingestion_agent.validators import validate_file, validate_dataframe_rows
from agents.ingestion_agent.csv_handler import load_csv, load_csv_sample, save_to_bronze, copy_csv_to_bronze
from agents.ingestion_agent.excel_handler import load_excel, save_excel_sheet_to_bronze
from agents.ingestion_agent.merge_service import merge_all_csv_files
from agents.ingestion_agent.profiler import clean_and_profile
from agents.statistical_agent.statistical_agent import run_statistical_tests
from agents.eda_agent.eda_agent import run_eda
from agents.predictive_agent.predictive_agent import run_predictive

logger = get_logger("DataIngestionAgent")


def _log_percent_progress(task_name: str, percent: int, last_logged_percent: int) -> int:
    normalized_percent = max(0, min(100, percent))
    if normalized_percent >= last_logged_percent + 10 or normalized_percent == 100:
        logger.info(f"{task_name} progress: {normalized_percent}%")
        return normalized_percent
    return last_logged_percent


async def _persist_upload_to_temp(file: UploadFile) -> str:
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
                    f"Upload stream for {file.filename}", percent, last_logged_percent,
                )
        return tmp.name


def _build_table_entry(
    table_name: str,
    profiling: dict,
    meta: dict,
    source_type: str,
    sheet_name: Optional[str] = None,
) -> dict:
    return {
        "table_name": table_name,
        "source_type": source_type,
        "sheet_name": sheet_name,
        "profiling": profiling,
        "ingestion_meta": meta,
    }


async def ingest_files(files: list) -> dict:
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
        file_log: dict[str, Any] = {"file": file.filename, "status": "started", "steps": []}
        processing_log["events"].append(file_log)

        try:
            ext = validate_file(file)
            file_log["extension"] = ext
            file_log["steps"].append({"step": "validate_file", "status": "completed"})
            start_time = time.time()
            tmp_path = await _persist_upload_to_temp(file)
            file_log["tmp_path"] = tmp_path
            file_log["steps"].append({"step": "persist_temp_file", "status": "completed"})

            if ext == ".csv":
                file_size_bytes = os.path.getsize(tmp_path)
                is_large_csv = file_size_bytes >= LARGE_CSV_THRESHOLD_BYTES
                df, meta = load_csv_sample(tmp_path) if is_large_csv else load_csv(tmp_path)

                file_log["steps"].append({
                    "step": "load_csv_sample" if is_large_csv else "load_csv",
                    "status": "completed",
                    "details": {"rows": int(len(df)), "columns": int(len(df.columns)), "file_size_bytes": file_size_bytes, "sample_based": is_large_csv, **meta},
                })

                bronze_info = copy_csv_to_bronze(tmp_path, file.filename) if is_large_csv else save_to_bronze(df, file.filename)
                file_log["steps"].append({"step": "save_to_bronze", "status": "completed", "details": bronze_info})

                validation_metrics = validate_dataframe_rows(df)
                file_log["steps"].append({"step": "validate_rows", "status": "completed", "details": validation_metrics})

                _, profiling = clean_and_profile(df)
                file_log["steps"].append({
                    "step": "clean_and_profile", "status": "completed",
                    "details": {"shape_before": profiling["shape_before"], "shape_after": profiling["shape_after"], "outliers_detected": len(profiling.get("outlier_report", {}))},
                })

                latency_seconds = round(time.time() - start_time, 4)
                file_log["status"] = "completed"
                file_log["latency_seconds"] = latency_seconds

                csv_artifacts.append({"filename": file.filename, "tmp_path": tmp_path})
                deferred_cleanup_paths.append(tmp_path)
                tmp_path = None

                results.append(_build_table_entry(
                    table_name=file.filename,
                    profiling={**profiling, "validation_metrics": validation_metrics, "latency_seconds": latency_seconds},
                    meta={**meta, "bronze_info": bronze_info},
                    source_type="csv",
                ))

            else:
                sheets = load_excel(tmp_path)
                file_log["steps"].append({"step": "load_excel", "status": "completed", "details": {"sheet_count": len(sheets)}})
                for sheet_name, df in sheets.items():
                    validation_metrics = validate_dataframe_rows(df)
                    _, profiling = clean_and_profile(df)
                    bronze_info = save_excel_sheet_to_bronze(df, file.filename, sheet_name)
                    file_log["steps"].append({
                        "step": "clean_and_profile_sheet", "status": "completed",
                        "details": {"sheet_name": sheet_name, "shape_before": profiling["shape_before"], "shape_after": profiling["shape_after"]},
                    })
                    results.append(_build_table_entry(
                        table_name=f"{file.filename}__{sheet_name}",
                        profiling={**profiling, "validation_metrics": validation_metrics},
                        meta={"bronze_info": bronze_info},
                        source_type="excel",
                        sheet_name=sheet_name,
                    ))
                file_log["status"] = "completed"
                file_log["latency_seconds"] = round(time.time() - start_time, 4)

        except IngestionError as e:
            err = e.to_dict()
            err["file"] = file.filename
            errors.append(err)
            file_log["status"] = "failed"
            file_log["error"] = err
        except Exception as e:
            logger.exception(f"Unexpected error for file {file.filename}")
            err = {"file": file.filename, "error_code": "UNEXPECTED_ERROR", "message": "An unexpected error occurred.", "hint": str(e)}
            errors.append(err)
            file_log["status"] = "failed"
            file_log["error"] = err
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Merge ─────────────────────────────────────────────────────────────────
    merge_results: list[dict] = []
    if len(csv_artifacts) >= 1:
        merge_log_entry: dict[str, Any] = {"step": "merge_csv_files", "status": "started", "files": [a["filename"] for a in csv_artifacts]}
        processing_log["events"].append(merge_log_entry)
        try:
            merge_results = merge_all_csv_files(csv_artifacts)
            merge_log_entry["status"] = "completed"
        except IngestionError as e:
            err = e.to_dict()
            err["file"] = ",".join(a["filename"] for a in csv_artifacts)
            errors.append(err)
            merge_log_entry["status"] = "failed"
            merge_log_entry["error"] = err
        except Exception as e:
            err = {"file": ",".join(a["filename"] for a in csv_artifacts), "error_code": "UNEXPECTED_MERGE_ERROR", "message": "Merge failed.", "hint": str(e)}
            errors.append(err)
            merge_log_entry["status"] = "failed"
            merge_log_entry["error"] = err
    else:
        processing_log["events"].append({"step": "merge_csv_files", "status": "skipped", "reason": "No CSV files ingested."})

    processing_log["completed_at"] = datetime.utcnow().isoformat() + "Z"
    processing_log["status"] = "completed_with_errors" if errors else "completed"

    for path in deferred_cleanup_paths:
        if os.path.exists(path):
            os.remove(path)

    # ── Statistical Analysis ──────────────────────────────────────────────────
    statistical_results: list[dict] = []
    stat_log_entry: dict[str, Any] = {"step": "statistical_analysis", "status": "started"}
    processing_log["events"].append(stat_log_entry)

    try:
        datasets_for_stats: list[tuple[str, Any]] = []
        merged_paths = [mr for mr in merge_results if mr.get("status") == "completed" and mr.get("output_path")]

        if merged_paths:
            # Merge succeeded — run stats on merged output(s)
            for mr in merged_paths:
                try:
                    merged_df, _ = load_csv_sample(mr["output_path"])
                    datasets_for_stats.append((mr["output_filename"], merged_df))
                except Exception as e:
                    logger.warning("Could not load merged file for stats: %s", e)
        else:
            # No merge (no common cols, fan-out guard triggered, or single file) —
            # run stats on each individual processed file independently.
            # Build a lookup: source_name -> output_path from all copied_tables across all merge results
            processed_lookup: dict[str, str] = {}
            for mr in merge_results:
                for ct in mr.get("copied_tables", []):
                    src = ct.get("source_name")
                    path = ct.get("output_path")
                    if src and path:
                        processed_lookup[src] = path

            for table in results:
                try:
                    if table["source_type"] == "csv":
                        processed_path = processed_lookup.get(table["table_name"])
                        if processed_path and os.path.exists(processed_path):
                            # Use sample for large files to keep stats fast
                            df_stat, _ = load_csv_sample(processed_path)
                        else:
                            preview = table["profiling"].get("preview", [])
                            df_stat = pd.DataFrame(preview) if preview else None
                        if df_stat is not None:
                            datasets_for_stats.append((table["table_name"], df_stat))
                    elif table["source_type"] == "excel":
                        preview = table["profiling"].get("preview", [])
                        if preview:
                            datasets_for_stats.append((table["table_name"], pd.DataFrame(preview)))
                except Exception as e:
                    logger.warning("Could not prepare %s for stats: %s", table["table_name"], e)

        for dataset_name, df_stat in datasets_for_stats:
            logger.info("Running statistical tests on: %s", dataset_name)
            stat_result = run_statistical_tests(dataset_name, df_stat)
            statistical_results.append(stat_result)

        stat_log_entry["status"] = "completed"
        stat_log_entry["datasets_analyzed"] = [r["dataset_name"] for r in statistical_results]

    except Exception as e:
        logger.exception("Statistical analysis step failed")
        stat_log_entry["status"] = "failed"
        stat_log_entry["error"] = str(e)

    # ── EDA ───────────────────────────────────────────────────────────────────
    # Always 3 plots per dataset — LLM picks the most useful ones automatically.
    # datasets_for_stats already contains the right DataFrames (merged or individual).
    eda_results: list[dict] = []
    eda_log_entry: dict[str, Any] = {"step": "eda", "status": "started"}
    processing_log["events"].append(eda_log_entry)

    try:
        for dataset_name, df_eda in datasets_for_stats:
            logger.info("Running EDA on: %s", dataset_name)
            eda_result = run_eda(dataset_name, df_eda, n_plots=3)
            # Strip base64 from processing_log to keep it readable — still in eda_results
            eda_results.append(eda_result)

        eda_log_entry["status"] = "completed"
        eda_log_entry["datasets_analyzed"] = [r["dataset_name"] for r in eda_results]

    except Exception as e:
        logger.exception("EDA step failed")
        eda_log_entry["status"] = "failed"
        eda_log_entry["error"] = str(e)

    # ── Predictive Analysis ───────────────────────────────────────────────────
    predictive_results: list[dict] = []
    pred_log_entry: dict[str, Any] = {"step": "predictive_analysis", "status": "started"}
    processing_log["events"].append(pred_log_entry)

    try:
        # Build a lookup of eda_insights per dataset for the predictive agent
        eda_insights_lookup: dict[str, dict] = {
            r["dataset_name"]: r.get("eda_insights", {})
            for r in eda_results
        }

        for dataset_name, df_pred in datasets_for_stats:
            logger.info("Running predictive analysis on: %s", dataset_name)
            insights = eda_insights_lookup.get(dataset_name, {})
            pred_result = run_predictive(dataset_name, df_pred, eda_insights=insights)
            predictive_results.append(pred_result)

        pred_log_entry["status"] = "completed"
        pred_log_entry["datasets_analyzed"] = [r["dataset_name"] for r in predictive_results]

    except Exception as e:
        logger.exception("Predictive analysis step failed")
        pred_log_entry["status"] = "failed"
        pred_log_entry["error"] = str(e)

    return {
        "tables":              results,
        "errors":              errors,
        "merge_results":       merge_results,
        "statistical_results": statistical_results,
        "eda_results":         eda_results,
        "predictive_results":  predictive_results,
        "processing_log":      processing_log,
    }