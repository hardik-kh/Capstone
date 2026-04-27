# Core ingestion orchestration: validation, loading, cleaning, profiling

import os
import time
import tempfile
import asyncio
import copy
import inspect
from typing import Any, Optional, Callable
from datetime import datetime

import pandas as pd
from fastapi import UploadFile

from src.core.config import UPLOAD_CHUNK_SIZE_BYTES, LARGE_CSV_THRESHOLD_BYTES
from src.core.exceptions import IngestionError
from src.core.logger import get_logger
from src.agents.ingestion_agent.validators import validate_file, validate_dataframe_rows
from src.agents.ingestion_agent.csv_handler import load_csv, load_csv_sample, save_to_bronze, copy_csv_to_bronze
from src.agents.ingestion_agent.excel_handler import load_excel, save_excel_sheet_to_bronze
from src.agents.ingestion_agent.merge_service import merge_all_csv_files
from src.agents.ingestion_agent.profiler import clean_and_profile
from src.agents.statistical_agent.statistical_agent import run_statistical_tests
from src.agents.eda_agent.eda_agent import run_eda
from src.agents.predictive_agent.predictive_agent import run_predictive
from src.agents.reporting_agent.reporting_agent import run_reporting

logger = get_logger("DataIngestionAgent")


def _trace(message: str) -> None:
    ts = datetime.utcnow().isoformat() + "Z"
    print(f"[INGEST_TRACE {ts}] {message}", flush=True)


def _safe_remove_file(path: str, context: str, retries: int = 6, delay_seconds: float = 0.25) -> None:
    """Best-effort temp cleanup on Windows, where file handles may release slightly later."""
    for attempt in range(1, retries + 1):
        try:
            if os.path.exists(path):
                os.remove(path)
            _trace(f"{context}: removed temp file {path}")
            return
        except PermissionError as e:
            if attempt == retries:
                _trace(f"{context}: temp cleanup skipped after {retries} attempts ({e}) path={path}")
                return
            time.sleep(delay_seconds)
        except Exception as e:
            _trace(f"{context}: temp cleanup error ({e}) path={path}")
            return


def _log_percent_progress(task_name: str, percent: int, last_logged_percent: int) -> int:
    normalized_percent = max(0, min(100, percent))
    if normalized_percent >= last_logged_percent + 10 or normalized_percent == 100:
        logger.info(f"{task_name} progress: {normalized_percent}%")
        return normalized_percent
    return last_logged_percent


async def _persist_upload_to_temp(file: UploadFile, suffix: str = "") -> str:
    total_size = getattr(file, "size", None)
    bytes_written = 0
    last_logged_percent = -10
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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


async def _emit_progress(
    progress_callback: Optional[Callable[[dict[str, Any]], Any]],
    payload: dict[str, Any],
) -> None:
    if not progress_callback:
        return
    snapshot = copy.deepcopy(payload)
    maybe_awaitable = progress_callback(snapshot)
    if inspect.isawaitable(maybe_awaitable):
        await maybe_awaitable


async def ingest_files(
    files: list,
    progress_callback: Optional[Callable[[dict[str, Any]], Any]] = None,
) -> dict:
    _trace(f"ingest_files() start: files={len(files)}")
    results: list[dict] = []
    errors: list[dict] = []
    processing_log: dict[str, Any] = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "files_received": [file.filename for file in files],
        "events": [],
    }
    csv_artifacts: list[dict[str, Any]] = []
    deferred_cleanup_paths: list[str] = []
    payload: dict[str, Any] = {
        "tables": [],
        "errors": [],
        "merge_results": [],
        "statistical_results": [],
        "eda_results": [],
        "predictive_results": [],
        "reporting_results": [],
        "processing_log": processing_log,
    }

    _trace("initial progress emit")
    await _emit_progress(progress_callback, payload)

    for file in files:
        logger.info(f"Processing file: {file.filename}")
        _trace(f"file start: {file.filename}")
        tmp_path = None
        file_log: dict[str, Any] = {"file": file.filename, "status": "started", "steps": []}
        processing_log["events"].append(file_log)

        try:
            _trace(f"{file.filename}: validate_file()")
            ext = validate_file(file)
            _trace(f"{file.filename}: extension={ext}")
            file_log["extension"] = ext
            file_log["steps"].append({"step": "validate_file", "status": "completed"})
            start_time = time.time()
            _trace(f"{file.filename}: persist upload to temp start")
            tmp_path = await _persist_upload_to_temp(file, suffix=ext)
            _trace(f"{file.filename}: persist upload complete tmp_path={tmp_path}")
            file_log["tmp_path"] = tmp_path
            file_log["steps"].append({"step": "persist_temp_file", "status": "completed"})

            if ext == ".csv":
                file_size_bytes = os.path.getsize(tmp_path)
                is_large_csv = file_size_bytes >= LARGE_CSV_THRESHOLD_BYTES
                _trace(
                    f"{file.filename}: CSV branch start file_size={file_size_bytes} "
                    f"is_large_csv={is_large_csv}"
                )
                if is_large_csv:
                    _trace(f"{file.filename}: load_csv_sample() start")
                    df, meta = await asyncio.to_thread(load_csv_sample, tmp_path)
                else:
                    _trace(f"{file.filename}: load_csv() start")
                    df, meta = await asyncio.to_thread(load_csv, tmp_path)
                _trace(
                    f"{file.filename}: CSV loaded rows={len(df)} cols={len(df.columns)} meta={meta}"
                )

                file_log["steps"].append({
                    "step": "load_csv_sample" if is_large_csv else "load_csv",
                    "status": "completed",
                    "details": {"rows": int(len(df)), "columns": int(len(df.columns)), "file_size_bytes": file_size_bytes, "sample_based": is_large_csv, **meta},
                })

                if is_large_csv:
                    _trace(f"{file.filename}: copy_csv_to_bronze() start")
                    bronze_info = await asyncio.to_thread(copy_csv_to_bronze, tmp_path, file.filename)
                else:
                    _trace(f"{file.filename}: save_to_bronze() start")
                    bronze_info = await asyncio.to_thread(save_to_bronze, df, file.filename)
                _trace(f"{file.filename}: bronze save done {bronze_info}")
                file_log["steps"].append({"step": "save_to_bronze", "status": "completed", "details": bronze_info})

                _trace(f"{file.filename}: validate_dataframe_rows()")
                validation_metrics = validate_dataframe_rows(df)
                _trace(f"{file.filename}: validation done {validation_metrics}")
                file_log["steps"].append({"step": "validate_rows", "status": "completed", "details": validation_metrics})

                _trace(f"{file.filename}: clean_and_profile() start")
                _, profiling = await asyncio.to_thread(clean_and_profile, df)
                _trace(
                    f"{file.filename}: clean_and_profile done "
                    f"shape_before={profiling.get('shape_before')} shape_after={profiling.get('shape_after')}"
                )
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
                payload["tables"] = results
                _trace(f"{file.filename}: CSV table added; tables_count={len(results)}")
                await _emit_progress(progress_callback, payload)

            else:
                file_log["steps"].append({"step": "load_excel", "status": "started"})
                _trace(f"{file.filename}: Excel branch start ext={ext}")
                await _emit_progress(progress_callback, payload)
                logger.info("Loading Excel workbook: %s", file.filename)
                _trace(f"{file.filename}: load_excel() start")
                sheets = await asyncio.wait_for(
                    asyncio.to_thread(load_excel, tmp_path, ext),
                    timeout=180,
                )
                _trace(f"{file.filename}: load_excel() done sheet_count={len(sheets)}")
                logger.info("Excel workbook loaded: %s (%d sheet(s))", file.filename, len(sheets))
                file_log["steps"].append({"step": "load_excel", "status": "completed", "details": {"sheet_count": len(sheets)}})
                for sheet_name, df in sheets.items():
                    _trace(
                        f"{file.filename}: sheet start name={sheet_name} rows={len(df)} cols={len(df.columns)}"
                    )
                    _trace(f"{file.filename}:{sheet_name}: validate_dataframe_rows()")
                    validation_metrics = validate_dataframe_rows(df)
                    _trace(f"{file.filename}:{sheet_name}: clean_and_profile() start")
                    _, profiling = await asyncio.to_thread(clean_and_profile, df)
                    _trace(f"{file.filename}:{sheet_name}: save_excel_sheet_to_bronze() start")
                    bronze_info = await asyncio.to_thread(save_excel_sheet_to_bronze, df, file.filename, sheet_name)
                    _trace(
                        f"{file.filename}:{sheet_name}: sheet done "
                        f"shape_before={profiling.get('shape_before')} "
                        f"shape_after={profiling.get('shape_after')} "
                        f"bronze={bronze_info.get('output_path')}"
                    )
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
                    payload["tables"] = results
                    _trace(f"{file.filename}: table added for sheet={sheet_name}; tables_count={len(results)}")
                    await _emit_progress(progress_callback, payload)
                file_log["status"] = "completed"
                file_log["latency_seconds"] = round(time.time() - start_time, 4)
                _trace(f"{file.filename}: Excel file completed latency={file_log['latency_seconds']}s")

        except asyncio.TimeoutError:
            err = {
                "file": file.filename,
                "error_code": "EXCEL_READ_TIMEOUT",
                "message": "Excel parsing timed out.",
                "hint": "The workbook may be very large or malformed. Try a smaller file or simplify formulas/formatting.",
            }
            _trace(f"{file.filename}: ERROR EXCEL_READ_TIMEOUT")
            errors.append(err)
            payload["errors"] = errors
            file_log["status"] = "failed"
            file_log["error"] = err
            await _emit_progress(progress_callback, payload)
        except IngestionError as e:
            err = e.to_dict()
            err["file"] = file.filename
            _trace(f"{file.filename}: IngestionError {err}")
            errors.append(err)
            payload["errors"] = errors
            file_log["status"] = "failed"
            file_log["error"] = err
            await _emit_progress(progress_callback, payload)
        except Exception as e:
            logger.exception(f"Unexpected error for file {file.filename}")
            err = {"file": file.filename, "error_code": "UNEXPECTED_ERROR", "message": "An unexpected error occurred.", "hint": str(e)}
            _trace(f"{file.filename}: Unexpected error {err}")
            errors.append(err)
            payload["errors"] = errors
            file_log["status"] = "failed"
            file_log["error"] = err
            await _emit_progress(progress_callback, payload)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                _safe_remove_file(tmp_path, context=file.filename)

    # ── Merge ─────────────────────────────────────────────────────────────────
    _trace(f"merge stage start csv_artifacts={len(csv_artifacts)}")
    merge_results: list[dict] = []
    if len(csv_artifacts) >= 1:
        merge_log_entry: dict[str, Any] = {"step": "merge_csv_files", "status": "started", "files": [a["filename"] for a in csv_artifacts]}
        processing_log["events"].append(merge_log_entry)
        try:
            _trace("merge_all_csv_files() start")
            merge_results = await asyncio.to_thread(merge_all_csv_files, csv_artifacts)
            _trace(f"merge_all_csv_files() done merge_results={len(merge_results)}")
            merge_log_entry["status"] = "completed"
            payload["merge_results"] = merge_results
            await _emit_progress(progress_callback, payload)
        except IngestionError as e:
            err = e.to_dict()
            err["file"] = ",".join(a["filename"] for a in csv_artifacts)
            errors.append(err)
            payload["errors"] = errors
            merge_log_entry["status"] = "failed"
            merge_log_entry["error"] = err
            _trace(f"merge stage IngestionError {err}")
            await _emit_progress(progress_callback, payload)
        except Exception as e:
            err = {"file": ",".join(a["filename"] for a in csv_artifacts), "error_code": "UNEXPECTED_MERGE_ERROR", "message": "Merge failed.", "hint": str(e)}
            errors.append(err)
            payload["errors"] = errors
            merge_log_entry["status"] = "failed"
            merge_log_entry["error"] = err
            _trace(f"merge stage Unexpected error {err}")
            await _emit_progress(progress_callback, payload)
    else:
        processing_log["events"].append({"step": "merge_csv_files", "status": "skipped", "reason": "No CSV files ingested."})
        _trace("merge stage skipped (no csv files)")
        await _emit_progress(progress_callback, payload)

    for path in deferred_cleanup_paths:
        if os.path.exists(path):
            _safe_remove_file(path, context="deferred_cleanup")

    # ── Statistical Analysis ──────────────────────────────────────────────────
    _trace("statistical_analysis stage start")
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

        _trace(f"statistical_analysis datasets prepared count={len(datasets_for_stats)}")
        for dataset_name, df_stat in datasets_for_stats:
            logger.info("Running statistical tests on: %s", dataset_name)
            _trace(f"statistical_analysis running dataset={dataset_name}")
            stat_result = await asyncio.to_thread(run_statistical_tests, dataset_name, df_stat)
            statistical_results.append(stat_result)
            payload["statistical_results"] = statistical_results
            await _emit_progress(progress_callback, payload)

        stat_log_entry["status"] = "completed"
        stat_log_entry["datasets_analyzed"] = [r["dataset_name"] for r in statistical_results]
        _trace(f"statistical_analysis completed results={len(statistical_results)}")
        await _emit_progress(progress_callback, payload)

    except Exception as e:
        logger.exception("Statistical analysis step failed")
        stat_log_entry["status"] = "failed"
        stat_log_entry["error"] = str(e)
        _trace(f"statistical_analysis failed error={e}")
        await _emit_progress(progress_callback, payload)

    # ── EDA ───────────────────────────────────────────────────────────────────
    # Always 3 plots per dataset — LLM picks the most useful ones automatically.
    # datasets_for_stats already contains the right DataFrames (merged or individual).
    _trace("eda stage start")
    eda_results: list[dict] = []
    eda_log_entry: dict[str, Any] = {"step": "eda", "status": "started"}
    processing_log["events"].append(eda_log_entry)

    try:
        for dataset_name, df_eda in datasets_for_stats:
            logger.info("Running EDA on: %s", dataset_name)
            _trace(f"eda running dataset={dataset_name}")
            eda_result = await asyncio.to_thread(run_eda, dataset_name, df_eda, 3)
            # Strip base64 from processing_log to keep it readable — still in eda_results
            eda_results.append(eda_result)
            payload["eda_results"] = eda_results
            await _emit_progress(progress_callback, payload)

        eda_log_entry["status"] = "completed"
        eda_log_entry["datasets_analyzed"] = [r["dataset_name"] for r in eda_results]
        _trace(f"eda completed results={len(eda_results)}")
        await _emit_progress(progress_callback, payload)

    except Exception as e:
        logger.exception("EDA step failed")
        eda_log_entry["status"] = "failed"
        eda_log_entry["error"] = str(e)
        _trace(f"eda failed error={e}")
        await _emit_progress(progress_callback, payload)

    # ── Predictive Analysis ───────────────────────────────────────────────────
    _trace("predictive_analysis stage start")
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
            _trace(f"predictive_analysis running dataset={dataset_name}")
            insights = eda_insights_lookup.get(dataset_name, {})
            pred_result = await asyncio.to_thread(run_predictive, dataset_name, df_pred, insights)
            predictive_results.append(pred_result)
            payload["predictive_results"] = predictive_results
            await _emit_progress(progress_callback, payload)

        pred_log_entry["status"] = "completed"
        pred_log_entry["datasets_analyzed"] = [r["dataset_name"] for r in predictive_results]
        _trace(f"predictive_analysis completed results={len(predictive_results)}")
        await _emit_progress(progress_callback, payload)

    except Exception as e:
        logger.exception("Predictive analysis step failed")
        pred_log_entry["status"] = "failed"
        pred_log_entry["error"] = str(e)
        _trace(f"predictive_analysis failed error={e}")
        await _emit_progress(progress_callback, payload)

    # ── Reporting ─────────────────────────────────────────────────────────────
    _trace("reporting stage start")
    reporting_results: list[dict] = []
    report_log_entry: dict[str, Any] = {"step": "reporting", "status": "started"}
    processing_log["events"].append(report_log_entry)

    try:
        eda_insights_for_report: dict[str, dict] = {
            r["dataset_name"]: r.get("eda_insights", {})
            for r in eda_results
        }
        for dataset_name, df_report in datasets_for_stats:
            logger.info("Running reporting agent on: %s", dataset_name)
            _trace(f"reporting running dataset={dataset_name}")
            insights = eda_insights_for_report.get(dataset_name, {})
            report_result = await asyncio.to_thread(run_reporting, dataset_name, df_report, insights)
            reporting_results.append(report_result)
            payload["reporting_results"] = reporting_results
            await _emit_progress(progress_callback, payload)

        report_log_entry["status"] = "completed"
        report_log_entry["datasets_reported"] = [r["dataset_name"] for r in reporting_results]
        _trace(f"reporting completed results={len(reporting_results)}")
        await _emit_progress(progress_callback, payload)

    except Exception as e:
        logger.exception("Reporting step failed")
        report_log_entry["status"] = "failed"
        report_log_entry["error"] = str(e)
        _trace(f"reporting failed error={e}")
        await _emit_progress(progress_callback, payload)

    processing_log["completed_at"] = datetime.utcnow().isoformat() + "Z"
    processing_log["status"] = "completed_with_errors" if errors else "completed"
    _trace(
        "ingest_files() end "
        f"tables={len(results)} errors={len(errors)} "
        f"merge_results={len(merge_results)} stats={len(statistical_results)} "
        f"eda={len(eda_results)} predictive={len(predictive_results)} reporting={len(reporting_results)}"
    )

    final_payload = {
        "tables":              results,
        "errors":              errors,
        "merge_results":       merge_results,
        "statistical_results": statistical_results,
        "eda_results":         eda_results,
        "predictive_results":  predictive_results,
        "reporting_results":   reporting_results,
        "processing_log":      processing_log,
    }
    await _emit_progress(progress_callback, final_payload)
    return final_payload
