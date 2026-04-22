import os
from typing import Any, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from agents.ingestion_agent.router import router as ingestion_router
from agents.ingestion_agent.csv_handler import load_csv, load_csv_sample
from agents.reporting_agent.router import router as reporting_router
from agents.reporting_agent.reporting_agent import run_reporting
from core.config import PROCESSED_OUTPUT_DIR
from core.logger import get_logger

logger = get_logger("Main")

app = FastAPI(title="Autonomous Analytics", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Home ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    return """<html><body style="font-family:Arial;padding:40px;">
    <h1>Autonomous Analytics</h1>
    <p>Visit <a href="/docs">/docs</a> for API docs.</p>
    </body></html>"""

# ── Existing routes (untouched) ───────────────────────────────────────────────
app.include_router(ingestion_router, prefix="/ingest",     tags=["Data Ingestion"])
app.include_router(reporting_router,  prefix="/reporting",  tags=["Reporting"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _datasets_from_response(data: dict) -> list[tuple[str, pd.DataFrame]]:
    """
    Derives (clean_dataset_name, DataFrame) pairs from the ingestion response.
    Priority: merged output file → individual processed files → profiling preview.
    clean_dataset_name has no file extension so it matches the reporting router lookup.
    """
    merge_results = data.get("merge_results", [])
    tables        = data.get("tables", [])
    datasets: list[tuple[str, pd.DataFrame]] = []

    # ── Merged files (happy path) ─────────────────────────────────────────────
    for mr in merge_results:
        if mr.get("status") == "completed" and mr.get("output_path"):
            path = mr["output_path"]
            if os.path.exists(path):
                try:
                    file_size = os.path.getsize(path)
                    # Use full load for files under 200MB so time filter works correctly
                    # load_csv_sample reads only first N rows (oldest data) which breaks recency filter
                    if file_size < 200 * 1024 * 1024:
                        df, _ = load_csv(path)
                    else:
                        df, _ = load_csv_sample(path)
                    clean = os.path.splitext(mr["output_filename"])[0]
                    datasets.append((clean, df))
                    logger.info("Reporting: loaded merged file %s (%d rows, %.1fMB)", clean, len(df), file_size/1e6)
                except Exception as e:
                    logger.warning("Reporting: could not load merged file %s — %s", path, e)
            else:
                logger.warning("Reporting: merged path does not exist: %s", path)

    if datasets:
        return datasets

    # ── Individual processed files (fallback) ─────────────────────────────────
    processed_lookup: dict[str, str] = {}
    for mr in merge_results:
        for ct in mr.get("copied_tables", []):
            src  = ct.get("source_name")
            path = ct.get("output_path")
            if src and path:
                processed_lookup[src] = path

    for table in tables:
        name = table.get("table_name", "")
        clean = os.path.splitext(name)[0]
        try:
            if table["source_type"] == "csv":
                path = processed_lookup.get(name)
                if path and os.path.exists(path):
                    file_size = os.path.getsize(path)
                    if file_size < 200 * 1024 * 1024:
                        df, _ = load_csv(path)
                    else:
                        df, _ = load_csv_sample(path)
                    datasets.append((clean, df))
                    logger.info("Reporting: loaded processed file %s (%d rows)", clean, len(df))
                    continue
            # Last resort: use profiling preview (only 10 rows — limited analysis)
            preview = table.get("profiling", {}).get("preview", [])
            if preview:
                datasets.append((clean, pd.DataFrame(preview)))
                logger.warning("Reporting: using 10-row preview for %s — processed file not found", clean)
        except Exception as e:
            logger.warning("Reporting: could not prepare %s — %s", clean, e)

    return datasets


# ── Enriched endpoint ─────────────────────────────────────────────────────────
@app.post("/ingest-and-report/", tags=["Data Ingestion"])
async def ingest_and_report(
    files: List[UploadFile] = File(...),
    reporting_months: int = Form(default=12),
):
    """
    Runs the full ingestion pipeline then appends reporting_results.
    ingestion_agent.py is never modified.
    """
    from agents.ingestion_agent.ingestion_agent import ingest_files
    data: dict[str, Any] = await ingest_files(files)

    reporting_results: list[dict] = []
    try:
        datasets = _datasets_from_response(data)
        logger.info("Reporting: %d dataset(s) to process", len(datasets))

        eda_lookup: dict[str, dict] = {
            r["dataset_name"]: r.get("eda_insights", {})
            for r in data.get("eda_results", [])
        }

        for clean_name, df in datasets:
            logger.info("Reporting: running agent on '%s' (%d rows)", clean_name, len(df))
            insights = eda_lookup.get(clean_name, {})
            result = run_reporting(
                clean_name, df,
                eda_insights=insights,
                reporting_months=reporting_months,
            )
            if result.get("status") == "failed":
                logger.error("Reporting: agent failed for '%s' — %s", clean_name, result.get("error"))
            else:
                logger.info("Reporting: completed '%s' — html=%s pdf=%s",
                            clean_name, result.get("report_html_path"), result.get("report_pdf_path"))
            reporting_results.append(result)

    except Exception as e:
        logger.exception("Reporting step failed: %s", e)

    data["reporting_results"] = reporting_results
    return data