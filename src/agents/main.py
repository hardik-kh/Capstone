import os
import logging
from typing import Any, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from agents.ingestion_agent.router import router as ingestion_router
from agents.ingestion_agent.csv_handler import load_csv, load_csv_sample
from agents.reporting_agent.router import router as reporting_router
from agents.reporting_agent.reporting_agent import run_reporting
from core.config import LARGE_CSV_THRESHOLD_BYTES, PROCESSED_OUTPUT_DIR
from core.logger import get_logger

logger = get_logger("Main")

app = FastAPI(
    title="Autonomous Analytics",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Home route ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Autonomous Analytics</title>
        </head>
        <body style="font-family: Arial; padding: 40px;">
            <h1>Autonomous Analytics</h1>
            <p>Welcome to the Autonomous Analytics API.</p>
            <p>Visit <a href="/docs">/docs</a> to use the ingestion agent.</p>
        </body>
    </html>
    """

# ── Existing ingestion routes (untouched) ────────────────────────────────────
app.include_router(ingestion_router, prefix="/ingest", tags=["Data Ingestion"])

# ── Reporting routes — HTML preview + PDF download ───────────────────────────
app.include_router(reporting_router, prefix="/reporting", tags=["Reporting"])


# ── Enriched ingest endpoint ─────────────────────────────────────────────────
# Wraps the original /ingest/ pipeline and appends reporting_results.
# ingestion_agent.py is never modified.

def _build_datasets_for_reporting(data: dict) -> list[tuple[str, pd.DataFrame]]:
    """
    Re-derives the list of (dataset_name, DataFrame) pairs from the
    ingestion response — same logic the ingestion agent uses internally
    for stats/EDA/predictive, but done here so we never touch that file.
    """
    merge_results = data.get("merge_results", [])
    tables        = data.get("tables", [])
    datasets      = []

    merged_paths = [
        mr for mr in merge_results
        if mr.get("status") == "completed" and mr.get("output_path")
    ]

    if merged_paths:
        for mr in merged_paths:
            try:
                df, _ = load_csv_sample(mr["output_path"])
                datasets.append((mr["output_filename"], df))
            except Exception as e:
                logger.warning("Could not load merged file for reporting: %s", e)
    else:
        # Build lookup: source_name → output_path from copied_tables
        processed_lookup: dict[str, str] = {}
        for mr in merge_results:
            for ct in mr.get("copied_tables", []):
                src  = ct.get("source_name")
                path = ct.get("output_path")
                if src and path:
                    processed_lookup[src] = path

        for table in tables:
            try:
                if table["source_type"] == "csv":
                    processed_path = processed_lookup.get(table["table_name"])
                    if processed_path and os.path.exists(processed_path):
                        df, _ = load_csv_sample(processed_path)
                    else:
                        preview = table.get("profiling", {}).get("preview", [])
                        df = pd.DataFrame(preview) if preview else None
                    if df is not None:
                        datasets.append((table["table_name"], df))
                elif table["source_type"] == "excel":
                    preview = table.get("profiling", {}).get("preview", [])
                    if preview:
                        datasets.append((table["table_name"], pd.DataFrame(preview)))
            except Exception as e:
                logger.warning("Could not prepare %s for reporting: %s", table.get("table_name"), e)

    return datasets


@app.post("/ingest-and-report/", tags=["Data Ingestion"])
async def ingest_and_report(files: List[UploadFile] = File(...)):
    """
    Runs the full ingestion pipeline (identical to /ingest/) then appends
    reporting_results from the Reporting Agent.
    ingestion_agent.py is not modified — reporting is bolted on here.
    """
    from agents.ingestion_agent.ingestion_agent import ingest_files
    data: dict[str, Any] = await ingest_files(files)

    # ── Reporting step ────────────────────────────────────────────────────────
    reporting_results: list[dict] = []
    try:
        datasets = _build_datasets_for_reporting(data)

        eda_insights_lookup: dict[str, dict] = {
            r["dataset_name"]: r.get("eda_insights", {})
            for r in data.get("eda_results", [])
        }

        for dataset_name, df in datasets:
            logger.info("Running reporting agent on: %s", dataset_name)
            insights = eda_insights_lookup.get(dataset_name, {})
            result = run_reporting(dataset_name, df, eda_insights=insights)
            reporting_results.append(result)

    except Exception as e:
        logger.exception("Reporting step failed in main.py: %s", e)

    data["reporting_results"] = reporting_results
    return data
