# Reporting / Insights Agent
# - Public entry point: run_reporting(dataset_name, df, eda_insights=None)
# - Orchestrates: analytics_engine → chart_generator → llm_writer → report_composer
# - Returns structured result dict matching the pattern of other agents

from __future__ import annotations

import os
import time
from typing import Any, Optional

import pandas as pd

from core.config import DATA_DIR
from core.logger import get_logger
from agents.reporting_agent.analytics_engine import run_analytics
from agents.reporting_agent.chart_generator import generate_charts
from agents.reporting_agent.llm_writer import write_report_sections
from agents.reporting_agent.report_composer import compose_report

logger = get_logger("ReportingAgent")

REPORTING_OUTPUT_DIR = str(DATA_DIR / "reporting")


def run_reporting(
    dataset_name: str,
    df: pd.DataFrame,
    eda_insights: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Main entry point for the Reporting / Insights Agent.

    Args:
        dataset_name: Human-readable name used in titles and file paths.
        df:           The DataFrame to analyse (full dataset).
        eda_insights: Optional insights dict from the EDA agent.

    Returns a dict with:
        - dataset_name
        - report_html_path (None when PDF is available and HTML fallback is not kept)
        - report_pdf_path  (None if WeasyPrint not installed)
        - charts           list of {type, path, caption}
        - kpis             computed KPI dict
        - findings         full structured analytics output
        - duration_seconds
        - status           "completed" | "failed"
        - error            (only present on failure)
    """
    started_at = time.time()
    logger.info("Reporting agent starting: %s (%d rows, %d cols)", dataset_name, len(df), len(df.columns))

    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in dataset_name).strip("_") or "report"
    out_dir   = os.path.join(REPORTING_OUTPUT_DIR, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # ── Step 1: Analytics engine ──────────────────────────────────────────
        findings = run_analytics(dataset_name, df)
        logger.info("Analytics engine complete for %s", dataset_name)

        # ── Step 2: Chart generation ──────────────────────────────────────────
        charts = generate_charts(findings, out_dir)
        logger.info("Generated %d charts for %s", len(charts), dataset_name)

        # ── Step 3: LLM report writer ─────────────────────────────────────────
        sections = write_report_sections(findings, charts)
        logger.info("Report sections written for %s", dataset_name)

        # ── Step 4: Report composer ───────────────────────────────────────────
        report_paths = compose_report(findings, sections, charts, out_dir, keep_html=False)
        logger.info("Report composed for %s — HTML: %s | PDF: %s",
                    dataset_name, report_paths.get("html_path"), report_paths.get("pdf_path"))

        duration = round(time.time() - started_at, 4)
        logger.info("Reporting agent complete: %s in %.2fs", dataset_name, duration)

        return {
            "dataset_name":     dataset_name,
            "report_html_path": report_paths.get("html_path"),
            "report_pdf_path":  report_paths.get("pdf_path"),
            "charts":           charts,
            "kpis":             findings.get("kpis", {}),
            "findings":         findings,
            "output_dir":       out_dir,
            "duration_seconds": duration,
            "status":           "completed",
        }

    except Exception as e:
        logger.exception("Reporting agent failed for %s: %s", dataset_name, e)
        return {
            "dataset_name":     dataset_name,
            "report_html_path": None,
            "report_pdf_path":  None,
            "charts":           [],
            "kpis":             {},
            "findings":         {},
            "output_dir":       out_dir,
            "duration_seconds": round(time.time() - started_at, 4),
            "status":           "failed",
            "error":            str(e),
        }
