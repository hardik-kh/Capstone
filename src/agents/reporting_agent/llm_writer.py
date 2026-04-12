# LLM Report Writer
# - Builds a compact, structured prompt from findings (never the raw CSV)
# - Calls Azure OpenAI for: executive summary, key findings, risks, recommendations
# - Falls back to template strings if LLM is unavailable or fails

from __future__ import annotations

import json
from typing import Any

from core.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
)
from core.logger import get_logger

logger = get_logger("LLMReportWriter")


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(findings: dict, charts: list[dict]) -> str:
    kpis = findings.get("kpis", {})
    top_products  = findings.get("top_5_products", [])
    top_regions   = findings.get("top_5_regions", [])
    top_cats      = findings.get("category_contribution", [])[:5]
    bw_period     = findings.get("best_worst_period", {})
    anomalies     = findings.get("anomalies", [])
    pareto        = findings.get("pareto", {})
    reporting_period = findings.get("reporting_period", "All available data")
    chart_captions   = [c.get("caption", "") for c in charts if c.get("caption")]

    prompt_data = {
        "task": (
            "You are a senior business analyst writing a concise 2-page executive report. "
            "Produce exactly 4 sections: executive_summary, key_findings, risks_and_anomalies, "
            "recommended_actions. Each section is plain-English prose (2–4 sentences). "
            "Be specific — reference actual numbers from the KPI table. "
            "Tone: professional, direct, actionable. No jargon. No markdown headers inside values."
        ),
        "dataset_name":      findings.get("dataset_name"),
        "reporting_period":  reporting_period,
        "row_count":         findings.get("row_count"),
        "kpi_table": {
            "total_sales":              kpis.get("total_sales"),
            "total_orders":             kpis.get("total_orders"),
            "units_sold":               kpis.get("units_sold"),
            "average_order_value":      kpis.get("average_order_value"),
            "growth_over_prior_period": kpis.get("growth_over_previous_period"),
            "trend_summary":            kpis.get("trend_summary"),
        },
        "top_5_products":  [{"name": p["name"], "sales": p["sales"], "share_pct": p["share_pct"]} for p in top_products],
        "top_5_regions":   [{"name": r["name"], "sales": r["sales"], "share_pct": r["share_pct"]} for r in top_regions],
        "top_5_categories":[{"name": c["name"], "share_pct": c["share_pct"]} for c in top_cats],
        "best_period":     bw_period.get("best_period"),
        "worst_period":    bw_period.get("worst_period"),
        "pareto_insight":  pareto,
        "anomalies":       anomalies[:5],  # send at most 5 to keep prompt tight
        "seasonality":     findings.get("seasonality_hint"),
        "chart_captions":  chart_captions,
        "output_format": {
            "description": "Return a JSON object with exactly these 4 keys (string values only):",
            "keys": [
                "executive_summary",
                "key_findings",
                "risks_and_anomalies",
                "recommended_actions",
            ],
        },
    }
    return json.dumps(prompt_data, default=str)


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> dict[str, str]:
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.warning("openai not installed — using fallback report text")
        return {}

    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        logger.warning("Azure OpenAI not configured — using fallback report text")
        return {}

    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        # Validate expected keys are present
        required = {"executive_summary", "key_findings", "risks_and_anomalies", "recommended_actions"}
        if not required.issubset(parsed.keys()):
            logger.warning("LLM response missing keys — falling back")
            return {}
        return parsed
    except Exception as e:
        logger.warning("LLM call failed: %s — using fallback", e)
        return {}


# ── Fallback template ─────────────────────────────────────────────────────────

def _fallback_sections(findings: dict) -> dict[str, str]:
    kpis = findings.get("kpis", {})
    name = findings.get("dataset_name", "the dataset")
    period = findings.get("reporting_period", "the reporting period")

    total_sales = kpis.get("total_sales")
    orders      = kpis.get("total_orders")
    growth      = kpis.get("growth_over_previous_period")
    trend       = kpis.get("trend_summary", "Trend data unavailable.")

    sales_str   = f"${total_sales:,.2f}" if total_sales is not None else "N/A"
    orders_str  = f"{orders:,}" if orders is not None else "N/A"
    growth_str  = (f"{growth:+.1f}%" if growth is not None else "N/A")

    top_product = findings.get("top_5_products", [{}])[0].get("name", "N/A")
    top_region  = findings.get("top_5_regions",  [{}])[0].get("name", "N/A")
    anomalies   = findings.get("anomalies", [])
    bw          = findings.get("best_worst_period", {})
    best_p      = bw.get("best_period", {}).get("period", "N/A")
    worst_p     = bw.get("worst_period", {}).get("period", "N/A")

    exec_summary = (
        f"This report summarises sales performance for {name} during {period}. "
        f"Total revenue reached {sales_str} across {orders_str} orders, "
        f"with growth of {growth_str} versus the prior period. {trend}"
    )

    key_findings = (
        f"The top-performing product was '{top_product}' and the leading region was '{top_region}'. "
        f"The strongest sales month was {best_p} and the weakest was {worst_p}. "
        f"Category and product concentration analyses are detailed in the charts below."
    )

    risks = (
        f"{len(anomalies)} anomalous period(s) were detected in the sales data. "
        "These periods show unusually high or low revenue compared to the overall trend "
        "and warrant further investigation for root cause analysis."
    ) if anomalies else (
        "No significant anomalies were detected in the reporting period. "
        "Monitoring should continue to catch early signals of performance deviation."
    )

    recommendations = (
        f"Focus retention and upsell efforts on the top-performing region ({top_region}) "
        f"and product ({top_product}) to consolidate existing revenue strength. "
        "Investigate the causes of weak periods and apply corrective actions for the next cycle."
    )

    return {
        "executive_summary":    exec_summary,
        "key_findings":         key_findings,
        "risks_and_anomalies":  risks,
        "recommended_actions":  recommendations,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def write_report_sections(findings: dict, charts: list[dict]) -> dict[str, str]:
    """
    Calls Azure OpenAI with a structured prompt derived from findings.
    Falls back to template strings if LLM is unavailable.

    Returns dict with keys:
        executive_summary, key_findings, risks_and_anomalies, recommended_actions
    """
    prompt   = _build_prompt(findings, charts)
    sections = _call_llm(prompt)

    if not sections:
        logger.info("Using fallback report sections")
        sections = _fallback_sections(findings)

    return sections
