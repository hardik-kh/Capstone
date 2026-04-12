# Report Composer
# - Renders a 2-page HTML report from findings + sections + charts
# - Page 1: title, period, executive summary, KPI cards, 1 major chart
# - Page 2: detailed findings, 2–3 supporting charts, recommendations
# - Saves HTML + PDF to out_dir via WeasyPrint
# - Gracefully falls back to HTML-only if WeasyPrint is not installed

from __future__ import annotations

import base64
import os
from datetime import datetime
from typing import Any

from core.logger import get_logger

logger = get_logger("ReportComposer")


# ── Base64 image helper ───────────────────────────────────────────────────────

def _img_to_b64(path: str) -> str:
    """Embed image as inline base64 data URI so the PDF is self-contained."""
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


# ── KPI card helpers ──────────────────────────────────────────────────────────

def _fmt_val(val: Any) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if abs(val) >= 1_000_000:
            return f"${val/1_000_000:.2f}M"
        if abs(val) >= 1_000:
            return f"${val/1_000:.1f}K"
        return f"${val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _fmt_growth(val: Any) -> str:
    if val is None:
        return "N/A"
    sign = "▲" if float(val) >= 0 else "▼"
    color = "#16A34A" if float(val) >= 0 else "#DC2626"
    return f'<span style="color:{color}">{sign} {abs(float(val)):.1f}%</span>'


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{report_title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    color: #1F2937;
    background: #FFFFFF;
  }}

  /* ── Page breaks ── */
  .page {{
    width: 210mm;
    min-height: 297mm;
    padding: 18mm 18mm 16mm 18mm;
    page-break-after: always;
  }}
  .page:last-child {{ page-break-after: auto; }}

  @page {{
    size: A4;
    margin: 0;
  }}

  /* ── Header bar ── */
  .header-bar {{
    background: #1E3A5F;
    color: #FFFFFF;
    padding: 18px 24px 14px 24px;
    border-radius: 6px 6px 0 0;
    margin-bottom: 0;
  }}
  .header-bar h1 {{
    font-size: 20pt;
    font-weight: 700;
    letter-spacing: -0.3px;
  }}
  .header-bar .sub {{
    font-size: 10pt;
    opacity: 0.8;
    margin-top: 4px;
  }}

  /* ── Section labels ── */
  .section-label {{
    font-size: 7pt;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6B7280;
    margin: 20px 0 6px 0;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 4px;
  }}

  /* ── Executive summary ── */
  .exec-summary {{
    background: #F0F7FF;
    border-left: 4px solid #2563EB;
    padding: 12px 16px;
    border-radius: 0 4px 4px 0;
    font-size: 11pt;
    line-height: 1.6;
    color: #1F2937;
    margin-bottom: 16px;
  }}

  /* ── KPI cards ── */
  .kpi-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 16px;
  }}
  .kpi-card {{
    flex: 1 1 140px;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 10px 14px;
  }}
  .kpi-card .kpi-label {{
    font-size: 8pt;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }}
  .kpi-card .kpi-value {{
    font-size: 17pt;
    font-weight: 700;
    color: #1E3A5F;
    margin-top: 2px;
  }}
  .kpi-card .kpi-sub {{
    font-size: 9pt;
    color: #6B7280;
    margin-top: 2px;
  }}

  /* ── Charts ── */
  .chart-block {{
    margin: 14px 0;
    text-align: center;
  }}
  .chart-block img {{
    max-width: 100%;
    border-radius: 4px;
    border: 1px solid #E5E7EB;
  }}
  .chart-caption {{
    font-size: 9pt;
    color: #6B7280;
    margin-top: 5px;
    font-style: italic;
  }}

  .chart-row {{
    display: flex;
    gap: 12px;
    margin: 14px 0;
  }}
  .chart-row .chart-block {{
    flex: 1;
    margin: 0;
  }}

  /* ── Rankings table ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 10pt;
    margin: 8px 0 14px 0;
  }}
  th {{
    background: #1E3A5F;
    color: #FFFFFF;
    font-weight: 600;
    padding: 7px 10px;
    text-align: left;
    font-size: 9pt;
  }}
  td {{
    padding: 6px 10px;
    border-bottom: 1px solid #F3F4F6;
  }}
  tr:nth-child(even) td {{ background: #F9FAFB; }}

  /* ── Prose sections ── */
  .prose {{
    line-height: 1.65;
    color: #374151;
    font-size: 10.5pt;
    margin-bottom: 12px;
  }}

  /* ── Anomaly badges ── */
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 8.5pt;
    font-weight: 600;
    margin: 2px 3px;
  }}
  .badge-high {{ background: #FEF3C7; color: #92400E; }}
  .badge-low  {{ background: #FEE2E2; color: #991B1B; }}

  /* ── Footer ── */
  .footer {{
    margin-top: 20px;
    padding-top: 8px;
    border-top: 1px solid #E5E7EB;
    font-size: 8pt;
    color: #9CA3AF;
    display: flex;
    justify-content: space-between;
  }}

  /* ── Page 2 header ── */
  .page2-header {{
    background: #F3F4F6;
    border-radius: 4px;
    padding: 10px 16px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .page2-header .p2-title {{
    font-size: 13pt;
    font-weight: 700;
    color: #1E3A5F;
  }}
  .page2-header .p2-sub {{
    font-size: 9pt;
    color: #6B7280;
  }}
</style>
</head>
<body>

<!-- ══════════════════════════════════════════════════════════════
     PAGE 1: Title · Executive Summary · KPIs · Main Chart
     ══════════════════════════════════════════════════════════════ -->
<div class="page">

  <div class="header-bar">
    <h1>{report_title}</h1>
    <div class="sub">Reporting Period: {reporting_period} &nbsp;|&nbsp; Generated: {generated_at}</div>
  </div>

  <div class="section-label">Executive Summary</div>
  <div class="exec-summary">{executive_summary}</div>

  <div class="section-label">Key Performance Indicators</div>
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Total Sales</div>
      <div class="kpi-value">{kpi_total_sales}</div>
      <div class="kpi-sub">Reporting period</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Orders</div>
      <div class="kpi-value">{kpi_total_orders}</div>
      <div class="kpi-sub">Unique transactions</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Units Sold</div>
      <div class="kpi-value">{kpi_units_sold}</div>
      <div class="kpi-sub">Total quantity</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Avg. Order Value</div>
      <div class="kpi-value">{kpi_aov}</div>
      <div class="kpi-sub">Per transaction</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Period Growth</div>
      <div class="kpi-value" style="font-size:14pt">{kpi_growth}</div>
      <div class="kpi-sub">vs prior period</div>
    </div>
  </div>

  {main_chart_html}

  <div class="footer">
    <span>Autonomous Analytics — Reporting Agent</span>
    <span>Page 1 of 2</span>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════
     PAGE 2: Detailed Findings · Charts · Recommendations
     ══════════════════════════════════════════════════════════════ -->
<div class="page">

  <div class="page2-header">
    <div>
      <div class="p2-title">Detailed Analysis — {dataset_name}</div>
      <div class="p2-sub">{reporting_period}</div>
    </div>
    <div class="p2-sub">Page 2 of 2</div>
  </div>

  <div class="section-label">Key Findings</div>
  <div class="prose">{key_findings}</div>

  {rankings_html}

  {supporting_charts_html}

  <div class="section-label">Risks &amp; Anomalies</div>
  <div class="prose">{risks_and_anomalies}</div>
  {anomaly_badges_html}

  <div class="section-label">Recommended Actions</div>
  <div class="prose">{recommended_actions}</div>

  {best_worst_html}

  <div class="footer">
    <span>Autonomous Analytics — Reporting Agent</span>
    <span>Page 2 of 2</span>
  </div>
</div>

</body>
</html>
"""


# ── HTML fragment builders ────────────────────────────────────────────────────

def _chart_html(chart: dict, width: str = "100%") -> str:
    b64 = _img_to_b64(chart["path"])
    if not b64:
        return ""
    caption = chart.get("caption", "")
    return (
        f'<div class="chart-block">'
        f'<img src="{b64}" style="width:{width}" alt="{chart["type"]}" />'
        f'<div class="chart-caption">{caption}</div>'
        f"</div>"
    )


def _rankings_html(findings: dict) -> str:
    products = findings.get("top_5_products", [])
    regions  = findings.get("top_5_regions", [])
    if not products and not regions:
        return ""

    html = '<div style="display:flex;gap:16px;margin-bottom:4px;">'

    if products:
        html += '<div style="flex:1"><div class="section-label" style="margin-top:8px">Top 5 Products</div>'
        html += "<table><tr><th>#</th><th>Product</th><th>Sales</th><th>Share</th></tr>"
        for i, p in enumerate(products, 1):
            html += f"<tr><td>{i}</td><td>{p['name']}</td><td>{_fmt_val(p['sales'])}</td><td>{p['share_pct']:.1f}%</td></tr>"
        html += "</table></div>"

    if regions:
        html += '<div style="flex:1"><div class="section-label" style="margin-top:8px">Top 5 Regions</div>'
        html += "<table><tr><th>#</th><th>Region</th><th>Sales</th><th>Share</th></tr>"
        for i, r in enumerate(regions, 1):
            html += f"<tr><td>{i}</td><td>{r['name']}</td><td>{_fmt_val(r['sales'])}</td><td>{r['share_pct']:.1f}%</td></tr>"
        html += "</table></div>"

    html += "</div>"
    return html


def _supporting_charts_html(charts: list[dict]) -> str:
    # Show charts 2 and 3 (region + category) side by side, then chart 4 full-width
    if len(charts) < 2:
        return "".join(_chart_html(c) for c in charts[1:])

    html = ""
    side_charts = [c for c in charts[1:] if c["type"] in ("sales_by_region", "sales_by_category")]
    remaining   = [c for c in charts[1:] if c["type"] not in ("sales_by_region", "sales_by_category")]

    if len(side_charts) >= 2:
        html += '<div class="chart-row">'
        html += _chart_html(side_charts[0], "100%")
        html += _chart_html(side_charts[1], "100%")
        html += "</div>"
        remaining = side_charts[2:] + remaining
    elif side_charts:
        html += _chart_html(side_charts[0])

    for c in remaining[:2]:   # at most 2 more
        html += _chart_html(c)

    return html


def _anomaly_badges_html(anomalies: list[dict]) -> str:
    if not anomalies:
        return ""
    badges = ""
    for a in anomalies:
        cls = "badge-high" if a["type"] == "high" else "badge-low"
        label = "HIGH" if a["type"] == "high" else "LOW"
        badges += f'<span class="badge {cls}">{label}: {a["period"]} (${a["sales"]:,.0f})</span>'
    return f"<div style='margin:6px 0 10px 0'>{badges}</div>"


def _best_worst_html(findings: dict) -> str:
    bw = findings.get("best_worst_period", {})
    best  = bw.get("best_period")
    worst = bw.get("worst_period")
    if not best and not worst:
        return ""
    html = '<div style="display:flex;gap:16px;margin-top:10px">'
    if best:
        html += (
            f'<div style="flex:1;background:#F0FDF4;border:1px solid #BBF7D0;'
            f'border-radius:6px;padding:10px 14px">'
            f'<div style="font-size:8pt;font-weight:600;color:#15803D;text-transform:uppercase;'
            f'letter-spacing:0.8px">Best Period</div>'
            f'<div style="font-size:15pt;font-weight:700;color:#14532D">{best["period"]}</div>'
            f'<div style="font-size:9pt;color:#166534">{_fmt_val(best["sales"])}</div></div>'
        )
    if worst:
        html += (
            f'<div style="flex:1;background:#FFF7ED;border:1px solid #FED7AA;'
            f'border-radius:6px;padding:10px 14px">'
            f'<div style="font-size:8pt;font-weight:600;color:#C2410C;text-transform:uppercase;'
            f'letter-spacing:0.8px">Worst Period</div>'
            f'<div style="font-size:15pt;font-weight:700;color:#7C2D12">{worst["period"]}</div>'
            f'<div style="font-size:9pt;color:#9A3412">{_fmt_val(worst["sales"])}</div></div>'
        )
    html += "</div>"
    return html


# ── Public entry point ────────────────────────────────────────────────────────

def compose_report(
    findings: dict,
    sections: dict[str, str],
    charts: list[dict],
    out_dir: str,
) -> dict[str, str]:
    """
    Renders the 2-page HTML report and converts to PDF via WeasyPrint.

    Returns:
        {"html_path": ..., "pdf_path": ... (or None if WeasyPrint unavailable)}
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_"
                        for c in findings.get("dataset_name", "report")).strip("_") or "report"

    kpis = findings.get("kpis", {})

    # Build the main chart (sales trend) for page 1
    main_chart = next((c for c in charts if c["type"] == "sales_trend"), None)
    main_chart_html = _chart_html(main_chart) if main_chart else ""

    html = _HTML_TEMPLATE.format(
        report_title        = f"Sales Performance Report — {findings.get('dataset_name', '')}",
        dataset_name        = findings.get("dataset_name", ""),
        reporting_period    = findings.get("reporting_period", "All available data"),
        generated_at        = datetime.now().strftime("%B %d, %Y %H:%M"),
        executive_summary   = sections.get("executive_summary", ""),
        kpi_total_sales     = _fmt_val(kpis.get("total_sales")),
        kpi_total_orders    = _fmt_val(kpis.get("total_orders")),
        kpi_units_sold      = _fmt_val(kpis.get("units_sold")),
        kpi_aov             = _fmt_val(kpis.get("average_order_value")),
        kpi_growth          = _fmt_growth(kpis.get("growth_over_previous_period")),
        main_chart_html     = main_chart_html,
        key_findings        = sections.get("key_findings", ""),
        rankings_html       = _rankings_html(findings),
        supporting_charts_html = _supporting_charts_html(charts),
        risks_and_anomalies = sections.get("risks_and_anomalies", ""),
        anomaly_badges_html = _anomaly_badges_html(findings.get("anomalies", [])),
        recommended_actions = sections.get("recommended_actions", ""),
        best_worst_html     = _best_worst_html(findings),
    )

    html_path = os.path.join(out_dir, f"{safe_name}_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML report saved: %s", html_path)

    # ── PDF via WeasyPrint ─────────────────────────────────────────────────────
    pdf_path = None
    try:
        from weasyprint import HTML as WP_HTML
        pdf_path = os.path.join(out_dir, f"{safe_name}_report.pdf")
        WP_HTML(string=html).write_pdf(pdf_path)
        logger.info("PDF report saved: %s", pdf_path)
    except ImportError:
        logger.warning("WeasyPrint not installed — PDF not generated. Install with: pip install weasyprint")
    except Exception as e:
        logger.warning("PDF generation failed: %s", e)

    return {"html_path": html_path, "pdf_path": pdf_path}
