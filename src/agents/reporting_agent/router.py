# FastAPI router for the Reporting Agent
# GET /reporting/{dataset_name}/html  — serves the HTML report in-browser
# GET /reporting/{dataset_name}/pdf   — streams the PDF as a download

import os
import urllib.parse

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from src.core.config import DATA_DIR
from src.core.logger import get_logger

logger = get_logger("ReportingRouter")

router = APIRouter()

REPORTING_DIR = str(DATA_DIR / "reporting")


def _safe_name(dataset_name: str) -> str:
    """Reproduce the same sanitisation used by reporting_agent.py."""
    return "".join(c if c.isalnum() or c in "_-" else "_"
                   for c in dataset_name).strip("_") or "report"


def _report_path(dataset_name: str, ext: str) -> str:
    safe = _safe_name(urllib.parse.unquote(dataset_name))
    return os.path.join(REPORTING_DIR, safe, f"{safe}_report.{ext}")


@router.get(
    "/{dataset_name}/html",
    summary="View HTML business report in browser",
    response_class=HTMLResponse,
)
async def get_report_html(dataset_name: str):
    html_path = _report_path(dataset_name, "html")
    pdf_path = _report_path(dataset_name, "pdf")

    # If HTML was not persisted (PDF-first mode), serve the PDF inline.
    if not os.path.exists(html_path) and os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf")

    if not os.path.exists(html_path):
        raise HTTPException(
            status_code=404,
            detail=f"HTML report not found for dataset '{dataset_name}'. "
                   f"Run the ingestion pipeline first.",
        )
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.get(
    "/{dataset_name}/pdf",
    summary="Download PDF business report",
)
async def get_report_pdf(dataset_name: str):
    pdf_path  = _report_path(dataset_name, "pdf")
    html_path = _report_path(dataset_name, "html")
    safe = _safe_name(urllib.parse.unquote(dataset_name))

    # Happy path — WeasyPrint generated a real PDF
    if os.path.exists(pdf_path):
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=f"{safe}_report.pdf",
        )

    # Fallback — WeasyPrint not installed; serve HTML with auto-print so the
    # browser's native print-to-PDF produces an equivalent result.
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        # Inject a one-shot print trigger before </body>
        print_script = """
<script>
  window.addEventListener('load', function () {
    var banner = document.createElement('div');
    banner.style.cssText = (
      'position:fixed;top:0;left:0;right:0;z-index:9999;'
      'background:#1E3A5F;color:#fff;padding:10px 20px;'
      'font-family:sans-serif;font-size:13px;'
      'display:flex;align-items:center;justify-content:space-between;'
    );
    banner.innerHTML = (
      '<span>💡 PDF export: use <strong>File → Print → Save as PDF</strong> '
      'in your browser, or click the button →</span>'
      '<button onclick="window.print()" style="'
        'padding:6px 16px;background:#fff;color:#1E3A5F;'
        'border:none;border-radius:4px;font-weight:700;cursor:pointer;">'
        '🖨 Print / Save as PDF'
      '</button>'
    );
    document.body.prepend(banner);
    // small delay so the banner renders before the dialog opens
    setTimeout(function () { window.print(); }, 600);
  });
</script>
"""
        html_content = html_content.replace("</body>", print_script + "\n</body>")
        return HTMLResponse(
            content=html_content,
            headers={
                "Content-Disposition": f'inline; filename="{safe}_report.html"'
            },
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"No report found for dataset '{dataset_name}'. "
            "Run the ingestion pipeline first."
        ),
    )


@router.get(
    "/{dataset_name}/pdf/download",
    summary="Download PDF business report as attachment",
)
async def download_report_pdf(dataset_name: str):
    """Returns a strict PDF attachment. Does not fallback to HTML."""
    pdf_path = _report_path(dataset_name, "pdf")
    safe = _safe_name(urllib.parse.unquote(dataset_name))
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"PDF report not found for dataset '{dataset_name}'. "
                "Generate PDF (WeasyPrint) and retry."
            ),
        )
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{safe}_report.pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe}_report.pdf"'},
    )


@router.get(
    "/",
    summary="List available reports",
)
async def list_reports():
    """Returns a list of all generated reports with their available formats."""
    if not os.path.exists(REPORTING_DIR):
        return {"reports": []}

    reports = []
    for entry in sorted(os.listdir(REPORTING_DIR)):
        folder = os.path.join(REPORTING_DIR, entry)
        if not os.path.isdir(folder):
            continue
        html_path = os.path.join(folder, f"{entry}_report.html")
        pdf_path  = os.path.join(folder, f"{entry}_report.pdf")
        reports.append({
            "dataset_name": entry,
            "html_available": os.path.exists(html_path),
            "pdf_available":  os.path.exists(pdf_path),
            "html_url": f"/reporting/{entry}/html",
            "pdf_url":  f"/reporting/{entry}/pdf",
        })

    return {"reports": reports}
