# Chart Generator
# - Produces 3–5 business charts from the analytics findings dict
# - All charts saved as PNG to data/reporting/<safe_name>/
# - Returns list of {type, path, caption} — same shape as EDA plots

from __future__ import annotations

import base64
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from core.logger import get_logger

logger = get_logger("ChartGenerator")

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
             "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#4F46E5"]
BG_COLOR  = "#FFFFFF"
GRID_COLOR = "#E5E7EB"
FONT_SIZE  = 11


def _apply_style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(labelsize=FONT_SIZE - 1)


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def _encode(path: str) -> str:
    """Return base64-encoded PNG for inline rendering in the frontend."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def _fmt_currency(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.0f}"


# ── Chart 1: Sales trend over time ───────────────────────────────────────────

def _chart_sales_trend(findings: dict, out_dir: str, safe_name: str) -> dict | None:
    monthly = findings.get("time_trends", {}).get("monthly", [])
    if not monthly:
        return None
    try:
        df = pd.DataFrame(monthly)
        df["period_dt"] = pd.to_datetime(df["period"].astype(str), errors="coerce")
        df = df.dropna(subset=["period_dt"]).sort_values("period_dt")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["period_dt"], df["sales"], color=PALETTE[0], linewidth=2.5, marker="o", markersize=4)
        ax.fill_between(df["period_dt"], df["sales"], alpha=0.1, color=PALETTE[0])
        _apply_style(ax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_currency(x)))
        ax.set_title("Sales Trend Over Time", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Period", fontsize=FONT_SIZE)
        ax.set_ylabel("Sales", fontsize=FONT_SIZE)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{safe_name}_sales_trend.png")
        _save(fig, path)
        direction = findings.get("time_trends", {}).get("trend_direction", "flat")
        return {"type": "sales_trend", "path": path, "base64": _encode(path),
                "caption": f"Monthly sales show a {direction} trend over the reporting period."}
    except Exception as e:
        logger.warning("Sales trend chart failed: %s", e)
        return None


# ── Chart 2: Sales by region ─────────────────────────────────────────────────

def _chart_sales_by_region(findings: dict, out_dir: str, safe_name: str) -> dict | None:
    regions = findings.get("top_5_regions", [])
    if not regions:
        return None
    try:
        names  = [r["name"] for r in regions]
        sales  = [r["sales"] for r in regions]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(names[::-1], sales[::-1], color=PALETTE[:len(names)], height=0.55)
        _apply_style(ax)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, axis="x")
        ax.grid(False, axis="y")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_currency(x)))
        ax.set_title("Sales by Region (Top 5)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Total Sales", fontsize=FONT_SIZE)

        for bar, val in zip(bars, sales[::-1]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    _fmt_currency(val), va="center", fontsize=FONT_SIZE - 1)

        fig.tight_layout()
        path = os.path.join(out_dir, f"{safe_name}_sales_by_region.png")
        _save(fig, path)
        top = regions[0]["name"] if regions else "N/A"
        return {"type": "sales_by_region", "path": path, "base64": _encode(path),
                "caption": f"Top performing region is {top}, contributing {regions[0]['share_pct']:.1f}% of total sales."}
    except Exception as e:
        logger.warning("Region chart failed: %s", e)
        return None


# ── Chart 3: Sales by category ───────────────────────────────────────────────

def _chart_sales_by_category(findings: dict, out_dir: str, safe_name: str) -> dict | None:
    cats = findings.get("category_contribution", [])
    if not cats:
        return None
    try:
        top_cats = cats[:8]
        labels = [c["name"] for c in top_cats]
        shares = [c["share_pct"] for c in top_cats]

        # Combine remainder into "Other"
        other = round(100 - sum(shares), 2)
        if other > 0.5:
            labels.append("Other")
            shares.append(other)

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            shares, labels=None, autopct="%1.1f%%",
            colors=PALETTE[:len(shares)], startangle=140,
            pctdistance=0.78, wedgeprops={"linewidth": 1, "edgecolor": "white"}
        )
        for at in autotexts:
            at.set_fontsize(FONT_SIZE - 1)
        ax.legend(wedges, labels, title="Category", loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=FONT_SIZE - 1)
        ax.set_title("Sales by Category", fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{safe_name}_sales_by_category.png")
        _save(fig, path)
        top_cat = labels[0] if labels else "N/A"
        return {"type": "sales_by_category", "path": path, "base64": _encode(path),
                "caption": f"{top_cat} is the leading category with {shares[0]:.1f}% of total revenue."}
    except Exception as e:
        logger.warning("Category chart failed: %s", e)
        return None


# ── Chart 4: Top 10 products ─────────────────────────────────────────────────

def _chart_top_products(findings: dict, out_dir: str, safe_name: str) -> dict | None:
    products = findings.get("top_10_products", [])
    if not products:
        return None
    try:
        names = [p["name"][:30] for p in products]  # truncate long names
        sales = [p["sales"] for p in products]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(names)), sales, color=PALETTE[0], width=0.6)
        _apply_style(ax)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=FONT_SIZE - 1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_currency(x)))
        ax.set_title("Top 10 Products by Sales", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Total Sales", fontsize=FONT_SIZE)

        # Colour top bar distinctly
        bars[0].set_color(PALETTE[4])
        fig.tight_layout()

        path = os.path.join(out_dir, f"{safe_name}_top_products.png")
        _save(fig, path)
        top = names[0] if names else "N/A"
        return {"type": "top_products", "path": path, "base64": _encode(path),
                "caption": f"Top product '{top}' leads with {_fmt_currency(sales[0])} in sales."}
    except Exception as e:
        logger.warning("Top products chart failed: %s", e)
        return None


# ── Chart 5 (optional): Anomaly overlay ──────────────────────────────────────

def _chart_anomalies(findings: dict, out_dir: str, safe_name: str) -> dict | None:
    anomalies = findings.get("anomalies", [])
    monthly   = findings.get("time_trends", {}).get("monthly", [])
    if not anomalies or not monthly:
        return None
    try:
        df = pd.DataFrame(monthly)
        df["period_dt"] = pd.to_datetime(df["period"].astype(str), errors="coerce")
        df = df.dropna(subset=["period_dt"]).sort_values("period_dt")

        anomaly_periods = {a["period"]: a["type"] for a in anomalies}

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["period_dt"], df["sales"], color=PALETTE[0], linewidth=2, zorder=2)
        _apply_style(ax)

        for _, row in df.iterrows():
            period_str = str(row["period_dt"].to_period("M")) if hasattr(row["period_dt"], "to_period") else str(row["period"])[:7]
            if period_str in anomaly_periods:
                color = PALETTE[2] if anomaly_periods[period_str] == "low" else PALETTE[3]
                ax.scatter(row["period_dt"], row["sales"], color=color, s=80, zorder=3)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_currency(x)))
        ax.set_title("Sales with Anomaly Detection", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Period", fontsize=FONT_SIZE)
        ax.set_ylabel("Sales", fontsize=FONT_SIZE)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=PALETTE[0], linewidth=2, label="Monthly Sales"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[3], markersize=9, label="High Anomaly"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[2], markersize=9, label="Low Anomaly"),
        ]
        ax.legend(handles=legend_elements, fontsize=FONT_SIZE - 1)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{safe_name}_anomalies.png")
        _save(fig, path)
        return {"type": "anomalies", "path": path, "base64": _encode(path),
                "caption": f"{len(anomalies)} anomalous period(s) identified — highlighted above."}
    except Exception as e:
        logger.warning("Anomaly chart failed: %s", e)
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def generate_charts(findings: dict, out_dir: str) -> list[dict[str, Any]]:
    """
    Generates 3–5 charts from the findings dict.
    Saves PNGs to out_dir.
    Returns list of {type, path, caption}.
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_"
                        for c in findings.get("dataset_name", "report")).strip("_") or "report"

    charts = []
    for fn in [_chart_sales_trend, _chart_sales_by_region,
               _chart_sales_by_category, _chart_top_products, _chart_anomalies]:
        result = fn(findings, out_dir, safe_name)
        if result:
            charts.append(result)

    logger.info("Generated %d charts for %s", len(charts), findings.get("dataset_name"))
    return charts
