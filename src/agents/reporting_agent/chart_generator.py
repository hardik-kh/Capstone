"""Generalized chart generator for reporting findings."""

from __future__ import annotations

import base64
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("ChartGenerator")

PALETTE = [
    "#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#4F46E5",
]
BG_COLOR = "#FFFFFF"
GRID_COLOR = "#E5E7EB"
FONT_SIZE = 11


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
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def _fmt_number(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.1f}K"
    return f"{val:.0f}"


def _chart_trend(findings: dict, out_dir: str, safe_name: str) -> dict | None:
    metric_label = findings.get("metric", {}).get("label", "Metric")
    monthly = findings.get("time_trends", {}).get("monthly", [])
    if not monthly:
        return None
    try:
        df = pd.DataFrame(monthly)
        if "value" not in df.columns:
            return None
        df["period_dt"] = pd.to_datetime(df["period"].astype(str), errors="coerce")
        df = df.dropna(subset=["period_dt"]).sort_values("period_dt")
        if df.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["period_dt"], df["value"], color=PALETTE[0], linewidth=2.5, marker="o", markersize=4)
        ax.fill_between(df["period_dt"], df["value"], alpha=0.1, color=PALETTE[0])
        _apply_style(ax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_number(float(x))))
        ax.set_title(f"{metric_label} Trend Over Time", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Period", fontsize=FONT_SIZE)
        ax.set_ylabel(metric_label, fontsize=FONT_SIZE)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{safe_name}_trend.png")
        _save(fig, path)
        direction = findings.get("time_trends", {}).get("trend_direction", "flat")
        return {
            "type": "metric_trend",
            "path": path,
            "base64": _encode(path),
            "caption": f"{metric_label} shows a {direction} trend over the reporting period.",
        }
    except Exception as e:
        logger.warning("Trend chart failed: %s", e)
        return None


def _chart_ranking(ranking: dict, out_dir: str, safe_name: str, idx: int) -> dict | None:
    metric_label = str(ranking.get("metric_label", "Value"))
    label = ranking.get("label", "Dimension")
    items = ranking.get("items", [])[:10]
    if not items:
        return None
    try:
        names = [str(r["name"])[:32] for r in items]
        vals = [float(r["value"]) for r in items]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.barh(names[::-1], vals[::-1], color=PALETTE[:len(vals)], height=0.55)
        _apply_style(ax)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, axis="x")
        ax.grid(False, axis="y")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_number(float(x))))
        ax.set_title(f"{metric_label} by {label}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(metric_label, fontsize=FONT_SIZE)
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2, _fmt_number(val), va="center", fontsize=FONT_SIZE - 1)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{safe_name}_ranking_{idx}.png")
        _save(fig, path)
        return {
            "type": f"ranking_{idx}",
            "path": path,
            "base64": _encode(path),
            "caption": f"Top {label} by {metric_label}.",
        }
    except Exception as e:
        logger.warning("Ranking chart failed (%s): %s", label, e)
        return None


def _can_use_pie(items: list[dict]) -> bool:
    """Pie charts are readable only for small, positive, not-overly-dominant slices."""
    if len(items) < 3 or len(items) > 7:
        return False
    vals = []
    for item in items:
        try:
            vals.append(float(item.get("value", 0)))
        except Exception:
            return False
    if any(v < 0 for v in vals):
        return False
    total = sum(vals)
    if total <= 0:
        return False
    top_share = max(vals) / total
    return top_share <= 0.8


def _chart_ranking_pie(ranking: dict, out_dir: str, safe_name: str, idx: int) -> dict | None:
    metric_label = str(ranking.get("metric_label", "Value"))
    label = ranking.get("label", "Dimension")
    items = ranking.get("items", [])[:7]
    if not _can_use_pie(items):
        return None
    try:
        names = [str(r["name"])[:26] for r in items]
        vals = [float(r["value"]) for r in items]
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        wedges, _, autotexts = ax.pie(
            vals,
            labels=None,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.78,
            colors=PALETTE[:len(vals)],
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        for t in autotexts:
            t.set_fontsize(FONT_SIZE - 1)
        ax.legend(
            wedges,
            names,
            title=label,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=FONT_SIZE - 1,
        )
        ax.set_title(f"{metric_label} Share by {label}", fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{safe_name}_ranking_pie_{idx}.png")
        _save(fig, path)
        return {
            "type": f"ranking_pie_{idx}",
            "path": path,
            "base64": _encode(path),
            "caption": f"{label} contribution split for {metric_label}.",
        }
    except Exception as e:
        logger.warning("Ranking pie chart failed (%s): %s", label, e)
        return None


def _chart_anomalies(findings: dict, out_dir: str, safe_name: str) -> dict | None:
    anomalies = findings.get("anomalies", [])
    monthly = findings.get("time_trends", {}).get("monthly", [])
    if not anomalies or not monthly:
        return None
    try:
        df = pd.DataFrame(monthly)
        if "value" not in df.columns:
            return None
        df["period_dt"] = pd.to_datetime(df["period"].astype(str), errors="coerce")
        df = df.dropna(subset=["period_dt"]).sort_values("period_dt")
        if df.empty:
            return None

        anomaly_map = {str(a["period"]): a["type"] for a in anomalies}
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["period_dt"], df["value"], color=PALETTE[0], linewidth=2, zorder=2)
        _apply_style(ax)
        for _, row in df.iterrows():
            period_str = str(row["period_dt"].to_period("M"))
            if period_str in anomaly_map:
                color = PALETTE[2] if anomaly_map[period_str] == "low" else PALETTE[3]
                ax.scatter(row["period_dt"], row["value"], color=color, s=80, zorder=3)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_number(float(x))))
        ax.set_title("Anomaly Highlights", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Period", fontsize=FONT_SIZE)
        ax.set_ylabel("Value", fontsize=FONT_SIZE)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{safe_name}_anomalies.png")
        _save(fig, path)
        return {
            "type": "anomalies",
            "path": path,
            "base64": _encode(path),
            "caption": f"{len(anomalies)} anomalous period(s) detected.",
        }
    except Exception as e:
        logger.warning("Anomaly chart failed: %s", e)
        return None


def generate_charts(findings: dict, out_dir: str) -> list[dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in findings.get("dataset_name", "report")).strip("_") or "report"

    charts: list[dict[str, Any]] = []

    trend = _chart_trend(findings, out_dir, safe_name)
    if trend:
        charts.append(trend)

    # Supporting charts: cap at 2 (avoid repetitive horizontal bars).
    rankings = findings.get("top_rankings", [])[:3]
    supporting: list[dict[str, Any]] = []

    pie_used = False
    for i, ranking in enumerate(rankings, start=1):
        if len(supporting) >= 2:
            break
        items = ranking.get("items", [])[:7]
        if not pie_used and _can_use_pie(items):
            pie_chart = _chart_ranking_pie(ranking, out_dir, safe_name, i)
            if pie_chart:
                supporting.append(pie_chart)
                pie_used = True
                continue
        bar_chart = _chart_ranking(ranking, out_dir, safe_name, i)
        if bar_chart:
            supporting.append(bar_chart)

    # If no rankings were chartable, use anomaly chart as a fallback signal.
    if not supporting:
        anomaly_chart = _chart_anomalies(findings, out_dir, safe_name)
        if anomaly_chart:
            supporting.append(anomaly_chart)

    charts.extend(supporting[:2])

    logger.info("Generated %d charts for %s", len(charts), findings.get("dataset_name"))
    return charts
