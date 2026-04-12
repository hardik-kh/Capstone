# Analytics Engine
# - Auto-detects sales, order, date, region, category, product columns
# - Computes all core KPIs and segmentation analyses
# - Pure pandas — no I/O, no LLM calls
# - Returns a single structured `findings` dict consumed by chart_generator and llm_writer

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.logger import get_logger

logger = get_logger("AnalyticsEngine")

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Column detection ──────────────────────────────────────────────────────────

_SALES_HINTS    = ["sales", "revenue", "amount", "total", "price", "value", "gmv", "income"]
_ORDER_HINTS    = ["order_id", "orderid", "order id", "transaction_id", "txn_id", "invoice"]
_PRODUCT_HINTS  = ["product", "item", "sku", "article", "good", "service", "name"]
_REGION_HINTS   = ["region", "country", "state", "city", "territory", "zone", "location", "area", "store"]
_CATEGORY_HINTS = ["category", "segment", "department", "division", "type", "class", "group", "channel"]
_DATE_HINTS     = ["date", "time", "period", "day", "week", "month", "year", "created", "ordered", "shipped"]
_QTY_HINTS      = ["qty", "quantity", "units", "count", "volume", "pieces", "sold"]


def _match_col(df: pd.DataFrame, hints: list[str], dtype_filter=None) -> Optional[str]:
    """Return the first column whose lowercased name contains any hint keyword."""
    cols = df.columns.tolist()
    if dtype_filter:
        cols = [c for c in cols if pd.api.types.is_dtype_equal(df[c].dtype, dtype_filter)
                or str(df[c].dtype).startswith(dtype_filter)]
    for hint in hints:
        for col in cols:
            if hint in col.lower().replace(" ", "_"):
                return col
    return None


def _detect_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Auto-detect the semantic role of columns."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    object_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols    = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    # Also try parsing object cols that look like dates
    if not date_cols:
        for col in object_cols:
            sample = df[col].dropna().head(10).astype(str)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum() >= 7:
                    date_cols.append(col)
            except Exception:
                pass

    # Sales column: best-matching numeric by name, else highest-sum numeric
    sales_col = _match_col(df[numeric_cols] if numeric_cols else df, _SALES_HINTS)
    if sales_col is None and numeric_cols:
        sales_col = df[numeric_cols].sum().idxmax()

    # Quantity column
    qty_col = _match_col(df[numeric_cols] if numeric_cols else df, _QTY_HINTS)
    if qty_col == sales_col:
        qty_col = None

    detected = {
        "sales_col":    sales_col,
        "order_col":    _match_col(df, _ORDER_HINTS),
        "product_col":  _match_col(df, _PRODUCT_HINTS),
        "region_col":   _match_col(df, _REGION_HINTS),
        "category_col": _match_col(df, _CATEGORY_HINTS),
        "date_col":     date_cols[0] if date_cols else None,
        "qty_col":      qty_col,
    }
    logger.info("Detected columns: %s", {k: v for k, v in detected.items() if v})
    return detected


# ── KPI computation ───────────────────────────────────────────────────────────

def _compute_kpis(df: pd.DataFrame, cols: dict) -> dict[str, Any]:
    sales_col = cols["sales_col"]
    kpis: dict[str, Any] = {}

    if sales_col:
        kpis["total_sales"]  = round(float(df[sales_col].sum()), 2)
        kpis["average_order_value"] = round(float(df[sales_col].mean()), 2)
    else:
        kpis["total_sales"] = None
        kpis["average_order_value"] = None

    kpis["total_orders"] = int(df[cols["order_col"]].nunique()) if cols["order_col"] else int(len(df))
    kpis["units_sold"]   = int(df[cols["qty_col"]].sum()) if cols["qty_col"] else int(len(df))

    # Period-over-period growth (requires date)
    kpis["growth_over_previous_period"] = None
    if sales_col and cols["date_col"]:
        try:
            tmp = df.copy()
            tmp["_date"] = pd.to_datetime(tmp[cols["date_col"]], errors="coerce")
            tmp = tmp.dropna(subset=["_date"])
            midpoint = tmp["_date"].min() + (tmp["_date"].max() - tmp["_date"].min()) / 2
            first_half  = tmp[tmp["_date"] <= midpoint][sales_col].sum()
            second_half = tmp[tmp["_date"] >  midpoint][sales_col].sum()
            if first_half and first_half != 0:
                growth = round(((second_half - first_half) / first_half) * 100, 2)
                kpis["growth_over_previous_period"] = growth
        except Exception as e:
            logger.warning("Growth calc failed: %s", e)

    return kpis


# ── Dimension rankings ────────────────────────────────────────────────────────

def _top_n(df: pd.DataFrame, dim_col: str, sales_col: str, n: int = 5) -> list[dict]:
    if not dim_col or not sales_col or dim_col not in df.columns:
        return []
    grouped = (
        df.groupby(dim_col, observed=True)[sales_col]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    total = df[sales_col].sum() or 1
    result = []
    for _, row in grouped.iterrows():
        result.append({
            "name":  str(row[dim_col]),
            "sales": round(float(row[sales_col]), 2),
            "share_pct": round(float(row[sales_col]) / total * 100, 2),
        })
    return result


def _category_contribution(df: pd.DataFrame, cols: dict) -> list[dict]:
    if not cols["category_col"] or not cols["sales_col"]:
        return []
    return _top_n(df, cols["category_col"], cols["sales_col"], n=20)


def _best_worst_period(df: pd.DataFrame, cols: dict) -> dict:
    if not cols["date_col"] or not cols["sales_col"]:
        return {}
    try:
        tmp = df.copy()
        tmp["_date"] = pd.to_datetime(tmp[cols["date_col"]], errors="coerce")
        tmp = tmp.dropna(subset=["_date"])
        monthly = (
            tmp.groupby(tmp["_date"].dt.to_period("M"))[cols["sales_col"]]
            .sum()
            .sort_values()
        )
        if monthly.empty:
            return {}
        return {
            "best_period":  {"period": str(monthly.index[-1]), "sales": round(float(monthly.iloc[-1]), 2)},
            "worst_period": {"period": str(monthly.index[0]),  "sales": round(float(monthly.iloc[0]),  2)},
        }
    except Exception as e:
        logger.warning("Best/worst period failed: %s", e)
        return {}


# ── Time trends ───────────────────────────────────────────────────────────────

def _time_trends(df: pd.DataFrame, cols: dict) -> dict[str, Any]:
    if not cols["date_col"] or not cols["sales_col"]:
        return {}
    try:
        tmp = df.copy()
        tmp["_date"] = pd.to_datetime(tmp[cols["date_col"]], errors="coerce")
        tmp = tmp.dropna(subset=["_date"]).sort_values("_date")
        sc = cols["sales_col"]

        daily   = tmp.groupby("_date")[sc].sum().reset_index()
        weekly  = tmp.groupby(tmp["_date"].dt.to_period("W"))[sc].sum().reset_index()
        monthly = tmp.groupby(tmp["_date"].dt.to_period("M"))[sc].sum().reset_index()

        def to_records(frame: pd.DataFrame) -> list[dict]:
            frame = frame.copy()
            frame.columns = ["period", "sales"]
            frame["period"] = frame["period"].astype(str)
            frame["sales"]  = frame["sales"].round(2)
            return frame.to_dict("records")

        # Trend direction from linear regression on monthly
        trend_direction = "flat"
        if len(monthly) >= 3:
            x = np.arange(len(monthly))
            y = monthly[sc].values.astype(float)
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.01 * y.mean():
                trend_direction = "upward"
            elif slope < -0.01 * y.mean():
                trend_direction = "downward"

        return {
            "daily":           to_records(daily),
            "weekly":          to_records(weekly),
            "monthly":         to_records(monthly),
            "trend_direction": trend_direction,
            "data_range_days": int((tmp["_date"].max() - tmp["_date"].min()).days),
        }
    except Exception as e:
        logger.warning("Time trends failed: %s", e)
        return {}


# ── Pareto / concentration ────────────────────────────────────────────────────

def _pareto_analysis(df: pd.DataFrame, cols: dict) -> dict:
    """What % of sales come from the top 20% of products."""
    if not cols["product_col"] or not cols["sales_col"]:
        return {}
    try:
        by_product = df.groupby(cols["product_col"], observed=True)[cols["sales_col"]].sum().sort_values(ascending=False)
        total = by_product.sum()
        top_20_count = max(1, int(len(by_product) * 0.20))
        top_20_sales = by_product.head(top_20_count).sum()
        return {
            "top_20pct_products_count": top_20_count,
            "top_20pct_sales_share":    round(float(top_20_sales / total * 100), 2) if total else None,
            "total_products":           int(len(by_product)),
        }
    except Exception as e:
        logger.warning("Pareto analysis failed: %s", e)
        return {}


# ── Anomaly detection ─────────────────────────────────────────────────────────

def _detect_anomalies(df: pd.DataFrame, cols: dict) -> list[dict]:
    """IQR-based anomaly detection on monthly sales totals."""
    if not cols["date_col"] or not cols["sales_col"]:
        return []
    try:
        tmp = df.copy()
        tmp["_date"] = pd.to_datetime(tmp[cols["date_col"]], errors="coerce")
        tmp = tmp.dropna(subset=["_date"])
        monthly = tmp.groupby(tmp["_date"].dt.to_period("M"))[cols["sales_col"]].sum()
        if len(monthly) < 4:
            return []
        q1, q3 = monthly.quantile(0.25), monthly.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        anomalies = []
        for period, val in monthly.items():
            if val < lower or val > upper:
                anomalies.append({
                    "period": str(period),
                    "sales":  round(float(val), 2),
                    "type":   "high" if val > upper else "low",
                })
        return anomalies
    except Exception as e:
        logger.warning("Anomaly detection failed: %s", e)
        return []


# ── Seasonality hint ──────────────────────────────────────────────────────────

def _seasonality_hint(trends: dict) -> Optional[str]:
    """Simple month-of-year variance check to hint at seasonality."""
    monthly = trends.get("monthly", [])
    if len(monthly) < 13:
        return None
    try:
        records = pd.DataFrame(monthly)
        records["month"] = pd.to_datetime(records["period"].astype(str)).dt.month
        monthly_mean = records.groupby("month")["sales"].mean()
        cv = monthly_mean.std() / monthly_mean.mean() if monthly_mean.mean() else 0
        if cv > 0.15:
            peak_month = monthly_mean.idxmax()
            import calendar
            return f"Possible seasonality detected — peak month is typically {calendar.month_name[peak_month]}."
        return "No strong seasonality pattern detected in available history."
    except Exception:
        return None


# ── Trend summary (plain English) ─────────────────────────────────────────────

def _trend_summary(kpis: dict, trends: dict, anomalies: list) -> str:
    parts = []
    direction = trends.get("trend_direction", "flat")
    parts.append(f"Sales trend is {direction}.")
    growth = kpis.get("growth_over_previous_period")
    if growth is not None:
        label = "grew" if growth >= 0 else "declined"
        parts.append(f"Revenue {label} {abs(growth):.1f}% versus the prior period.")
    if anomalies:
        highs = [a for a in anomalies if a["type"] == "high"]
        lows  = [a for a in anomalies if a["type"] == "low"]
        if highs:
            parts.append(f"{len(highs)} unusually high period(s) detected ({', '.join(a['period'] for a in highs[:2])}).")
        if lows:
            parts.append(f"{len(lows)} unusually low period(s) detected ({', '.join(a['period'] for a in lows[:2])}).")
    return " ".join(parts)


# ── Public entry point ────────────────────────────────────────────────────────

def run_analytics(dataset_name: str, df: pd.DataFrame) -> dict[str, Any]:
    """
    Runs the full rules-based analytics pipeline on a single DataFrame.

    Returns a structured `findings` dict ready for chart_generator and llm_writer.
    No I/O — pure computation.
    """
    logger.info("Analytics engine starting: %s (%d rows, %d cols)", dataset_name, len(df), len(df.columns))

    cols    = _detect_columns(df)
    kpis    = _compute_kpis(df, cols)
    trends  = _time_trends(df, cols)
    anomalies = _detect_anomalies(df, cols)

    kpis["trend_summary"] = _trend_summary(kpis, trends, anomalies)

    findings: dict[str, Any] = {
        "dataset_name":         dataset_name,
        "row_count":            int(len(df)),
        "detected_columns":     cols,
        "kpis":                 kpis,
        "top_5_products":       _top_n(df, cols["product_col"],  cols["sales_col"], n=5),
        "top_10_products":      _top_n(df, cols["product_col"],  cols["sales_col"], n=10),
        "top_5_regions":        _top_n(df, cols["region_col"],   cols["sales_col"], n=5),
        "category_contribution":_category_contribution(df, cols),
        "best_worst_period":    _best_worst_period(df, cols),
        "time_trends":          trends,
        "pareto":               _pareto_analysis(df, cols),
        "anomalies":            anomalies,
        "seasonality_hint":     _seasonality_hint(trends),
        "reporting_period": (
            f"{pd.to_datetime(df[cols['date_col']], errors='coerce').min().strftime('%b %d, %Y')} – "
            f"{pd.to_datetime(df[cols['date_col']], errors='coerce').max().strftime('%b %d, %Y')}"
        ) if cols["date_col"] else "All available data",
    }

    logger.info("Analytics engine complete: %s", dataset_name)
    return findings
