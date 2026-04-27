"""Generalized analytics engine for reporting.

Design goals:
- Work across arbitrary business datasets (not just sales schemas).
- Avoid picking ID-like columns as the primary metric.
- Use robust fallbacks when semantic hints are missing.
- Return stable, structured outputs for charting and report composition.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("AnalyticsEngine")


METRIC_HINTS = [
    "sales",
    "revenue",
    "amount",
    "total",
    "price",
    "value",
    "gmv",
    "income",
    "transaction",
    "transactions",
    "count",
    "qty",
    "quantity",
    "units",
    "volume",
]
ID_HINTS = ["id", "_id", "code", "key", "index", "nbr", "num", "no"]
DATE_HINTS = ["date", "time", "timestamp", "created", "updated", "month", "year"]


MAX_MONTHS = 60
MIN_MONTHS = 6


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _is_probable_id(series: pd.Series, col_name: str) -> bool:
    n = _norm(col_name)
    if any(h in n for h in ID_HINTS):
        return True
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique(dropna=True) / max(1, len(non_null))
    int_like = bool((non_null % 1 == 0).all())
    return int_like and unique_ratio > 0.95


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in object_cols:
        if col in date_cols:
            continue
        name = _norm(col)
        if not any(h in name for h in DATE_HINTS):
            continue
        sample = df[col].dropna().head(20)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.7:
            date_cols.append(col)
    return date_cols


def _score_metric_candidate(df: pd.DataFrame, col: str) -> float:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return -1e9
    name = _norm(col)
    score = 0.0
    if any(h in name for h in METRIC_HINTS):
        score += 8.0
    if _is_probable_id(df[col], col):
        score -= 12.0
    unique_ratio = s.nunique(dropna=True) / max(1, len(s))
    score += min(4.0, float(np.log1p(s.std(ddof=0) if len(s) > 1 else 0.0)))
    score += 2.0 if unique_ratio < 0.9 else -2.0
    score += 1.5 if (s >= 0).mean() > 0.9 else 0.0
    score += 1.0 if s.quantile(0.99) > s.quantile(0.5) else 0.0
    return score


def _choose_primary_metric(df: pd.DataFrame) -> Optional[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return None
    ranked = sorted(
        ((col, _score_metric_candidate(df, col)) for col in numeric_cols),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[0][0] if ranked else None


def _detect_dimensions(df: pd.DataFrame, metric_col: Optional[str], date_cols: list[str]) -> list[str]:
    dims: list[str] = []
    candidates = df.columns.tolist()
    for col in candidates:
        if col == metric_col or col in date_cols:
            continue
        s = df[col]
        # object/category dimensions
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            nunique = s.nunique(dropna=True)
            if 2 <= nunique <= 100:
                dims.append(col)
                continue
        # numeric low-cardinality dimensions (e.g., store_nbr)
        if pd.api.types.is_numeric_dtype(s):
            if _is_probable_id(s, col):
                # Keep some ID-like dimensions if they are low-cardinality enough.
                pass
            nunique = s.nunique(dropna=True)
            if 2 <= nunique <= 60:
                dims.append(col)
    return dims


def _choose_trend_agg_mode(df: pd.DataFrame, metric_col: str, date_col: str) -> str:
    """Choose trend aggregation mode.

    - mean: preserves per-record scale when there are many rows per day and metric is bounded.
    - sum: for additive metrics where totals are more meaningful.
    """
    name = _norm(metric_col)
    s = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if s.empty:
        return "sum"
    if any(tok in name for tok in ["rate", "ratio", "score", "avg", "average"]):
        return "mean"
    tmp = pd.DataFrame(
        {
            "_date": pd.to_datetime(df[date_col], errors="coerce"),
            "_metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()
    if tmp.empty:
        return "sum"
    uniq_days = max(1, tmp["_date"].dt.date.nunique())
    rows_per_day = len(tmp) / uniq_days
    if rows_per_day > 1.5 and float(s.quantile(0.99)) <= 10_000:
        return "mean"
    return "sum"


def _rank_dimension(
    df: pd.DataFrame,
    dim_col: str,
    metric_col: str,
    agg_mode: str,
    n: int = 10,
) -> list[dict[str, Any]]:
    if dim_col not in df.columns or metric_col not in df.columns:
        return []
    tmp = pd.DataFrame(
        {
            "_dim": df[dim_col].astype(str),
            "_metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()
    if tmp.empty:
        return []
    grouped = tmp.groupby("_dim", observed=True)["_metric"]
    if agg_mode == "mean":
        values = grouped.mean().sort_values(ascending=False).head(n)
    else:
        values = grouped.sum().sort_values(ascending=False).head(n)
    total = float(values.sum()) or 1.0
    rows = []
    for name, val in values.items():
        rows.append(
            {
                "name": str(name),
                "value": round(float(val), 2),
                "share_pct": round((float(val) / total) * 100, 2),
            }
        )
    return rows


def _build_time_trends(df: pd.DataFrame, date_col: str, metric_col: str, agg_mode: str) -> dict[str, Any]:
    tmp = pd.DataFrame(
        {
            "_date": pd.to_datetime(df[date_col], errors="coerce"),
            "_metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()
    if tmp.empty:
        return {}
    tmp = tmp.sort_values("_date")
    g = tmp.groupby("_date")["_metric"]
    daily = (g.mean() if agg_mode == "mean" else g.sum()).reset_index()

    weekly_group = tmp.groupby(tmp["_date"].dt.to_period("W"))["_metric"]
    monthly_group = tmp.groupby(tmp["_date"].dt.to_period("M"))["_metric"]
    weekly = (weekly_group.mean() if agg_mode == "mean" else weekly_group.sum()).reset_index()
    monthly = (monthly_group.mean() if agg_mode == "mean" else monthly_group.sum()).reset_index()

    def _to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        f = frame.copy()
        f.columns = ["period", "value"]
        f["period"] = f["period"].astype(str)
        f["value"] = f["value"].astype(float).round(2)
        return f.to_dict(orient="records")

    trend_direction = "flat"
    if len(monthly) >= 3:
        y = monthly.iloc[:, 1].astype(float).values
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0])
        baseline = float(np.mean(np.abs(y))) or 1.0
        if slope > 0.01 * baseline:
            trend_direction = "upward"
        elif slope < -0.01 * baseline:
            trend_direction = "downward"

    return {
        "daily": _to_records(daily),
        "weekly": _to_records(weekly),
        "monthly": _to_records(monthly),
        "trend_direction": trend_direction,
        "agg_mode": agg_mode,
        "data_range_days": int((tmp["_date"].max() - tmp["_date"].min()).days),
    }


def _detect_anomalies(trends: dict[str, Any]) -> list[dict[str, Any]]:
    monthly = trends.get("monthly", [])
    if len(monthly) < 4:
        return []
    series = pd.Series([row.get("value", 0) for row in monthly], dtype="float64")
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return []
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out: list[dict[str, Any]] = []
    for row in monthly:
        val = float(row.get("value", 0))
        if val < lower or val > upper:
            out.append(
                {
                    "period": str(row.get("period")),
                    "value": round(val, 2),
                    "type": "high" if val > upper else "low",
                }
            )
    return out


def _apply_time_window(
    df: pd.DataFrame,
    date_col: str,
    reporting_months: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    warnings_out: list[str] = []
    if reporting_months > MAX_MONTHS:
        warnings_out.append(
            f"Requested window of {reporting_months} months exceeds the maximum of {MAX_MONTHS}. "
            f"Analysis capped at {MAX_MONTHS} months."
        )
        reporting_months = MAX_MONTHS

    tmp = df.copy()
    tmp["_date"] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=["_date"])
    if tmp.empty:
        return df, {}, warnings_out

    data_min = tmp["_date"].min()
    data_max = tmp["_date"].max()
    data_months = (data_max.year - data_min.year) * 12 + (data_max.month - data_min.month)
    if data_months < MIN_MONTHS:
        warnings_out.append(
            f"Dataset has about {data_months} month(s) of history. "
            f"At least {MIN_MONTHS} months is recommended for stable trend analysis."
        )

    cutoff = data_max - pd.DateOffset(months=reporting_months)
    filtered = tmp[tmp["_date"] > cutoff].drop(columns=["_date"])
    if filtered.empty:
        warnings_out.append(
            f"No rows found within the last {reporting_months} months; full dataset was used."
        )
        filtered = df

    reporting_window = {
        "requested_months": reporting_months,
        "cutoff_date": cutoff.strftime("%b %d, %Y"),
        "actual_to": data_max.strftime("%b %d, %Y"),
        "rows_after_filter": int(len(filtered)),
    }
    return filtered, reporting_window, warnings_out


def run_analytics(
    dataset_name: str,
    df: pd.DataFrame,
    reporting_months: int = 0,
) -> dict[str, Any]:
    logger.info("Analytics engine starting: %s (%d rows, %d cols)", dataset_name, len(df), len(df.columns))

    findings: dict[str, Any] = {
        "dataset_name": dataset_name,
        "row_count": int(len(df)),
        "detected_columns": {},
        "reporting_window": {},
        "guardrail_warnings": [],
        "kpis": {},
        "kpi_cards": [],
        "time_trends": {},
        "anomalies": [],
        "top_rankings": [],
        "best_worst_period": {},
        "reporting_period": "All available data",
        "metric": {},
    }

    if df.empty:
        findings["guardrail_warnings"].append("Dataset is empty; report contains only fallback summaries.")
        return findings

    date_cols = _detect_datetime_columns(df)
    metric_col = _choose_primary_metric(df)

    if not metric_col:
        findings["guardrail_warnings"].append(
            "No usable numeric metric column was detected. Charts and KPI values are limited."
        )
        return findings

    metric_label = _norm(metric_col).replace("_", " ").title()
    dims = _detect_dimensions(df, metric_col, date_cols)
    date_col = date_cols[0] if date_cols else None

    findings["detected_columns"] = {
        "metric_col": metric_col,
        "date_col": date_col,
        "dimension_cols": dims,
    }

    working_df = df
    if reporting_months and reporting_months > 0 and date_col:
        working_df, window, warns = _apply_time_window(df, date_col, reporting_months)
        findings["reporting_window"] = window
        findings["guardrail_warnings"].extend(warns)

    metric_series = pd.to_numeric(working_df[metric_col], errors="coerce").dropna()
    if metric_series.empty:
        findings["guardrail_warnings"].append(
            f"Metric column '{metric_col}' has no numeric values after cleaning."
        )
        return findings

    agg_mode = _choose_trend_agg_mode(working_df, metric_col, date_col) if date_col else "sum"
    metric_total = float(metric_series.sum())
    metric_avg = float(metric_series.mean())
    metric_median = float(metric_series.median())

    kpis = {
        "total_sales": round(metric_total, 2),  # backward compatibility with existing consumers
        "average_order_value": round(metric_avg, 2),
        "total_orders": int(len(working_df)),
        "units_sold": int(len(working_df)),
        "growth_over_previous_period": None,
        "trend_summary": "",
        "metric_total": round(metric_total, 2),
        "metric_average": round(metric_avg, 2),
        "metric_median": round(metric_median, 2),
        "metric_column": metric_col,
        "metric_label": metric_label,
    }

    trends = _build_time_trends(working_df, date_col, metric_col, agg_mode) if date_col else {}
    anomalies = _detect_anomalies(trends)

    monthly = trends.get("monthly", [])
    if len(monthly) >= 2:
        first_val = float(monthly[0]["value"])
        last_val = float(monthly[-1]["value"])
        if abs(first_val) > 1e-9:
            kpis["growth_over_previous_period"] = round(((last_val - first_val) / abs(first_val)) * 100, 2)

    trend_dir = trends.get("trend_direction", "flat")
    kpis["trend_summary"] = f"{metric_label} trend is {trend_dir}."

    rankings: list[dict[str, Any]] = []
    for dim in dims[:3]:
        items = _rank_dimension(working_df, dim, metric_col, agg_mode, n=10)
        if items:
            rankings.append(
                {
                    "dimension": dim,
                    "label": _norm(dim).replace("_", " ").title(),
                    "metric_label": metric_label,
                    "items": items,
                }
            )

    best_worst = {}
    if monthly:
        sorted_monthly = sorted(monthly, key=lambda x: float(x["value"]))
        best_worst = {
            "best_period": {
                "period": str(sorted_monthly[-1]["period"]),
                "value": round(float(sorted_monthly[-1]["value"]), 2),
            },
            "worst_period": {
                "period": str(sorted_monthly[0]["period"]),
                "value": round(float(sorted_monthly[0]["value"]), 2),
            },
        }

    if agg_mode == "mean":
        kpi_cards = [
            {"label": f"Average {metric_label}", "value": metric_avg, "subtitle": "Per record"},
            {"label": f"Median {metric_label}", "value": metric_median, "subtitle": "Central tendency"},
            {"label": "Records", "value": int(len(working_df)), "subtitle": "Rows analyzed"},
            {"label": "Distinct Columns", "value": int(len(working_df.columns)), "subtitle": "Dataset width"},
            {"label": "Period Growth", "value": kpis["growth_over_previous_period"], "subtitle": "First vs last period"},
        ]
    else:
        kpi_cards = [
            {"label": f"Total {metric_label}", "value": metric_total, "subtitle": "Across reporting period"},
            {"label": f"Average {metric_label}", "value": metric_avg, "subtitle": "Per record"},
            {"label": "Records", "value": int(len(working_df)), "subtitle": "Rows analyzed"},
            {"label": "Distinct Columns", "value": int(len(working_df.columns)), "subtitle": "Dataset width"},
            {"label": "Period Growth", "value": kpis["growth_over_previous_period"], "subtitle": "First vs last period"},
        ]

    if date_col:
        date_vals = pd.to_datetime(working_df[date_col], errors="coerce").dropna()
        if not date_vals.empty:
            findings["reporting_period"] = (
                f"{date_vals.min().strftime('%b %d, %Y')} - {date_vals.max().strftime('%b %d, %Y')}"
            )

    findings.update(
        {
            "row_count": int(len(working_df)),
            "kpis": kpis,
            "kpi_cards": kpi_cards,
            "time_trends": trends,
            "anomalies": anomalies,
            "top_rankings": rankings,
            "best_worst_period": best_worst,
            "metric": {
                "column": metric_col,
                "label": metric_label,
                "trend_agg_mode": agg_mode,
            },
            # Backward-compatible keys expected by older report templates/readers
            "top_5_products": rankings[0]["items"][:5] if rankings else [],
            "top_10_products": rankings[0]["items"][:10] if rankings else [],
            "top_5_regions": rankings[1]["items"][:5] if len(rankings) > 1 else [],
            "category_contribution": rankings[2]["items"][:20] if len(rankings) > 2 else [],
            "pareto": {},
            "seasonality_hint": None,
        }
    )

    if not date_col:
        findings["guardrail_warnings"].append(
            "No date column detected; time-trend and growth analyses are limited."
        )

    logger.info("Analytics engine complete: %s", dataset_name)
    return findings
