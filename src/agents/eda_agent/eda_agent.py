# EDA Agent
# - LLM decides which 3 plots are most useful per dataset
# - Plots generated on sampled data (100k rows max) for performance
# - Time series uses time-based resampling, not random sampling
# - Outputs: PNG files on disk + base64 in response + eda_insights for predictive agent

import base64
import io
import json
import os
import time
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from core.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_ENDPOINT,
    DATA_DIR,
)
from core.logger import get_logger

logger = get_logger("EDAAgent")

# ── Constants ─────────────────────────────────────────────────────────────────

EDA_OUTPUT_DIR       = str(DATA_DIR / "eda")
PLOT_SAMPLE_ROWS     = 100_000   # max rows passed to matplotlib
PLOTS_PER_DATASET    = 3         # always 3 per dataset
CATEGORY_MAX_BARS    = 15        # top N categories shown in bar chart
HEATMAP_MAX_COLS     = 20        # cap correlation matrix size
TIME_RESAMPLE_TARGET = 200       # target number of points on time series plot

AVAILABLE_PLOT_TYPES = [
    "histogram",          # distribution of a numeric column
    "kde",                # smooth density curve for a numeric column
    "boxplot",            # outlier detection per numeric column
    "correlation_heatmap",# correlation matrix of all numeric columns
    "missing_heatmap",    # visual of null values across columns
    "timeseries",         # value over time — triggers seasonality detection
    "categorical_bar",    # top-N value counts for a categorical column
    "scatter",            # relationship between two numeric columns
]

# Seaborn theme applied once at module level
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _sample_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Random sample capped at PLOT_SAMPLE_ROWS. Returns full df if small."""
    if len(df) <= PLOT_SAMPLE_ROWS:
        return df
    return df.sample(n=PLOT_SAMPLE_ROWS, random_state=42).reset_index(drop=True)


def _resample_timeseries(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Resamples a time series to ~TIME_RESAMPLE_TARGET points by auto-picking frequency.
    Preserves trend shape better than random sampling.
    """
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")

    n = len(ts)
    if n <= TIME_RESAMPLE_TARGET:
        return ts.reset_index()

    # Pick resample frequency based on date range
    date_range_days = (ts.index.max() - ts.index.min()).days
    if date_range_days > 365 * 3:
        freq = "ME"       # monthly
    elif date_range_days > 90:
        freq = "W"        # weekly
    else:
        freq = "D"        # daily

    resampled = ts[value_col].resample(freq).mean().dropna().reset_index()
    return resampled


# ── Metadata builder ──────────────────────────────────────────────────────────

def _build_metadata(df: pd.DataFrame) -> dict:
    """Builds a rich but compact metadata dict for the LLM.
    Uses full dataset stats (not sample) so the LLM sees accurate numbers.
    """
    numeric_cols    = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols   = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    # Also detect string columns that look like dates
    import warnings
    for col in categorical_cols:
        sample = df[col].dropna().head(5).astype(str).tolist()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(sample, errors="raise")
            datetime_cols.append(col)
        except Exception:
            pass

    col_stats = []
    for col in df.columns:
        series = df[col].dropna()
        entry: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(series.nunique()),
            "sample_values": series.head(5).astype(str).tolist(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            entry["min"]      = float(series.min())   if len(series) else None
            entry["max"]      = float(series.max())   if len(series) else None
            entry["mean"]     = float(series.mean())  if len(series) else None
            entry["std"]      = float(series.std())   if len(series) else None
            entry["skewness"] = float(series.skew())  if len(series) else None
        elif col in categorical_cols:
            entry["top_values"] = series.value_counts().head(5).to_dict()
        col_stats.append(entry)

    return {
        "row_count":        len(df),
        "column_count":     len(df.columns),
        "numeric_columns":  numeric_cols,
        "categorical_columns": [c for c in categorical_cols if c not in datetime_cols],
        "datetime_columns": datetime_cols,
        "has_missing":      bool(df.isna().any().any()),
        "total_null_pct":   round(df.isna().mean().mean() * 100, 2),
        "columns":          col_stats,
        "available_plot_types": AVAILABLE_PLOT_TYPES,
    }


# ── LLM plan selection ────────────────────────────────────────────────────────

def _select_plots_with_llm(dataset_name: str, metadata: dict, n_plots: int) -> list[dict]:
    """Asks Azure OpenAI to pick the n_plots most useful visualizations.
    Falls back to heuristics if LLM is unavailable.
    """
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.warning("openai not installed — using heuristic fallback")
        return _heuristic_plan(metadata, n_plots)

    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        logger.warning("Azure OpenAI not configured — using heuristic fallback")
        return _heuristic_plan(metadata, n_plots)

    system_prompt = (
        "You are a senior data analyst helping small business owners understand their data. "
        "Pick the most insightful visualizations that reveal patterns, outliers, trends, or relationships "
        "a business owner would actually care about. "
        "Think about what will later help a predictive model selection agent — "
        "e.g. if you see date columns with many rows, a timeseries plot helps detect seasonality. "
        "If you see high skewness, a histogram or boxplot is useful. "
        "If many columns are numeric, a correlation heatmap reveals feature relationships."
    )

    user_prompt = {
        "task": (
            f"Given the dataset metadata below, select exactly {n_plots} visualizations "
            f"that are most useful for understanding this data. "
            f"Choose only from the available_plot_types. "
            f"Use only column names that exist exactly in the metadata."
        ),
        "dataset_name": dataset_name,
        "metadata": metadata,
        "HARD_RULES": [
            "Return exactly a JSON array with exactly {n_plots} plot objects — no more, no less.".replace("{n_plots}", str(n_plots)),
            "Only use column names from metadata.columns[*].name.",
            "For timeseries: date_column must be from datetime_columns, value_column must be from numeric_columns.",
            "For histogram, kde, boxplot: column must be from numeric_columns.",
            "For categorical_bar: column must be from categorical_columns.",
            "For scatter: both x_column and y_column must be from numeric_columns.",
            "For correlation_heatmap and missing_heatmap: no column fields needed.",
            "Do not pick missing_heatmap if has_missing is false.",
            "Do not pick timeseries if datetime_columns is empty.",
            "Do not pick correlation_heatmap or scatter if numeric_columns has fewer than 2 columns.",
        ],
        "output_format": {
            "description": "Return a JSON array of plot objects only. No explanation.",
            "plot_object_fields": {
                "type": "one of the available_plot_types",
                "column": "for histogram, kde, boxplot",
                "date_column": "for timeseries",
                "value_column": "for timeseries",
                "x_column": "for scatter",
                "y_column": "for scatter",
                "reason": "one sentence why this plot is useful for this dataset",
            },
            "example": [
                {"type": "timeseries", "date_column": "date", "value_column": "sales", "reason": "Reveals seasonal sales patterns"},
                {"type": "correlation_heatmap", "reason": "Shows which features move together"},
                {"type": "categorical_bar", "column": "store_type", "reason": "Shows revenue distribution by store type"},
            ],
        },
    }

    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(user_prompt)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        # LLM sometimes wraps array in a key
        plots = parsed if isinstance(parsed, list) else next(
            (v for v in parsed.values() if isinstance(v, list)), []
        )

        validated = _validate_plot_specs(plots, metadata)
        if len(validated) < n_plots:
            logger.warning("LLM returned %d valid plots, needed %d — padding with heuristics", len(validated), n_plots)
            fallback = _heuristic_plan(metadata, n_plots)
            seen_types = {p["type"] for p in validated}
            for fb in fallback:
                if len(validated) >= n_plots:
                    break
                if fb["type"] not in seen_types:
                    validated.append(fb)
                    seen_types.add(fb["type"])

        logger.info("LLM selected %d plot(s) for %s", len(validated[:n_plots]), dataset_name)
        return validated[:n_plots]

    except Exception as e:
        logger.warning("LLM plot selection failed: %s — using heuristic fallback", e)
        return _heuristic_plan(metadata, n_plots)


def _validate_plot_specs(plots: list, metadata: dict) -> list:
    """Filters out plot specs that reference non-existent columns or break rules."""
    all_cols      = {c["name"] for c in metadata.get("columns", [])}
    numeric_cols  = set(metadata.get("numeric_columns", []))
    cat_cols      = set(metadata.get("categorical_columns", []))
    date_cols     = set(metadata.get("datetime_columns", []))
    has_missing   = metadata.get("has_missing", False)

    valid = []
    for p in plots:
        t = p.get("type", "")
        try:
            if t == "histogram" or t == "kde" or t == "boxplot":
                assert p.get("column") in numeric_cols
            elif t == "timeseries":
                assert p.get("date_column") in date_cols | all_cols
                assert p.get("value_column") in numeric_cols
            elif t == "categorical_bar":
                assert p.get("column") in cat_cols | all_cols
            elif t == "scatter":
                assert p.get("x_column") in numeric_cols
                assert p.get("y_column") in numeric_cols
            elif t == "correlation_heatmap":
                assert len(numeric_cols) >= 2
            elif t == "missing_heatmap":
                assert has_missing
            elif t not in AVAILABLE_PLOT_TYPES:
                continue
            valid.append(p)
        except AssertionError:
            logger.warning("Skipping invalid plot spec: %s", p)
    return valid


def _heuristic_plan(metadata: dict, n_plots: int) -> list[dict]:
    """Rule-based fallback when LLM is unavailable."""
    numeric  = metadata.get("numeric_columns", [])
    cats     = metadata.get("categorical_columns", [])
    dates    = metadata.get("datetime_columns", [])
    missing  = metadata.get("has_missing", False)

    candidates = []

    if dates and numeric:
        candidates.append({"type": "timeseries", "date_column": dates[0], "value_column": numeric[0],
                           "reason": f"Trend of {numeric[0]} over time"})
    if len(numeric) >= 2:
        candidates.append({"type": "correlation_heatmap", "reason": "Feature relationships"})
    if numeric:
        candidates.append({"type": "histogram", "column": numeric[0], "reason": f"Distribution of {numeric[0]}"})
    if numeric:
        candidates.append({"type": "boxplot", "column": numeric[0], "reason": f"Outliers in {numeric[0]}"})
    if cats:
        candidates.append({"type": "categorical_bar", "column": cats[0], "reason": f"Distribution of {cats[0]}"})
    if missing:
        candidates.append({"type": "missing_heatmap", "reason": "Null value pattern"})
    if len(numeric) >= 2:
        candidates.append({"type": "scatter", "x_column": numeric[0], "y_column": numeric[1],
                           "reason": f"Relationship between {numeric[0]} and {numeric[1]}"})

    return candidates[:n_plots]


# ── Plot renderers ────────────────────────────────────────────────────────────

def _fig_to_base64(fig: plt.Figure) -> str:
    """Converts a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _save_and_encode(fig: plt.Figure, path: str) -> str:
    """Saves figure to disk AND returns base64."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=120)
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _render_histogram(df: pd.DataFrame, col: str, title_prefix: str, out_path: str) -> dict:
    sample = _sample_for_plot(df)
    data   = pd.to_numeric(sample[col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(f"{title_prefix} — Distribution of {col}", fontsize=13)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    b64 = _save_and_encode(fig, out_path)
    return {"type": "histogram", "column": col, "path": out_path, "base64": b64}


def _render_kde(df: pd.DataFrame, col: str, title_prefix: str, out_path: str) -> dict:
    sample = _sample_for_plot(df)
    data   = pd.to_numeric(sample[col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data, ax=ax, fill=True, color="#4C72B0", alpha=0.6)
    ax.set_title(f"{title_prefix} — Density of {col}", fontsize=13)
    ax.set_xlabel(col)
    b64 = _save_and_encode(fig, out_path)
    return {"type": "kde", "column": col, "path": out_path, "base64": b64}


def _render_boxplot(df: pd.DataFrame, col: str, title_prefix: str, out_path: str) -> dict:
    sample = _sample_for_plot(df)
    data   = pd.to_numeric(sample[col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    ax.set_title(f"{title_prefix} — Outliers in {col}", fontsize=13)
    ax.set_ylabel(col)
    b64 = _save_and_encode(fig, out_path)
    return {"type": "boxplot", "column": col, "path": out_path, "base64": b64}


def _render_correlation_heatmap(df: pd.DataFrame, title_prefix: str, out_path: str) -> dict:
    sample      = _sample_for_plot(df)
    numeric_df  = sample.select_dtypes(include=["number"]).iloc[:, :HEATMAP_MAX_COLS]

    if numeric_df.shape[1] < 2:
        raise ValueError("Not enough numeric columns for correlation heatmap.")

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(min(14, corr.shape[1] + 2), min(12, corr.shape[0] + 2)))
    sns.heatmap(corr, annot=corr.shape[1] <= 12, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.4, square=True)
    ax.set_title(f"{title_prefix} — Correlation Heatmap", fontsize=13)
    b64 = _save_and_encode(fig, out_path)
    return {"type": "correlation_heatmap", "path": out_path, "base64": b64}


def _render_missing_heatmap(df: pd.DataFrame, title_prefix: str, out_path: str) -> dict:
    # Cap columns for readability
    sample = df.iloc[:min(500, len(df)), :30]
    null_matrix = sample.isna()

    fig, ax = plt.subplots(figsize=(min(16, null_matrix.shape[1] + 2), 6))
    sns.heatmap(null_matrix, cbar=False, yticklabels=False,
                cmap=["#4C72B0", "#f5f5f5"], ax=ax)
    ax.set_title(f"{title_prefix} — Missing Values", fontsize=13)
    ax.set_xlabel("Columns")
    b64 = _save_and_encode(fig, out_path)
    return {"type": "missing_heatmap", "path": out_path, "base64": b64}


def _render_timeseries(df: pd.DataFrame, date_col: str, value_col: str,
                       title_prefix: str, out_path: str) -> dict:
    resampled = _resample_timeseries(df, date_col, value_col)
    x = resampled.iloc[:, 0]
    y = pd.to_numeric(resampled[value_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, color="#4C72B0", linewidth=1.5)
    ax.fill_between(x, y, alpha=0.15, color="#4C72B0")
    ax.set_title(f"{title_prefix} — {value_col} over Time", fontsize=13)
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    fig.autofmt_xdate()
    b64 = _save_and_encode(fig, out_path)
    return {"type": "timeseries", "date_column": date_col, "value_column": value_col,
            "path": out_path, "base64": b64}


def _render_categorical_bar(df: pd.DataFrame, col: str, title_prefix: str, out_path: str) -> dict:
    counts = df[col].astype(str).value_counts().head(CATEGORY_MAX_BARS)

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(f"{title_prefix} — {col} Distribution", fontsize=13)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=35, ha="right")
    b64 = _save_and_encode(fig, out_path)
    return {"type": "categorical_bar", "column": col, "path": out_path, "base64": b64}


def _render_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                    title_prefix: str, out_path: str) -> dict:
    sample = _sample_for_plot(df)
    x = pd.to_numeric(sample[x_col], errors="coerce")
    y = pd.to_numeric(sample[y_col], errors="coerce")
    mask = x.notna() & y.notna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[mask], y[mask], alpha=0.3, s=10, color="#4C72B0")
    ax.set_title(f"{title_prefix} — {x_col} vs {y_col}", fontsize=13)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    b64 = _save_and_encode(fig, out_path)
    return {"type": "scatter", "x_column": x_col, "y_column": y_col,
            "path": out_path, "base64": b64}


def _render_plot(spec: dict, df: pd.DataFrame, title_prefix: str, out_dir: str, slug: str) -> Optional[dict]:
    """Dispatches a plot spec to the correct renderer. Returns result dict or None on failure."""
    t = spec.get("type", "")
    path = os.path.join(out_dir, f"{slug}_{t}.png")

    try:
        if t == "histogram":
            return _render_histogram(df, spec["column"], title_prefix, path)
        elif t == "kde":
            return _render_kde(df, spec["column"], title_prefix, path)
        elif t == "boxplot":
            return _render_boxplot(df, spec["column"], title_prefix, path)
        elif t == "correlation_heatmap":
            return _render_correlation_heatmap(df, title_prefix, path)
        elif t == "missing_heatmap":
            return _render_missing_heatmap(df, title_prefix, path)
        elif t == "timeseries":
            return _render_timeseries(df, spec["date_column"], spec["value_column"], title_prefix, path)
        elif t == "categorical_bar":
            return _render_categorical_bar(df, spec["column"], title_prefix, path)
        elif t == "scatter":
            return _render_scatter(df, spec["x_column"], spec["y_column"], title_prefix, path)
        else:
            logger.warning("Unknown plot type: %s", t)
            return None
    except Exception as e:
        logger.warning("Plot '%s' failed for %s: %s", t, title_prefix, e)
        return None


# ── LLM insight generation ────────────────────────────────────────────────────

def _generate_eda_insights(dataset_name: str, metadata: dict, plot_specs: list[dict]) -> dict:
    """Calls Azure OpenAI to produce structured insights for the predictive agent.
    Falls back to a safe default dict on any failure.
    """
    try:
        from openai import AzureOpenAI
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("Azure OpenAI not configured")

        context = {
            "dataset_name":   dataset_name,
            "row_count":      metadata["row_count"],
            "numeric_columns": metadata["numeric_columns"],
            "datetime_columns": metadata["datetime_columns"],
            "categorical_columns": metadata["categorical_columns"],
            "has_missing":    metadata["has_missing"],
            "total_null_pct": metadata["total_null_pct"],
            "column_stats":   metadata["columns"],
            "plots_generated": [{"type": p.get("type"), "reason": p.get("reason")} for p in plot_specs],
        }

        prompt = (
            "You are a data science advisor. Based on the dataset metadata below, "
            "produce a structured JSON insight object that will guide a predictive modelling agent. "
            "Be specific — use actual column names and real values from the metadata.\n\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            "{\n"
            '  "has_seasonality": true/false,\n'
            '  "trend_direction": "upward" | "downward" | "flat" | "unknown",\n'
            '  "skewed_columns": ["col1", ...],\n'
            '  "high_correlations": [["col_a", "col_b"], ...],\n'
            '  "dominant_categories": {"col": "top_value", ...},\n'
            '  "data_quality_issues": ["issue1", ...],\n'
            '  "recommended_models": ["SARIMA", "XGBoost", ...],\n'
            '  "reasoning": "one paragraph plain English explanation"\n'
            "}\n\n"
            f"Dataset context:\n{json.dumps(context)}"
        )

        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)

    except Exception as e:
        logger.warning("EDA insight generation failed: %s — returning defaults", e)
        return {
            "has_seasonality":    bool(metadata.get("datetime_columns")),
            "trend_direction":    "unknown",
            "skewed_columns":     [c["name"] for c in metadata.get("columns", [])
                                   if isinstance(c.get("skewness"), float) and abs(c["skewness"]) > 1],
            "high_correlations":  [],
            "dominant_categories": {},
            "data_quality_issues": ["high null rate"] if metadata.get("total_null_pct", 0) > 10 else [],
            "recommended_models": ["SARIMA"] if metadata.get("datetime_columns") else ["XGBoost", "LinearRegression"],
            "reasoning":          "Insights generated via fallback heuristics — LLM unavailable.",
        }


# ── Public entry point ────────────────────────────────────────────────────────

def run_eda(
    dataset_name: str,
    df: pd.DataFrame,
    n_plots: int = PLOTS_PER_DATASET,
) -> dict:
    """Main entry point. Runs full EDA on a single dataset.

    Args:
        dataset_name: Human-readable name (used in titles and filenames).
        df:           The DataFrame to analyse (full dataset — sampling handled internally).
        n_plots:      Number of plots to generate (default 3).

    Returns a dict with:
        - plots:        list of {type, path, base64, ...}
        - eda_insights: structured insights for the predictive agent
        - plot_specs:   the LLM-chosen plan
        - duration_seconds
        - status
    """
    started_at = time.time()
    logger.info("EDA starting: %s (%d rows, %d cols)", dataset_name, len(df), len(df.columns))

    # Sanitise name for use in filenames
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in dataset_name).strip("_") or "dataset"
    out_dir   = os.path.join(EDA_OUTPUT_DIR, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    # Build metadata from full dataset
    metadata = _build_metadata(df)

    # LLM picks the plots
    plot_specs = _select_plots_with_llm(dataset_name, metadata, n_plots)
    logger.info("Plot plan for %s: %s", dataset_name, [p["type"] for p in plot_specs])

    # Render each plot
    plots = []
    for spec in plot_specs:
        slug   = f"{safe_name}_{len(plots)}"
        result = _render_plot(spec, df, dataset_name, out_dir, slug)
        if result:
            result["reason"] = spec.get("reason", "")
            plots.append(result)
        else:
            logger.warning("Plot failed, skipping: %s", spec)

    # Generate structured insights for predictive agent
    eda_insights = _generate_eda_insights(dataset_name, metadata, plot_specs)

    duration = round(time.time() - started_at, 4)
    logger.info("EDA complete: %s — %d plots in %.2fs", dataset_name, len(plots), duration)

    return {
        "dataset_name":    dataset_name,
        "row_count":       len(df),
        "column_count":    len(df.columns),
        "sampled_for_plots": len(df) > PLOT_SAMPLE_ROWS,
        "plot_specs":      plot_specs,
        "plots":           plots,
        "eda_insights":    eda_insights,
        "output_dir":      out_dir,
        "duration_seconds": duration,
        "status":          "completed" if plots else "failed",
    }