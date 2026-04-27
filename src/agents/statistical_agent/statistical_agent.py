# Statistical Testing Agent
# - No hardcoded test list — LLM picks freely from all scipy tests
# - Dynamic runner executes any test the LLM specifies
# - Business insight generated per test in plain English
# - Robust fallbacks for edge cases (too few rows, wrong column types, etc.)

import json
import time
from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
)
from src.core.logger import get_logger

logger = get_logger("StatisticalAgent")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    except Exception:
        return None


def _describe_schema(df: pd.DataFrame) -> dict:
    """Builds a rich schema description including column stats and sample values."""
    schema = []
    for col in df.columns:
        series = df[col].dropna()
        entry = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(series.count()),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(series.nunique()),
            "sample_values": series.head(5).astype(str).tolist(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            entry["min"] = _safe_float(series.min())
            entry["max"] = _safe_float(series.max())
            entry["mean"] = _safe_float(series.mean())
            entry["std"] = _safe_float(series.std())
            entry["usable_for"] = ["normality", "correlation", "ttest", "anova", "adfuller"]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            entry["date_range"] = f"{series.min()} to {series.max()}"
            entry["usable_for"] = ["context_only — DO NOT pass to any test as a column"]
        else:
            entry["usable_for"] = ["group_column in group tests", "chi_square", "fisher_exact"]
        schema.append(entry)

    numeric_cols = [c["name"] for c in schema if "int" in c["dtype"] or "float" in c["dtype"]]
    categorical_cols = [c["name"] for c in schema if c["dtype"] == "object"]
    datetime_cols = [c["name"] for c in schema if "datetime" in c["dtype"]]

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "IMPORTANT_RULES": [
            "datetime_columns are CONTEXT ONLY — never put them in 'columns', 'group_column', or 'value_column'.",
            "For adfuller: 'columns' must contain exactly ONE column from numeric_columns.",
            "For correlation tests: both columns in 'columns' must be from numeric_columns.",
            "For group tests (kruskal, f_oneway, mannwhitneyu, ttest_ind): 'group_column' must be from categorical_columns, 'value_column' must be from numeric_columns.",
            "For chi2_contingency: both columns must be from categorical_columns.",
        ],
        "columns": schema,
    }


# ── LLM test selection ────────────────────────────────────────────────────────

def _select_tests_with_llm(dataset_name: str, schema: dict) -> list:
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.warning("openai not installed — using heuristic fallback")
        return _heuristic_test_selection(schema)

    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        logger.warning("Azure OpenAI not configured — using heuristic fallback")
        return _heuristic_test_selection(schema)

    system_prompt = (
        "You are a senior data analyst helping small business owners understand their data. "
        "Pick the most BUSINESS-RELEVANT statistical tests — focus on questions a business owner "
        "would actually care about: Are sales different across categories? Is there a trend over time? "
        "Do two metrics move together? Which group performs best? "
        "You have access to the full scipy.stats library and statsmodels — pick any test that fits. "
        "Always think: what business decision could this test help make?"
    )

    user_prompt = {
        "task": (
            "Given the dataset schema below, select up to 3 statistical tests that would be most "
            "meaningful and actionable for a small business owner. "
            "Choose only tests that genuinely fit the data — return fewer than 3 if appropriate. "
            "You can use ANY scipy.stats or statsmodels test, not just a predefined list."
        ),
        "dataset_name": dataset_name,
        "schema": schema,
        "HARD_RULES_YOU_MUST_FOLLOW": [
            "NEVER put a datetime column in 'columns', 'group_column', or 'value_column'. Datetime columns are context only.",
            "For adfuller: 'columns' must contain exactly ONE column from schema.numeric_columns. Never use a datetime column.",
            "For pearsonr, spearmanr, kendalltau: both entries in 'columns' must be from schema.numeric_columns.",
            "For shapiro, normaltest, kstest: 'columns' must contain ONE column from schema.numeric_columns.",
            "For kruskal, f_oneway, mannwhitneyu, ttest_ind, levene, bartlett: 'group_column' must be a low-cardinality column (categorical string OR integer ID like store_nbr, product_id) AND 'value_column' must be from schema.numeric_columns.",
            "For chi2_contingency, fisher_exact: both columns must be from schema.categorical_columns.",
            "If schema.numeric_columns is empty, only select chi2_contingency or fisher_exact.",
            "Only use column names that exist EXACTLY in the schema.",
        ],
        "output_format": {
            "description": "Return a JSON object with key 'tests' containing an array of test objects.",
            "test_object_fields": {
                "test_id": "A short snake_case identifier you choose (e.g. 'pearson_correlation', 'anova_sales_by_category', 'adf_revenue_trend')",
                "test_type": (
                    "The scipy/statsmodels function to call. Must be one of: "
                    "'shapiro', 'kstest', 'pearsonr', 'spearmanr', 'kendalltau', "
                    "'ttest_ind', 'ttest_1samp', 'ttest_rel', "
                    "'f_oneway', 'chi2_contingency', 'fisher_exact', "
                    "'mannwhitneyu', 'kruskal', 'friedmanchisquare', "
                    "'levene', 'bartlett', 'mood', 'wilcoxon', "
                    "'pointbiserialr', 'adfuller', 'normaltest'"
                ),
                "columns": "List of column names — must follow HARD_RULES above",
                "group_column": "For group-based tests: one column from schema.categorical_columns",
                "value_column": "For group-based tests: one column from schema.numeric_columns",
                "reason": "One sentence: why is this test relevant for a business owner?",
            }
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
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        tests = parsed.get("tests", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(tests, list):
            tests = []

        numeric_cols = set(schema.get("numeric_columns", []))
        categorical_cols = set(schema.get("categorical_columns", []))
        datetime_cols = set(schema.get("datetime_columns", []))
        all_cols = set(c["name"] for c in schema.get("columns", []))

        # Build a set of all columns that are usable as group columns:
        # - categorical (object) columns always qualify
        # - low-cardinality numeric columns qualify (e.g. store_nbr, product_id)
        col_unique_counts = {c["name"]: c["unique_count"] for c in schema.get("columns", [])}
        row_count = schema.get("row_count", 1)
        valid_group_cols = set()
        for col_name, unique_count in col_unique_counts.items():
            if col_name in datetime_cols:
                continue
            # Categorical columns always valid as group col
            if col_name in categorical_cols:
                valid_group_cols.add(col_name)
            # Numeric columns with low cardinality are valid group cols (IDs, categories encoded as int)
            elif col_name in numeric_cols and unique_count <= max(50, row_count * 0.05):
                valid_group_cols.add(col_name)

        col_non_null_counts = {c["name"]: int(c.get("non_null_count", 0)) for c in schema.get("columns", [])}
        valid = []
        for t in tests:
            test_type = t.get("test_type", "")
            columns = t.get("columns", [])
            group_col = t.get("group_column", "")
            value_col = t.get("value_column", "")

            # Skip if any column doesn't exist
            all_used = [c for c in columns + [group_col, value_col] if c]
            if any(c not in all_cols for c in all_used):
                logger.warning("Skipping test %s — references non-existent columns: %s", test_type, all_used)
                continue

            # Skip if datetime column used where it shouldn't be
            if any(c in datetime_cols for c in all_used):
                logger.warning("Skipping test %s — datetime column used incorrectly", test_type)
                continue

            # adfuller columns must all be numeric
            if test_type == "adfuller" and not all(c in numeric_cols for c in columns):
                logger.warning("Skipping adfuller — non-numeric column in columns list: %s", columns)
                continue
            if test_type == "adfuller":
                # ADF needs a single numeric series with enough observations.
                adf_candidates = [c for c in columns if c in numeric_cols]
                if value_col and value_col in numeric_cols:
                    adf_candidates.append(value_col)
                has_sufficient_series = any(col_non_null_counts.get(c, 0) >= 12 for c in adf_candidates)
                if not has_sufficient_series:
                    logger.warning(
                        "Skipping adfuller — no numeric series with 12+ observations in %s",
                        adf_candidates,
                    )
                    continue

            # Correlation tests must use numeric columns
            if test_type in ("pearsonr", "spearmanr", "kendalltau") and not all(c in numeric_cols for c in columns):
                logger.warning("Skipping %s — non-numeric column used", test_type)
                continue

            # Group tests: group_column must be a valid grouping col, value_column must be numeric
            if test_type in ("kruskal", "f_oneway", "mannwhitneyu", "ttest_ind", "levene", "bartlett"):
                if group_col and group_col not in valid_group_cols:
                    logger.warning("Skipping %s — group_column '%s' is not a valid grouping column (too many unique values or is datetime)", test_type, group_col)
                    continue
                if value_col and value_col not in numeric_cols:
                    logger.warning("Skipping %s — value_column '%s' is not numeric", test_type, value_col)
                    continue

            if t.get("test_type") and t.get("columns") is not None:
                valid.append(t)

        logger.info("LLM selected %d valid test(s) for %s", len(valid), dataset_name)
        return valid[:3]
    except Exception as e:
        logger.warning("LLM test selection failed: %s — using heuristic fallback", e)
        return _heuristic_test_selection(schema)


def _heuristic_test_selection(schema: dict) -> list:
    """Fallback: rule-based test selection when LLM is unavailable."""
    tests = []
    numeric = schema.get("numeric_columns", [])
    categorical = schema.get("categorical_columns", [])
    datetime_cols = schema.get("datetime_columns", [])

    if len(numeric) >= 1:
        tests.append({
            "test_id": "normality_check",
            "test_type": "shapiro",
            "columns": [numeric[0]],
            "reason": f"Check whether {numeric[0]} is normally distributed.",
        })
    if len(numeric) >= 2:
        tests.append({
            "test_id": "correlation_check",
            "test_type": "pearsonr",
            "columns": numeric[:2],
            "reason": f"Check whether {numeric[0]} and {numeric[1]} move together.",
        })
    non_null_map = {c["name"]: int(c.get("non_null_count", 0)) for c in schema.get("columns", [])}
    adf_candidate = next((c for c in numeric if non_null_map.get(c, 0) >= 12), None)

    if datetime_cols and adf_candidate:
        tests.append({
            "test_id": "trend_stationarity",
            "test_type": "adfuller",
            "columns": [adf_candidate],
            "reason": f"Check whether {adf_candidate} has a stable trend over time.",
        })
    elif categorical and len(numeric) >= 1:
        tests.append({
            "test_id": "group_comparison",
            "test_type": "kruskal",
            "columns": [numeric[0]],
            "group_column": categorical[0],
            "value_column": numeric[0],
            "reason": f"Compare {numeric[0]} across {categorical[0]} groups.",
        })
    return tests[:3]


# ── Dynamic test runner ───────────────────────────────────────────────────────

def _run_test_dynamic(test_spec: dict, df: pd.DataFrame) -> tuple:
    """
    Dynamically runs any supported scipy/statsmodels test.
    Returns (statistic, p_value, additional_metrics, extra_for_interpret).
    Raises ValueError or RuntimeError on invalid inputs.
    """
    test_type = test_spec.get("test_type", "")
    columns = test_spec.get("columns", [])
    group_col = test_spec.get("group_column", "")
    value_col = test_spec.get("value_column", columns[0] if columns else "")

    # Validate columns exist
    all_cols = columns + ([group_col] if group_col else []) + ([value_col] if value_col and value_col not in columns else [])
    missing = [c for c in all_cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")

    def get_numeric(col):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 3:
            raise ValueError(f"Column '{col}' has fewer than 3 numeric values — test cannot run.")
        return s

    def get_groups(group_col, value_col, min_groups=2):
        groups = {name: pd.to_numeric(g[value_col], errors="coerce").dropna().values
                  for name, g in df.groupby(group_col)}
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        if len(groups) < min_groups:
            raise ValueError(f"Need at least {min_groups} non-empty groups in '{group_col}'. Found: {len(groups)}.")
        return groups

    stat, p, additional, extra = None, None, {}, {}

    # ── Normality tests ───────────────────────────────────────────────────────
    if test_type == "shapiro":
        col = get_numeric(columns[0])
        sample = col.sample(min(len(col), 5000), random_state=42) if len(col) > 5000 else col
        stat, p = stats.shapiro(sample)
        additional = {
            "sample_size": len(sample),
            "mean": _safe_float(col.mean()),
            "std": _safe_float(col.std()),
            "skewness": _safe_float(col.skew()),
            "kurtosis": _safe_float(col.kurtosis()),
        }

    elif test_type == "normaltest":
        col = get_numeric(columns[0])
        stat, p = stats.normaltest(col)
        additional = {"sample_size": len(col), "mean": _safe_float(col.mean()), "std": _safe_float(col.std())}

    elif test_type == "kstest":
        col = get_numeric(columns[0])
        stat, p = stats.kstest(col, "norm", args=(float(col.mean()), float(col.std())))
        additional = {"sample_size": len(col)}

    # ── Correlation tests ─────────────────────────────────────────────────────
    elif test_type == "pearsonr":
        common = df[[columns[0], columns[1]]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(common) < 3:
            raise ValueError("Not enough overlapping numeric rows for Pearson correlation.")
        corr, p = stats.pearsonr(common[columns[0]], common[columns[1]])
        stat = corr
        extra["correlation"] = _safe_float(corr)
        additional = {
            "correlation_coefficient": _safe_float(corr),
            "r_squared": _safe_float(corr ** 2),
            "sample_size": len(common),
        }

    elif test_type == "spearmanr":
        common = df[[columns[0], columns[1]]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(common) < 3:
            raise ValueError("Not enough overlapping rows for Spearman correlation.")
        corr, p = stats.spearmanr(common[columns[0]], common[columns[1]])
        stat = corr
        extra["correlation"] = _safe_float(corr)
        additional = {"rho": _safe_float(corr), "sample_size": len(common)}

    elif test_type == "kendalltau":
        common = df[[columns[0], columns[1]]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(common) < 3:
            raise ValueError("Not enough overlapping rows for Kendall's tau.")
        tau, p = stats.kendalltau(common[columns[0]], common[columns[1]])
        stat = tau
        extra["correlation"] = _safe_float(tau)
        additional = {"tau": _safe_float(tau), "sample_size": len(common)}

    elif test_type == "pointbiserialr":
        binary = df[columns[0]].dropna()
        numeric = get_numeric(columns[1])
        common = df[[columns[0], columns[1]]].dropna()
        corr, p = stats.pointbiserialr(common[columns[0]], pd.to_numeric(common[columns[1]], errors="coerce"))
        stat = corr
        extra["correlation"] = _safe_float(corr)
        additional = {"correlation": _safe_float(corr), "sample_size": len(common)}

    # ── T-tests ───────────────────────────────────────────────────────────────
    elif test_type == "ttest_ind":
        groups = get_groups(group_col, value_col, min_groups=2)
        keys = list(groups.keys())
        g1, g2 = groups[keys[0]], groups[keys[1]]
        stat, p = stats.ttest_ind(g1, g2)
        additional = {
            "group_1": str(keys[0]), "group_1_mean": _safe_float(g1.mean()), "group_1_n": len(g1),
            "group_2": str(keys[1]), "group_2_mean": _safe_float(g2.mean()), "group_2_n": len(g2),
            "mean_difference": _safe_float(g1.mean() - g2.mean()),
        }

    elif test_type == "ttest_1samp":
        col = get_numeric(columns[0])
        popmean = float(col.mean())
        stat, p = stats.ttest_1samp(col, popmean)
        additional = {"sample_mean": _safe_float(col.mean()), "hypothesized_mean": _safe_float(popmean), "sample_size": len(col)}

    elif test_type == "ttest_rel":
        common = df[[columns[0], columns[1]]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(common) < 3:
            raise ValueError("Not enough paired rows for paired t-test.")
        stat, p = stats.ttest_rel(common[columns[0]], common[columns[1]])
        additional = {
            "mean_diff": _safe_float((common[columns[0]] - common[columns[1]]).mean()),
            "sample_size": len(common),
        }

    elif test_type == "wilcoxon":
        common = df[[columns[0], columns[1]]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(common) < 3:
            raise ValueError("Not enough paired rows for Wilcoxon test.")
        stat, p = stats.wilcoxon(common[columns[0]], common[columns[1]])
        additional = {"sample_size": len(common)}

    # ── ANOVA / non-parametric group tests ────────────────────────────────────
    elif test_type == "f_oneway":
        groups = get_groups(group_col, value_col)
        stat, p = stats.f_oneway(*groups.values())
        additional = {
            "n_groups": len(groups),
            "group_column": group_col,
            "value_column": value_col,
            "group_means": {str(k): _safe_float(v.mean()) for k, v in groups.items()},
        }

    elif test_type == "kruskal":
        groups = get_groups(group_col, value_col)
        stat, p = stats.kruskal(*groups.values())
        additional = {
            "n_groups": len(groups),
            "group_column": group_col,
            "group_medians": {str(k): _safe_float(np.median(v)) for k, v in groups.items()},
        }

    elif test_type == "mannwhitneyu":
        groups = get_groups(group_col, value_col, min_groups=2)
        keys = list(groups.keys())
        g1, g2 = groups[keys[0]], groups[keys[1]]
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        additional = {
            "group_1": str(keys[0]), "group_1_n": len(g1), "group_1_median": _safe_float(np.median(g1)),
            "group_2": str(keys[1]), "group_2_n": len(g2), "group_2_median": _safe_float(np.median(g2)),
        }

    elif test_type == "friedmanchisquare":
        # Each column is a treatment group
        numeric_cols = [pd.to_numeric(df[c], errors="coerce").dropna() for c in columns]
        min_len = min(len(c) for c in numeric_cols)
        if min_len < 3:
            raise ValueError("Not enough rows for Friedman test.")
        trimmed = [c.iloc[:min_len].values for c in numeric_cols]
        stat, p = stats.friedmanchisquare(*trimmed)
        additional = {"n_groups": len(columns), "n_observations": min_len}

    elif test_type == "mood":
        groups = get_groups(group_col, value_col, min_groups=2)
        keys = list(groups.keys())
        stat, p = stats.mood(groups[keys[0]], groups[keys[1]])
        additional = {"group_1": str(keys[0]), "group_2": str(keys[1])}

    # ── Categorical tests ─────────────────────────────────────────────────────
    elif test_type == "chi2_contingency":
        contingency = pd.crosstab(df[columns[0]], df[columns[1]])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            raise ValueError("Chi-square requires at least 2 categories in each column.")
        stat, p, dof, _ = stats.chi2_contingency(contingency)
        additional = {
            "degrees_of_freedom": int(dof),
            "contingency_shape": list(contingency.shape),
            "n_categories_col1": int(contingency.shape[0]),
            "n_categories_col2": int(contingency.shape[1]),
        }

    elif test_type == "fisher_exact":
        contingency = pd.crosstab(df[columns[0]], df[columns[1]])
        if contingency.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 contingency table (exactly 2 categories in each column).")
        odds, p = stats.fisher_exact(contingency.values)
        stat = odds
        additional = {"odds_ratio": _safe_float(odds), "contingency_shape": list(contingency.shape)}

    # ── Variance tests ────────────────────────────────────────────────────────
    elif test_type == "levene":
        groups = get_groups(group_col, value_col)
        stat, p = stats.levene(*groups.values())
        additional = {"n_groups": len(groups), "group_column": group_col}

    elif test_type == "bartlett":
        groups = get_groups(group_col, value_col)
        stat, p = stats.bartlett(*groups.values())
        additional = {"n_groups": len(groups), "group_column": group_col}

    # ── Time series ───────────────────────────────────────────────────────────
    elif test_type == "adfuller":
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            raise ImportError("Install statsmodels: pip install statsmodels")

        # Find a usable numeric column — skip any datetime columns
        adf_col = None
        for c in columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            candidate = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(candidate) >= 12:
                adf_col = candidate
                break

        if adf_col is None:
            # Try value_col as fallback
            if value_col and value_col in df.columns:
                candidate = pd.to_numeric(df[value_col], errors="coerce").dropna()
                if len(candidate) >= 12:
                    adf_col = candidate

        if adf_col is None:
            raise ValueError(
                f"adfuller requires at least one numeric column with 12+ observations. "
                f"Columns provided: {columns}. Datetime columns cannot be used as the series."
            )

        result = adfuller(adf_col, autolag="AIC")
        stat, p = result[0], result[1]
        additional = {
            "n_lags_used": int(result[2]),
            "n_observations": int(result[3]),
            "critical_values": {k: _safe_float(v) for k, v in result[4].items()},
        }

    else:
        raise ValueError(
            f"Unknown test_type: '{test_type}'. "
            "Supported: shapiro, normaltest, kstest, pearsonr, spearmanr, kendalltau, pointbiserialr, "
            "ttest_ind, ttest_1samp, ttest_rel, wilcoxon, f_oneway, kruskal, mannwhitneyu, "
            "friedmanchisquare, mood, chi2_contingency, fisher_exact, levene, bartlett, adfuller."
        )

    return _safe_float(stat), _safe_float(p), additional, extra


# ── Interpretation ────────────────────────────────────────────────────────────

def _technical_interpret(test_type: str, test_id: str, p_value: Optional[float], stat: Optional[float], extra: dict) -> str:
    if p_value is None:
        return "Test could not be completed."
    sig = p_value < 0.05
    pstr = "<0.001" if p_value < 0.001 else str(round(p_value, 4))
    sstr = str(round(stat, 4)) if stat is not None else "N/A"

    templates = {
        "shapiro":        f"Shapiro-Wilk W={sstr}, p={pstr}. Data {'does not follow' if sig else 'follows'} a normal distribution.",
        "normaltest":     f"D'Agostino-Pearson statistic={sstr}, p={pstr}. Normality {'rejected' if sig else 'not rejected'}.",
        "kstest":         f"KS statistic={sstr}, p={pstr}. Normality {'rejected' if sig else 'not rejected'}.",
        "pearsonr":       f"Pearson r={sstr}, p={pstr}. {'Significant' if sig else 'No significant'} linear correlation. R²={round(extra.get('correlation',0)**2,4) if extra.get('correlation') else 'N/A'}.",
        "spearmanr":      f"Spearman ρ={sstr}, p={pstr}. {'Significant' if sig else 'No significant'} monotonic relationship.",
        "kendalltau":     f"Kendall τ={sstr}, p={pstr}. {'Significant' if sig else 'No significant'} rank correlation.",
        "pointbiserialr": f"Point-biserial r={sstr}, p={pstr}. {'Significant' if sig else 'No significant'} correlation between binary and continuous variable.",
        "ttest_ind":      f"Independent t-test: t={sstr}, p={pstr}. Group means {'significantly differ' if sig else 'do not significantly differ'}.",
        "ttest_1samp":    f"One-sample t-test: t={sstr}, p={pstr}. Mean {'significantly differs from' if sig else 'does not differ from'} hypothesized value.",
        "ttest_rel":      f"Paired t-test: t={sstr}, p={pstr}. Paired means {'significantly differ' if sig else 'do not significantly differ'}.",
        "wilcoxon":       f"Wilcoxon signed-rank: W={sstr}, p={pstr}. Paired distributions {'significantly differ' if sig else 'do not significantly differ'}.",
        "f_oneway":       f"One-way ANOVA: F={sstr}, p={pstr}. {'Significant differences' if sig else 'No significant differences'} across groups.",
        "kruskal":        f"Kruskal-Wallis H={sstr}, p={pstr}. {'Significant differences' if sig else 'No significant differences'} across groups.",
        "mannwhitneyu":   f"Mann-Whitney U={sstr}, p={pstr}. Distributions {'significantly differ' if sig else 'do not significantly differ'}.",
        "friedmanchisquare": f"Friedman χ²={sstr}, p={pstr}. {'Significant differences' if sig else 'No significant differences'} across repeated measures.",
        "mood":           f"Mood's median test: statistic={sstr}, p={pstr}. Medians {'significantly differ' if sig else 'do not significantly differ'}.",
        "chi2_contingency": f"Chi-square χ²={sstr}, p={pstr}. Variables are {'dependent' if sig else 'independent'}.",
        "fisher_exact":   f"Fisher's exact: odds ratio={sstr}, p={pstr}. {'Significant' if sig else 'No significant'} association.",
        "levene":         f"Levene's W={sstr}, p={pstr}. Variances {'significantly differ' if sig else 'do not significantly differ'} across groups.",
        "bartlett":       f"Bartlett's statistic={sstr}, p={pstr}. Variances {'significantly differ' if sig else 'do not significantly differ'} across groups.",
        "adfuller":       f"ADF statistic={sstr}, p={pstr}. Series is {'stationary' if sig else 'non-stationary (has a trend or unit root)'}.",
    }
    return templates.get(test_type, f"Statistic={sstr}, p={pstr}. {'Significant' if sig else 'Not significant'} at α=0.05.")


def _generate_business_insight(test_type: str, test_spec: dict, p_value: Optional[float],
                                stat: Optional[float], additional_metrics: dict, df: pd.DataFrame) -> str:
    """Calls GPT to generate a plain-English business insight."""
    if p_value is None:
        return "This test could not be completed for this dataset."
    try:
        from openai import AzureOpenAI
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("Not configured")
        sig = p_value < 0.05
        columns = test_spec.get("columns", [])
        group_col = test_spec.get("group_column", "")
        value_col = test_spec.get("value_column", "")
        col_context = []
        for c in list(set(columns + ([group_col] if group_col else []) + ([value_col] if value_col else []))):
            if c and c in df.columns:
                series = df[c].dropna()
                col_context.append({
                    "column": c,
                    "sample_values": series.head(5).astype(str).tolist(),
                    "unique_count": int(series.nunique()),
                })
        context = {
            "test_type": test_type,
            "significant": sig,
            "p_value": round(p_value, 4) if p_value >= 0.001 else "<0.001",
            "statistic": round(stat, 4) if stat else None,
            "columns": columns,
            "group_column": group_col,
            "value_column": value_col,
            "key_metrics": {k: v for k, v in additional_metrics.items() if not isinstance(v, dict)},
            "column_context": col_context,
        }
        prompt = (
            "You are explaining a statistical result to a small business owner with no data science background. "
            "Write ONE or TWO plain-English sentences that: "
            "1) State what was found using the actual column names from the data, "
            "2) Tell the owner what this means for their business or what action to consider. "
            "Do NOT use words like: p-value, null hypothesis, statistically significant, test statistic, alpha, chi-square, ANOVA, t-test. "
            "Be specific, practical, and use the actual column names. "
            "Good example: 'Your holiday promotions drive significantly more transactions than regular days — "
            "consider increasing stock and staffing on these events.' "
            f"\n\nResult: {json.dumps(context)}"
        )
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Business insight generation failed: %s", e)
        return _fallback_business_insight(test_type, test_spec, p_value, additional_metrics)


def _fallback_business_insight(test_type: str, test_spec: dict, p_value: Optional[float], additional_metrics: dict) -> str:
    if p_value is None:
        return "This analysis could not be completed for this dataset."
    sig = p_value < 0.05
    cols = test_spec.get("columns", ["your data"])
    group_col = test_spec.get("group_column", "category")
    value_col = test_spec.get("value_column", cols[0] if cols else "values")
    c0 = cols[0] if cols else "your data"
    c1 = cols[1] if len(cols) > 1 else "the other variable"

    fallbacks = {
        "shapiro":         f"Your {c0} {'has an unusual distribution with outliers or skew' if sig else 'is evenly spread'} — {'investigate extreme values before drawing conclusions.' if sig else 'averages are a reliable summary of this data.'}",
        "normaltest":      f"Your {c0} {'is unevenly distributed' if sig else 'is evenly distributed'} — {'some values are pulling the average up or down.' if sig else 'averages give a reliable picture.'}",
        "kstest":          f"Your {c0} {'does not follow a typical bell-curve pattern' if sig else 'follows a typical bell-curve pattern'} — {'consider looking at the median instead of the mean.' if sig else 'the mean is a reliable summary.'}",
        "pearsonr":        f"{c0} and {c1} {'move together strongly' if sig else 'do not appear related'} — {'changes in one are likely reflected in the other, which could help with forecasting.' if sig else 'they behave independently of each other.'}",
        "spearmanr":       f"{'There is a consistent relationship' if sig else 'There is no consistent relationship'} between {c0} and {c1} — {'higher values in one tend to go with higher values in the other.' if sig else 'they do not follow a predictable pattern together.'}",
        "kendalltau":      f"{'A consistent ranking pattern exists' if sig else 'No consistent ranking pattern found'} between {c0} and {c1}.",
        "pointbiserialr":  f"{'A meaningful link exists' if sig else 'No meaningful link found'} between {c0} and {c1}.",
        "ttest_ind":       f"The two {group_col} groups have {'meaningfully different' if sig else 'similar'} {value_col} averages — {'the difference is real and worth investigating.' if sig else 'no action needed based on this difference.'}",
        "ttest_1samp":     f"Your {c0} average {'differs meaningfully from the expected value' if sig else 'is in line with what is expected'}.",
        "ttest_rel":       f"{'A meaningful difference exists' if sig else 'No meaningful difference found'} between {c0} and {c1} for the same observations.",
        "wilcoxon":        f"{'A consistent difference exists' if sig else 'No consistent difference found'} between {c0} and {c1} when comparing the same items.",
        "f_oneway":        f"{value_col} varies {'significantly' if sig else 'only slightly'} across different {group_col} groups — {'some groups are clearly outperforming others. Investigate which groups and why.' if sig else 'all groups perform at a similar level.'}",
        "kruskal":         f"{'At least one' if sig else 'No'} {group_col} group performs differently on {value_col} — {'dig into which groups stand out and consider adjusting your strategy accordingly.' if sig else 'all groups are performing at a similar level.'}",
        "mannwhitneyu":    f"The two {group_col} groups have {'different' if sig else 'similar'} {value_col} patterns — {'one group consistently outperforms the other.' if sig else 'both groups behave similarly.'}",
        "friedmanchisquare": f"{'Meaningful differences exist' if sig else 'No meaningful differences found'} across the measurement groups — {'some time periods or conditions outperform others.' if sig else 'performance is consistent across all conditions.'}",
        "mood":            f"The typical {value_col} {'differs meaningfully' if sig else 'is similar'} between the two {group_col} groups.",
        "chi2_contingency": f"{c0} and {c1} are {'related to each other' if sig else 'independent of each other'} — {'knowing one helps predict the other, which could guide targeting or segmentation.' if sig else 'they do not influence each other.'}",
        "fisher_exact":    f"{'A significant association exists' if sig else 'No significant association found'} between {c0} and {c1}.",
        "levene":          f"The {group_col} groups have {'different levels of consistency' if sig else 'similar levels of consistency'} in {value_col} — {'some groups are more predictable than others.' if sig else 'all groups show similar variability.'}",
        "bartlett":        f"The spread of {value_col} {'varies significantly' if sig else 'is consistent'} across {group_col} groups.",
        "adfuller":        f"Your {c0} is {'stable and predictable over time' if sig else 'trending or shifting over time'} — {'you can reliably forecast future values.' if sig else 'consider accounting for this trend before making forecasts.'}",
    }
    return fallbacks.get(test_type, "See the technical result above for details.")


# ── Single test entry point ───────────────────────────────────────────────────

def _run_single_test(test_spec: dict, df: pd.DataFrame) -> dict:
    test_type = test_spec.get("test_type", "")
    test_id = test_spec.get("test_id", test_type)
    columns = test_spec.get("columns", [])
    reason = test_spec.get("reason", "")

    result = {
        "test_id": test_id,
        "test_name": test_id.replace("_", " ").title(),
        "test_type": test_type,
        "columns_used": columns,
        "reason": reason,
        "status": "completed",
        "statistic": None,
        "p_value": None,
        "additional_metrics": {},
        "interpretation": "",
        "business_insight": "",
        "error": None,
    }

    try:
        stat, p, additional, extra = _run_test_dynamic(test_spec, df)
        result["statistic"] = stat
        result["p_value"] = p
        result["additional_metrics"] = additional
        result["interpretation"] = _technical_interpret(test_type, test_id, p, stat, extra)
        result["business_insight"] = _generate_business_insight(test_type, test_spec, p, stat, additional, df)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["interpretation"] = f"Test could not be completed: {e}"
        result["business_insight"] = "This test could not be completed for this dataset."
        logger.warning("Test %s (%s) failed: %s", test_id, test_type, e)

    return result


# ── Public entry point ────────────────────────────────────────────────────────

def run_statistical_tests(dataset_name: str, df: pd.DataFrame) -> dict:
    started_at = time.time()
    logger.info("Statistical analysis starting: %s (%d rows, %d cols)", dataset_name, len(df), len(df.columns))

    schema = _describe_schema(df)
    selected_tests = []
    try:
        selected_tests = _select_tests_with_llm(dataset_name, schema)
    except Exception as e:
        logger.error("Test selection failed: %s", e)

    # Final runtime guard: drop tests that are invalid for actual coerced data.
    filtered_tests = []
    for t in selected_tests:
        if t.get("test_type") == "adfuller":
            cols = t.get("columns", []) or []
            value_col = t.get("value_column")
            candidates = [c for c in cols if c in df.columns]
            if value_col and value_col in df.columns and value_col not in candidates:
                candidates.append(value_col)
            has_series = any(len(pd.to_numeric(df[c], errors="coerce").dropna()) >= 12 for c in candidates)
            if not has_series:
                logger.info("Dropping adfuller at runtime for %s (insufficient numeric observations).", dataset_name)
                continue
        filtered_tests.append(t)
    selected_tests = filtered_tests

    if not selected_tests:
        return {
            "dataset_name": dataset_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "selected_tests": [],
            "results": [],
            "duration_seconds": round(time.time() - started_at, 4),
            "status": "skipped",
            "reason": "No appropriate statistical tests could be identified for this dataset.",
        }

    results = []
    for test_spec in selected_tests:
        logger.info("Running %s (%s) on %s", test_spec.get("test_id"), test_spec.get("test_type"), dataset_name)
        results.append(_run_single_test(test_spec, df))

    duration = round(time.time() - started_at, 4)
    completed = sum(1 for r in results if r["status"] == "completed")
    logger.info("Done: %s — %d/%d tests completed in %.2fs", dataset_name, completed, len(results), duration)

    return {
        "dataset_name": dataset_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "selected_tests": selected_tests,
        "results": results,
        "duration_seconds": duration,
        "status": "completed" if completed > 0 else "failed",
    }
