# Predictive Agent
# - LLM freely picks model, target column, features, and transform from whatever is installed
# - No hardcoded columns, paths, or model list
# - Resampling guard keeps total runtime under 10 seconds
# - Graceful fallback at every step — never crashes the pipeline
# - Accepts DataFrame + eda_insights directly, no disk I/O inside

import json
import time
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_ENDPOINT,
    PROCESSED_OUTPUT_DIR,
)
from src.core.logger import get_logger

logger = get_logger("PredictiveAgent")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Sampling thresholds ───────────────────────────────────────────────────────
XGBOOST_ROW_CAP      = 50_000   # rows above this → time-based sample
TIMESERIES_POINT_CAP = 3_650    # daily points for SARIMA / Prophet (10-year cap)
MIN_ROWS_FOR_PREDICTIVE = 12
MIN_TEST_ROWS_FOR_RELIABLE_METRICS = 5

# ── Model registry ────────────────────────────────────────────────────────────

# Models that use only the target column (no feature matrix needed)
TIMESERIES_MODELS = {
    "SARIMA", "AutoARIMA", "Prophet", "ExponentialSmoothing", "Theta"
}

def _discover_available_models(has_date: bool) -> dict:
    """Returns {model_name: trainer_fn} for every model importable at runtime.

    Current catalogue (14 models):
    ── Tree / Ensemble ──────────────────────────────────────
      XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees,
      GradientBoosting
    ── Linear / Regularised ─────────────────────────────────
      Ridge, ElasticNet, LinearRegression
    ── Time-series (requires date column) ───────────────────
      SARIMA, AutoARIMA, Prophet, ExponentialSmoothing, Theta
    """
    models = {}

    # ── Tree / Ensemble ───────────────────────────────────────────────────────
    try:
        from xgboost import XGBRegressor  # noqa: F401
        models["XGBoost"] = _train_xgboost
    except ImportError:
        pass

    try:
        import lightgbm  # noqa: F401
        models["LightGBM"] = _train_lightgbm
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor  # noqa: F401
        models["CatBoost"] = _train_catboost
    except ImportError:
        pass

    try:
        from sklearn.ensemble import RandomForestRegressor  # noqa: F401
        models["RandomForest"] = _train_random_forest
    except ImportError:
        pass

    try:
        from sklearn.ensemble import ExtraTreesRegressor  # noqa: F401
        models["ExtraTrees"] = _train_extra_trees
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        models["GradientBoosting"] = _train_gradient_boosting
    except ImportError:
        pass

    # ── Linear / Regularised ──────────────────────────────────────────────────
    try:
        from sklearn.linear_model import Ridge  # noqa: F401
        models["Ridge"] = _train_ridge
    except ImportError:
        pass

    try:
        from sklearn.linear_model import ElasticNet  # noqa: F401
        models["ElasticNet"] = _train_elasticnet
    except ImportError:
        pass

    try:
        from sklearn.linear_model import LinearRegression  # noqa: F401
        models["LinearRegression"] = _train_linear_regression
    except ImportError:
        pass

    # ── Time-series (only when a date column is detected) ─────────────────────
    if has_date:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: F401
            models["SARIMA"] = _train_sarima
        except ImportError:
            pass

        try:
            from statsforecast.models import AutoARIMA as _AutoARIMA  # noqa: F401
            models["AutoARIMA"] = _train_autoarima
        except ImportError:
            pass

        try:
            from prophet import Prophet  # noqa: F401
            models["Prophet"] = _train_prophet
        except ImportError:
            pass

        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: F401
            models["ExponentialSmoothing"] = _train_exp_smoothing
        except ImportError:
            pass

        try:
            from statsforecast.models import Theta as _Theta  # noqa: F401
            models["Theta"] = _train_theta
        except ImportError:
            pass

    return models


# ── Schema builder for LLM ────────────────────────────────────────────────────

def _build_schema(df: pd.DataFrame) -> dict:
    """Compact schema with enough info for the LLM to make smart choices."""
    numeric_cols    = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols   = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    # Detect string columns that look like dates
    for col in categorical_cols:
        try:
            sample = df[col].dropna().head(5).astype(str).tolist()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(sample, errors="raise")
            if col not in datetime_cols:
                datetime_cols.append(col)
        except Exception:
            pass

    col_profiles = []
    for col in df.columns:
        series = df[col].dropna()
        profile: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_pct": round(df[col].isna().mean() * 100, 1),
            "unique_count": int(series.nunique()),
            "sample_values": series.head(5).astype(str).tolist(),
        }
        if pd.api.types.is_numeric_dtype(df[col]) and len(series):
            profile["min"]      = float(series.min())
            profile["max"]      = float(series.max())
            profile["mean"]     = float(series.mean())
            profile["std"]      = float(series.std())
            profile["skewness"] = float(series.skew())
            # Flag likely ID columns so LLM avoids picking them as target
            is_id = (
                series.nunique() / len(series) > 0.95
                and (series % 1 == 0).all()
                and series.min() >= 0
            )
            profile["likely_id"] = bool(is_id)
        col_profiles.append(profile)

    return {
        "row_count":          len(df),
        "column_count":       len(df.columns),
        "numeric_columns":    numeric_cols,
        "categorical_columns": [c for c in categorical_cols if c not in datetime_cols],
        "datetime_columns":   datetime_cols,
        "columns":            col_profiles,
    }


# ── LLM plan selection ────────────────────────────────────────────────────────

def _select_plan_with_llm(
    dataset_name: str,
    schema: dict,
    eda_insights: dict,
    available_models: list[str],
) -> dict:
    """Ask Azure OpenAI to pick model, target, features, and transform.
    Returns a validated plan dict. Falls back to heuristics on any failure.
    """
    try:
        from openai import AzureOpenAI
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("Azure OpenAI not configured")

        numeric_cols  = schema.get("numeric_columns", [])
        datetime_cols = schema.get("datetime_columns", [])
        cat_cols      = schema.get("categorical_columns", [])

        # Filter out likely-ID columns from target candidates
        id_cols = {c["name"] for c in schema.get("columns", []) if c.get("likely_id")}
        target_candidates = [c for c in numeric_cols if c not in id_cols]

        system_prompt = (
            "You are a machine learning model selector for a small business analytics platform. "
            "Given a dataset schema and EDA insights, choose the best regression/forecasting model "
            "to predict the most business-relevant numeric column. "
            "Think practically — what would help a business owner most?"
        )

        user_prompt = {
            "task": (
                "Choose the best predictive model setup for this dataset. "
                "Return ONLY a JSON object with exactly these keys: "
                "target_column, selected_model, selected_features, target_transform, reason."
            ),
            "dataset_name": dataset_name,
            "schema": schema,
            "eda_insights": eda_insights,
            "available_models": available_models,
            "target_candidates": target_candidates,
            "HARD_RULES": [
                "target_column must be from target_candidates (non-ID numeric columns).",
                "selected_model must be exactly one name from available_models.",
                "selected_features must be a list of column names from schema that are NOT the target_column.",
                f"For time-series models ({', '.join(sorted(TIMESERIES_MODELS))}): selected_features must be empty [].",
                "target_transform must be 'none' or 'log1p'. Use log1p if target skewness > 1.",
                "Do NOT pick date/datetime columns as target_column.",
                "Do NOT pick ID columns (likely_id=true) as target_column or features.",
                f"datetime_columns available: {datetime_cols}",
                f"categorical_columns available: {cat_cols}",
                "Model selection guidance:",
                "  - If has_seasonality=true in eda_insights AND a time-series model is available → prefer AutoARIMA or Prophet (they auto-tune; better than SARIMA).",
                "  - If dataset is large (>10k rows) with many numeric features → prefer XGBoost, LightGBM, or CatBoost.",
                "  - If dataset is small (<1k rows) or features are sparse → prefer Ridge, ElasticNet, or ExtraTrees.",
                "  - If data has strong non-linearity but no dates → prefer GradientBoosting or ExtraTrees.",
                "  - Theta is ideal for short time series (<200 points) without complex seasonality.",
                "  - CatBoost handles mixed numeric/categorical features natively without encoding.",
                "reason: one sentence plain English explaining the model choice for a business owner.",
            ],
        }

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
        raw  = response.choices[0].message.content or "{}"
        plan = json.loads(raw)
        return _validate_plan(plan, schema, available_models)

    except Exception as e:
        logger.warning("LLM plan selection failed: %s — using heuristic fallback", e)
        return _heuristic_plan(schema, available_models)


def _validate_plan(plan: dict, schema: dict, available_models: list[str]) -> dict:
    """Validates and sanitises the LLM plan. Falls back field-by-field."""
    numeric_cols  = schema.get("numeric_columns", [])
    all_cols      = {c["name"] for c in schema.get("columns", [])}
    datetime_cols = set(schema.get("datetime_columns", []))
    id_cols       = {c["name"] for c in schema.get("columns", []) if c.get("likely_id")}
    target_candidates = [c for c in numeric_cols if c not in id_cols and c not in datetime_cols]

    # target_column
    target = plan.get("target_column", "")
    if target not in target_candidates:
        target = target_candidates[0] if target_candidates else (numeric_cols[0] if numeric_cols else "")
        logger.warning("LLM target invalid — using fallback: %s", target)

    # selected_model
    model = str(plan.get("selected_model", "")).strip()
    if model not in available_models:
        model = available_models[0] if available_models else "XGBoost"
        logger.warning("LLM model invalid — using fallback: %s", model)

    # selected_features
    features = plan.get("selected_features") or []
    if not isinstance(features, list):
        features = []
    if model in TIMESERIES_MODELS:
        features = []
    else:
        features = [
            f for f in features
            if f in all_cols
            and f != target
            and f not in datetime_cols
            and f not in id_cols
        ]
        # If LLM gave nothing useful, auto-select
        if not features:
            features = [
                c for c in numeric_cols + schema.get("categorical_columns", [])
                if c != target and c not in datetime_cols and c not in id_cols
            ][:10]

    # target_transform
    transform = str(plan.get("target_transform", "none")).strip().lower()
    if transform not in {"none", "log1p"}:
        transform = "none"

    return {
        "target_column":    target,
        "selected_model":   model,
        "selected_features": features,
        "target_transform": transform,
        "reason":           plan.get("reason", ""),
    }


def _heuristic_plan(schema: dict, available_models: list[str]) -> dict:
    """Rule-based fallback when LLM is unavailable."""
    numeric_cols  = schema.get("numeric_columns", [])
    datetime_cols = set(schema.get("datetime_columns", []))
    id_cols       = {c["name"] for c in schema.get("columns", []) if c.get("likely_id")}

    target_candidates = [c for c in numeric_cols if c not in id_cols and c not in datetime_cols]

    # Pick the numeric column with highest std as target (most interesting to predict)
    best_target = target_candidates[0] if target_candidates else (numeric_cols[0] if numeric_cols else "")
    best_std = -1
    for col_info in schema.get("columns", []):
        if col_info["name"] in target_candidates and col_info.get("std", 0) > best_std:
            best_std = col_info.get("std", 0)
            best_target = col_info["name"]

    # Prefer AutoARIMA when date col exists, else XGBoost, else first available
    if datetime_cols:
        preferred_ts = ["AutoARIMA", "Prophet", "ExponentialSmoothing", "Theta", "SARIMA"]
        model = next((m for m in preferred_ts if m in available_models), None)
    else:
        model = None

    if not model:
        preferred_tree = ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting", "RandomForest"]
        model = next((m for m in preferred_tree if m in available_models), None)

    if not model:
        model = available_models[0] if available_models else "XGBoost"

    features = [] if model in TIMESERIES_MODELS else [
        c for c in numeric_cols + schema.get("categorical_columns", [])
        if c != best_target and c not in datetime_cols and c not in id_cols
    ][:10]

    skewness = next(
        (c.get("skewness", 0) for c in schema.get("columns", []) if c["name"] == best_target), 0
    )
    transform = "log1p" if abs(skewness or 0) > 1 else "none"

    return {
        "target_column":    best_target,
        "selected_model":   model,
        "selected_features": features,
        "target_transform": transform,
        "reason":           "Heuristic fallback — LLM unavailable.",
    }


# ── Resampling guard ──────────────────────────────────────────────────────────

def _sample_for_speed(df: pd.DataFrame, date_col: Optional[str], model_name: str, target_col: str = "") -> tuple[pd.DataFrame, bool]:
    """Caps dataset size to stay reasonable. Returns (sampled_df, was_sampled).

    For time-series models:
    - Aggregate by date using MEAN of the target (avoids nonsensical sums across stores/categories)
    - Keep all other numeric cols as means too; keep first value for categoricals
    - Only cap if > TIMESERIES_POINT_CAP unique dates (very long series)

    For tree/linear models:
    - Only sample if truly huge (> XGBOOST_ROW_CAP)
    """
    time_series_models = {"SARIMA", "Prophet", "ExponentialSmoothing"}

    if model_name in time_series_models and date_col:
        # Build per-date aggregation: mean for numerics, first for non-numerics
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = [c for c in df.columns if c != date_col and c not in numeric_cols]

        agg_dict = {c: "mean" for c in numeric_cols}
        agg_dict.update({c: "first" for c in non_numeric_cols})

        daily = df.groupby(date_col, as_index=False).agg(agg_dict)
        daily = daily.sort_values(date_col).reset_index(drop=True)

        original_rows = len(df)
        was_aggregated = len(daily) < original_rows

        if len(daily) > TIMESERIES_POINT_CAP:
            logger.info("Resampling time series from %d to %d daily points", len(daily), TIMESERIES_POINT_CAP)
            step = max(1, len(daily) // TIMESERIES_POINT_CAP)
            daily = daily.iloc[::step].reset_index(drop=True)
            return daily, True

        return daily, was_aggregated

    if len(df) > XGBOOST_ROW_CAP:
        logger.info("Sampling %d rows down to %d for %s", len(df), XGBOOST_ROW_CAP, model_name)
        if date_col and date_col in df.columns:
            df_sorted = df.sort_values(date_col).reset_index(drop=True)
        else:
            df_sorted = df.reset_index(drop=True)
        step = len(df_sorted) // XGBOOST_ROW_CAP
        return df_sorted.iloc[::step].reset_index(drop=True), True

    return df, False


# ── Train / test split ────────────────────────────────────────────────────────

def _split(df: pd.DataFrame, date_col: Optional[str], ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n <= 1:
        return df.iloc[:0].copy(), df.copy()
    min_test = min(MIN_TEST_ROWS_FOR_RELIABLE_METRICS, n - 1)
    test_size = max(min_test, int(round(n * (1 - ratio))))
    test_size = min(test_size, n - 1)
    cutoff = n - test_size
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


# ── Feature encoding ──────────────────────────────────────────────────────────

def _encode_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encodes categoricals, fills nulls, returns X_train, X_test."""
    from sklearn.preprocessing import LabelEncoder

    train = train.copy()
    test  = test.copy()
    encoders = {}

    for col in feature_cols:
        if col not in train.columns:
            train[col] = 0
            test[col]  = 0
            continue
        if train[col].dtype == object or str(train[col].dtype) == "category":
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col]]).astype(str).fillna("__null__")
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str).fillna("__null__"))
            test[col]  = le.transform(test[col].astype(str).fillna("__null__"))
            encoders[col] = le

    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test  = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return X_train, X_test


# ── Model trainers ────────────────────────────────────────────────────────────

def _apply_transform(y: pd.Series, transform: str) -> pd.Series:
    if transform == "log1p":
        y = y.clip(lower=0)
        return np.log1p(y)
    return y

def _invert_transform(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(y)
    return y


def _train_xgboost(train: pd.DataFrame, test: pd.DataFrame, target: str,
                   features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from xgboost import XGBRegressor

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    model = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = _invert_transform(model.predict(X_test), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_lightgbm(train: pd.DataFrame, test: pd.DataFrame, target: str,
                    features: list[str], transform: str) -> tuple[Any, pd.Series]:
    import lightgbm as lgb

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)
    preds = _invert_transform(model.predict(X_test), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_random_forest(train: pd.DataFrame, test: pd.DataFrame, target: str,
                         features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    model = RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = _invert_transform(model.predict(X_test), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_ridge(train: pd.DataFrame, test: pd.DataFrame, target: str,
                 features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    preds = _invert_transform(model.predict(X_test_s), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_linear_regression(train: pd.DataFrame, test: pd.DataFrame, target: str,
                              features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    preds = _invert_transform(model.predict(X_test_s), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_catboost(train: pd.DataFrame, test: pd.DataFrame, target: str,
                    features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from catboost import CatBoostRegressor

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    # Detect which feature cols are still categorical after encoding attempt
    cat_features = [i for i, col in enumerate(features)
                    if train[col].dtype == object or str(train[col].dtype) == "category"]

    model = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6,
        loss_function="RMSE", random_seed=42,
        verbose=0, allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    preds = _invert_transform(model.predict(X_test), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_extra_trees(train: pd.DataFrame, test: pd.DataFrame, target: str,
                       features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from sklearn.ensemble import ExtraTreesRegressor

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    model = ExtraTreesRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = _invert_transform(model.predict(X_test), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_gradient_boosting(train: pd.DataFrame, test: pd.DataFrame, target: str,
                              features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from sklearn.ensemble import GradientBoostingRegressor

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    preds = _invert_transform(model.predict(X_test), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_elasticnet(train: pd.DataFrame, test: pd.DataFrame, target: str,
                      features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler

    X_train, X_test = _encode_features(train, test, features)
    y_train = _apply_transform(train[target].fillna(0), transform)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # alpha=0.1 (mild regularisation), l1_ratio=0.5 (equal L1+L2 mix)
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42)
    model.fit(X_train_s, y_train)
    preds = _invert_transform(model.predict(X_test_s), transform)
    return model, pd.Series(preds, index=test.index, name=f"predicted_{target}")


def _train_sarima(train: pd.DataFrame, test: pd.DataFrame, target: str,
                  features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y_train = train[target].astype(float)
    if transform == "log1p":
        y_train = np.log1p(y_train.clip(lower=0))

    # Auto-detect seasonal period from date index spacing
    # Daily data → 7, Weekly → 52, Monthly → 12, fallback → 1 (no seasonality)
    seasonal_period = 1
    date_col = next((c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c])), None)
    if date_col:
        try:
            dates = pd.to_datetime(train[date_col].dropna()).sort_values()
            median_gap_days = dates.diff().dt.days.median()
            if median_gap_days is not None:
                if median_gap_days <= 1.5:
                    seasonal_period = 7    # daily → weekly seasonality
                elif median_gap_days <= 8:
                    seasonal_period = 52   # weekly → yearly
                elif median_gap_days <= 32:
                    seasonal_period = 12   # monthly → yearly
        except Exception:
            pass

    use_seasonal = seasonal_period > 1
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period) if use_seasonal else (0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=100)
    forecast = fitted.forecast(steps=len(test))

    preds = pd.Series(
        _invert_transform(np.array(forecast), transform),
        index=test.index,
        name=f"predicted_{target}",
    )
    return fitted, preds


def _train_prophet(train: pd.DataFrame, test: pd.DataFrame, target: str,
                   features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from prophet import Prophet

    # Find the date column in train
    date_col = next(
        (c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c])), None
    ) or next(
        (c for c in train.columns if "date" in c.lower()), None
    )
    if not date_col:
        raise ValueError("Prophet requires a date column — none found after resampling.")

    fit_df = train[[date_col, target]].rename(columns={date_col: "ds", target: "y"}).copy()
    fit_df["ds"] = pd.to_datetime(fit_df["ds"], errors="coerce")
    fit_df = fit_df.dropna(subset=["ds", "y"])

    if transform == "log1p":
        fit_df["y"] = np.log1p(fit_df["y"].clip(lower=0))

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(fit_df)

    future = test[[date_col]].rename(columns={date_col: "ds"}).copy()
    future["ds"] = pd.to_datetime(future["ds"], errors="coerce")
    forecast = model.predict(future)

    preds_raw = forecast["yhat"].values
    preds = pd.Series(
        _invert_transform(preds_raw, transform),
        index=test.index,
        name=f"predicted_{target}",
    )
    return model, preds


def _train_exp_smoothing(train: pd.DataFrame, test: pd.DataFrame, target: str,
                         features: list[str], transform: str) -> tuple[Any, pd.Series]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_train = train[target].astype(float)
    if transform == "log1p":
        y_train = np.log1p(y_train.clip(lower=0))

    # Need at least 2× the seasonal period of data to fit seasonal model
    seasonal_periods = min(7, len(y_train) // 2)
    use_seasonal = seasonal_periods >= 2 and len(y_train) >= seasonal_periods * 2

    model = ExponentialSmoothing(
        y_train,
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=seasonal_periods if use_seasonal else None,
    )
    fitted = model.fit(optimized=True, use_brute=False)
    forecast = fitted.forecast(steps=len(test))

    preds = pd.Series(
        _invert_transform(np.array(forecast), transform),
        index=test.index,
        name=f"predicted_{target}",
    )
    return fitted, preds


def _train_autoarima(train: pd.DataFrame, test: pd.DataFrame, target: str,
                     features: list[str], transform: str) -> tuple[Any, pd.Series]:
    """AutoARIMA via statsforecast — automatically tunes (p,d,q)(P,D,Q,m).
    Far more robust than hardcoded SARIMA orders.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    y_train = train[target].astype(float)
    if transform == "log1p":
        y_train = np.log1p(y_train.clip(lower=0))

    # Detect seasonal period the same way SARIMA does
    seasonal_period = 1
    date_col = next((c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c])), None)
    if date_col:
        try:
            dates = pd.to_datetime(train[date_col].dropna()).sort_values()
            median_gap_days = dates.diff().dt.days.median()
            if median_gap_days is not None:
                if median_gap_days <= 1.5:
                    seasonal_period = 7
                elif median_gap_days <= 8:
                    seasonal_period = 52
                elif median_gap_days <= 32:
                    seasonal_period = 12
        except Exception:
            pass

    # statsforecast expects a DataFrame with columns: unique_id, ds, y
    sf_train = pd.DataFrame({
        "unique_id": "series_1",
        "ds": pd.RangeIndex(len(y_train)),
        "y": y_train.values,
    })

    sf = StatsForecast(
        models=[AutoARIMA(season_length=seasonal_period, approximation=True)],
        freq=1,
        n_jobs=1,
    )
    sf.fit(sf_train)
    forecast_df = sf.predict(h=len(test))
    raw_preds = forecast_df["AutoARIMA"].values

    preds = pd.Series(
        _invert_transform(raw_preds, transform),
        index=test.index,
        name=f"predicted_{target}",
    )
    return sf, preds


def _train_theta(train: pd.DataFrame, test: pd.DataFrame, target: str,
                 features: list[str], transform: str) -> tuple[Any, pd.Series]:
    """Theta model via statsforecast — simple, robust, often beats ARIMA on short series."""
    from statsforecast import StatsForecast
    from statsforecast.models import Theta

    y_train = train[target].astype(float)
    if transform == "log1p":
        y_train = np.log1p(y_train.clip(lower=0))

    seasonal_period = 1
    date_col = next((c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c])), None)
    if date_col:
        try:
            dates = pd.to_datetime(train[date_col].dropna()).sort_values()
            median_gap_days = dates.diff().dt.days.median()
            if median_gap_days is not None:
                if median_gap_days <= 1.5:
                    seasonal_period = 7
                elif median_gap_days <= 8:
                    seasonal_period = 52
                elif median_gap_days <= 32:
                    seasonal_period = 12
        except Exception:
            pass

    sf_train = pd.DataFrame({
        "unique_id": "series_1",
        "ds": pd.RangeIndex(len(y_train)),
        "y": y_train.values,
    })

    sf = StatsForecast(
        models=[Theta(season_length=seasonal_period)],
        freq=1,
        n_jobs=1,
    )
    sf.fit(sf_train)
    forecast_df = sf.predict(h=len(test))
    raw_preds = forecast_df["Theta"].values

    preds = pd.Series(
        _invert_transform(raw_preds, transform),
        index=test.index,
        name=f"predicted_{target}",
    )
    return sf, preds


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0).reset_index(drop=True)
    y_pred = pd.to_numeric(y_pred, errors="coerce").fillna(0).reset_index(drop=True)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0

    # WAPE — weighted absolute percentage error
    actual_sum = float(y_true.abs().sum())
    wape_pct = round((float((y_true - y_pred).abs().sum()) / actual_sum) * 100, 2) if actual_sum else 0.0
    score_pct = round(max(0.0, 100.0 - wape_pct), 2)

    return {
        "mae":       round(mae, 4),
        "rmse":      round(rmse, 4),
        "r2":        round(r2, 4),
        "wape_pct":  wape_pct,
        "score_pct": score_pct,
    }


# ── Save predictions ──────────────────────────────────────────────────────────

def _save_predictions(test_df: pd.DataFrame, preds: pd.Series,
                      target: str, model_name: str, dataset_name: str) -> str:
    import os
    os.makedirs(str(PROCESSED_OUTPUT_DIR), exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in dataset_name).strip("_")
    out_path  = os.path.join(str(PROCESSED_OUTPUT_DIR), f"{safe_name}__predictions.csv")

    output = test_df.copy()
    output[f"predicted_{target}"] = preds.values
    output["model_used"] = model_name
    output.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)
    return out_path


# ── Public entry point ────────────────────────────────────────────────────────

def run_predictive(
    dataset_name: str,
    df: pd.DataFrame,
    eda_insights: Optional[dict] = None,
) -> dict:
    """Main entry point. Accepts a DataFrame and optional EDA insights dict.

    Args:
        dataset_name:  Human-readable name used in filenames and logs.
        df:            The full DataFrame (cleaning already done by ingestion agent).
        eda_insights:  Optional dict from EDA agent with has_seasonality,
                       recommended_models, etc.

    Returns a dict with model_used, target_column, metrics, predictions_path,
    plan_reason, duration_seconds, status — same shape as stat/EDA results.
    """
    started_at = time.time()
    logger.info("Predictive agent starting: %s (%d rows, %d cols)", dataset_name, len(df), len(df.columns))

    eda_insights = eda_insights or {}

    # ── Detect date column ────────────────────────────────────────────────────
    date_col: Optional[str] = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
    if not date_col:
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[col].dropna().head(10), errors="raise")
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    date_col = col
                    break
                except Exception:
                    pass

    has_date = date_col is not None
    logger.info("Date column detected: %s", date_col or "None")

    # ── Discover available models ─────────────────────────────────────────────
    available_models_map  = _discover_available_models(has_date)
    available_model_names = list(available_models_map.keys())
    logger.info("Available models: %s", available_model_names)

    if not available_model_names:
        return {
            "dataset_name": dataset_name, "status": "failed",
            "error": "No ML libraries found. Install at least scikit-learn.",
            "duration_seconds": round(time.time() - started_at, 4),
        }

    # ── Build schema & select plan ────────────────────────────────────────────
    schema = _build_schema(df)
    plan   = _select_plan_with_llm(dataset_name, schema, eda_insights, available_model_names)

    target    = plan["target_column"]
    model_name = plan["selected_model"]
    features  = plan["selected_features"]
    transform = plan["target_transform"]

    if not target or target not in df.columns:
        return {
            "dataset_name": dataset_name, "status": "failed",
            "error": f"Could not determine a valid target column. Schema: {schema['numeric_columns']}",
            "duration_seconds": round(time.time() - started_at, 4),
        }

    logger.info("Plan — model: %s | target: %s | features: %s | transform: %s",
                model_name, target, features, transform)

    # ── Drop rows where target is null ───────────────────────────────────────
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # ── Resampling guard ──────────────────────────────────────────────────────
    df_model, was_sampled = _sample_for_speed(df, date_col, model_name, target_col=target)
    if was_sampled:
        logger.info("Dataset sampled for speed: %d rows → %d", len(df), len(df_model))

    # ── Train / test split ────────────────────────────────────────────────────
    if len(df_model) < MIN_ROWS_FOR_PREDICTIVE:
        return {
            "dataset_name": dataset_name,
            "status": "skipped",
            "error": (
                f"Not enough rows for predictive modelling. "
                f"Need at least {MIN_ROWS_FOR_PREDICTIVE} rows after cleaning; got {len(df_model)}."
            ),
            "duration_seconds": round(time.time() - started_at, 4),
        }
    train_df, test_df = _split(df_model, date_col)

    if len(test_df) == 0:
        return {
            "dataset_name": dataset_name, "status": "failed",
            "error": "Not enough data to create a test set (need at least 2 rows).",
            "duration_seconds": round(time.time() - started_at, 4),
        }

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = available_models_map.get(model_name)
    if not trainer:
        # Fallback to first available
        model_name = available_model_names[0]
        trainer    = available_models_map[model_name]
        logger.warning("Selected model not available — falling back to %s", model_name)

    try:
        model_obj, preds = trainer(train_df, test_df, target, features, transform)
    except Exception as e:
        logger.warning("Model %s failed (%s) — falling back to XGBoost", model_name, e)
        try:
            model_obj, preds = _train_xgboost(train_df, test_df, target, features, transform)
            model_name = "XGBoost (fallback)"
        except Exception as e2:
            return {
                "dataset_name": dataset_name, "status": "failed",
                "error": f"Training failed: {e2}",
                "duration_seconds": round(time.time() - started_at, 4),
            }

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true = test_df[target].reset_index(drop=True)
    y_pred = preds.reset_index(drop=True)
    metrics = _compute_metrics(y_true, y_pred)
    metrics_reliable = len(test_df) >= MIN_TEST_ROWS_FOR_RELIABLE_METRICS
    quality_note = None
    if not metrics_reliable:
        quality_note = (
            f"Only {len(test_df)} test rows were available. "
            "Metrics are unstable; treat this as directional only."
        )
        metrics["r2"] = None
        metrics["score_pct"] = None


    # ── Build chart_data for frontend ─────────────────────────────────────────
    # Combine train actuals + test actuals + test predictions into a compact payload
    # Cap at 500 points for JSON size — evenly spaced if longer
    try:
        train_actual = train_df[target].reset_index(drop=True)
        all_actual   = pd.concat([train_actual, y_true], ignore_index=True)

        # Build label from date col if available, else use index
        if date_col and date_col in train_df.columns:
            train_dates = train_df[date_col].reset_index(drop=True).astype(str)
            test_dates  = test_df[date_col].reset_index(drop=True).astype(str)
            all_labels  = pd.concat([train_dates, test_dates], ignore_index=True).tolist()
        else:
            all_labels = list(range(len(all_actual)))

        split_idx = len(train_actual)

        # Pad predicted series with None for the train portion
        predicted_full = [None] * split_idx + y_pred.tolist()

        # Cap to 500 points for JSON size
        MAX_CHART_PTS = 500
        n = len(all_actual)
        if n > MAX_CHART_PTS:
            step = max(1, n // MAX_CHART_PTS)
            idxs = list(range(0, n, step))
            all_labels     = [all_labels[i] for i in idxs]
            all_actual     = [round(float(all_actual.iloc[i]), 4) for i in idxs]
            predicted_full = [round(float(predicted_full[i]), 4) if predicted_full[i] is not None else None for i in idxs]
            split_idx      = next((j for j, i in enumerate(idxs) if i >= split_idx), len(idxs))
        else:
            all_actual     = [round(float(v), 4) for v in all_actual]
            predicted_full = [round(float(v), 4) if v is not None else None for v in predicted_full]

        chart_data = {
            "labels":    all_labels,
            "actual":    all_actual,
            "predicted": predicted_full,
            "splitIdx":  split_idx,
        }
    except Exception as e:
        logger.warning("Could not build chart_data: %s", e)
        chart_data = None

    # ── Save predictions ──────────────────────────────────────────────────────
    try:
        predictions_path = _save_predictions(test_df, preds, target, model_name, dataset_name)
    except Exception as e:
        predictions_path = None
        logger.warning("Could not save predictions: %s", e)

    duration = round(time.time() - started_at, 4)
    score_for_log = metrics.get("score_pct")
    score_label = "N/A" if score_for_log is None else f"{float(score_for_log):.1f}%"
    logger.info(
        "Predictive complete: %s — model=%s target=%s score=%s in %.2fs",
        dataset_name, model_name, target, score_label, duration,
    )

    return {
        "dataset_name":      dataset_name,
        "model_used":        model_name,
        "target_column":     target,
        "selected_features": features,
        "target_transform":  transform,
        "train_rows":        len(train_df),
        "test_rows":         len(test_df),
        "was_sampled":       was_sampled,
        "metrics_reliable":  metrics_reliable,
        "quality_note":      quality_note,
        "metrics":           metrics,
        "chart_data":        chart_data,
        "predictions_path":  predictions_path,
        "plan_reason":       plan.get("reason", ""),
        "duration_seconds":  duration,
        "status":            "completed",
    }

