import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import ast
import os
import time
from typing import List, Optional
import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False

# ===========================
# 1. CONFIG
# ===========================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = "all_products_actuals_and_bookings.xlsx"  # <--- change to your file
REVISED_ACTUALS_FILE = "all_products_actuals_and_bookings_revised.xlsx"
REVISED_ACTUALS_SHEET = "Revised Actuals"
OUTPUT_FILE = "sarima_multi_sku_summary.xlsx"
ORDER_FILE = "sarimax_order_search_summary.xlsx"       # per-SKU SARIMA orders
NOTES_FILE = "Notes.xlsx"                              # manual order overrides
ENABLE_ML_CHALLENGER = True
ENABLE_PROPHET_CHALLENGER = True

# Column names – adjust if your master file uses different names
COL_PRODUCT = "Product"
COL_DIVISION = "Division"
COL_DATE = "Month"
COL_ACTUALS = "Actuals"
COL_NEW_OPPORTUNITIES = "New_Opportunities"
COL_OPEN_OPPS = "Open_Opportunities"
COL_BOOKINGS = "Bookings"
COL_MEDIAN_MONTHS_SINCE_LAST_ACTIVITY = "Median_Months_Since_Last_Activity"
COL_OPEN_NOT_MODIFIED_90_DAYS = "Open_Not_Modified_90_Days"
COL_PCT_OPEN_NOT_MODIFIED_90_DAYS = "Pct_Open_Not_Modified_90_Days"
COL_EARLY_TO_LATE_RATIO = "Early_to_Late_Ratio"

# Evaluation config
TEST_HORIZON = 12       # last 12 months as holdout for Test MAE/RMSE
MIN_OBS_TOTAL = 30      # minimum total points to attempt modeling
ROCV_MIN_OBS = 24       # minimum obs for rolling-origin CV
ROCV_HORIZON = 1        # 1-step ahead in ROCV

# Weighted MAE buckets for the holdout horizon (month index from holdout start).
WEIGHTED_MAE_BUCKETS = [
    (1, 3, 0.50),   # months 1-3 (closest to forecast origin)
    (4, 6, 0.30),   # months 4-6
    (7, 12, 0.20),  # months 7-12
]

# Acceptance thresholds
EPSILON_IMPROVEMENT = 0.02  # model must improve Test MAE by at least 2%
DELTA_ROCV_TOLERANCE = 0.05 # ROCV_MAE can be up to 5% worse than baseline
BASELINE_MAE_ZERO_EPS = 1e-9  # treat MAE at or below this as effectively zero
ETS_SEASONAL_PERIODS = 12     # monthly data


# ===========================
# 2. HELPER FUNCTIONS
# ===========================

def _output_path(filename: str) -> str:
    if not filename:
        return filename
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    return os.path.join(OUTPUT_DIR, filename)


def _parse_args():
    parser = argparse.ArgumentParser(description="Multi-SKU SARIMA/SARIMAX/ETS/ML evaluation engine")
    parser.add_argument("--disable-ml", action="store_true", help="Disable ML challenger evaluation")
    parser.add_argument("--disable-prophet", action="store_true", help="Disable Prophet challenger evaluation")
    return parser.parse_args()


def aggregate_monthly_duplicates(
    df: pd.DataFrame,
    product_col: str = COL_PRODUCT,
    division_col: str = COL_DIVISION,
    date_col: str = COL_DATE,
    sum_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Collapse duplicate Product/Division/Month rows by summing key numeric fields.
    Any extra columns are preserved by taking the first value within each group.
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    # Only sum true additive series; keep others (e.g., opportunities) as first.
    sum_cols = sum_cols or [COL_ACTUALS, COL_BOOKINGS]
    present_sum_cols = [c for c in sum_cols if c in df.columns]

    group_cols = [c for c in [product_col, division_col, date_col] if c in df.columns]
    agg_dict = {col: "sum" for col in present_sum_cols}
    for col in df.columns:
        if col in group_cols or col in agg_dict:
            continue
        agg_dict[col] = "first"

    aggregated = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(agg_dict)
        .sort_values(group_cols)
    )
    return aggregated


def mark_prelaunch_actuals_as_missing(
    df: pd.DataFrame,
    product_col: str = COL_PRODUCT,
    division_col: str = COL_DIVISION,
    date_col: str = COL_DATE,
    actuals_col: str = COL_ACTUALS,
    output_excel_path: Optional[str] = None,
    output_sheet: str = REVISED_ACTUALS_SHEET,
) -> pd.DataFrame:
    """
    Replace pre-launch zeros in Actuals with NaN per Product+Division.
    Pre-launch is defined as any date before the first positive actual.
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if actuals_col in df.columns:
        df[actuals_col] = pd.to_numeric(df[actuals_col], errors="coerce")

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(date_col).copy()
        first_active = group.loc[group[actuals_col] > 0, date_col].min()
        if pd.isna(first_active):
            group[actuals_col] = np.nan
            return group
        prelaunch_mask = (group[date_col] < first_active) & (group[actuals_col] == 0)
        group.loc[prelaunch_mask, actuals_col] = np.nan
        return group

    group_cols = [c for c in [product_col, division_col] if c in df.columns]
    if group_cols:
        df = df.groupby(group_cols, dropna=False, group_keys=False).apply(_apply)
    else:
        df = _apply(df)

    if output_excel_path:
        output_excel_path = _output_path(output_excel_path)
        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=output_sheet, index=False)
        print(f"Wrote revised actuals to {output_excel_path} ({output_sheet}).")

    return df


def safe_mae(y_true, y_pred):
    """Compute MAE robustly."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[mask], y_pred[mask])


def _build_holdout_weights(horizon: int, buckets=WEIGHTED_MAE_BUCKETS) -> np.ndarray:
    """Build normalized per-step weights for a holdout horizon (1-based month index)."""
    if horizon <= 0:
        return np.array([], dtype="float64")
    weights = np.zeros(horizon, dtype="float64")
    for start, end, total_weight in buckets:
        if horizon < start:
            continue
        bucket_end = min(end, horizon)
        bucket_len = bucket_end - start + 1
        if bucket_len <= 0:
            continue
        weights[start - 1:bucket_end] = total_weight / bucket_len
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


def safe_weighted_mae(y_true, y_pred, buckets=WEIGHTED_MAE_BUCKETS):
    """Compute weighted MAE, renormalizing weights over valid (finite) points."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return np.nan
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    weights = _build_holdout_weights(n, buckets=buckets)
    weights = weights[mask]
    total = weights.sum()
    if total <= 0:
        return np.nan
    weights = weights / total
    errors = np.abs(y_true[mask] - y_pred[mask])
    return float(np.average(errors, weights=weights))


def safe_rmse(y_true, y_pred):
    """RMSE that ignores NaNs and infs in either array."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    mse = mean_squared_error(y_true[mask], y_pred[mask])  # no squared arg
    return np.sqrt(mse)


def _compute_test_window(history_months: int) -> int:
    """Match forecast script: smaller test windows for short histories."""
    if history_months < 24:
        return min(TEST_HORIZON, max(6, int(np.floor(0.25 * history_months))))
    return TEST_HORIZON


def _regressor_failure_metrics(model_name):
    """Consistent fallback metrics when a regressor model cannot be fit."""
    return {
        "Model": model_name,
        "Test_MAE": 1e9,
        "Test_WMAE": 1e9,
        "Test_RMSE": 1e9,
        "AIC": np.nan,
        "BIC": np.nan,
        "ROCV_MAE": np.nan,
        "Regressor_coef": np.nan,
        "Regressor_pvalue": np.nan
    }


def build_rocv_spec(y_series, exog_series=None, min_obs=ROCV_MIN_OBS, horizon=ROCV_HORIZON):
    """
    Precompute the rolling-origin windows and cleaned series used in ROCV.
    This avoids recomputing masks/splits for every candidate.
    """
    y_clean = pd.Series(y_series).copy()
    if exog_series is not None:
        exog_clean = pd.Series(exog_series).copy()

        # Drop rows where exog or y are NaN to avoid exog NaN errors
        mask = (~y_clean.isna()) & (~exog_clean.isna())
        y_clean = y_clean[mask]
        exog_clean = exog_clean[mask]
    else:
        y_clean = y_clean.dropna()
        exog_clean = None

    n = len(y_clean)
    origins = []
    if n >= min_obs + horizon:
        for origin in range(min_obs, n - horizon + 1):
            if exog_clean is not None:
                exog_train = exog_clean.iloc[:origin]
                exog_future = exog_clean.iloc[origin:origin + horizon]
                if exog_train.isna().any() or exog_future.isna().any():
                    continue
            origins.append(origin)

    return {
        "y_clean": y_clean,
        "exog_clean": exog_clean,
        "origins": origins,
        "horizon": horizon
    }


def rolling_origin_cv_precomputed(rocv_spec, order=None, seasonal_order=None):
    """
    Rolling-origin cross validation using precomputed splits/cleaned series.
    """
    if rocv_spec is None:
        return np.nan

    y_clean = rocv_spec["y_clean"]
    exog_clean = rocv_spec.get("exog_clean")
    origins = rocv_spec.get("origins", [])
    horizon = rocv_spec.get("horizon", ROCV_HORIZON)

    if not origins:
        return np.nan

    errors = []
    for origin in origins:
        y_train = y_clean.iloc[:origin]
        y_test_point = y_clean.iloc[origin:origin + horizon]

        try:
            if exog_clean is not None:
                exog_train = exog_clean.iloc[:origin]
                exog_future = exog_clean.iloc[origin:origin + horizon]
                res = fit_sarimax(y_train, exog_train, order, seasonal_order)
                preds = res.get_forecast(steps=horizon, exog=exog_future).predicted_mean
            else:
                res = fit_sarima_baseline(y_train, order, seasonal_order)
                preds = res.get_forecast(steps=horizon).predicted_mean
        except Exception:
            continue

        err = safe_mae(y_test_point.values, preds.values)
        if not np.isnan(err):
            errors.append(err)

    if len(errors) == 0:
        return np.nan

    return float(np.mean(errors))


def _series_is_nonnegative(y_series) -> bool:
    """Treat zeros as positive for multiplicative eligibility checks."""
    y_clean = pd.Series(y_series).dropna()
    if y_clean.empty:
        return False
    return (y_clean >= 0).all()


def _ets_candidate_specs(y_series) -> List[dict]:
    """Build a small, industry-standard ETS candidate grid."""
    allow_mul = _series_is_nonnegative(y_series)
    error_opts = ["add"] + (["mul"] if allow_mul else [])
    trend_opts = [None, "add"]
    seasonal_opts = [None, "add"] + (["mul"] if allow_mul else [])

    candidates = []
    for error in error_opts:
        for trend in trend_opts:
            damped_opts = [False] if trend is None else [False, True]
            for damped in damped_opts:
                for seasonal in seasonal_opts:
                    candidates.append({
                        "error": error,
                        "trend": trend,
                        "damped_trend": damped,
                        "seasonal": seasonal,
                        "seasonal_periods": ETS_SEASONAL_PERIODS if seasonal is not None else None,
                    })
    return candidates


def _info_criterion(res) -> float:
    """Prefer AICc when available; fall back to AIC."""
    aicc = getattr(res, "aicc", None)
    if aicc is not None and np.isfinite(aicc):
        return float(aicc)
    aic = getattr(res, "aic", None)
    if aic is not None and np.isfinite(aic):
        return float(aic)
    return np.inf


def _fit_ets_candidate(y_train, spec, use_state_space: bool):
    """Fit a single ETS candidate using the requested backend."""
    if use_state_space:
        model = ETSModel(
            y_train,
            error=spec["error"],
            trend=spec["trend"],
            damped_trend=spec["damped_trend"],
            seasonal=spec["seasonal"],
            seasonal_periods=spec["seasonal_periods"],
            initialization_method="estimated",
        )
        return model.fit()

    model = ExponentialSmoothing(
        y_train,
        trend=spec["trend"],
        damped_trend=spec["damped_trend"] if spec["trend"] is not None else False,
        seasonal=spec["seasonal"],
        seasonal_periods=spec["seasonal_periods"] if spec["seasonal"] is not None else None,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)


def _select_best_ets_model(y_train, context: str = ""):
    """Choose the best ETS candidate by AICc (or AIC) within a SKU."""
    best_res = None
    best_spec = None
    best_score = np.inf

    for spec in _ets_candidate_specs(y_train):
        try:
            res = _fit_ets_candidate(y_train, spec, use_state_space=True)
            score = _info_criterion(res)
        except Exception as exc:
            print(f"[WARN] ETSModel failed ({spec}){context}: {exc}")
            continue

        if score < best_score:
            best_res = res
            best_spec = spec
            best_score = score

    if best_res is not None:
        return best_res, best_spec, "ETSModel"

    # Fallback: only if ETSModel cannot fit anything
    for spec in _ets_candidate_specs(y_train):
        if spec["error"] != "add":
            continue
        try:
            res = _fit_ets_candidate(y_train, spec, use_state_space=False)
            score = _info_criterion(res)
        except Exception as exc:
            print(f"[WARN] ExponentialSmoothing failed ({spec}){context}: {exc}")
            continue

        if score < best_score:
            best_res = res
            best_spec = spec
            best_score = score

    if best_res is None:
        return None, None, None
    return best_res, best_spec, "ExponentialSmoothing"


def _forecast_ets(res, steps, index):
    """Forecast ETS model and align to the provided index."""
    mean = None
    try:
        forecast_res = res.get_forecast(steps=steps)
        mean = forecast_res.predicted_mean
    except Exception:
        if hasattr(res, "forecast"):
            mean = res.forecast(steps=steps)

    if mean is None:
        raise ValueError("ETS forecast failed")
    return pd.Series(mean, index=index)


def rolling_origin_cv_ets_precomputed(rocv_spec, ets_spec, impl, horizon=ROCV_HORIZON):
    """Rolling-origin CV for ETS using a fixed spec and backend."""
    if rocv_spec is None:
        return np.nan

    y_clean = rocv_spec["y_clean"]
    origins = rocv_spec.get("origins", [])
    if not origins:
        return np.nan

    errors = []
    for origin in origins:
        y_train = y_clean.iloc[:origin]
        y_test_point = y_clean.iloc[origin:origin + horizon]

        try:
            res = _fit_ets_candidate(y_train, ets_spec, use_state_space=(impl == "ETSModel"))
            preds = _forecast_ets(res, steps=horizon, index=y_test_point.index)
        except Exception:
            continue

        err = safe_mae(y_test_point.values, preds.values)
        if not np.isnan(err):
            errors.append(err)

    if len(errors) == 0:
        return np.nan
    return float(np.mean(errors))


def build_ml_features(df_sku: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML features for a single SKU.
    Features are lagged target, calendar, and optional historical exogenous fields.
    """
    df = df_sku.copy()
    y = df[COL_ACTUALS].astype(float)

    features = pd.DataFrame(index=df.index)
    features["y_lag1"] = y.shift(1)
    features["y_lag2"] = y.shift(2)
    features["y_lag3"] = y.shift(3)
    if len(y.dropna()) >= 24:
        features["y_lag12"] = y.shift(12)

    features["month_of_year"] = features.index.month
    features["trend_index"] = np.arange(len(features))

    # Optional exogenous features (historical only)
    if "New_Quotes" in df.columns:
        features["New_Quotes_l0"] = pd.to_numeric(df["New_Quotes"], errors="coerce")
        features["New_Quotes_l1"] = pd.to_numeric(df["New_Quotes"], errors="coerce").shift(1)
    if COL_OPEN_OPPS in df.columns:
        features["Open_Opportunities_l0"] = pd.to_numeric(df[COL_OPEN_OPPS], errors="coerce")
    if COL_BOOKINGS in df.columns:
        features["Bookings_l0"] = pd.to_numeric(df[COL_BOOKINGS], errors="coerce")
        features["Bookings_l1"] = pd.to_numeric(df[COL_BOOKINGS], errors="coerce").shift(1)

    return features


def _ml_exog_columns(columns: List[str]) -> List[str]:
    prefixes = ("New_Quotes", "Open_Opportunities", "Bookings")
    return [c for c in columns if c.startswith(prefixes)]


def forecast_ml_recursive(
    model: GradientBoostingRegressor,
    y_history: List[float],
    last_date: pd.Timestamp,
    horizon: int,
    feature_columns: List[str],
) -> pd.Series:
    """
    Recursive ML forecast using only lagged target + calendar + trend features.
    """
    preds = []
    history = list(y_history)
    start_trend = len(history)
    include_lag12 = "y_lag12" in feature_columns

    for step in range(1, horizon + 1):
        future_date = last_date + pd.DateOffset(months=step)
        month_of_year = future_date.month
        trend_index = start_trend + (step - 1)

        if len(history) < 3 or (include_lag12 and len(history) < 12):
            raise ValueError("Insufficient history for recursive ML forecast.")

        row = {
            "y_lag1": history[-1],
            "y_lag2": history[-2],
            "y_lag3": history[-3],
            "month_of_year": month_of_year,
            "trend_index": trend_index,
        }
        if include_lag12:
            row["y_lag12"] = history[-12]

        X_next = pd.DataFrame([row])[feature_columns]
        y_next = float(model.predict(X_next)[0])
        preds.append(y_next)
        history.append(y_next)

    index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return pd.Series(preds, index=index)


def rolling_origin_cv_ml(
    df_sku: pd.DataFrame,
    min_obs: int = ROCV_MIN_OBS,
    horizon: int = ROCV_HORIZON,
):
    """Rolling-origin CV for ML, rebuilding features up to each origin."""
    y = df_sku[COL_ACTUALS].astype(float)
    base_features = build_ml_features(df_sku)
    df_ml_full = pd.concat([base_features, y.rename("y")], axis=1).dropna()
    if len(df_ml_full) < min_obs + horizon:
        return np.nan, "Insufficient rows for ROCV."

    errors = []
    for origin in range(min_obs, len(df_ml_full) - horizon + 1):
        cutoff_idx = df_ml_full.index[origin + horizon - 1]
        df_subset = df_sku.loc[:cutoff_idx]
        features_subset = build_ml_features(df_subset)
        df_ml_subset = pd.concat(
            [features_subset, df_subset[COL_ACTUALS].astype(float).rename("y")],
            axis=1
        ).dropna()
        if len(df_ml_subset) < origin + horizon:
            continue

        train = df_ml_subset.iloc[:origin]
        test = df_ml_subset.iloc[origin:origin + horizon]
        try:
            model = GradientBoostingRegressor(random_state=42)
            model.fit(train.drop(columns=["y"]), train["y"])
            preds = model.predict(test.drop(columns=["y"]))
            err = safe_mae(test["y"].values, preds)
            if not np.isnan(err):
                errors.append(err)
        except Exception:
            continue

    if not errors:
        return np.nan, "No valid ROCV origins."
    return float(np.mean(errors)), ""


def evaluate_ml_candidate(
    df_sku: pd.DataFrame,
    horizon: int = TEST_HORIZON,
):
    """
    Fit ML challenger and return metrics + skip reasons (if any).
    """
    y = df_sku[COL_ACTUALS].astype(float)
    features = build_ml_features(df_sku)
    df_ml = pd.concat([features, y.rename("y")], axis=1).dropna()

    metrics = {
        "Model": "ML_GBR",
        "Test_MAE": np.nan,
        "Test_WMAE": np.nan,
        "Test_RMSE": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
        "ROCV_MAE": np.nan,
        "Regressor_coef": np.nan,
        "Regressor_pvalue": np.nan,
        "Skip_Reason": "",
        "Requires_Future_Exog": False,
        "Forecast_valid": False,
    }

    if len(df_ml) <= horizon + 5:
        metrics["Skip_Reason"] = "Insufficient ML rows after dropping NaNs."
        return metrics

    train_end = len(df_ml) - horizon
    train = df_ml.iloc[:train_end]
    test = df_ml.iloc[train_end:]

    exog_cols = _ml_exog_columns(list(train.columns))
    exog_nonnull = [c for c in exog_cols if train[c].notna().any()]
    feature_cols = [c for c in train.columns if c != "y"]
    if exog_nonnull:
        # Lag-only fallback when exogenous values exist but future exog isn't available.
        feature_cols = [c for c in feature_cols if c not in exog_nonnull]
    if not feature_cols:
        metrics["Skip_Reason"] = "No usable ML features after exog handling."
        return metrics

    try:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(train[feature_cols], train["y"])
        preds = model.predict(test[feature_cols])
    except Exception as exc:
        metrics["Skip_Reason"] = f"ML fit failed: {exc}"
        return metrics

    metrics["Test_MAE"] = safe_mae(test["y"].values, preds)
    metrics["Test_WMAE"] = safe_weighted_mae(test["y"].values, preds)
    metrics["Test_RMSE"] = safe_rmse(test["y"].values, preds)

    rocv_mae, rocv_reason = rolling_origin_cv_ml(df_sku)
    metrics["ROCV_MAE"] = rocv_mae
    if rocv_reason and not metrics["Skip_Reason"]:
        metrics["Skip_Reason"] = rocv_reason

    if not metrics["Requires_Future_Exog"]:
        try:
            y_history = list(y.dropna().astype(float).values)
            if y_history:
                feature_columns = list(feature_cols)
                fc = forecast_ml_recursive(
                    model=model,
                    y_history=y_history,
                    last_date=y.dropna().index.max(),
                    horizon=horizon,
                    feature_columns=feature_columns,
                )
                metrics["Forecast_valid"] = (len(fc) == horizon and np.isfinite(fc.values).all())
        except Exception as exc:
            metrics["Skip_Reason"] = metrics["Skip_Reason"] or f"ML forecast failed: {exc}"

    if metrics["Requires_Future_Exog"] and not metrics["Skip_Reason"]:
        metrics["Skip_Reason"] = "ML uses exogenous features; future exog not allowed for recommendation."

    return metrics


def fit_sarima_baseline(y_train, order, seasonal_order):
    """Fit baseline SARIMA model without exogenous regressors."""
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    return res


def fit_sarimax(y_train, exog_train, order, seasonal_order):
    """Fit SARIMAX with exogenous regressor."""
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    return res


def rolling_origin_cv(y, exog=None, order=None, seasonal_order=None,
                      min_obs=ROCV_MIN_OBS, horizon=ROCV_HORIZON):
    """
    Very simple rolling-origin cross validation for 1-step ahead.
    Returns ROCV MAE (or NaN if not enough usable origins).
    """
    y = pd.Series(y).copy()
    if exog is not None:
        exog = pd.Series(exog).copy()

        # Drop rows where exog or y are NaN to avoid exog NaN errors
        mask = (~y.isna()) & (~exog.isna())
        y = y[mask]
        exog = exog[mask]
    else:
        y = y.dropna()

    n = len(y)
    if n < min_obs + horizon:
        return np.nan

    errors = []
    # origins: we need at least 'min_obs' points to fit
    for origin in range(min_obs, n - horizon + 1):
        y_train = y.iloc[:origin]
        y_test_point = y.iloc[origin:origin + horizon]

        if exog is not None:
            exog_train = exog.iloc[:origin]
            exog_future = exog.iloc[origin:origin + horizon]

            # Require no NaNs in training or future exog for this origin
            if exog_train.isna().any() or exog_future.isna().any():
                continue

            try:
                res = fit_sarimax(y_train, exog_train, order, seasonal_order)
                preds = res.get_forecast(steps=horizon, exog=exog_future).predicted_mean
            except Exception:
                continue
        else:
            try:
                res = fit_sarima_baseline(y_train, order, seasonal_order)
                preds = res.get_forecast(steps=horizon).predicted_mean
            except Exception:
                continue

        err = safe_mae(y_test_point.values, preds.values)
        if not np.isnan(err):
            errors.append(err)

    if len(errors) == 0:
        return np.nan

    return float(np.mean(errors))


def evaluate_single_model(
    y,
    exog=None,
    model_name="SARIMA_baseline",
    horizon=TEST_HORIZON,
    order=None,
    seasonal_order=None,
    sku=None,
    bu=None,
    exog_clean_full=None,
    rocv_spec=None,
    exog_invalid=False
):
    """
    Fit model, compute test MAE/RMSE on last 'horizon' points,
    compute ROCV MAE, and return a dict of metrics.
    """
    y = pd.Series(y).astype(float)
    n_total = len(y)

    # If too short, bail with NaNs
    if n_total <= horizon + 5:  # need at least some training
        return {
            "Model": model_name,
            "Test_MAE": np.nan,
            "Test_WMAE": np.nan,
            "Test_RMSE": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "ROCV_MAE": np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }

    if exog is not None:
        if exog_invalid:
            return _regressor_failure_metrics(model_name)

        train_end = n_total - horizon
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:]

        exog_clean_full = exog_clean_full if exog_clean_full is not None else pd.Series(exog).astype(float).ffill().bfill()
        exog_train = exog_clean_full.iloc[:train_end]
        exog_test = exog_clean_full.iloc[train_end:]

        # Coverage warning only
        coverage = exog_train.notna().mean()
        if coverage < 0.7:
            sku_txt = f"{sku}/{bu}" if sku or bu else ""
            print(f"[WARN] Low exog coverage ({coverage:.2%}) for {sku_txt}. Proceeding with cleaned exog.")

        # If still all NaN after cleaning, bail with large MAE
        if exog_train.notna().sum() == 0:
            return _regressor_failure_metrics(model_name)

        # Fit SARIMAX
        try:
            res = fit_sarimax(y_train, exog_train, order=order, seasonal_order=seasonal_order)
        except Exception:
            return _regressor_failure_metrics(model_name)

        # Forecast
        try:
            forecast_res = res.get_forecast(steps=horizon, exog=exog_test)
            y_pred = forecast_res.predicted_mean
        except Exception:
            y_pred = pd.Series(index=y_test.index, data=np.nan)

        # ROCV
        rocv_mae = rolling_origin_cv_precomputed(rocv_spec, order=order, seasonal_order=seasonal_order)

        # Extract regressor coefficient & pvalue (first exog param)
        reg_coef = np.nan
        reg_pval = np.nan
        try:
            # res.params is an ordered index; exog coeffs come first by default
            reg_coef = float(res.params.iloc[0])
            reg_pval = float(res.pvalues.iloc[0])
        except Exception:
            pass

        return {
            "Model": model_name,
            "Test_MAE": safe_mae(y_test.values, y_pred.values) if not y_pred.isna().all() else 1e9,
            "Test_WMAE": safe_weighted_mae(y_test.values, y_pred.values) if not y_pred.isna().all() else 1e9,
            "Test_RMSE": safe_rmse(y_test.values, y_pred.values) if not y_pred.isna().all() else 1e9,
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "ROCV_MAE": float(rocv_mae) if rocv_mae is not None else np.nan,
            "Regressor_coef": reg_coef,
            "Regressor_pvalue": reg_pval
        }

    else:
        train_end = n_total - horizon
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:]

        # Fit baseline SARIMA
        try:
            res = fit_sarima_baseline(y_train, order=order, seasonal_order=seasonal_order)
        except Exception:
            return {
                "Model": model_name,
                "Test_MAE": np.nan,
                "Test_WMAE": np.nan,
                "Test_RMSE": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "ROCV_MAE": np.nan,
                "Regressor_coef": np.nan,
                "Regressor_pvalue": np.nan
            }

        try:
            forecast_res = res.get_forecast(steps=horizon)
            y_pred = forecast_res.predicted_mean
        except Exception:
            y_pred = pd.Series(index=y_test.index, data=np.nan)

        rocv_mae = rolling_origin_cv_precomputed(rocv_spec, order=order, seasonal_order=seasonal_order)

        return {
            "Model": model_name,
            "Test_MAE": safe_mae(y_test.values, y_pred.values),
            "Test_WMAE": safe_weighted_mae(y_test.values, y_pred.values),
            "Test_RMSE": safe_rmse(y_test.values, y_pred.values),
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "ROCV_MAE": float(rocv_mae) if rocv_mae is not None else np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }


def evaluate_ets_model(
    y,
    model_name="ETS_baseline",
    horizon=TEST_HORIZON,
    sku=None,
    bu=None,
    rocv_spec=None,
):
    """
    Fit ETS (best by AICc/AIC), compute test MAE/RMSE on last 'horizon' points,
    compute ROCV MAE, and return a dict of metrics.
    """
    y = pd.Series(y).astype(float)
    y_clean = y.dropna()
    n_total = len(y_clean)

    if n_total <= horizon:
        return {
            "Model": model_name,
            "Test_MAE": np.nan,
            "Test_WMAE": np.nan,
            "Test_RMSE": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "ROCV_MAE": np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }

    train_end = n_total - horizon
    y_train = y_clean.iloc[:train_end]
    y_test = y_clean.iloc[train_end:]

    context = f" for {sku}/{bu}" if sku or bu else ""
    ets_res, ets_spec, ets_impl = _select_best_ets_model(y_train, context=context)
    if ets_res is None:
        return {
            "Model": model_name,
            "Test_MAE": np.nan,
            "Test_WMAE": np.nan,
            "Test_RMSE": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "ROCV_MAE": np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }

    try:
        y_pred = _forecast_ets(ets_res, steps=horizon, index=y_test.index)
    except Exception:
        y_pred = pd.Series(index=y_test.index, data=np.nan)

    rocv_mae = rolling_origin_cv_ets_precomputed(rocv_spec, ets_spec, ets_impl)

    aic = getattr(ets_res, "aic", np.nan)
    bic = getattr(ets_res, "bic", np.nan)

    return {
        "Model": model_name,
        "Test_MAE": safe_mae(y_test.values, y_pred.values),
        "Test_WMAE": safe_weighted_mae(y_test.values, y_pred.values),
        "Test_RMSE": safe_rmse(y_test.values, y_pred.values),
        "AIC": float(aic) if np.isfinite(aic) else np.nan,
        "BIC": float(bic) if np.isfinite(bic) else np.nan,
        "ROCV_MAE": float(rocv_mae) if rocv_mae is not None else np.nan,
        "Regressor_coef": np.nan,
        "Regressor_pvalue": np.nan
    }


def _prepare_prophet_df(y_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        "ds": pd.to_datetime(y_series.index).tz_localize(None),
        "y": pd.to_numeric(y_series.values, errors="coerce"),
    })
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds")
    df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
    return df


def _coerce_month_start_series(y_series: pd.Series) -> pd.Series:
    """Normalize index to month starts without dropping valid month-end data."""
    y_clean = pd.Series(y_series).astype(float).dropna()
    if y_clean.empty:
        return y_clean
    idx = pd.to_datetime(y_clean.index).to_period("M").to_timestamp()
    idx = pd.DatetimeIndex(idx)
    y_clean.index = idx
    y_clean = y_clean.groupby(level=0).sum().sort_index()
    return y_clean


def _evaluate_prophet_metrics(y_series, horizon=TEST_HORIZON, sku=None, bu=None):
    metrics = {
        "Model": "PROPHET",
        "Test_MAE": np.nan,
        "Test_WMAE": np.nan,
        "Test_RMSE": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
        "ROCV_MAE": np.nan,
        "Regressor_coef": np.nan,
        "Regressor_pvalue": np.nan,
        "Skip_Reason": "",
        "Forecast_valid": False,
    }

    if not PROPHET_AVAILABLE:
        metrics["Skip_Reason"] = "Prophet not installed."
        return metrics

    y_clean = _coerce_month_start_series(y_series)
    if len(y_clean) < 24:
        metrics["Skip_Reason"] = "Insufficient history (<24 months)."
        return metrics

    if len(y_clean) <= horizon + 5:
        metrics["Skip_Reason"] = "Insufficient history after holdout."
        return metrics

    train_end = len(y_clean) - horizon
    y_train = y_clean.iloc[:train_end]
    y_test = y_clean.iloc[train_end:]

    try:
        train_df = _prepare_prophet_df(y_train)
        if train_df.empty:
            metrics["Skip_Reason"] = "No usable Prophet training data."
            return metrics

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
        )
        model.fit(train_df)

        future_df = pd.DataFrame({"ds": y_test.index})
        future_df["ds"] = pd.to_datetime(future_df["ds"]).dt.tz_localize(None)
        future_df["ds"] = future_df["ds"].dt.to_period("M").dt.to_timestamp()
        forecast = model.predict(future_df)
        y_pred = pd.Series(forecast["yhat"].values, index=y_test.index)

        metrics["Test_MAE"] = safe_mae(y_test.values, y_pred.values)
        metrics["Test_WMAE"] = safe_weighted_mae(y_test.values, y_pred.values)
        metrics["Test_RMSE"] = safe_rmse(y_test.values, y_pred.values)
        metrics["Forecast_valid"] = np.isfinite(y_pred.values).all() and len(y_pred) == len(y_test)
    except Exception as exc:
        sku_txt = f"{sku}/{bu}" if sku or bu else ""
        print(f"Prophet failed for SKU {sku_txt}: {exc}")
        metrics["Skip_Reason"] = f"Prophet failed: {exc}"
        return metrics

    # ROCV
    try:
        rocv_maes = []
        min_train = max(24, ROCV_MIN_OBS)
        for origin in range(min_train, len(y_clean) - ROCV_HORIZON + 1):
            y_train_rocv = y_clean.iloc[:origin]
            y_test_rocv = y_clean.iloc[origin:origin + ROCV_HORIZON]
            train_df = _prepare_prophet_df(y_train_rocv)
            if train_df.empty:
                continue
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="additive",
                changepoint_prior_scale=0.05,
            )
            model.fit(train_df)
            future_df = pd.DataFrame({"ds": y_test_rocv.index})
            future_df["ds"] = pd.to_datetime(future_df["ds"]).dt.tz_localize(None)
            future_df["ds"] = future_df["ds"].dt.to_period("M").dt.to_timestamp()
            forecast = model.predict(future_df)
            preds = pd.Series(forecast["yhat"].values, index=y_test_rocv.index)
            err = safe_mae(y_test_rocv.values, preds.values)
            if not np.isnan(err):
                rocv_maes.append(err)

        if rocv_maes:
            metrics["ROCV_MAE"] = float(np.mean(rocv_maes))
        else:
            metrics["ROCV_MAE"] = np.nan
            if not metrics["Skip_Reason"]:
                metrics["Skip_Reason"] = "No valid ROCV origins."
    except Exception:
        metrics["ROCV_MAE"] = np.nan

    return metrics


def _candidate_passes_rules(m, baseline, epsilon=EPSILON_IMPROVEMENT, delta=DELTA_ROCV_TOLERANCE):
    if baseline is None:
        return False

    if m.get("Requires_Future_Exog"):
        return False
    if m.get("Forecast_valid") is False:
        return False

    mae = m.get("Test_MAE", np.nan)
    rocv = m.get("ROCV_MAE", np.nan)
    baseline_mae = baseline.get("Test_MAE", np.nan)
    baseline_rocv = baseline.get("ROCV_MAE", np.nan)

    if np.isnan(mae) or np.isnan(baseline_mae):
        return False

    if mae > baseline_mae * (1.0 - epsilon):
        return False

    if (not np.isnan(baseline_rocv)) and (not np.isnan(rocv)):
        if rocv > baseline_rocv * (1.0 + delta):
            return False

    if np.isnan(rocv) and not np.isnan(baseline_rocv):
        return False

    return True


def choose_best_model(metrics_list, epsilon=EPSILON_IMPROVEMENT, delta=DELTA_ROCV_TOLERANCE):
    """
    Given a list of metrics dicts (all for same SKU),
    apply acceptance rules and pick a best model.
    Returns chosen dict + baseline dict + some derived fields.
    """
    if not metrics_list:
        return None, None

    def _weighted_mae(m):
        wmae = m.get("Test_WMAE", np.nan)
        if np.isfinite(wmae):
            return wmae
        return m.get("Test_MAE", np.nan)

    def wmae_rmse_key(m):
        mae = _weighted_mae(m)
        rmse = m.get("Test_RMSE", np.nan)
        return (
            mae if np.isfinite(mae) else np.inf,
            rmse if np.isfinite(rmse) else np.inf,
        )

    baseline_models = {"SARIMA_baseline", "ETS_baseline"}
    baseline_candidates = [m for m in metrics_list if m["Model"] in baseline_models]
    baseline = min(baseline_candidates, key=wmae_rmse_key) if baseline_candidates else None

    if baseline is None:
        # Fallback: choose minimum weighted MAE
        valid = [m for m in metrics_list if np.isfinite(_weighted_mae(m))]
        if not valid:
            return None, None
        best = min(valid, key=wmae_rmse_key)
        return best, None

    baseline_mae = baseline["Test_MAE"]
    baseline_rocv = baseline["ROCV_MAE"]

    # If baseline itself is NaN on Test_MAE, just choose lowest weighted MAE overall
    if np.isnan(baseline_mae):
        valid = [m for m in metrics_list if np.isfinite(_weighted_mae(m))]
        if not valid:
            return None, baseline
        best = min(valid, key=wmae_rmse_key)
        return best, baseline

    accepted = []
    for m in metrics_list:
        if m["Model"] in baseline_models:
            continue  # baseline is always a fallback, not an "accepted regressor" candidate

        if _candidate_passes_rules(m, baseline, epsilon=epsilon, delta=delta):
            accepted.append(m)

    if not accepted:
        # No regressor model passes the rules → choose baseline
        return baseline, baseline

    # If multiple accepted, choose the one with lowest weighted MAE
    best = min(accepted, key=wmae_rmse_key)
    return best, baseline


def _model_type(model_name: str) -> str:
    if model_name == "ML_GBR":
        return "ML"
    if model_name == "PROPHET":
        return "PROPHET"
    if model_name == "ETS_baseline":
        return "ETS"
    if model_name == "SARIMA_baseline":
        return "SARIMA"
    if model_name.startswith("SARIMAX"):
        return "SARIMAX"
    return "Other"


def engineer_regressors(df_sku):
    """
    Given a single-SKU dataframe (sorted by Month),
    engineer lagged exogenous regressors we want to test.
    NOTE: We intentionally exclude lag-0 regressors; only lags 1-3 are kept.
    """
    df = df_sku.copy()
    # Ensure numeric
    for col in [
        COL_NEW_OPPORTUNITIES,
        COL_OPEN_OPPS,
        COL_BOOKINGS,
        COL_MEDIAN_MONTHS_SINCE_LAST_ACTIVITY,
        COL_OPEN_NOT_MODIFIED_90_DAYS,
        COL_PCT_OPEN_NOT_MODIFIED_90_DAYS,
        COL_EARLY_TO_LATE_RATIO,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Lagged features we are willing to use (1-3 where applicable)
    if COL_NEW_OPPORTUNITIES in df.columns:
        df["New_Opportunities_l1"] = df[COL_NEW_OPPORTUNITIES].shift(1)
        df["New_Opportunities_l2"] = df[COL_NEW_OPPORTUNITIES].shift(2)
        df["New_Opportunities_l3"] = df[COL_NEW_OPPORTUNITIES].shift(3)

    if COL_OPEN_OPPS in df.columns:
        df["Open_Opportunities_l1"] = df[COL_OPEN_OPPS].shift(1)
        df["Open_Opportunities_l2"] = df[COL_OPEN_OPPS].shift(2)
        df["Open_Opportunities_l3"] = df[COL_OPEN_OPPS].shift(3)

    if COL_BOOKINGS in df.columns:
        df["Bookings_l1"] = df[COL_BOOKINGS].shift(1)
        df["Bookings_l2"] = df[COL_BOOKINGS].shift(2)

    if COL_MEDIAN_MONTHS_SINCE_LAST_ACTIVITY in df.columns:
        df["Median_Months_Since_Last_Activity_l1"] = df[
            COL_MEDIAN_MONTHS_SINCE_LAST_ACTIVITY
        ].shift(1)
        df["Median_Months_Since_Last_Activity_l2"] = df[
            COL_MEDIAN_MONTHS_SINCE_LAST_ACTIVITY
        ].shift(2)

    if COL_OPEN_NOT_MODIFIED_90_DAYS in df.columns:
        df["Open_Not_Modified_90_Days_l1"] = df[COL_OPEN_NOT_MODIFIED_90_DAYS].shift(1)

    if COL_PCT_OPEN_NOT_MODIFIED_90_DAYS in df.columns:
        df["Pct_Open_Not_Modified_90_Days_l1"] = df[
            COL_PCT_OPEN_NOT_MODIFIED_90_DAYS
        ].shift(1)
        df["Pct_Open_Not_Modified_90_Days_l2"] = df[
            COL_PCT_OPEN_NOT_MODIFIED_90_DAYS
        ].shift(2)

    if COL_EARLY_TO_LATE_RATIO in df.columns:
        df["Early_to_Late_Ratio_l1"] = df[COL_EARLY_TO_LATE_RATIO].shift(1)
        df["Early_to_Late_Ratio_l2"] = df[COL_EARLY_TO_LATE_RATIO].shift(2)

    return df


# ===========================
# 3. MAIN ENGINE
# ===========================

def _parse_order_cell(value, expected_len):
    """Parse order tuple stored in Excel as tuple/list/string."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        if value.strip() == "":
            return None
        try:
            candidate = ast.literal_eval(value)
        except Exception:
            return None
    elif isinstance(value, (list, tuple)):
        candidate = tuple(value)
    else:
        try:
            candidate = ast.literal_eval(str(value))
        except Exception:
            return None
    if not isinstance(candidate, (list, tuple)):
        return None
    if len(candidate) != expected_len:
        return None
    try:
        return tuple(int(x) for x in candidate)
    except Exception:
        return None


def _load_manual_order_overrides(path=NOTES_FILE):
    """Load manual SARIMA orders from Notes.xlsx if available."""
    if path is None or not os.path.exists(path):
        return {}
    try:
        df_notes = pd.read_excel(path)
    except Exception as exc:
        print(f"Could not read manual overrides from {path}: {exc}")
        return {}

    # Support either the main column names or the Notes.xlsx schema (group_key/BU).
    product_col = COL_PRODUCT if COL_PRODUCT in df_notes.columns else None
    division_col = COL_DIVISION if COL_DIVISION in df_notes.columns else None
    if product_col is None and "group_key" in df_notes.columns:
        product_col = "group_key"
    if division_col is None and "BU" in df_notes.columns:
        division_col = "BU"

    required_cols = {product_col, division_col, "Chosen_Order", "Chosen_Seasonal_Order"}
    if None in required_cols or not required_cols.issubset(df_notes.columns):
        print(f"Manual overrides file {path} is missing required columns; skipping.")
        return {}

    overrides = {}
    for _, row in df_notes.iterrows():
        order = _parse_order_cell(row["Chosen_Order"], 3)
        seas_order = _parse_order_cell(row["Chosen_Seasonal_Order"], 4)
        if order is None or seas_order is None:
            continue
        prod = row[product_col]
        div = row[division_col]
        overrides[(prod, div)] = (order, seas_order)
    return overrides


def load_orders_map(path=ORDER_FILE, overrides_path=None):
    """
    Load per-SKU SARIMA orders from Excel.
    Returns dict keyed by (Product, Division) with (order, seasonal_order) tuples.
    """
    df_orders = pd.read_excel(path)
    required_cols = {COL_PRODUCT, COL_DIVISION, "Chosen_Order", "Chosen_Seasonal_Order"}
    if not required_cols.issubset(df_orders.columns):
        missing = required_cols - set(df_orders.columns)
        raise ValueError(f"Missing required columns in {path}: {missing}")

    orders_map = {}
    for _, row in df_orders.iterrows():
        prod = row[COL_PRODUCT]
        div = row[COL_DIVISION]
        order = _parse_order_cell(row["Chosen_Order"], 3)
        seas_order = _parse_order_cell(row["Chosen_Seasonal_Order"], 4)
        if order is None or seas_order is None:
            continue
        orders_map[(prod, div)] = (order, seas_order)

    overrides = _load_manual_order_overrides(overrides_path or NOTES_FILE)
    if overrides:
        orders_map.update(overrides)
        print(f"Applied {len(overrides)} manual order override(s) from {overrides_path or NOTES_FILE}.")
    return orders_map


def main():
    args = _parse_args()
    enable_ml = ENABLE_ML_CHALLENGER and (not args.disable_ml)
    enable_prophet = ENABLE_PROPHET_CHALLENGER and (not args.disable_prophet)
    if enable_ml:
        print("ML challenger enabled — evaluating ML_GBR candidate.")
    else:
        print("ML challenger disabled (ENABLE_ML_CHALLENGER=False) — skipping ML candidate evaluation.")
    if enable_prophet:
        if PROPHET_AVAILABLE:
            print("Prophet challenger enabled — evaluating PROPHET candidate.")
        else:
            print("Prophet not installed; skipping Prophet challenger.")
            enable_prophet = False
    else:
        print("Prophet challenger disabled (ENABLE_PROPHET_CHALLENGER=False) — skipping Prophet candidate evaluation.")

    print("Loading data...")
    df_all = pd.read_excel(INPUT_FILE)
    df_all = aggregate_monthly_duplicates(df_all)
    df_all = mark_prelaunch_actuals_as_missing(
        df_all,
        output_excel_path=_output_path(REVISED_ACTUALS_FILE),
        output_sheet=REVISED_ACTUALS_SHEET,
    )
    print(f"Loading per-SKU SARIMA orders from {ORDER_FILE}...")
    orders_map = load_orders_map(ORDER_FILE, overrides_path=NOTES_FILE)

    # Basic cleaning
    df_all[COL_DATE] = pd.to_datetime(df_all[COL_DATE])
    df_all = df_all.sort_values([COL_PRODUCT, COL_DIVISION, COL_DATE])

    results_rows = []
    ranking_rows_all = []
    skipped_no_order = []

    # Loop over SKUs (by Product + Division)
    sku_groups = df_all.groupby([COL_PRODUCT, COL_DIVISION], dropna=False)
    total_skus = len(sku_groups)
    print(f"Found {total_skus} product+division combinations.")

    for (prod, div), df_sku in sku_groups:
        print(f"\n=== Processing SKU: {prod}, Division: {div} ===")
        order_pair = orders_map.get((prod, div))
        if order_pair is None:
            print("  -> Skipping: no order/seasonal_order found in sarimax_order_search_summary.")
            skipped_no_order.append({"Product": prod, "Division": div, "Reason": "No order in order search summary"})
            results_rows.append({
                "Product": prod,
                "Division": div,
                "Chosen_Model": "NO_ORDER_IN_FILE",
                "Baseline_MAE": np.nan,
                "Baseline_ROCV_MAE": np.nan,
                "Chosen_MAE": np.nan,
                "Chosen_RMSE": np.nan,
                "Chosen_ROCV_MAE": np.nan,
                "MAE_Improvement_Pct": np.nan,
                "Regressor_Name": None,
                "Regressor_Lag": None,
                "Regressor_Coef": np.nan,
                "Regressor_pvalue": np.nan,
                "Baseline_AIC": np.nan,
                "Baseline_BIC": np.nan,
                "Chosen_AIC": np.nan,
                "Chosen_BIC": np.nan,
                "Accepted_by_rules": False
            })
            continue

        order, seasonal_order = order_pair
        df_sku = df_sku.sort_values(COL_DATE).set_index(COL_DATE)

        # Engineer regressors
        df_sku = engineer_regressors(df_sku)

        y = df_sku[COL_ACTUALS].astype(float)
        n = len(y)
        y_nonnull = y.dropna()
        last_actual_dt = y_nonnull.index.max() if not y_nonnull.empty else None
        nonzero_mask = y_nonnull != 0
        first_nonzero_dt = y_nonnull[nonzero_mask].index.min() if nonzero_mask.any() else None
        if first_nonzero_dt is not None and last_actual_dt is not None:
            months_since_first_nonzero = (
                last_actual_dt.to_period("M") - first_nonzero_dt.to_period("M")
            ).n
        else:
            months_since_first_nonzero = -1
        history_months = len(y_nonnull)
        allow_sarima = history_months >= 12 and months_since_first_nonzero >= 11
        allow_sarimax = history_months >= 36 and allow_sarima
        baseline_rocv_spec = build_rocv_spec(y, exog_series=None)
        test_horizon = _compute_test_window(history_months)

        if n < MIN_OBS_TOTAL:
            print(f"  -> Skipping (only {n} observations; need at least {MIN_OBS_TOTAL}).")
            results_rows.append({
                "Product": prod,
                "Division": div,
                "Chosen_Model": "INSUFFICIENT_HISTORY",
                "Baseline_MAE": np.nan,
                "Baseline_ROCV_MAE": np.nan,
                "Chosen_MAE": np.nan,
                "Chosen_RMSE": np.nan,
                "Chosen_ROCV_MAE": np.nan,
                "MAE_Improvement_Pct": np.nan,
                "Regressor_Name": None,
                "Regressor_Lag": None,
                "Regressor_Coef": np.nan,
                "Regressor_pvalue": np.nan,
                "Baseline_AIC": np.nan,
                "Baseline_BIC": np.nan,
                "Chosen_AIC": np.nan,
                "Chosen_BIC": np.nan,
                "Accepted_by_rules": False
            })
            continue

        metrics_list = []
        ranking_rows = []

        # --- Baseline model ---
        if allow_sarima:
            print("  Fitting SARIMA_baseline...")
            baseline_metrics = evaluate_single_model(
                y=y,
                exog=None,
                model_name="SARIMA_baseline",
                horizon=test_horizon,
                order=order,
                seasonal_order=seasonal_order,
                sku=prod,
                bu=div,
                rocv_spec=baseline_rocv_spec
            )
            if (
                baseline_metrics is not None
                and not np.isnan(baseline_metrics.get("Test_MAE", np.nan))
                and abs(baseline_metrics["Test_MAE"]) <= BASELINE_MAE_ZERO_EPS
            ):
                print(f"[INFO] Baseline MAE ~0 for {prod}/{div}; treating as perfect baseline.")
            metrics_list.append(baseline_metrics)
            ranking_rows.append({
                "Product": prod,
                "Division": div,
                "Model": baseline_metrics["Model"],
                "Test_MAE": baseline_metrics["Test_MAE"],
                "Test_RMSE": baseline_metrics["Test_RMSE"],
                "ROCV_MAE": baseline_metrics["ROCV_MAE"],
                "AIC": baseline_metrics["AIC"],
                "BIC": baseline_metrics["BIC"],
                "Regressor_Name": None,
                "Regressor_Lag": None,
                "Accepted_by_rules": False  # baseline not part of acceptance set
            })
        else:
            print("  Skipping SARIMA_baseline: history <12 months or first non-zero <12 months ago.")

        # --- ETS baseline ---
        print("  Fitting ETS_baseline...")
        ets_metrics = evaluate_ets_model(
            y=y,
            model_name="ETS_baseline",
            horizon=test_horizon,
            sku=prod,
            bu=div,
            rocv_spec=baseline_rocv_spec
        )
        metrics_list.append(ets_metrics)
        ranking_rows.append({
            "Product": prod,
            "Division": div,
            "Model": ets_metrics["Model"],
            "Test_MAE": ets_metrics["Test_MAE"],
            "Test_RMSE": ets_metrics["Test_RMSE"],
            "ROCV_MAE": ets_metrics["ROCV_MAE"],
            "AIC": ets_metrics["AIC"],
            "BIC": ets_metrics["BIC"],
            "Regressor_Name": None,
            "Regressor_Lag": None,
            "Accepted_by_rules": False
        })

        # --- SARIMAX with New_Opportunities lags 1-3 ---
        if allow_sarimax:
            for lag in [1, 2, 3]:
                col = f"New_Opportunities_l{lag}"
                if col in df_sku.columns:
                    print(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_list.append(
                        evaluate_single_model(
                            y=y,
                            exog=exog_series,
                            model_name=f"SARIMAX_{col}",
                        horizon=test_horizon,
                        order=order,
                        seasonal_order=seasonal_order,
                        sku=prod,
                            bu=div,
                            exog_clean_full=exog_clean_full,
                            rocv_spec=rocv_spec,
                            exog_invalid=exog_invalid
                        )
                    )
                    metrics_row = metrics_list[-1]
                    ranking_rows.append({
                        "Product": prod,
                        "Division": div,
                        "Model": metrics_row["Model"],
                        "Test_MAE": metrics_row["Test_MAE"],
                        "Test_RMSE": metrics_row["Test_RMSE"],
                        "ROCV_MAE": metrics_row["ROCV_MAE"],
                        "AIC": metrics_row["AIC"],
                        "BIC": metrics_row["BIC"],
                        "Regressor_Name": "New_Opportunities",
                        "Regressor_Lag": lag,
                        "Accepted_by_rules": False  # updated later once chosen
                    })
        else:
            print("  Skipping SARIMAX (New_Opportunities): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Open_Opportunities lags 1-3 ---
        if allow_sarimax:
            for lag in [1, 2, 3]:
                col = f"Open_Opportunities_l{lag}"
                if col in df_sku.columns:
                    print(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_list.append(
                        evaluate_single_model(
                            y=y,
                            exog=exog_series,
                            model_name=f"SARIMAX_{col}",
                        horizon=test_horizon,
                        order=order,
                        seasonal_order=seasonal_order,
                        sku=prod,
                            bu=div,
                            exog_clean_full=exog_clean_full,
                            rocv_spec=rocv_spec,
                            exog_invalid=exog_invalid
                        )
                    )
                    metrics_row = metrics_list[-1]
                    ranking_rows.append({
                        "Product": prod,
                        "Division": div,
                        "Model": metrics_row["Model"],
                        "Test_MAE": metrics_row["Test_MAE"],
                        "Test_RMSE": metrics_row["Test_RMSE"],
                        "ROCV_MAE": metrics_row["ROCV_MAE"],
                        "AIC": metrics_row["AIC"],
                        "BIC": metrics_row["BIC"],
                        "Regressor_Name": "Open_Opportunities",
                        "Regressor_Lag": lag,
                        "Accepted_by_rules": False
                    })
        else:
            print("  Skipping SARIMAX (Open_Opportunities): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Bookings lags 1-2 ---
        if allow_sarimax:
            for lag in [1, 2]:
                col = f"Bookings_l{lag}"
                if col in df_sku.columns:
                    print(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_list.append(
                        evaluate_single_model(
                            y=y,
                            exog=exog_series,
                            model_name=f"SARIMAX_{col}",
                        horizon=test_horizon,
                        order=order,
                        seasonal_order=seasonal_order,
                        sku=prod,
                            bu=div,
                            exog_clean_full=exog_clean_full,
                            rocv_spec=rocv_spec,
                            exog_invalid=exog_invalid
                        )
                    )
                    metrics_row = metrics_list[-1]
                    ranking_rows.append({
                        "Product": prod,
                        "Division": div,
                        "Model": metrics_row["Model"],
                        "Test_MAE": metrics_row["Test_MAE"],
                        "Test_RMSE": metrics_row["Test_RMSE"],
                        "ROCV_MAE": metrics_row["ROCV_MAE"],
                        "AIC": metrics_row["AIC"],
                        "BIC": metrics_row["BIC"],
                        "Regressor_Name": "Bookings",
                        "Regressor_Lag": lag,
                        "Accepted_by_rules": False
                    })
        else:
            print("  Skipping SARIMAX (Bookings): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Median_Months_Since_Last_Activity lags 1-2 ---
        if allow_sarimax:
            for lag in [1, 2]:
                col = f"Median_Months_Since_Last_Activity_l{lag}"
                if col in df_sku.columns:
                    print(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_list.append(
                        evaluate_single_model(
                            y=y,
                            exog=exog_series,
                            model_name=f"SARIMAX_{col}",
                            horizon=test_horizon,
                            order=order,
                            seasonal_order=seasonal_order,
                            sku=prod,
                            bu=div,
                            exog_clean_full=exog_clean_full,
                            rocv_spec=rocv_spec,
                            exog_invalid=exog_invalid
                        )
                    )
                    metrics_row = metrics_list[-1]
                    ranking_rows.append({
                        "Product": prod,
                        "Division": div,
                        "Model": metrics_row["Model"],
                        "Test_MAE": metrics_row["Test_MAE"],
                        "Test_RMSE": metrics_row["Test_RMSE"],
                        "ROCV_MAE": metrics_row["ROCV_MAE"],
                        "AIC": metrics_row["AIC"],
                        "BIC": metrics_row["BIC"],
                        "Regressor_Name": "Median_Months_Since_Last_Activity",
                        "Regressor_Lag": lag,
                        "Accepted_by_rules": False
                    })
        else:
            print("  Skipping SARIMAX (Median_Months_Since_Last_Activity): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Open_Not_Modified_90_Days lag 1 ---
        if allow_sarimax:
            for lag in [1]:
                col = f"Open_Not_Modified_90_Days_l{lag}"
                if col in df_sku.columns:
                    print(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_list.append(
                        evaluate_single_model(
                            y=y,
                            exog=exog_series,
                            model_name=f"SARIMAX_{col}",
                            horizon=test_horizon,
                            order=order,
                            seasonal_order=seasonal_order,
                            sku=prod,
                            bu=div,
                            exog_clean_full=exog_clean_full,
                            rocv_spec=rocv_spec,
                            exog_invalid=exog_invalid
                        )
                    )
                    metrics_row = metrics_list[-1]
                    ranking_rows.append({
                        "Product": prod,
                        "Division": div,
                        "Model": metrics_row["Model"],
                        "Test_MAE": metrics_row["Test_MAE"],
                        "Test_RMSE": metrics_row["Test_RMSE"],
                        "ROCV_MAE": metrics_row["ROCV_MAE"],
                        "AIC": metrics_row["AIC"],
                        "BIC": metrics_row["BIC"],
                        "Regressor_Name": "Open_Not_Modified_90_Days",
                        "Regressor_Lag": lag,
                        "Accepted_by_rules": False
                    })
        else:
            print("  Skipping SARIMAX (Open_Not_Modified_90_Days): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Pct_Open_Not_Modified_90_Days lags 1-2 ---
        if allow_sarimax:
            for lag in [1, 2]:
                col = f"Pct_Open_Not_Modified_90_Days_l{lag}"
                if col in df_sku.columns:
                    print(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_list.append(
                        evaluate_single_model(
                            y=y,
                            exog=exog_series,
                            model_name=f"SARIMAX_{col}",
                            horizon=test_horizon,
                            order=order,
                            seasonal_order=seasonal_order,
                            sku=prod,
                            bu=div,
                            exog_clean_full=exog_clean_full,
                            rocv_spec=rocv_spec,
                            exog_invalid=exog_invalid
                        )
                    )
                    metrics_row = metrics_list[-1]
                    ranking_rows.append({
                        "Product": prod,
                        "Division": div,
                        "Model": metrics_row["Model"],
                        "Test_MAE": metrics_row["Test_MAE"],
                        "Test_RMSE": metrics_row["Test_RMSE"],
                        "ROCV_MAE": metrics_row["ROCV_MAE"],
                        "AIC": metrics_row["AIC"],
                        "BIC": metrics_row["BIC"],
                        "Regressor_Name": "Pct_Open_Not_Modified_90_Days",
                        "Regressor_Lag": lag,
                        "Accepted_by_rules": False
                    })
        else:
            print("  Skipping SARIMAX (Pct_Open_Not_Modified_90_Days): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Early_to_Late_Ratio lags 1-2 ---
        if allow_sarimax:
            for lag in [1, 2]:
                col = f"Early_to_Late_Ratio_l{lag}"
                if col in df_sku.columns:
                    print(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_list.append(
                        evaluate_single_model(
                            y=y,
                            exog=exog_series,
                            model_name=f"SARIMAX_{col}",
                            horizon=test_horizon,
                            order=order,
                            seasonal_order=seasonal_order,
                            sku=prod,
                            bu=div,
                            exog_clean_full=exog_clean_full,
                            rocv_spec=rocv_spec,
                            exog_invalid=exog_invalid
                        )
                    )
                    metrics_row = metrics_list[-1]
                    ranking_rows.append({
                        "Product": prod,
                        "Division": div,
                        "Model": metrics_row["Model"],
                        "Test_MAE": metrics_row["Test_MAE"],
                        "Test_RMSE": metrics_row["Test_RMSE"],
                        "ROCV_MAE": metrics_row["ROCV_MAE"],
                        "AIC": metrics_row["AIC"],
                        "BIC": metrics_row["BIC"],
                        "Regressor_Name": "Early_to_Late_Ratio",
                        "Regressor_Lag": lag,
                        "Accepted_by_rules": False
                    })
        else:
            print("  Skipping SARIMAX (Early_to_Late_Ratio): history <36 months or SARIMA gate not met.")

        # --- ML challenger (GradientBoostingRegressor) ---
        if enable_ml:
            print("  Evaluating ML_GBR...")
            ml_metrics = evaluate_ml_candidate(df_sku, horizon=test_horizon)
            metrics_list.append(ml_metrics)
            if ml_metrics.get("Skip_Reason"):
                print(f"  -> ML skipped: {ml_metrics['Skip_Reason']}")
            ranking_rows.append({
                "Product": prod,
                "Division": div,
                "Model": ml_metrics["Model"],
                "Model_Type": _model_type(ml_metrics["Model"]),
                "Test_MAE": ml_metrics["Test_MAE"],
                "Test_RMSE": ml_metrics["Test_RMSE"],
                "ROCV_MAE": ml_metrics["ROCV_MAE"],
                "AIC": ml_metrics["AIC"],
                "BIC": ml_metrics["BIC"],
                "Regressor_Name": None,
                "Regressor_Lag": None,
                "Accepted_by_rules": False,
                "Skip_Reason": ml_metrics.get("Skip_Reason", ""),
            })

        # --- Prophet challenger ---
        if enable_prophet:
            print("  Evaluating PROPHET...")
            prophet_metrics = _evaluate_prophet_metrics(y, horizon=test_horizon, sku=prod, bu=div)
            metrics_list.append(prophet_metrics)
            if prophet_metrics.get("Skip_Reason"):
                print(f"  -> Prophet skipped: {prophet_metrics['Skip_Reason']}")
            ranking_rows.append({
                "Product": prod,
                "Division": div,
                "Model": prophet_metrics["Model"],
                "Model_Type": _model_type(prophet_metrics["Model"]),
                "Test_MAE": prophet_metrics["Test_MAE"],
                "Test_RMSE": prophet_metrics["Test_RMSE"],
                "ROCV_MAE": prophet_metrics["ROCV_MAE"],
                "AIC": prophet_metrics["AIC"],
                "BIC": prophet_metrics["BIC"],
                "Regressor_Name": None,
                "Regressor_Lag": None,
                "Accepted_by_rules": False,
                "Skip_Reason": prophet_metrics.get("Skip_Reason", ""),
            })

        # --- Choose best model according to rules ---
        best_model, baseline = choose_best_model(metrics_list)
        if (enable_ml or enable_prophet) and baseline is not None:
            baseline_models = {"SARIMA_baseline", "ETS_baseline"}
            accepted_models = {
                m["Model"] for m in metrics_list
                if m["Model"] not in baseline_models and _candidate_passes_rules(m, baseline)
            }
            for row in ranking_rows:
                row["Accepted_by_rules"] = row["Model"] in accepted_models

        # Identify best non-baseline candidate by weighted MAE.
        # Prefer a SARIMAX candidate when available so regressor fields populate.
        non_baseline_candidates = [
            m for m in metrics_list
            if m["Model"] not in {"SARIMA_baseline", "ETS_baseline"} and np.isfinite(m.get("Test_WMAE", np.nan))
        ]
        sarimax_candidates = [
            m for m in non_baseline_candidates
            if isinstance(m.get("Model"), str) and m["Model"].startswith("SARIMAX")
        ]
        def wmae_rmse_key_local(m):
            mae = m.get("Test_WMAE", np.nan)
            rmse = m.get("Test_RMSE", np.nan)
            return (
                mae if np.isfinite(mae) else np.inf,
                rmse if np.isfinite(rmse) else np.inf,
            )

        best_nonbaseline = None
        if sarimax_candidates:
            best_nonbaseline = min(sarimax_candidates, key=wmae_rmse_key_local)
        elif non_baseline_candidates:
            best_nonbaseline = min(non_baseline_candidates, key=wmae_rmse_key_local)

        if best_model is None:
            print("  -> No valid model metrics; marking as FAILED.")
            results_rows.append({
                "Product": prod,
                "Division": div,
                "Chosen_Model": "FAILED",
                "Best_NonBaseline_Model": None,
                "Best_NonBaseline_Regressor_Name": None,
                "Best_NonBaseline_Regressor_Lag": None,
                "Baseline_MAE": np.nan,
                "Baseline_ROCV_MAE": np.nan,
                "Chosen_MAE": np.nan,
                "Chosen_RMSE": np.nan,
                "Chosen_ROCV_MAE": np.nan,
                "MAE_Improvement_Pct": np.nan,
                "Regressor_Name": None,
                "Regressor_Lag": None,
                "Regressor_Coef": np.nan,
                "Regressor_pvalue": np.nan,
                "Baseline_AIC": np.nan,
                "Baseline_BIC": np.nan,
                "Chosen_AIC": np.nan,
                "Chosen_BIC": np.nan,
                "Accepted_by_rules": False
            })
            continue

        # Baseline info
        baseline_mae = baseline["Test_MAE"] if baseline is not None else np.nan
        baseline_rocv = baseline["ROCV_MAE"] if baseline is not None else np.nan

        chosen_mae = best_model["Test_MAE"]
        chosen_rmse = best_model["Test_RMSE"]
        chosen_rocv = best_model["ROCV_MAE"]
        chosen_aic = best_model["AIC"]
        chosen_bic = best_model["BIC"]

        # Improvement %
        if not np.isnan(baseline_mae) and not np.isnan(chosen_mae):
            if abs(baseline_mae) <= BASELINE_MAE_ZERO_EPS:
                mae_improvement_pct = np.nan
            else:
                mae_improvement_pct = (baseline_mae - chosen_mae) / baseline_mae
        else:
            mae_improvement_pct = np.nan

        # Determine regressor name + lag
        regressor_name = None
        regressor_lag = None
        if best_model["Model"].startswith("SARIMAX"):
            if "New_Opportunities" in best_model["Model"]:
                regressor_name = "New_Opportunities"
                if "_l" in best_model["Model"]:
                    regressor_lag = int(best_model["Model"].split("_l")[-1])
            elif "Open_Opportunities" in best_model["Model"]:
                regressor_name = "Open_Opportunities"
                if "_l" in best_model["Model"]:
                    regressor_lag = int(best_model["Model"].split("_l")[-1])
            elif "Bookings" in best_model["Model"]:
                regressor_name = "Bookings"
                if "_l" in best_model["Model"]:
                    regressor_lag = int(best_model["Model"].split("_l")[-1])
            elif "Median_Months_Since_Last_Activity" in best_model["Model"]:
                regressor_name = "Median_Months_Since_Last_Activity"
                if "_l" in best_model["Model"]:
                    regressor_lag = int(best_model["Model"].split("_l")[-1])
            elif "Open_Not_Modified_90_Days" in best_model["Model"]:
                regressor_name = "Open_Not_Modified_90_Days"
                if "_l" in best_model["Model"]:
                    regressor_lag = int(best_model["Model"].split("_l")[-1])
            elif "Pct_Open_Not_Modified_90_Days" in best_model["Model"]:
                regressor_name = "Pct_Open_Not_Modified_90_Days"
                if "_l" in best_model["Model"]:
                    regressor_lag = int(best_model["Model"].split("_l")[-1])
            elif "Early_to_Late_Ratio" in best_model["Model"]:
                regressor_name = "Early_to_Late_Ratio"
                if "_l" in best_model["Model"]:
                    regressor_lag = int(best_model["Model"].split("_l")[-1])

        # Best non-baseline info
        best_nonbaseline_model = best_nonbaseline["Model"] if best_nonbaseline is not None else None
        best_nonbaseline_reg_name = None
        best_nonbaseline_reg_lag = None
        if best_nonbaseline_model is not None and best_nonbaseline_model.startswith("SARIMAX"):
            if "New_Opportunities" in best_nonbaseline_model:
                best_nonbaseline_reg_name = "New_Opportunities"
                if "_l" in best_nonbaseline_model:
                    best_nonbaseline_reg_lag = int(best_nonbaseline_model.split("_l")[-1])
            elif "Open_Opportunities" in best_nonbaseline_model:
                best_nonbaseline_reg_name = "Open_Opportunities"
                if "_l" in best_nonbaseline_model:
                    best_nonbaseline_reg_lag = int(best_nonbaseline_model.split("_l")[-1])
            elif "Bookings" in best_nonbaseline_model:
                best_nonbaseline_reg_name = "Bookings"
                if "_l" in best_nonbaseline_model:
                    best_nonbaseline_reg_lag = int(best_nonbaseline_model.split("_l")[-1])
            elif "Median_Months_Since_Last_Activity" in best_nonbaseline_model:
                best_nonbaseline_reg_name = "Median_Months_Since_Last_Activity"
                if "_l" in best_nonbaseline_model:
                    best_nonbaseline_reg_lag = int(best_nonbaseline_model.split("_l")[-1])
            elif "Open_Not_Modified_90_Days" in best_nonbaseline_model:
                best_nonbaseline_reg_name = "Open_Not_Modified_90_Days"
                if "_l" in best_nonbaseline_model:
                    best_nonbaseline_reg_lag = int(best_nonbaseline_model.split("_l")[-1])
            elif "Pct_Open_Not_Modified_90_Days" in best_nonbaseline_model:
                best_nonbaseline_reg_name = "Pct_Open_Not_Modified_90_Days"
                if "_l" in best_nonbaseline_model:
                    best_nonbaseline_reg_lag = int(best_nonbaseline_model.split("_l")[-1])
            elif "Early_to_Late_Ratio" in best_nonbaseline_model:
                best_nonbaseline_reg_name = "Early_to_Late_Ratio"
                if "_l" in best_nonbaseline_model:
                    best_nonbaseline_reg_lag = int(best_nonbaseline_model.split("_l")[-1])

        accepted_by_rules = (best_model["Model"] not in {"SARIMA_baseline", "ETS_baseline"})

        print(f"  -> Chosen model: {best_model['Model']}")
        print(f"     Baseline MAE: {baseline_mae:.3f} | Chosen MAE: {chosen_mae:.3f} | Improvement: {mae_improvement_pct * 100 if not np.isnan(mae_improvement_pct) else np.nan:.2f}%")

        model_type = _model_type(best_model["Model"]) if (enable_ml or enable_prophet) else None
        results_rows.append({
            "Product": prod,
            "Division": div,
            "Chosen_Model": best_model["Model"],
            **({"Model_Type": model_type} if (enable_ml or enable_prophet) else {}),
            "Best_NonBaseline_Model": best_nonbaseline_model,
            "Best_NonBaseline_Regressor_Name": best_nonbaseline_reg_name,
            "Best_NonBaseline_Regressor_Lag": best_nonbaseline_reg_lag,
            "Baseline_MAE": baseline_mae,
            "Baseline_ROCV_MAE": baseline_rocv,
            "Chosen_MAE": chosen_mae,
            "Chosen_RMSE": chosen_rmse,
            "Chosen_ROCV_MAE": chosen_rocv,
            "MAE_Improvement_Pct": mae_improvement_pct,
            "Regressor_Name": regressor_name,
            "Regressor_Lag": regressor_lag,
            "Regressor_Coef": best_model["Regressor_coef"],
            "Regressor_pvalue": best_model["Regressor_pvalue"],
            "Baseline_AIC": baseline["AIC"] if baseline is not None else np.nan,
            "Baseline_BIC": baseline["BIC"] if baseline is not None else np.nan,
            "Chosen_AIC": chosen_aic,
            "Chosen_BIC": chosen_bic,
            "Accepted_by_rules": accepted_by_rules
        })

        ranking_rows_all.extend(ranking_rows)

    # ===========================
    # 4. EXPORT RESULTS
    # ===========================
    print("\nWriting Excel summary...")
    df_results = pd.DataFrame(results_rows)

    # Nice ordering of columns
    col_order = [
        "Product", "Division",
        "Chosen_Model",
    ]
    if enable_ml or enable_prophet:
        col_order.append("Model_Type")
    col_order.extend([
        "Best_NonBaseline_Model", "Accepted_by_rules",
        "Baseline_MAE", "Baseline_ROCV_MAE",
        "Chosen_MAE", "Chosen_RMSE", "Chosen_ROCV_MAE",
        "MAE_Improvement_Pct",
        "Regressor_Name", "Regressor_Lag",
        "Best_NonBaseline_Regressor_Name", "Best_NonBaseline_Regressor_Lag",
        "Regressor_Coef", "Regressor_pvalue",
        "Baseline_AIC", "Baseline_BIC",
        "Chosen_AIC", "Chosen_BIC"
    ])
    # Ensure any missing columns exist (e.g., when rows were skipped or failed early)
    for col in col_order:
        if col not in df_results.columns:
            df_results[col] = np.nan
    df_results = df_results[col_order]

    # Build ranking table (one row per model candidate) if available
    df_rankings = pd.DataFrame(ranking_rows_all)
    if not df_rankings.empty:
        if enable_ml or enable_prophet:
            if "Model_Type" not in df_rankings.columns:
                df_rankings["Model_Type"] = df_rankings["Model"].apply(_model_type)
            if "Skip_Reason" not in df_rankings.columns:
                df_rankings["Skip_Reason"] = ""
        # rank within SKU by Test_MAE (ascending)
        df_rankings["Rank_by_Test_MAE"] = (
            df_rankings.groupby(["Product", "Division"])["Test_MAE"]
            .rank(method="first")
        )

    output_file = _output_path(OUTPUT_FILE)
    with pd.ExcelWriter(output_file) as writer:
        df_results.to_excel(writer, index=False, sheet_name="Model_Summary")
        if not df_rankings.empty:
            df_rankings.to_excel(writer, index=False, sheet_name="Model_Rankings")
        if skipped_no_order:
            pd.DataFrame(skipped_no_order).to_excel(
                writer, index=False, sheet_name="Skipped_No_Order"
            )

    if skipped_no_order:
        print("\nSkipped SKUs (no order found):")
        for row in skipped_no_order:
            print(f"  - Product: {row['Product']}, Division: {row['Division']}")

    print(f"\nDone. Summary written to: {output_file}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    elapsed = time.perf_counter() - t0
    minutes = elapsed / 60.0
    print(f"\nTotal runtime: {elapsed:,.2f} seconds ({minutes:,.2f} minutes)")
