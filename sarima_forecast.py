import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import ast
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False


# ===========================
# CONFIG
# ===========================

INPUT_FILE = "all_products_with_sf_and_bookings.xlsx"
REVISED_ACTUALS_FILE = "all_products_with_sf_and_bookings_revised.xlsx"
REVISED_ACTUALS_SHEET = "Revised Actuals"
SUMMARY_FILE = "sarima_multi_sku_summary.xlsx"          # chosen model per SKU
ORDER_FILE = "sarimax_order_search_summary.xlsx"        # chosen (p,d,q)(P,D,Q,s) per SKU
NOTES_FILE = "Notes.xlsx"                               # manual order overrides
OUTPUT_FILE_BASE = "stats_model_forecasts.xlsx"
# If a chosen regressor has lag 0, we fall back to baseline (no exog).
# If future exogenous values are missing, we do NOT fill; the SKU will fall back to baseline instead.
FILL_MISSING_FUTURE_EXOG_WITH_LAST = True
ENABLE_ML_CHALLENGER = True
ENABLE_PROPHET_CHALLENGER = True

COL_PRODUCT = "Product"
COL_DIVISION = "Division"
COL_DATE = "Month"
COL_ACTUALS = "Actuals"
COL_NEW_OPPORTUNITIES = "New_Opportunities"
COL_OPEN_OPPS = "Open_Opportunities"
COL_BOOKINGS = "Bookings"

FORECAST_HORIZON = 12
FORECAST_FLOOR = 0.0
ETS_MAX_SCALE_MULTIPLIER = 20.0
ETS_STABILITY_EPS = 1e-6
TEST_HORIZON_DEFAULT = 12
MIN_TEST_WINDOW = 6
ROCV_HORIZON = 12
ROCV_MAX_ORIGINS = 12
EPSILON_IMPROVEMENT_ABS = 10.0
IMPROVEMENT_PCT = 0.05
ROCV_TOLERANCE_PCT = 0.10

# Labels for outward-facing model descriptions
MODEL_LABELS = {
    "baseline_sarima": "Seasonal Baseline",
    "baseline_ets": "ETS Baseline",
    "ml_gbr": "ML GBR",
    "prophet": "PROPHET",
}


def _build_output_filename(base_name: str, first_forecast_dt: Optional[pd.Timestamp]) -> str:
    """Append YYYY-Mon suffix to the base filename using first forecast month."""
    if first_forecast_dt is None or pd.isna(first_forecast_dt):
        return base_name
    try:
        suffix = first_forecast_dt.strftime("%Y-%b")
    except Exception:
        return base_name
    if "." in base_name:
        stem, ext = base_name.rsplit(".", 1)
        return f"{stem}_{suffix}.{ext}"
    return f"{base_name}_{suffix}"


# ===========================
# HELPERS
# ===========================

def aggregate_monthly_duplicates(
    df: pd.DataFrame,
    product_col: str = COL_PRODUCT,
    division_col: str = COL_DIVISION,
    date_col: str = COL_DATE,
    sum_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Collapse duplicate Product/Division/Month rows by summing key numeric fields.
    Preserves any extra columns by taking the first value within each group.
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    # Only sum true additive series; keep others (e.g., opportunities, bookings) as first.
    sum_cols = sum_cols or [COL_ACTUALS]
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
        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=output_sheet, index=False)
        print(f"Wrote revised actuals to {output_excel_path} ({output_sheet}).")

    return df


def _friendly_regressor_name(reg_name: Optional[str]) -> str:
    """Return a natural-language regressor name."""
    if reg_name is None:
        return "Regressor"
    mapping = {
        "New_Opportunities": "Salesforce",
        "Open_Opportunities": "Salesforce",
        "Bookings": "Bookings",
    }
    return mapping.get(reg_name, reg_name.replace("_", " "))


def _natural_model_label(model_group: str,
                         regressor_name: Optional[str] = None,
                         regressor_lag: Optional[int] = None) -> str:
    """Map model group to a natural-language label."""
    if model_group in MODEL_LABELS:
        return MODEL_LABELS[model_group]

    if model_group == "with_regressor":
        base = _friendly_regressor_name(regressor_name)
        lag_part = f"Lag{regressor_lag}" if regressor_lag is not None else ""
        # e.g., "Salesforce Lag1 Regressor"
        pieces = [base, lag_part, "Regressor"]
        return " ".join([p for p in pieces if p])

    return model_group


def _model_group_from_choice(choice: dict) -> Optional[str]:
    """Map the chosen model from the summary file to a forecast model_group."""
    if not choice:
        return None
    model_name = choice.get("model")
    if not isinstance(model_name, str):
        return None
    model_name = model_name.strip()
    if model_name == "SARIMA_baseline":
        return "baseline_sarima"
    if model_name == "ETS_baseline":
        return "baseline_ets"
    if model_name.startswith("SARIMAX"):
        return "with_regressor"
    if model_name == "ML_GBR":
        return "ml_gbr"
    if model_name == "PROPHET":
        return "prophet"
    return None


def _parse_order_cell(value, expected_len: int) -> Optional[Tuple[int, ...]]:
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


def _load_manual_order_overrides(path: str) -> Dict[Tuple[str, str], Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Load manual SARIMA order overrides from Notes.xlsx if present."""
    if path is None or not os.path.exists(path):
        return {}
    try:
        df_notes = pd.read_excel(path)
    except Exception as exc:
        print(f"Could not read manual overrides from {path}: {exc}")
        return {}

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

    overrides: Dict[Tuple[str, str], Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for _, row in df_notes.iterrows():
        order = _parse_order_cell(row["Chosen_Order"], 3)
        seas_order = _parse_order_cell(row["Chosen_Seasonal_Order"], 4)
        if order is None or seas_order is None:
            continue
        prod = row[product_col]
        div = row[division_col]
        overrides[(prod, div)] = (order, seas_order)
    return overrides


def load_orders_map(path: str, overrides_path: Optional[str] = None) -> Dict[Tuple[str, str], Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Load per-SKU SARIMA orders from Excel."""
    df_orders = pd.read_excel(path)
    required_cols = {COL_PRODUCT, COL_DIVISION, "Chosen_Order", "Chosen_Seasonal_Order"}
    if not required_cols.issubset(df_orders.columns):
        missing = required_cols - set(df_orders.columns)
        raise ValueError(f"Missing required columns in {path}: {missing}")

    orders_map: Dict[Tuple[str, str], Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
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


def load_model_choices(path: str) -> Dict[Tuple[str, str], dict]:
    """
    Load chosen model summary per SKU.
    Returns {(Product, Division): {"model": str, "reg_name": str|None, "reg_lag": int|None, "best_nonbaseline": {...}}}
    """
    df = pd.read_excel(path)
    required_cols = {COL_PRODUCT, COL_DIVISION, "Chosen_Model"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in {path}: {missing}")

    def infer_regressor(model_name: str) -> Tuple[Optional[str], Optional[int]]:
        """Map model string to regressor name/lag when explicit cols are absent."""
        if not isinstance(model_name, str):
            return None, None
        model_name = model_name.strip()
        if model_name in {"SARIMA_baseline", "ETS_baseline"}:
            return None, None
        if "New_Opportunities" in model_name:
            return "New_Opportunities", 1
        if "Open_Opportunities" in model_name:
            return "Open_Opportunities", 0
        if "Bookings" in model_name and "_l1" in model_name:
            return "Bookings", 1
        if "Bookings" in model_name:
            return "Bookings", 0
        return None, None

    choices: Dict[Tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        prod = row[COL_PRODUCT]
        div = row[COL_DIVISION]
        model_name = row["Chosen_Model"]
        best_non_model = row.get("Best_NonBaseline_Model")
        best_non_reg_name = row.get("Best_NonBaseline_Regressor_Name")
        best_non_reg_lag = row.get("Best_NonBaseline_Regressor_Lag")
        reg_name = row.get("Regressor_Name")
        reg_lag = row.get("Regressor_Lag")

        # Normalize regressor info
        if pd.isna(reg_name):
            reg_name = None
        if pd.isna(reg_lag):
            reg_lag = None
        if pd.isna(best_non_reg_name):
            best_non_reg_name = None
        if pd.isna(best_non_reg_lag):
            best_non_reg_lag = None
        if reg_lag is not None:
            try:
                reg_lag = int(reg_lag)
            except Exception:
                reg_lag = None
        if best_non_reg_lag is not None:
            try:
                best_non_reg_lag = int(best_non_reg_lag)
            except Exception:
                best_non_reg_lag = None

        if reg_name is None or reg_lag is None:
            reg_name, reg_lag = infer_regressor(str(model_name))

        # Choose regressor details for the SARIMAX variant without overriding the chosen model.
        sarimax_reg_name = reg_name
        sarimax_reg_lag = reg_lag
        if not (isinstance(model_name, str) and model_name.startswith("SARIMAX")):
            if isinstance(best_non_model, str) and best_non_model.startswith("SARIMAX"):
                sarimax_reg_name = best_non_reg_name
                sarimax_reg_lag = best_non_reg_lag

        # Drop any lag-0 regressor: leave model choice untouched
        if sarimax_reg_lag == 0:
            sarimax_reg_name = None
            sarimax_reg_lag = None

        choices[(prod, div)] = {
            "model": str(model_name),
            "reg_name": sarimax_reg_name,
            "reg_lag": sarimax_reg_lag,
            "best_nonbaseline_model": best_non_model,
            "best_nonbaseline_reg_name": best_non_reg_name,
            "best_nonbaseline_reg_lag": best_non_reg_lag,
        }
    return choices


def engineer_regressors(df_sku: pd.DataFrame) -> pd.DataFrame:
    """Create lagged regressor features used by SARIMAX candidates."""
    df = df_sku.copy()
    for col in [COL_NEW_OPPORTUNITIES, COL_OPEN_OPPS, COL_BOOKINGS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if COL_NEW_OPPORTUNITIES in df.columns:
        df["New_Opportunities_l1"] = df[COL_NEW_OPPORTUNITIES].shift(1)
        df["New_Opportunities_l2"] = df[COL_NEW_OPPORTUNITIES].shift(2)
        df["New_Opportunities_l3"] = df[COL_NEW_OPPORTUNITIES].shift(3)
    if COL_OPEN_OPPS in df.columns:
        df["Open_Opportunities_l0"] = df[COL_OPEN_OPPS]
        df["Open_Opportunities_l1"] = df[COL_OPEN_OPPS].shift(1)
        df["Open_Opportunities_l2"] = df[COL_OPEN_OPPS].shift(2)
        df["Open_Opportunities_l3"] = df[COL_OPEN_OPPS].shift(3)
    if COL_BOOKINGS in df.columns:
        df["Bookings_l0"] = df[COL_BOOKINGS]
        df["Bookings_l1"] = df[COL_BOOKINGS].shift(1)
        df["Bookings_l2"] = df[COL_BOOKINGS].shift(2)
    return df


def get_exog_series(df: pd.DataFrame, reg_name: Optional[str], reg_lag: Optional[int]) -> Optional[pd.Series]:
    """Return the appropriate regressor series (with lag applied)."""
    if reg_name is None:
        return None

    col_map = {
        ("New_Opportunities", 1): "New_Opportunities_l1",
        ("New_Opportunities", 2): "New_Opportunities_l2",
        ("New_Opportunities", 3): "New_Opportunities_l3",
        ("Open_Opportunities", 0): "Open_Opportunities_l0",
        ("Open_Opportunities", 1): "Open_Opportunities_l1",
        ("Open_Opportunities", 2): "Open_Opportunities_l2",
        ("Open_Opportunities", 3): "Open_Opportunities_l3",
        ("Bookings", 0): "Bookings_l0",
        ("Bookings", 1): "Bookings_l1",
        ("Bookings", 2): "Bookings_l2",
    }
    col = col_map.get((reg_name, reg_lag))

    if col and col in df.columns:
        return df[col]

    # Fallback: shift the base column dynamically
    if reg_name not in df.columns:
        return None
    lag = reg_lag or 0
    return df[reg_name].shift(lag)


def select_default_regressor(df: pd.DataFrame) -> Tuple[Optional[str], Optional[int], Optional[pd.Series]]:
    """
    Pick a reasonable SARIMAX regressor when none was selected.
    Chooses the candidate with the most usable history.
    """
    candidates = [
        ("New_Opportunities", 1, "New_Opportunities_l1"),
        ("New_Opportunities", 2, "New_Opportunities_l2"),
        ("New_Opportunities", 3, "New_Opportunities_l3"),
        ("Open_Opportunities", 1, "Open_Opportunities_l1"),
        ("Open_Opportunities", 2, "Open_Opportunities_l2"),
        ("Open_Opportunities", 3, "Open_Opportunities_l3"),
        ("Bookings", 1, "Bookings_l1"),
        ("Bookings", 2, "Bookings_l2"),
    ]

    best = (None, None, None)
    best_non_null = 0
    for reg_name, reg_lag, col in candidates:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = int(series.notna().sum())
        if non_null == 0:
            continue
        if series.dropna().nunique() <= 1:
            continue
        if non_null > best_non_null:
            best = (reg_name, reg_lag, df[col])
            best_non_null = non_null

    return best


def _build_exog_frames(exog_series: Optional[pd.Series],
                       forecast_index: pd.DatetimeIndex) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convert a regressor series into train/future frames aligned to the forecast index.
    Returns (X_train, X_future) or (None, None) if future values cannot be built.
    """
    if exog_series is None:
        return None, None

    exog_series = pd.Series(exog_series).copy()
    exog_series.index = pd.to_datetime(exog_series.index)
    exog_series.index = exog_series.index.to_period("M").to_timestamp()

    X_train = exog_series.to_frame()
    X_train.columns = [exog_series.name or "regressor"]

    X_future = exog_series.reindex(forecast_index)
    missing_before = X_future.isna().sum().sum()

    # forward fill where possible to extend latest signal
    X_future = X_future.ffill()
    missing_after_ffill = X_future.isna().sum().sum()

    if missing_after_ffill > 0 and FILL_MISSING_FUTURE_EXOG_WITH_LAST:
        last_vals = X_train.ffill().iloc[-1:]
        X_future = X_future.fillna(last_vals.squeeze())

    missing_after = X_future.isna().sum().sum()
    if missing_before > 0 and missing_after > 0:
        # still missing future exog; skip this variant
        return None, None

    return X_train, X_future


def _series_is_nonnegative(y: pd.Series) -> bool:
    """Treat zeros as positive for multiplicative eligibility checks."""
    y_clean = pd.Series(y).dropna()
    if y_clean.empty:
        return False
    return (y_clean >= 0).all()


def _ets_reference_scale(y: pd.Series) -> float:
    """Compute a robust scale for ETS stability checks."""
    y_clean = pd.Series(y).dropna().astype(float)
    if y_clean.empty:
        return 1.0

    recent = y_clean.tail(12)
    ref = np.nanmedian(np.abs(recent))
    if not np.isfinite(ref) or ref <= 0:
        ref = np.nanmedian(np.abs(y_clean))
    if not np.isfinite(ref) or ref <= 0:
        ref = np.nanmax(np.abs(y_clean))
    if not np.isfinite(ref) or ref <= 0:
        ref = 1.0
    return float(ref)


def _ets_is_stable(res, y_train: pd.Series, steps: int) -> Tuple[bool, str]:
    """Validate ETS forecasts to avoid unstable or implausible outputs."""
    try:
        forecast = res.get_forecast(steps=steps).predicted_mean
    except Exception:
        try:
            forecast = res.forecast(steps=steps)
        except Exception:
            return False, "forecast_failed"

    forecast = pd.Series(forecast).astype(float)
    if not np.isfinite(forecast).all():
        return False, "forecast_nonfinite"

    ref = _ets_reference_scale(y_train)
    max_abs = max(ref * ETS_MAX_SCALE_MULTIPLIER, 1.0)
    lower = -max_abs
    upper = max_abs

    if _series_is_nonnegative(y_train):
        if (forecast < -ETS_STABILITY_EPS).any():
            return False, "forecast_negative"
        lower = -ETS_STABILITY_EPS

    if (forecast < lower).any() or (forecast > upper).any():
        return False, "forecast_out_of_range"

    fitted = getattr(res, "fittedvalues", None)
    if fitted is not None:
        fitted = pd.Series(fitted).astype(float)
        if not np.isfinite(fitted).all():
            return False, "fitted_nonfinite"
        if _series_is_nonnegative(y_train) and (fitted < -ETS_STABILITY_EPS).any():
            return False, "fitted_negative"
        if (fitted < -max_abs).any() or (fitted > max_abs).any():
            return False, "fitted_out_of_range"

    return True, ""


def _ets_candidate_specs(y: pd.Series, allow_seasonal: bool) -> List[dict]:
    """Build a small, industry-standard ETS candidate grid."""
    allow_mul = _series_is_nonnegative(y)
    error_opts = ["add"] + (["mul"] if allow_mul else [])
    trend_opts = [None, "add"]
    if allow_seasonal:
        seasonal_opts = [None, "add"] + (["mul"] if allow_mul else [])
    else:
        seasonal_opts = [None]

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
                        "seasonal_periods": 12 if seasonal is not None else None,
                    })
    return candidates


def _build_ml_features(y: pd.Series) -> pd.DataFrame:
    """Build ML features from the target series only (no future exog)."""
    y = pd.Series(y).astype(float)
    features = pd.DataFrame(index=y.index)
    features["y_lag1"] = y.shift(1)
    features["y_lag2"] = y.shift(2)
    features["y_lag3"] = y.shift(3)
    if len(y.dropna()) >= 24:
        features["y_lag12"] = y.shift(12)

    features["month_of_year"] = y.index.month
    features["trend_index"] = np.arange(len(y))
    return features


def _forecast_ml_recursive(
    model: GradientBoostingRegressor,
    y_history: List[float],
    last_date: pd.Timestamp,
    horizon: int,
    feature_columns: List[str],
) -> pd.Series:
    """Recursive ML forecast using lagged target + calendar + trend features."""
    preds = []
    history = list(y_history)
    start_trend = len(history)
    include_lag12 = "y_lag12" in feature_columns

    for step in range(1, horizon + 1):
        future_date = last_date + pd.DateOffset(months=step)
        if len(history) < 3 or (include_lag12 and len(history) < 12):
            raise ValueError("Insufficient history for ML recursive forecast.")

        row = {
            "y_lag1": history[-1],
            "y_lag2": history[-2],
            "y_lag3": history[-3],
            "month_of_year": future_date.month,
            "trend_index": start_trend + (step - 1),
        }
        if include_lag12:
            row["y_lag12"] = history[-12]

        X_next = pd.DataFrame([row])[feature_columns]
        y_next = float(model.predict(X_next)[0])
        preds.append(y_next)
        history.append(y_next)

    index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return pd.Series(preds, index=index)


def _prepare_prophet_df(y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        "ds": pd.to_datetime(y.index).tz_localize(None),
        "y": pd.to_numeric(y.values, errors="coerce"),
    })
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds")
    df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
    return df


def _coerce_month_start_series(y: pd.Series) -> pd.Series:
    """Normalize index to month starts without dropping valid month-end data."""
    y_clean = pd.Series(y).astype(float).dropna()
    if y_clean.empty:
        return y_clean
    idx = pd.to_datetime(y_clean.index).to_period("M").to_timestamp()
    idx = pd.DatetimeIndex(idx)
    y_clean.index = idx
    y_clean = y_clean.groupby(level=0).sum().sort_index()
    return y_clean


def _info_criterion(res) -> float:
    """Prefer AICc when available; fall back to AIC."""
    aicc = getattr(res, "aicc", None)
    if aicc is not None and np.isfinite(aicc):
        return float(aicc)
    aic = getattr(res, "aic", None)
    if aic is not None and np.isfinite(aic):
        return float(aic)
    return np.inf


def _fit_ets_candidate(y_train: pd.Series, spec: dict, use_state_space: bool):
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


def _select_best_ets_model(y_train: pd.Series, allow_seasonal: bool, context: str = ""):
    """Choose the best ETS candidate by AICc (or AIC) within a SKU."""
    best_res = None
    best_spec = None
    best_score = np.inf

    for spec in _ets_candidate_specs(y_train, allow_seasonal=allow_seasonal):
        try:
            res = _fit_ets_candidate(y_train, spec, use_state_space=True)
            score = _info_criterion(res)
        except Exception as exc:
            print(f"[WARN] ETSModel failed ({spec}){context}: {exc}")
            continue

        is_stable, reason = _ets_is_stable(res, y_train, steps=FORECAST_HORIZON)
        if not is_stable:
            print(f"[WARN] ETSModel unstable ({spec}){context}: {reason}")
            continue

        if score < best_score:
            best_res = res
            best_spec = spec
            best_score = score

    if best_res is not None:
        return best_res, best_spec, "ETSModel"

    # Fallback: only if ETSModel cannot fit anything
    for spec in _ets_candidate_specs(y_train, allow_seasonal=allow_seasonal):
        if spec["error"] != "add":
            continue
        try:
            res = _fit_ets_candidate(y_train, spec, use_state_space=False)
            score = _info_criterion(res)
        except Exception as exc:
            print(f"[WARN] ExponentialSmoothing failed ({spec}){context}: {exc}")
            continue

        is_stable, reason = _ets_is_stable(res, y_train, steps=FORECAST_HORIZON)
        if not is_stable:
            print(f"[WARN] ExponentialSmoothing unstable ({spec}){context}: {reason}")
            continue

        if score < best_score:
            best_res = res
            best_spec = spec
            best_score = score

    if best_res is None:
        return None, None, None
    return best_res, best_spec, "ExponentialSmoothing"


def _forecast_ets(res, forecast_index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.DataFrame]:
    """Generate ETS forecast mean and confidence intervals when available."""
    steps = len(forecast_index)
    mean = None
    ci = None

    try:
        forecast_res = res.get_forecast(steps=steps)
        mean = forecast_res.predicted_mean
        try:
            ci = forecast_res.conf_int(alpha=0.05)
        except Exception:
            ci = None
    except Exception:
        if hasattr(res, "forecast"):
            mean = res.forecast(steps=steps)

    if mean is None:
        raise ValueError("ETS forecast failed")

    mean = pd.Series(mean, index=forecast_index)
    if ci is None:
        ci = pd.DataFrame(
            {"lower_ci": [np.nan] * steps, "upper_ci": [np.nan] * steps},
            index=forecast_index,
        )
    else:
        ci = pd.DataFrame(ci)
        ci.index = forecast_index
    return mean, ci


def _evaluate_ml_metrics(
    y: pd.Series,
    test_window: int,
    rocv_horizon: int = ROCV_HORIZON,
    rocv_max_origins: int = ROCV_MAX_ORIGINS,
) -> dict:
    """Fit ML challenger, compute test MAE/RMSE and ROCV MAE."""
    y_clean = y.dropna().astype(float)
    if len(y_clean) <= test_window:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    features = _build_ml_features(y_clean)
    df_ml = pd.concat([features, y_clean.rename("y")], axis=1).dropna()
    if len(df_ml) <= test_window + 5:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    train_end = len(df_ml) - test_window
    train = df_ml.iloc[:train_end]
    test = df_ml.iloc[train_end:]

    try:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(train.drop(columns=["y"]), train["y"])
        preds = model.predict(test.drop(columns=["y"]))
    except Exception:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    test_mae = _safe_mae(test["y"].values, preds)
    test_rmse = _safe_rmse(test["y"].values, preds)

    rocv_maes = []
    min_train = max(12, rocv_horizon)
    origins = _rocv_origins(len(df_ml), rocv_horizon, rocv_max_origins, min_train)
    for origin in origins:
        train_rocv = df_ml.iloc[:origin]
        try:
            model_rocv = GradientBoostingRegressor(random_state=42)
            model_rocv.fit(train_rocv.drop(columns=["y"]), train_rocv["y"])
            y_history = list(train_rocv["y"].values)
            last_dt = train_rocv.index.max()
            feature_columns = list(train_rocv.drop(columns=["y"]).columns)
            fc = _forecast_ml_recursive(
                model=model_rocv,
                y_history=y_history,
                last_date=last_dt,
                horizon=rocv_horizon,
                feature_columns=feature_columns,
            )
            y_test_rocv = df_ml.iloc[origin:origin + rocv_horizon]["y"]
            mae = _safe_mae(y_test_rocv.values, fc.values)
            if np.isfinite(mae):
                rocv_maes.append(mae)
        except Exception:
            continue

    rocv_mae = float(np.mean(rocv_maes)) if rocv_maes else np.nan
    return {"Test_MAE": test_mae, "Test_RMSE": test_rmse, "ROCV_MAE": rocv_mae}


def _evaluate_prophet_metrics(
    y: pd.Series,
    test_window: int,
    rocv_horizon: int = ROCV_HORIZON,
    rocv_max_origins: int = ROCV_MAX_ORIGINS,
) -> dict:
    """Fit Prophet challenger, compute test MAE/RMSE and ROCV MAE."""
    if not PROPHET_AVAILABLE:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    y_clean = _coerce_month_start_series(y)
    if len(y_clean) < 24:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}
    if len(y_clean) <= test_window:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    y_train = y_clean.iloc[:-test_window]
    y_test = y_clean.iloc[-test_window:]

    try:
        train_df = _prepare_prophet_df(y_train)
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
    except Exception:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    test_mae = _safe_mae(y_test.values, y_pred.values)
    test_rmse = _safe_rmse(y_test.values, y_pred.values)

    rocv_maes = []
    min_train = max(24, rocv_horizon)
    origins = _rocv_origins(len(y_clean), rocv_horizon, rocv_max_origins, min_train)
    for origin in origins:
        y_train_rocv = y_clean.iloc[:origin]
        y_test_rocv = y_clean.iloc[origin:origin + rocv_horizon]
        try:
            train_df = _prepare_prophet_df(y_train_rocv)
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
            mae = _safe_mae(y_test_rocv.values, preds.values)
            if np.isfinite(mae):
                rocv_maes.append(mae)
        except Exception:
            continue

    rocv_mae = float(np.mean(rocv_maes)) if rocv_maes else np.nan
    return {"Test_MAE": test_mae, "Test_RMSE": test_rmse, "ROCV_MAE": rocv_mae}


def _fit_variant(y_train: pd.Series,
                 order: Tuple[int, int, int],
                 seasonal_order: Tuple[int, int, int, int],
                 forecast_index: pd.DatetimeIndex,
                 exog_train: Optional[pd.DataFrame] = None,
                 exog_future: Optional[pd.DataFrame] = None):
    """Fit one SARIMA/X variant and return mean + CI DataFrames aligned to forecast_index."""
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=len(forecast_index), exog=exog_future)
    mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)
    mean.index = forecast_index
    ci.index = forecast_index
    return mean, ci


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE while ignoring NaNs/Infs."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE while ignoring NaNs/Infs."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _compute_test_window(history_months: int) -> int:
    """Compute test window length based on history length."""
    if history_months < 24:
        return min(TEST_HORIZON_DEFAULT, max(MIN_TEST_WINDOW, int(np.floor(0.25 * history_months))))
    return TEST_HORIZON_DEFAULT


def _rocv_origins(n_obs: int, horizon: int, max_origins: int, min_train: int) -> List[int]:
    """Return rolling-origin indices (train length) for ROCV."""
    last_origin = n_obs - horizon
    if last_origin <= min_train:
        return []
    origins = list(range(min_train, last_origin + 1))
    if len(origins) > max_origins:
        origins = origins[-max_origins:]
    return origins


def _sarimax_forecast_series(res, steps: int, exog_future: Optional[pd.DataFrame] = None) -> pd.Series:
    """Forecast using a fitted SARIMAX model."""
    forecast = res.get_forecast(steps=steps, exog=exog_future)
    return forecast.predicted_mean


def _evaluate_sarima_metrics(
    y: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    test_window: int,
    exog_series: Optional[pd.Series] = None,
    rocv_horizon: int = ROCV_HORIZON,
    rocv_max_origins: int = ROCV_MAX_ORIGINS,
) -> dict:
    """Fit SARIMA/X, compute test MAE/RMSE and ROCV MAE."""
    y_clean = y.dropna().astype(float)
    if len(y_clean) <= test_window:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    y_train = y_clean.iloc[:-test_window]
    y_test = y_clean.iloc[-test_window:]

    exog_train = None
    exog_test = None
    if exog_series is not None:
        exog_series = exog_series.reindex(y_clean.index)
        exog_train = exog_series.iloc[:-test_window].to_frame()
        exog_test = exog_series.iloc[-test_window:].to_frame()
        if exog_train.isna().any().any() or exog_test.isna().any().any():
            return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    try:
        res = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        y_pred = _sarimax_forecast_series(res, steps=test_window, exog_future=exog_test)
    except Exception:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    test_mae = _safe_mae(y_test.values, y_pred.values)
    test_rmse = _safe_rmse(y_test.values, y_pred.values)

    rocv_maes = []
    min_train = max(12, rocv_horizon)
    origins = _rocv_origins(len(y_clean), rocv_horizon, rocv_max_origins, min_train)
    for origin in origins:
        y_train_rocv = y_clean.iloc[:origin]
        y_test_rocv = y_clean.iloc[origin:origin + rocv_horizon]
        exog_train_rocv = None
        exog_test_rocv = None
        if exog_series is not None:
            exog_train_rocv = exog_series.iloc[:origin].to_frame()
            exog_test_rocv = exog_series.iloc[origin:origin + rocv_horizon].to_frame()
            if exog_train_rocv.isna().any().any() or exog_test_rocv.isna().any().any():
                continue
        try:
            res_rocv = SARIMAX(
                y_train_rocv,
                exog=exog_train_rocv,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            y_pred_rocv = _sarimax_forecast_series(res_rocv, steps=rocv_horizon, exog_future=exog_test_rocv)
            mae = _safe_mae(y_test_rocv.values, y_pred_rocv.values)
            if np.isfinite(mae):
                rocv_maes.append(mae)
        except Exception:
            continue

    rocv_mae = float(np.mean(rocv_maes)) if rocv_maes else np.nan
    return {"Test_MAE": test_mae, "Test_RMSE": test_rmse, "ROCV_MAE": rocv_mae}


def _evaluate_ets_metrics(
    y: pd.Series,
    test_window: int,
    allow_seasonal: bool,
    rocv_horizon: int = ROCV_HORIZON,
    rocv_max_origins: int = ROCV_MAX_ORIGINS,
) -> dict:
    """Fit ETS, compute test MAE/RMSE and ROCV MAE."""
    y_clean = y.dropna().astype(float)
    if len(y_clean) <= test_window:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    y_train = y_clean.iloc[:-test_window]
    y_test = y_clean.iloc[-test_window:]

    ets_res, ets_spec, ets_impl = _select_best_ets_model(y_train, allow_seasonal=allow_seasonal, context="")
    if ets_res is None:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    try:
        y_pred = _forecast_ets(ets_res, y_test.index)[0]
    except Exception:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    test_mae = _safe_mae(y_test.values, y_pred.values)
    test_rmse = _safe_rmse(y_test.values, y_pred.values)

    rocv_maes = []
    min_train = max(12, rocv_horizon)
    origins = _rocv_origins(len(y_clean), rocv_horizon, rocv_max_origins, min_train)
    for origin in origins:
        y_train_rocv = y_clean.iloc[:origin]
        y_test_rocv = y_clean.iloc[origin:origin + rocv_horizon]
        try:
            res = _fit_ets_candidate(y_train_rocv, ets_spec, use_state_space=(ets_impl == "ETSModel"))
            y_pred_rocv = _forecast_ets(res, y_test_rocv.index)[0]
            mae = _safe_mae(y_test_rocv.values, y_pred_rocv.values)
            if np.isfinite(mae):
                rocv_maes.append(mae)
        except Exception:
            continue

    rocv_mae = float(np.mean(rocv_maes)) if rocv_maes else np.nan
    return {"Test_MAE": test_mae, "Test_RMSE": test_rmse, "ROCV_MAE": rocv_mae}


def _accept_candidate(
    candidate: dict,
    baseline: dict,
    epsilon_abs: float = EPSILON_IMPROVEMENT_ABS,
    improvement_pct: float = IMPROVEMENT_PCT,
    rocv_tol_pct: float = ROCV_TOLERANCE_PCT,
) -> bool:
    """Apply acceptance rules for non-baseline model candidates."""
    cand_mae = candidate.get("Test_MAE", np.nan)
    cand_rocv = candidate.get("ROCV_MAE", np.nan)
    base_mae = baseline.get("Test_MAE", np.nan)
    base_rocv = baseline.get("ROCV_MAE", np.nan)

    if not np.isfinite(cand_mae) or not np.isfinite(base_mae):
        return False

    required_improvement = max(epsilon_abs, base_mae * improvement_pct)
    if (base_mae - cand_mae) < required_improvement:
        return False

    if np.isfinite(base_rocv) and np.isfinite(cand_rocv):
        if cand_rocv > base_rocv * (1.0 + rocv_tol_pct):
            return False
    elif np.isfinite(base_rocv) and not np.isfinite(cand_rocv):
        return False

    if np.isfinite(cand_rocv):
        if abs(cand_rocv - cand_mae) > (rocv_tol_pct * max(cand_mae, 1e-9)):
            return False

    return True


def _recommend_model_group(
    y: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    exog_series: Optional[pd.Series],
    allow_sarima: bool,
    allow_sarimax: bool,
    allow_ets_for_reco: bool,
    allow_ets_seasonal: bool,
    allow_ml: bool,
    allow_prophet: bool,
) -> Optional[str]:
    """Recommend a model group per SKU based on Test and ROCV metrics."""
    y_clean = y.dropna().astype(float)
    history_months = len(y_clean)
    if history_months == 0:
        return None

    test_window = _compute_test_window(history_months)
    metrics = []

    baseline_group = "baseline_sarima" if allow_sarima else "baseline_ets"

    if baseline_group == "baseline_sarima" and allow_sarima:
        baseline_metrics = _evaluate_sarima_metrics(
            y_clean, order, seasonal_order, test_window, exog_series=None
        )
    else:
        baseline_metrics = _evaluate_ets_metrics(
            y_clean, test_window, allow_seasonal=allow_ets_seasonal
        )
    baseline_metrics["model_group"] = baseline_group
    metrics.append(baseline_metrics)

    if allow_sarimax and exog_series is not None:
        sarimax_metrics = _evaluate_sarima_metrics(
            y_clean, order, seasonal_order, test_window, exog_series=exog_series
        )
        sarimax_metrics["model_group"] = "with_regressor"
        metrics.append(sarimax_metrics)

    if allow_ets_for_reco and baseline_group != "baseline_ets":
        ets_metrics = _evaluate_ets_metrics(
            y_clean, test_window, allow_seasonal=allow_ets_seasonal
        )
        ets_metrics["model_group"] = "baseline_ets"
        metrics.append(ets_metrics)

    if allow_ml:
        ml_metrics = _evaluate_ml_metrics(
            y_clean, test_window
        )
        ml_metrics["model_group"] = "ml_gbr"
        metrics.append(ml_metrics)

    if allow_prophet:
        prophet_metrics = _evaluate_prophet_metrics(
            y_clean, test_window
        )
        prophet_metrics["model_group"] = "prophet"
        metrics.append(prophet_metrics)

    baseline = next((m for m in metrics if m["model_group"] == baseline_group), None)
    if baseline is None:
        return None

    candidates = [m for m in metrics if m["model_group"] != baseline_group]
    accepted = [m for m in candidates if _accept_candidate(m, baseline)]
    if not accepted:
        return baseline_group

    best = min(
        accepted,
        key=lambda m: m.get("Test_MAE", np.inf)
    )
    return best.get("model_group", baseline_group)


def generate_forecast_variants(
    y: pd.Series,
    product_id: str,
    bu_id: str,
    sarima_order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    forecast_index: pd.DatetimeIndex,
    exog_series: Optional[pd.Series] = None,
    regressor_name: Optional[str] = None,
    regressor_lag: Optional[int] = None,
    allow_sarima: bool = True,
    allow_sarimax: bool = True,
    allow_ets: bool = True,
    allow_ets_seasonal: bool = True,
    allow_ml: bool = True,
    allow_prophet: bool = True,
    status_reasons: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Build a long-form table of forecasts for:
    - SARIMA baseline (seasonal)
    - ETS baseline (selected by AICc/AIC over a small candidate grid)
    - SARIMAX with regressor (if provided and future exog is available)
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    y = pd.Series(y).astype(float)
    y.index = pd.to_datetime(y.index)
    y.index = y.index.to_period("M").to_timestamp()
    y = y.sort_index()

    forecast_index = pd.to_datetime(forecast_index)
    forecast_index = forecast_index.to_period("M").to_timestamp()
    forecast_index = pd.DatetimeIndex(forecast_index).sort_values()
    dfs = []
    status_reasons = status_reasons or {}

    def _build_na_rows(
        model_group: str,
        model_type: str,
        model_label: str,
        reason: str,
        status: str = "N/A",
    ) -> pd.DataFrame:
        include_orders = model_group in {"baseline_sarima", "with_regressor"}
        df_na = pd.DataFrame({
            "forecast_month": forecast_index,
            "forecast_value": [np.nan] * len(forecast_index),
            "lower_ci": [np.nan] * len(forecast_index),
            "upper_ci": [np.nan] * len(forecast_index),
            "model_group": model_group,
            "model_type": model_type,
            "model_label": model_label,
            "p": sarima_order[0] if include_orders else np.nan,
            "d": sarima_order[1] if include_orders else np.nan,
            "q": sarima_order[2] if include_orders else np.nan,
            "P": seasonal_order[0] if include_orders else np.nan,
            "D": seasonal_order[1] if include_orders else np.nan,
            "Q": seasonal_order[2] if include_orders else np.nan,
            "s": seasonal_order[3] if include_orders else np.nan,
            "regressor_names": None,
            "regressor_details": None,
            "model_status": status,
            "model_status_reason": reason,
        })
        return df_na

    def _attach_status(df: pd.DataFrame, status: str, reason: str = "") -> pd.DataFrame:
        df["model_status"] = status
        df["model_status_reason"] = reason
        return df

    # Seasonal baseline
    if not allow_sarima:
        reason = status_reasons.get("baseline_sarima", "N/A: SARIMA disabled by history threshold.")
        dfs.append(_build_na_rows("baseline_sarima", "SARIMA", _natural_model_label("baseline_sarima"), reason))
    else:
        try:
            mean, ci = _fit_variant(
                y_train=y.dropna(),
                order=sarima_order,
                seasonal_order=seasonal_order,
                forecast_index=forecast_index,
            )
            sarima_df = pd.DataFrame({
                "forecast_month": forecast_index,
                "forecast_value": mean.values,
                "lower_ci": ci.iloc[:, 0].values,
                "upper_ci": ci.iloc[:, 1].values,
                "model_group": "baseline_sarima",
                "model_type": "SARIMA",
                "model_label": _natural_model_label("baseline_sarima"),
                "p": sarima_order[0],
                "d": sarima_order[1],
                "q": sarima_order[2],
                "P": seasonal_order[0],
                "D": seasonal_order[1],
                "Q": seasonal_order[2],
                "s": seasonal_order[3],
                "regressor_names": None,
                "regressor_details": None,
            })
            dfs.append(_attach_status(_apply_forecast_floor(sarima_df), "ok"))
        except Exception as exc:
            print(f"  Failed SARIMA baseline: {exc}")
            dfs.append(_build_na_rows(
                "baseline_sarima",
                "SARIMA",
                _natural_model_label("baseline_sarima"),
                f"Failed: {exc}",
                status="Failed",
            ))

    # ETS baseline (AICc/AIC selection)
    if not allow_ets:
        reason = status_reasons.get("baseline_ets", "N/A: ETS disabled by history threshold.")
        dfs.append(_build_na_rows("baseline_ets", "ETS", _natural_model_label("baseline_ets"), reason))
    else:
        try:
            y_train = y.dropna()
            ets_res, ets_spec, ets_impl = _select_best_ets_model(
                y_train,
                allow_seasonal=allow_ets_seasonal,
                context=f" for {product_id}/{bu_id}"
            )
            if ets_res is None:
                raise ValueError("No ETS candidate fit succeeded")

            mean, ci = _forecast_ets(ets_res, forecast_index)
            ets_df = pd.DataFrame({
                "forecast_month": forecast_index,
                "forecast_value": mean.values,
                "lower_ci": ci.iloc[:, 0].values,
                "upper_ci": ci.iloc[:, 1].values,
                "model_group": "baseline_ets",
                "model_type": "ETS",
                "model_label": _natural_model_label("baseline_ets"),
                "p": np.nan,
                "d": np.nan,
                "q": np.nan,
                "P": np.nan,
                "D": np.nan,
                "Q": np.nan,
                "s": np.nan,
                "regressor_names": None,
                "regressor_details": None,
            })
            dfs.append(_attach_status(_apply_forecast_floor(ets_df), "ok"))
        except Exception as exc:
            print(f"  Failed ETS baseline: {exc}")
            dfs.append(_build_na_rows(
                "baseline_ets",
                "ETS",
                _natural_model_label("baseline_ets"),
                f"Failed: {exc}",
                status="Failed",
            ))

    # With regressor (SARIMAX)
    if not allow_sarimax:
        reason = status_reasons.get("with_regressor", "N/A: SARIMAX disabled by history threshold.")
        reg_label = _natural_model_label("with_regressor", regressor_name, regressor_lag)
        dfs.append(_build_na_rows("with_regressor", "SARIMAX", reg_label, reason))
    else:
        X_train, X_future = _build_exog_frames(exog_series, forecast_index)
        if X_train is not None and X_future is not None:
            aligned = pd.concat([y, X_train], axis=1).dropna()
            if len(aligned) >= 5:
                y_train = aligned.iloc[:, 0]
                exog_train = aligned.iloc[:, 1:]
                try:
                    mean, ci = _fit_variant(
                        y_train=y_train,
                        order=sarima_order,
                        seasonal_order=seasonal_order,
                        forecast_index=forecast_index,
                        exog_train=exog_train,
                        exog_future=X_future,
                    )
                    regressor_cols = list(exog_train.columns)
                    regressor_name_str = ", ".join(regressor_cols)
                    reg_label = _natural_model_label("with_regressor", regressor_name, regressor_lag)
                    sarimax_df = pd.DataFrame({
                        "forecast_month": forecast_index,
                        "forecast_value": mean.values,
                        "lower_ci": ci.iloc[:, 0].values,
                        "upper_ci": ci.iloc[:, 1].values,
                        "model_group": "with_regressor",
                        "model_type": "SARIMAX",
                        "model_label": reg_label,
                        "p": sarima_order[0],
                        "d": sarima_order[1],
                        "q": sarima_order[2],
                        "P": seasonal_order[0],
                        "D": seasonal_order[1],
                        "Q": seasonal_order[2],
                        "s": seasonal_order[3],
                        "regressor_names": [regressor_name_str] * len(mean),
                        "regressor_details": None,
                    })
                    dfs.append(_attach_status(_apply_forecast_floor(sarimax_df), "ok"))
                except Exception as exc:
                    print(f"  Failed regressor variant: {exc}")
                    reg_label = _natural_model_label("with_regressor", regressor_name, regressor_lag)
                    dfs.append(_build_na_rows(
                        "with_regressor",
                        "SARIMAX",
                        reg_label,
                        f"Failed: {exc}",
                        status="Failed",
                    ))
        else:
            reg_label = _natural_model_label("with_regressor", regressor_name, regressor_lag)
            reason = status_reasons.get("with_regressor", "N/A: missing future exog values.")
            dfs.append(_build_na_rows("with_regressor", "SARIMAX", reg_label, reason))

    # ML challenger (GradientBoostingRegressor)
    if allow_ml:
        try:
            y_train = y.dropna()
            features = _build_ml_features(y_train)
            df_ml = pd.concat([features, y_train.rename("y")], axis=1).dropna()
            if len(df_ml) <= 5:
                raise ValueError("Insufficient ML rows after dropping NaNs.")

            model = GradientBoostingRegressor(random_state=42)
            model.fit(df_ml.drop(columns=["y"]), df_ml["y"])
            feature_columns = list(df_ml.drop(columns=["y"]).columns)
            y_history = list(y_train.values)
            mean = _forecast_ml_recursive(
                model=model,
                y_history=y_history,
                last_date=y_train.index.max(),
                horizon=len(forecast_index),
                feature_columns=feature_columns,
            )
            ml_df = pd.DataFrame({
                "forecast_month": forecast_index,
                "forecast_value": mean.values,
                "lower_ci": [np.nan] * len(forecast_index),
                "upper_ci": [np.nan] * len(forecast_index),
                "model_group": "ml_gbr",
                "model_type": "ML",
                "model_label": _natural_model_label("ml_gbr"),
                "p": np.nan,
                "d": np.nan,
                "q": np.nan,
                "P": np.nan,
                "D": np.nan,
                "Q": np.nan,
                "s": np.nan,
                "regressor_names": None,
                "regressor_details": None,
            })
            dfs.append(_attach_status(_apply_forecast_floor(ml_df), "ok"))
        except Exception as exc:
            reason = status_reasons.get("ml_gbr", f"Failed: {exc}")
            dfs.append(_build_na_rows(
                "ml_gbr",
                "ML",
                _natural_model_label("ml_gbr"),
                reason,
                status="Failed",
            ))

    # Prophet challenger
    if allow_prophet:
        if not PROPHET_AVAILABLE:
            reason = status_reasons.get("prophet", "N/A: Prophet not installed.")
            dfs.append(_build_na_rows(
                "prophet",
                "PROPHET",
                _natural_model_label("prophet"),
                reason,
            ))
        else:
            try:
                y_train = _coerce_month_start_series(y)
                if len(y_train) < 24:
                    raise ValueError("Insufficient history (<24 months).")

                train_df = _prepare_prophet_df(y_train)
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="additive",
                    changepoint_prior_scale=0.05,
                )
                model.fit(train_df)
                future_df = pd.DataFrame({"ds": forecast_index})
                future_df["ds"] = pd.to_datetime(future_df["ds"]).dt.tz_localize(None)
                future_df["ds"] = future_df["ds"].dt.to_period("M").dt.to_timestamp()
                forecast = model.predict(future_df)
                mean = pd.Series(forecast["yhat"].values, index=forecast_index)

                prophet_df = pd.DataFrame({
                    "forecast_month": forecast_index,
                    "forecast_value": mean.values,
                    "lower_ci": [np.nan] * len(forecast_index),
                    "upper_ci": [np.nan] * len(forecast_index),
                    "model_group": "prophet",
                    "model_type": "PROPHET",
                    "model_label": _natural_model_label("prophet"),
                    "p": np.nan,
                    "d": np.nan,
                    "q": np.nan,
                    "P": np.nan,
                    "D": np.nan,
                    "Q": np.nan,
                    "s": np.nan,
                    "regressor_names": None,
                    "regressor_details": None,
                })
                dfs.append(_attach_status(_apply_forecast_floor(prophet_df), "ok"))
            except Exception as exc:
                reason = status_reasons.get("prophet", f"Failed: {exc}")
                dfs.append(_build_na_rows(
                    "prophet",
                    "PROPHET",
                    _natural_model_label("prophet"),
                    reason,
                    status="Failed",
                ))

    if not dfs:
        return pd.DataFrame()

    all_df = pd.concat(dfs, axis=0)
    all_df["product_id"] = product_id
    all_df["bu_id"] = bu_id
    all_df["run_id"] = str(run_id)
    all_df["training_start"] = y.index.min()
    all_df["training_end"] = y.index.max()
    all_df["horizon_months_ahead"] = all_df.groupby(["product_id", "model_group"]).cumcount() + 1

    # Ensure forecast_month is month-start timestamp
    all_df["forecast_month"] = pd.to_datetime(all_df["forecast_month"]).dt.to_period("M").dt.to_timestamp()
    return all_df


# ===========================
# POST-PROCESSING
# ===========================

def _apply_forecast_floor(df: pd.DataFrame, floor: float = FORECAST_FLOOR) -> pd.DataFrame:
    """Clip forecast point estimates and lower bounds at the given floor."""
    if df.empty:
        return df
    for col in ["forecast_value", "lower_ci"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=floor)
    return df


# ===========================
# MAIN
# ===========================

def _print_total_runtime(start_time: float) -> None:
    """Print elapsed runtime in seconds and minutes."""
    elapsed = time.perf_counter() - start_time
    minutes = elapsed / 60.0
    print(f"Total runtime: {elapsed:,.2f} seconds ({minutes:,.2f} minutes)")


def main():
    if ENABLE_ML_CHALLENGER:
        print("ML challenger enabled — evaluating ML_GBR candidate.")
    else:
        print("ML challenger disabled (ENABLE_ML_CHALLENGER=False) — skipping ML candidate evaluation.")
    if ENABLE_PROPHET_CHALLENGER:
        if PROPHET_AVAILABLE:
            print("Prophet challenger enabled — evaluating PROPHET candidate.")
        else:
            print("Prophet not installed; skipping Prophet challenger.")
    else:
        print("Prophet challenger disabled (ENABLE_PROPHET_CHALLENGER=False) — skipping Prophet candidate evaluation.")

    print("Loading data...")
    df_all = pd.read_excel(INPUT_FILE)
    df_all = aggregate_monthly_duplicates(df_all)
    df_all = mark_prelaunch_actuals_as_missing(
        df_all,
        output_excel_path=REVISED_ACTUALS_FILE,
        output_sheet=REVISED_ACTUALS_SHEET,
    )
    df_all[COL_DATE] = pd.to_datetime(df_all[COL_DATE])
    df_all = df_all.sort_values([COL_PRODUCT, COL_DIVISION, COL_DATE])

    print("Loading chosen models...")
    model_choices = load_model_choices(SUMMARY_FILE)

    print("Loading per-SKU orders...")
    orders_map = load_orders_map(ORDER_FILE, overrides_path=NOTES_FILE)

    results_rows = []
    sku_groups = df_all.groupby([COL_PRODUCT, COL_DIVISION], dropna=False)
    print(f"Found {len(sku_groups)} product+division combinations.")

    for (prod, div), df_sku in sku_groups:
        print(f"\n=== {prod} / {div} ===")
        choice = model_choices.get((prod, div))
        order_pair = orders_map.get((prod, div))

        if choice is None:
            print("  Skipping: no chosen model found in summary file.")
            continue
        if order_pair is None:
            print("  Skipping: no SARIMA order found in order file.")
            continue

        order, seasonal_order = order_pair
        df_sku = df_sku.set_index(COL_DATE)
        df_features = engineer_regressors(df_sku)

        y = df_features[COL_ACTUALS].astype(float)
        last_actual_dt = y.dropna().index.max()
        if pd.isna(last_actual_dt):
            print("  Skipping: no actuals available.")
            continue
        y_nonnull = y.dropna()
        history_months = len(y_nonnull)
        nonzero_mask = y_nonnull != 0
        first_nonzero_dt = y_nonnull[nonzero_mask].index.min() if nonzero_mask.any() else None
        if first_nonzero_dt is not None:
            months_since_first_nonzero = (
                last_actual_dt.to_period("M") - first_nonzero_dt.to_period("M")
            ).n
        else:
            months_since_first_nonzero = -1
        allow_sarima = history_months >= 12 and months_since_first_nonzero >= 11
        allow_sarimax = history_months >= 36 and allow_sarima
        allow_ets = True
        allow_ets_seasonal = history_months >= 24
        allow_ets_for_reco = history_months < 24

        freq = y.index.freq or pd.infer_freq(y.index)
        if freq is None:
            freq = "MS"
        freq = to_offset(freq)
        forecast_index = pd.date_range(start=last_actual_dt + freq, periods=FORECAST_HORIZON, freq=freq)

        reg_name = choice.get("reg_name")
        reg_lag = choice.get("reg_lag")
        exog_series = get_exog_series(df_features, reg_name, reg_lag)
        if allow_sarimax and exog_series is None:
            reg_name, reg_lag, exog_series = select_default_regressor(df_features)
        if not allow_sarimax:
            exog_series = None

        status_reasons = {}
        if not allow_sarima:
            status_reasons["baseline_sarima"] = (
                "N/A: history <12 months or first non-zero actual is <12 months ago."
            )
        if not allow_sarimax:
            if history_months < 36:
                status_reasons["with_regressor"] = "N/A: history <36 months; SARIMAX disabled."
            else:
                status_reasons["with_regressor"] = "N/A: SARIMA prerequisites not met."
        elif exog_series is None:
            status_reasons["with_regressor"] = "N/A: no regressor selected or exog unavailable."

        allow_ml = ENABLE_ML_CHALLENGER
        if not allow_ml:
            status_reasons["ml_gbr"] = "N/A: ML challenger disabled."

        allow_prophet = ENABLE_PROPHET_CHALLENGER and PROPHET_AVAILABLE
        if not ENABLE_PROPHET_CHALLENGER:
            status_reasons["prophet"] = "N/A: Prophet challenger disabled."
        elif not PROPHET_AVAILABLE:
            status_reasons["prophet"] = "N/A: Prophet not installed."

        fc_df = generate_forecast_variants(
            y=y,
            product_id=prod,
            bu_id=div,
            sarima_order=order,
            seasonal_order=seasonal_order,
            forecast_index=forecast_index,
            exog_series=exog_series,
            regressor_name=reg_name,
            regressor_lag=reg_lag,
            allow_sarima=allow_sarima,
            allow_sarimax=allow_sarimax,
            allow_ets=allow_ets,
            allow_ets_seasonal=allow_ets_seasonal,
            allow_ml=allow_ml,
            allow_prophet=allow_prophet,
            status_reasons=status_reasons,
        )
        if fc_df.empty:
            print("  No forecasts generated for this SKU (all variants failed).")
            continue

        recommended_group = _model_group_from_choice(choice)
        fc_df["recommended_model"] = fc_df["model_group"] == recommended_group

        results_rows.append(fc_df)
        print(f"  Generated {fc_df['model_group'].nunique()} model variants.")

    if not results_rows:
        print("No forecasts generated. Check inputs.")
        return

    all_forecasts = pd.concat(results_rows, axis=0, ignore_index=True)
    # Format forecast_month for Excel as MM/DD/YYYY to avoid time components
    all_forecasts["forecast_month"] = pd.to_datetime(all_forecasts["forecast_month"]).dt.strftime("%m/%d/%Y")

    # Add concatenated key column (product|BU|forecast_month|model_type) with month in "MMM YY" format; move to first position
    fc_month_short = pd.to_datetime(all_forecasts["forecast_month"]).dt.strftime("%b %y")
    key_col = (
        all_forecasts["product_id"].astype(str)
        + "|"
        + all_forecasts["bu_id"].astype(str)
        + "|"
        + fc_month_short.astype(str)
        + "|"
        + all_forecasts["model_type"].astype(str)
    )
    all_forecasts.insert(0, "product_bu_forecast_month_model_type", key_col)

    print("\nWriting combined forecasts...")
    first_fc_dt = pd.to_datetime(all_forecasts["forecast_month"]).min()
    output_file = _build_output_filename(OUTPUT_FILE_BASE, first_fc_dt)
    all_forecasts.to_excel(output_file, index=False, sheet_name="Forecast_Library")
    print(f"Done. Saved to {output_file}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    _print_total_runtime(start_time)
