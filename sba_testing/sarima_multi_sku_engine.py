import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="No frequency information was provided*")
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning as SMValueWarning
    warnings.filterwarnings("ignore", category=SMValueWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

import ast
import io
import logging
import os
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False

os.environ.setdefault("CMDSTANPY_LOG_LEVEL", "WARNING")
os.environ.setdefault("STAN_LOG_LEVEL", "WARNING")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").propagate = False
logging.getLogger("prophet").setLevel(logging.WARNING)
try:
    import cmdstanpy
    cmdstanpy.utils.get_logger().setLevel(logging.WARNING)
except Exception:
    pass

# ===========================
# 1. CONFIG
# ===========================

BASE_DIR = Path(__file__).resolve().parent
SALESFORCE_DATA_DIR = BASE_DIR / "salesforce_data"
if not SALESFORCE_DATA_DIR.exists():
    SALESFORCE_DATA_DIR = BASE_DIR.parent / "salesforce_data"

INPUT_FILE = BASE_DIR / "all_products_actuals_and_bookings.xlsx"  # <--- change to your file
REVISED_ACTUALS_FILE = BASE_DIR / "all_products_actuals_and_bookings_revised.xlsx"
REVISED_ACTUALS_SHEET = "Revised Actuals"
OUTPUT_FILE = BASE_DIR / "sarima_multi_sku_summary.xlsx"
OUTPUT_DIR = BASE_DIR / "data_storage"
HOLDOUT_TS_CSV = OUTPUT_DIR / "holdout_forecast_actuals.csv"
HOLDOUT_TS_SHEET = "Holdout_Forecast_Actuals"
ORDER_FILE = BASE_DIR / "sarimax_order_search_summary.xlsx"       # per-SKU SARIMA orders
NOTES_FILE = BASE_DIR / "Notes.xlsx"                              # manual order overrides
ENABLE_ML_CHALLENGER = True
ENABLE_PROPHET_CHALLENGER = True
QUIET_MODEL_OUTPUT = True
SKU_SUMMARY_ONLY = True
PIPELINE_FILES = sorted(SALESFORCE_DATA_DIR.glob("Merged Salesforce Pipeline *.xlsx"))
SUMMARY_REPORT_NAME = "Salesforce Pipeline Monthly Summary.xlsx"
SUMMARY_PATH = SALESFORCE_DATA_DIR / SUMMARY_REPORT_NAME
PRODUCT_CATALOG_PATH = BASE_DIR / "product_catalog_master.xlsx"
SF_PRODUCT_REFERENCE_PATHS = [
    SALESFORCE_DATA_DIR / "sf_product_reference_key.csv",
    BASE_DIR / "sf_product_reference_key.csv",
    BASE_DIR / "ML_model_testing" / "sf_product_reference_key.csv",
]
PIPELINE_ML_MODEL_NAME = "ML_GBR_PIPELINE"

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

# Recommended model selection thresholds
MAE_TOLERANCE = 0.20
RMSE_TOLERANCE = 0.20
ROCV_HARD_MULTIPLIER = 1.50
ROCV_PREFERRED_LOWER = 0.85
ROCV_PREFERRED_UPPER = 1.15
ROCV_TOO_SMOOTH = 0.70
ROCV_TOO_NOISY = 1.5
DA_MIN_PERIODS = 6
DA_IMPROVEMENT_PP = 0.10
DA_CLOSE_MAE_TOL = 0.05
BASELINE_MAE_ZERO_EPS = 1e-9  # treat MAE at or below this as effectively zero
ETS_SEASONAL_PERIODS = 12     # monthly data
SBA_ALPHA = 0.1
BLEND_SOFTMAX_TAU = 0.10
BLEND_EPS = 1e-9
BLEND_MODEL_NAME = "BLENDED_SOFTMAX"


# ===========================
# 2. HELPER FUNCTIONS
# ===========================

def _parse_args():
    parser = argparse.ArgumentParser(description="Multi-SKU SARIMA/SARIMAX/ETS/ML evaluation engine")
    parser.add_argument("--disable-ml", action="store_true", help="Disable ML challenger evaluation")
    parser.add_argument("--disable-prophet", action="store_true", help="Disable Prophet challenger evaluation")
    parser.add_argument(
        "--compare-model-selection",
        action="store_true",
        help="Generate old vs new model selection comparison reports after the run.",
    )
    return parser.parse_args()


@contextmanager
def quiet_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


def sku_log(message: str):
    if not SKU_SUMMARY_ONLY:
        print(message)


def format_sku_summary(idx, total, prod, div, ran=None, skipped=None, chosen=None, note=None):
    ran = ran or []
    skipped = skipped or []
    parts = [f"[{idx}/{total}] {prod}/{div}"]
    if chosen:
        parts.append(f"chosen={chosen}")
    if ran:
        parts.append("ran=" + ",".join(ran))
    if skipped:
        parts.append("skipped=" + ",".join(skipped))
    if note:
        parts.append(f"note={note}")
    return " | ".join(parts)


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
        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=output_sheet, index=False)
        print(f"Wrote revised actuals to {output_excel_path} ({output_sheet}).")

    return df


def last_completed_month(reference_date: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Return the last fully completed month based on the reference date."""
    ref = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp.today()
    ref = ref.normalize()
    first_of_month = ref.replace(day=1)
    return first_of_month - pd.DateOffset(months=1)


def filter_to_completed_months(
    df: pd.DataFrame,
    date_col: str = COL_DATE,
    cutoff: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Drop rows after the last fully completed month."""
    df = df.copy()
    if date_col not in df.columns:
        return df
    cutoff = cutoff if cutoff is not None else last_completed_month()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[df[date_col] <= cutoff].copy()


def safe_mae(y_true, y_pred):
    """Compute MAE robustly."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[mask], y_pred[mask])


def safe_rmse(y_true, y_pred):
    """RMSE that ignores NaNs and infs in either array."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


def compute_wape(actual, forecast, eps: float = BLEND_EPS) -> float:
    """Weighted absolute percentage error using absolute-actuals denominator."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(forecast)
    if not mask.any():
        return np.nan
    denom = np.sum(np.abs(actual[mask]))
    if denom <= eps:
        return np.nan
    return float(np.sum(np.abs(actual[mask] - forecast[mask])) / denom)


def _add_holdout_horizon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Horizon"] = (
        df.groupby([COL_PRODUCT, COL_DIVISION])["Date"]
        .rank(method="dense")
        .astype(int)
    )
    return df


def _blend_family(model_name: str) -> Optional[str]:
    if model_name == "SARIMA_baseline":
        return "SARIMA"
    if model_name.startswith("SARIMAX"):
        return "SARIMAX"
    if model_name == "ETS_baseline":
        return "ETS"
    if model_name == "SBA":
        return "SBA"
    if model_name == "PROPHET":
        return "PROPHET"
    if model_name in {"ML_GBR", PIPELINE_ML_MODEL_NAME}:
        return "ML"
    return None


def _softmax_weights_from_wape(wape_by_family: Dict[str, float], tau: float) -> Dict[str, float]:
    if not wape_by_family:
        return {}
    finite = {k: v for k, v in wape_by_family.items() if np.isfinite(v)}
    if not finite:
        return {}
    best = min(finite.values())
    denom = max(best, BLEND_EPS)
    weights = {}
    for fam, wape in finite.items():
        gap = (wape - best) / denom
        weights[fam] = float(np.exp(-gap / max(tau, BLEND_EPS)))
    total = sum(weights.values())
    if total <= BLEND_EPS:
        return {}
    return {fam: w / total for fam, w in weights.items()}


def _build_softmax_blend(
    holdout_df: pd.DataFrame,
    metrics_list: List[dict],
    tau: float = BLEND_SOFTMAX_TAU,
) -> Tuple[Optional[dict], pd.DataFrame, List[dict]]:
    if holdout_df.empty:
        return None, pd.DataFrame(), []

    df = _add_holdout_horizon(holdout_df)
    df["Family"] = df["Model"].apply(_blend_family)
    df = df[df["Family"].notna()]
    if df.empty:
        return None, pd.DataFrame(), []

    # Choose best model per family by overall WAPE on the holdout.
    overall_wape = (
        df.groupby(["Family", "Model"], dropna=False)
        .apply(lambda g: compute_wape(g["Actual"], g["Forecast"]))
        .reset_index(name="WAPE")
    )
    overall_wape = overall_wape[np.isfinite(overall_wape["WAPE"])]
    if overall_wape.empty:
        return None, pd.DataFrame(), []

    selected = {}
    for family, group in overall_wape.groupby("Family"):
        best_row = group.sort_values("WAPE", ascending=True).iloc[0]
        selected[family] = best_row["Model"]

    overall_wape_map = {
        fam: overall_wape.loc[overall_wape["Model"] == model, "WAPE"].min()
        for fam, model in selected.items()
    }
    overall_weights = _softmax_weights_from_wape(overall_wape_map, tau)

    # Per-horizon weights (using chosen model per family).
    weight_rows = []
    weights_by_horizon = {}
    for horizon, h_df in df.groupby("Horizon"):
        wape_by_family = {}
        for family, model in selected.items():
            g = h_df[h_df["Model"] == model]
            if g.empty:
                continue
            wape = compute_wape(g["Actual"], g["Forecast"])
            if np.isfinite(wape):
                wape_by_family[family] = wape
        weights = _softmax_weights_from_wape(wape_by_family, tau)
        if not weights:
            continue
        weights_by_horizon[int(horizon)] = {
            selected[fam]: weight for fam, weight in weights.items()
        }
        for family, weight in weights.items():
            weight_rows.append({
                COL_PRODUCT: h_df[COL_PRODUCT].iloc[0],
                COL_DIVISION: h_df[COL_DIVISION].iloc[0],
                "Horizon": int(horizon),
                "Model_Family": family,
                "Model": selected[family],
                "WAPE": wape_by_family.get(family, np.nan),
                "Weight": weight,
                "Tau": tau,
                "Weight_Source": "holdout_horizon",
            })

    selected_models = list(selected.values())
    df_sel = df[df["Model"].isin(selected_models)]
    if df_sel.empty:
        return None, pd.DataFrame(), weight_rows

    pivot = df_sel.pivot_table(
        index=[COL_PRODUCT, COL_DIVISION, "Date", "Horizon"],
        columns="Model",
        values="Forecast",
        aggfunc="mean",
    )
    actuals = (
        df_sel.groupby([COL_PRODUCT, COL_DIVISION, "Date", "Horizon"])["Actual"]
        .mean()
    )
    blended_rows = []
    for idx, row in pivot.iterrows():
        prod, div, date, horizon = idx
        weights = weights_by_horizon.get(int(horizon))
        if not weights:
            weights = {selected[fam]: w for fam, w in overall_weights.items()}
        if not weights:
            continue
        total = 0.0
        weight_sum = 0.0
        for model_name, weight in weights.items():
            if model_name in row and np.isfinite(row[model_name]):
                total += weight * row[model_name]
                weight_sum += weight
        if weight_sum <= BLEND_EPS:
            continue
        blended_rows.append({
            COL_PRODUCT: prod,
            COL_DIVISION: div,
            "Model": BLEND_MODEL_NAME,
            "Date": pd.to_datetime(date),
            "Actual": actuals.loc[idx],
            "Forecast": total / weight_sum,
        })

    blended_holdout = pd.DataFrame(blended_rows)
    if blended_holdout.empty:
        return None, pd.DataFrame(), weight_rows

    blend_mae = safe_mae(blended_holdout["Actual"].values, blended_holdout["Forecast"].values)
    blend_rmse = safe_rmse(blended_holdout["Actual"].values, blended_holdout["Forecast"].values)
    blend_wape = compute_wape(blended_holdout["Actual"], blended_holdout["Forecast"])

    rocv_map = {m.get("Model"): m.get("ROCV_MAE", np.nan) for m in metrics_list}
    rocv_weight_sum = 0.0
    rocv_total = 0.0
    for family, model_name in selected.items():
        weight = overall_weights.get(family)
        rocv_val = rocv_map.get(model_name, np.nan)
        if weight is None or not np.isfinite(rocv_val):
            continue
        rocv_total += weight * rocv_val
        rocv_weight_sum += weight
    blend_rocv = rocv_total / rocv_weight_sum if rocv_weight_sum > BLEND_EPS else np.nan

    blend_metrics = {
        "Model": BLEND_MODEL_NAME,
        "Test_MAE": blend_mae,
        "Test_RMSE": blend_rmse,
        "ROCV_MAE": blend_rocv,
        "AIC": np.nan,
        "BIC": np.nan,
        "WAPE": blend_wape,
        "Skip_Reason": "",
        "Forecast_valid": True,
    }
    return blend_metrics, blended_holdout, weight_rows


def _sba_forecast_value(y_train: pd.Series, alpha: float = SBA_ALPHA) -> float:
    """
    Syntetos-Boylan Approximation (SBA) forecast for intermittent demand.
    Returns the constant forecast level for the horizon.
    """
    y = pd.Series(y_train).astype(float).dropna()
    if y.empty:
        return np.nan

    z = None
    p = None
    q = 0.0
    for demand in y.values:
        q += 1.0
        if demand > 0:
            if z is None:
                z = float(demand)
                p = float(q)
            else:
                z = z + alpha * (float(demand) - z)
                p = p + alpha * (q - p)
            q = 0.0

    if z is None or p is None or p == 0:
        return 0.0

    return float((1.0 - alpha / 2.0) * (z / p))


def rolling_origin_cv_sba(
    y: pd.Series,
    alpha: float = SBA_ALPHA,
    horizon: int = ROCV_HORIZON,
    min_obs: int = ROCV_MIN_OBS,
) -> float:
    """Rolling-origin MAE for SBA."""
    y = pd.Series(y).astype(float).dropna()
    n = len(y)
    if n < min_obs + horizon:
        return np.nan

    errors = []
    for origin in range(min_obs, n - horizon + 1):
        y_train = y.iloc[:origin]
        y_test = y.iloc[origin:origin + horizon]
        forecast_value = _sba_forecast_value(y_train, alpha=alpha)
        preds = np.repeat(forecast_value, len(y_test))
        err = safe_mae(y_test.values, preds)
        if not np.isnan(err):
            errors.append(err)

    if not errors:
        return np.nan
    return float(np.mean(errors))
    mse = mean_squared_error(y_true[mask], y_pred[mask])  # no squared arg
    return np.sqrt(mse)


def _build_holdout_df(product, division, model, dates, actuals, forecasts) -> pd.DataFrame:
    df = pd.DataFrame({
        COL_PRODUCT: product,
        COL_DIVISION: division,
        "Model": model,
        "Date": pd.to_datetime(dates, errors="coerce"),
        "Actual": pd.to_numeric(actuals, errors="coerce"),
        "Forecast": pd.to_numeric(forecasts, errors="coerce"),
    })
    return df[[COL_PRODUCT, COL_DIVISION, "Model", "Date", "Actual", "Forecast"]]


@dataclass
class ModelBundle:
    adjustment_model: Optional[Pipeline]
    adjustment_features: List[str]


def month_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def parse_snapshot_month(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    dt = pd.to_datetime(raw, format="%Y-%m", errors="coerce")
    if dt.isna().any():
        mask = dt.isna()
        dt2 = pd.to_datetime(raw[mask], errors="coerce")
        dt.loc[mask] = dt2
    return dt.dt.to_period("M").dt.to_timestamp()


def parse_close_date(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    dt = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
    if dt.isna().any():
        mask = dt.isna()
        dt2 = pd.to_datetime(raw[mask], errors="coerce")
        dt.loc[mask] = dt2
    return dt.dt.to_period("M").dt.to_timestamp()


def month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def safe_nanmean(values: List[Optional[Union[float, int]]]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def normalize_code(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    raw = str(value).strip()
    if raw.endswith(".0") and raw.replace(".", "", 1).isdigit():
        raw = raw[:-2]
    return raw.upper()


def normalize_key(product: object, division: object) -> Tuple[str, str]:
    prod = normalize_code(product).strip()
    div = normalize_code(division).strip()
    return prod, div


def load_feature_mode(product_id: str, division: str, catalog: pd.DataFrame) -> str:
    group_key = (
        catalog["group_key_norm"]
        if "group_key_norm" in catalog.columns
        else catalog["group_key"].astype(str).apply(normalize_code)
    )
    business_unit = (
        catalog["business_unit_code_str"]
        if "business_unit_code_str" in catalog.columns
        else catalog["business_unit_code"].astype(str).apply(normalize_code)
    )
    match = catalog[
        (group_key == normalize_code(product_id))
        & (business_unit == normalize_code(division))
    ]
    if match.empty or "salesforce_feature_mode" not in match.columns:
        raise RuntimeError(
            f"salesforce_feature_mode not found for product {product_id} / {division}."
        )
    mode_raw = str(match.iloc[0]["salesforce_feature_mode"]).strip().lower()
    if mode_raw == "dollars":
        mode_raw = "revenue"
    if mode_raw not in {"quantity", "revenue"}:
        raise RuntimeError(f"Unsupported salesforce_feature_mode: {mode_raw}")
    return mode_raw


def load_sf_product_reference() -> pd.DataFrame:
    for path in SF_PRODUCT_REFERENCE_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            required = {"business_unit_code", "group_key", "salesforce_product_name"}
            if not required.issubset(set(df.columns)):
                raise RuntimeError(
                    "sf_product_reference_key.csv missing required columns: "
                    f"{', '.join(sorted(required))}."
                )
            df["business_unit_code"] = df["business_unit_code"].astype(str)
            df["group_key"] = df["group_key"].apply(normalize_code)
            df["salesforce_product_name"] = df["salesforce_product_name"].apply(normalize_code)
            df = df[df["group_key"] != ""]
            return df
    return pd.DataFrame()


def load_summary_report(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"group_key": str, "BU": str})
    if "Month" in df.columns:
        df["Month"] = month_start(df["Month"])
    df["group_key"] = df["group_key"].apply(normalize_code)
    df["BU"] = df["BU"].astype(str)
    return df


def filter_summary_product(df: pd.DataFrame, product_id: str, division: str) -> pd.DataFrame:
    if df.empty:
        return df
    product_key = normalize_code(product_id)
    division_key = str(division)
    product_mask = df["group_key"].apply(normalize_code) == product_key
    division_mask = df["BU"].astype(str) == division_key
    return df[product_mask & division_mask].copy()


def filter_pipeline_product(
    df: pd.DataFrame, product_id: str, sf_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    product_mask = pd.Series([False] * len(df), index=df.index)
    codes = {normalize_code(product_id)}
    if sf_codes:
        codes.update(normalize_code(code) for code in sf_codes if normalize_code(code))

    if "Product Code" in df.columns:
        product_codes = (
            df["Product_Code_num"]
            if "Product_Code_num" in df.columns
            else pd.to_numeric(df["Product Code"], errors="coerce")
        )
        numeric_codes = []
        for code in codes:
            if code.replace(".", "", 1).isdigit():
                numeric_codes.append(float(code))
        if numeric_codes:
            product_mask |= product_codes.isin(numeric_codes)
        product_norm = (
            df["Product_Code_norm"]
            if "Product_Code_norm" in df.columns
            else df["Product Code"].apply(normalize_code)
        )
        product_mask |= product_norm.isin(codes)
    if "Current OSC Product Name" in df.columns:
        name_norm = (
            df["Current_OSC_Product_Name_norm"]
            if "Current_OSC_Product_Name_norm" in df.columns
            else df["Current OSC Product Name"].apply(normalize_code)
        )
        product_mask |= name_norm.isin(codes)
    return df[product_mask].copy()


def load_pipeline_history(
    product_id: str,
    division: str,
    sf_codes: Optional[List[str]] = None,
    pipeline_all: Optional[pd.DataFrame] = None,
    pipeline_preprocessed: bool = False,
) -> pd.DataFrame:
    frames = []
    slip_frames = []

    if pipeline_all is None:
        sources: List[pd.DataFrame] = []
        for path in PIPELINE_FILES:
            sources.append(pd.read_excel(path))
    elif "_source_file" in pipeline_all.columns:
        sources = [
            pipeline_all[pipeline_all["_source_file"] == source].copy()
            for source in pipeline_all["_source_file"].dropna().unique()
        ]
    else:
        sources = [pipeline_all.copy()]

    for df in sources:
        if "Business Unit" in df.columns:
            df = df[df["Business Unit"] == division]
        df = filter_pipeline_product(df, product_id, sf_codes=sf_codes)
        if df.empty:
            continue
        if not pipeline_preprocessed:
            if "snapshot_month" not in df.columns and "Month" in df.columns:
                df["snapshot_month"] = parse_snapshot_month(df["Month"])
            if "target_month" not in df.columns and "Close Date" in df.columns:
                df["target_month"] = parse_close_date(df["Close Date"])
            if "Probability" in df.columns:
                df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
            if "Total Price" in df.columns:
                df["Total Price"] = pd.to_numeric(df["Total Price"], errors="coerce")
            if "Factored Quantity" in df.columns:
                df["Factored Quantity"] = pd.to_numeric(df["Factored Quantity"], errors="coerce")
            if "Factored Revenue" in df.columns:
                df["Factored Revenue"] = pd.to_numeric(df["Factored Revenue"], errors="coerce")
            if "Quantity" in df.columns:
                df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
            if "Quantity W/ Decimal" in df.columns:
                df["Quantity W/ Decimal"] = pd.to_numeric(
                    df["Quantity W/ Decimal"], errors="coerce"
                )

        if "target_month" in df.columns:
            df = df.dropna(subset=["target_month"])
        else:
            continue

        quantity_col = None
        if "Quantity" in df.columns and df["Quantity"].notna().any():
            quantity_col = "Quantity"
        elif "Quantity W/ Decimal" in df.columns and df["Quantity W/ Decimal"].notna().any():
            quantity_col = "Quantity W/ Decimal"
        else:
            quantity_col = "Quantity" if "Quantity" in df.columns else "Quantity W/ Decimal"

        if "Probability" in df.columns:
            prob = df["Probability"] / 100.0
            df["stage_weighted_qty"] = df[quantity_col] * prob
            if "Total Price" in df.columns:
                df["stage_weighted_revenue"] = df["Total Price"] * prob
            elif "Factored Revenue" in df.columns:
                df["stage_weighted_revenue"] = df["Factored Revenue"]
            else:
                df["stage_weighted_revenue"] = np.nan
        else:
            df["stage_weighted_qty"] = np.nan
            df["stage_weighted_revenue"] = np.nan

        agg = (
            df.groupby(["snapshot_month", "target_month"], dropna=False)
            .agg(
                pipeline_qty=(quantity_col, "sum"),
                pipeline_factored_qty=("Factored Quantity", "sum"),
                pipeline_factored_revenue=("Factored Revenue", "sum"),
                pipeline_stage_weighted_qty=("stage_weighted_qty", "sum"),
                pipeline_stage_weighted_revenue=("stage_weighted_revenue", "sum"),
            )
            .reset_index()
        )
        frames.append(agg)

        if "Opportunity Name" in df.columns and "Account Name" in df.columns:
            slip_df = df[
                [
                    "snapshot_month",
                    "target_month",
                    "Opportunity Name",
                    "Account Name",
                ]
            ].copy()
            slip_df["opp_key"] = (
                slip_df["Account Name"].astype(str).str.strip()
                + "||"
                + slip_df["Opportunity Name"].astype(str).str.strip()
            )
            slip_frames.append(slip_df)

    if not frames:
        return pd.DataFrame()

    pipeline_history = pd.concat(frames, ignore_index=True)

    slippage_months_by_snapshot = {}
    if slip_frames:
        slip_all = pd.concat(slip_frames, ignore_index=True)
        slip_all = slip_all.dropna(subset=["snapshot_month", "target_month", "opp_key"])
        slip_all = slip_all.sort_values(["opp_key", "snapshot_month"])
        slip_all["prior_target_month"] = slip_all.groupby("opp_key")["target_month"].shift(1)
        prior = slip_all["prior_target_month"]
        curr = slip_all["target_month"]
        slip_months = (curr.dt.year - prior.dt.year) * 12 + (curr.dt.month - prior.dt.month)
        slip_all["slip_months"] = slip_months.where(prior.notna(), np.nan)
        slip_stats = slip_all.groupby("snapshot_month")["slip_months"].median()
        slippage_months_by_snapshot = slip_stats.to_dict()

    if slippage_months_by_snapshot:
        pipeline_history["slippage_months"] = pipeline_history["snapshot_month"].map(
            slippage_months_by_snapshot
        )
    else:
        pipeline_history["slippage_months"] = np.nan

    return pipeline_history


def build_feature_frame(
    actuals: pd.DataFrame,
    pipeline: pd.DataFrame,
    max_horizon: int,
    feature_mode: str,
    summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    actuals_series = actuals.set_index("Month")["Actuals"].sort_index()
    actuals_lookup = actuals_series.to_dict()
    all_actual_months = actuals_series.index
    if all_actual_months.empty:
        return pd.DataFrame()

    snapshots = (
        pipeline["snapshot_month"].dropna().sort_values().unique().tolist()
        if not pipeline.empty
        else []
    )
    if not snapshots:
        return pd.DataFrame()

    pipeline_keyed = pipeline.set_index(["snapshot_month", "target_month"])

    summary_features: Dict[pd.Timestamp, Dict[str, float]] = {}
    summary_cols = [
        "Sum_Total_Price",
        "Sum_Quantity",
        "Open_Opportunities",
        "New_Opportunities",
        "Median_Months_Since_Last_Activity",
        "Open_Not_Modified_90_Days",
        "Early_Open_Opportunities",
        "Late_Open_Opportunities",
        "Open_Expected_Close_0_1_Months",
        "Open_Expected_Close_2_3_Months",
        "Open_Expected_Close_4_6_Months",
        "Open_Expected_Close_7_12_Months",
        "Open_Expected_Close_13_Plus_Months",
        "Open_Top_Decile_Count",
        "Open_Top_Decile_Close_0_3_Months",
        "Pct_Open_Not_Modified_90_Days",
        "Early_to_Late_Ratio",
        "Closed_Won_Count_24m",
        "Closed_Lost_Count_24m",
        "Median_Days_To_Close_Won_24m",
        "Mean_Days_To_Close_Won_24m",
        "Won_Close_0_3_Count_24m",
        "Won_Close_4_6_Count_24m",
        "Won_Close_7_12_Count_24m",
        "Won_Close_13_Plus_Count_24m",
        "Closed_Win_Rate_24m",
        "Pct_Won_Close_0_3_24m",
        "Pct_Won_Close_4_6_24m",
        "Pct_Won_Close_7_12_24m",
        "Pct_Won_Close_13_Plus_24m",
    ]
    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df.iterrows():
            key = pd.Timestamp(row["Month"])
            summary_features[key] = {col: row.get(col, np.nan) for col in summary_cols}

    snapshot_metrics = {}
    snapshot_ts_list = [pd.Timestamp(s) for s in snapshots]
    for snapshot_ts in snapshot_ts_list:
        snap_rows = pipeline[pipeline["snapshot_month"] == snapshot_ts].copy()
        snap_rows = snap_rows[snap_rows["target_month"] >= snapshot_ts]
        if snap_rows.empty:
            snapshot_metrics[snapshot_ts] = {}
            continue

        if feature_mode == "quantity":
            snap_rows["pipeline_primary"] = snap_rows["pipeline_factored_qty"]
            snap_rows["pipeline_stage_weighted_primary"] = snap_rows[
                "pipeline_stage_weighted_qty"
            ]
        else:
            snap_rows["pipeline_primary"] = snap_rows["pipeline_factored_revenue"]
            snap_rows["pipeline_stage_weighted_primary"] = snap_rows[
                "pipeline_stage_weighted_revenue"
            ]

        slip_months = snap_rows["slippage_months"].median()
        slip_months = int(round(slip_months)) if pd.notna(slip_months) else 0
        if slip_months:
            snap_rows["slip_target_month"] = snap_rows["target_month"] + pd.DateOffset(
                months=slip_months
            )
        else:
            snap_rows["slip_target_month"] = snap_rows["target_month"]

        slip_target_sum = (
            snap_rows.groupby("slip_target_month")["pipeline_primary"].sum().to_dict()
        )
        metrics = {
            "slippage_months": slip_months,
            "pipeline_slip_adjusted_primary": slip_target_sum,
        }
        for win in [1, 2, 3]:
            cutoff = snapshot_ts + pd.DateOffset(months=win)
            due_mask = snap_rows["target_month"] <= cutoff
            slip_mask = snap_rows["slip_target_month"] <= cutoff
            metrics[f"pipeline_due_{win}m"] = float(
                snap_rows.loc[due_mask, "pipeline_primary"].sum()
            )
            metrics[f"pipeline_stage_weighted_due_{win}m"] = float(
                snap_rows.loc[due_mask, "pipeline_stage_weighted_primary"].sum()
            )
            metrics[f"pipeline_slip_adjusted_due_{win}m"] = float(
                snap_rows.loc[slip_mask, "pipeline_primary"].sum()
            )
        snapshot_metrics[snapshot_ts] = metrics

    rows = []
    first_month = all_actual_months.min()
    for snapshot_ts in snapshot_ts_list:
        snapshot_for_lags = snapshot_ts - pd.DateOffset(months=1)
        snap_metrics = snapshot_metrics.get(snapshot_ts, {})
        slip_primary_map = snap_metrics.get("pipeline_slip_adjusted_primary", {})
        summary_row = summary_features.get(snapshot_ts, {})

        lag_months = [snapshot_for_lags - pd.DateOffset(months=i) for i in range(0, 12)]
        lag_values = [
            np.nan if actuals_lookup.get(m) is None else actuals_lookup.get(m)
            for m in lag_months
        ]
        lag_1 = lag_values[0]
        lag_2 = lag_values[1] if len(lag_values) > 1 else np.nan
        lag_3 = lag_values[2] if len(lag_values) > 2 else np.nan

        roll_mean_3 = safe_nanmean(lag_values[:3])
        roll_mean_6 = safe_nanmean(lag_values[:6])
        roll_mean_12 = safe_nanmean(lag_values[:12])
        avg_actuals_12 = roll_mean_12

        summary_primary = (
            summary_row.get("Sum_Quantity")
            if feature_mode == "quantity"
            else summary_row.get("Sum_Total_Price")
        )
        if summary_primary is None:
            summary_primary = np.nan
        summary_primary_coverage = (
            summary_primary / avg_actuals_12
            if avg_actuals_12 and not np.isnan(avg_actuals_12)
            else np.nan
        )
        summary_payload = {f"summary_{key}": summary_row.get(key, np.nan) for key in summary_cols}

        trend_idx = month_diff(snapshot_for_lags, first_month)

        try:
            snap_slice = pipeline_keyed.xs(snapshot_ts, level=0)
        except KeyError:
            snap_slice = pd.DataFrame()
        snap_qty = snap_slice["pipeline_qty"].to_dict() if not snap_slice.empty else {}
        snap_factored_qty = (
            snap_slice["pipeline_factored_qty"].to_dict() if not snap_slice.empty else {}
        )
        snap_factored_revenue = (
            snap_slice["pipeline_factored_revenue"].to_dict() if not snap_slice.empty else {}
        )
        snap_stage_weighted_qty = (
            snap_slice["pipeline_stage_weighted_qty"].to_dict() if not snap_slice.empty else {}
        )
        snap_stage_weighted_revenue = (
            snap_slice["pipeline_stage_weighted_revenue"].to_dict() if not snap_slice.empty else {}
        )

        prior_snapshot = snapshot_ts - pd.DateOffset(months=1)
        try:
            prior_slice = pipeline_keyed.xs(prior_snapshot, level=0)
        except KeyError:
            prior_slice = pd.DataFrame()
        prior_factored_qty = (
            prior_slice["pipeline_factored_qty"].to_dict() if not prior_slice.empty else {}
        )
        prior_factored_revenue = (
            prior_slice["pipeline_factored_revenue"].to_dict() if not prior_slice.empty else {}
        )

        for target_month in all_actual_months:
            if target_month < snapshot_ts:
                continue
            months_ahead = month_diff(target_month, snapshot_ts)
            if months_ahead > max_horizon:
                continue

            pipeline_qty = float(snap_qty.get(target_month, 0.0))
            pipeline_factored_qty = float(snap_factored_qty.get(target_month, 0.0))
            pipeline_factored_revenue = float(snap_factored_revenue.get(target_month, 0.0))
            pipeline_stage_weighted_qty = float(snap_stage_weighted_qty.get(target_month, 0.0))
            pipeline_stage_weighted_revenue = float(
                snap_stage_weighted_revenue.get(target_month, 0.0)
            )

            pipeline_primary = (
                pipeline_factored_qty if feature_mode == "quantity" else pipeline_factored_revenue
            )
            pipeline_stage_weighted_primary = (
                pipeline_stage_weighted_qty
                if feature_mode == "quantity"
                else pipeline_stage_weighted_revenue
            )
            prior_primary = float(
                prior_factored_qty.get(target_month, 0.0)
                if feature_mode == "quantity"
                else prior_factored_revenue.get(target_month, 0.0)
            )
            delta_pipeline = pipeline_primary - prior_primary
            pct_delta_pipeline = (
                delta_pipeline / prior_primary if prior_primary not in (0.0, np.nan) else np.nan
            )
            pipeline_coverage = (
                pipeline_primary / avg_actuals_12
                if avg_actuals_12 and not np.isnan(avg_actuals_12)
                else np.nan
            )
            target_actual = actuals_series.get(target_month)
            if pd.isna(target_actual):
                continue

            rows.append(
                {
                    "snapshot_month": snapshot_ts,
                    "target_month": target_month,
                    "months_ahead": months_ahead,
                    "target_month_num": target_month.month,
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_3": lag_3,
                    "roll_mean_3": roll_mean_3,
                    "roll_mean_6": roll_mean_6,
                    "roll_mean_12": roll_mean_12,
                    "trend_idx": trend_idx,
                    "pipeline_qty": pipeline_qty,
                    "pipeline_factored_qty": pipeline_factored_qty,
                    "pipeline_factored_revenue": pipeline_factored_revenue,
                    "pipeline_stage_weighted_qty": pipeline_stage_weighted_qty,
                    "pipeline_stage_weighted_revenue": pipeline_stage_weighted_revenue,
                    "pipeline_primary": pipeline_primary,
                    "pipeline_stage_weighted_primary": pipeline_stage_weighted_primary,
                    "prior_pipeline_primary": prior_primary,
                    "delta_pipeline": delta_pipeline,
                    "pct_delta_pipeline": pct_delta_pipeline,
                    "pipeline_coverage": pipeline_coverage,
                    "pipeline_slip_adjusted_primary": float(
                        slip_primary_map.get(target_month, 0.0)
                    ),
                    "pipeline_due_1m": snap_metrics.get("pipeline_due_1m", np.nan),
                    "pipeline_due_2m": snap_metrics.get("pipeline_due_2m", np.nan),
                    "pipeline_due_3m": snap_metrics.get("pipeline_due_3m", np.nan),
                    "pipeline_stage_weighted_due_1m": snap_metrics.get(
                        "pipeline_stage_weighted_due_1m", np.nan
                    ),
                    "pipeline_stage_weighted_due_2m": snap_metrics.get(
                        "pipeline_stage_weighted_due_2m", np.nan
                    ),
                    "pipeline_stage_weighted_due_3m": snap_metrics.get(
                        "pipeline_stage_weighted_due_3m", np.nan
                    ),
                    "pipeline_slip_adjusted_due_1m": snap_metrics.get(
                        "pipeline_slip_adjusted_due_1m", np.nan
                    ),
                    "pipeline_slip_adjusted_due_2m": snap_metrics.get(
                        "pipeline_slip_adjusted_due_2m", np.nan
                    ),
                    "pipeline_slip_adjusted_due_3m": snap_metrics.get(
                        "pipeline_slip_adjusted_due_3m", np.nan
                    ),
                    "slippage_months": snap_metrics.get("slippage_months", np.nan),
                    "summary_primary": summary_primary,
                    "summary_primary_coverage": summary_primary_coverage,
                    **summary_payload,
                    "actuals": target_actual,
                }
            )

    return pd.DataFrame(rows)


def build_baseline_features(y: pd.Series) -> pd.DataFrame:
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


def train_baseline_model(actuals_series: pd.Series):
    features = build_baseline_features(actuals_series)
    df_ml = pd.concat([features, actuals_series.rename("y")], axis=1).dropna()
    if len(df_ml) <= 5:
        raise RuntimeError("Insufficient rows for baseline ML training.")

    model = GradientBoostingRegressor(random_state=42)
    model.fit(df_ml.drop(columns=["y"]), df_ml["y"])
    preds = pd.Series(model.predict(df_ml.drop(columns=["y"])), index=df_ml.index)
    feature_columns = list(df_ml.drop(columns=["y"]).columns)
    return model, feature_columns, preds


def forecast_baseline_recursive(
    model: GradientBoostingRegressor,
    y_history: List[float],
    last_date: pd.Timestamp,
    horizon: int,
    feature_columns: List[str],
) -> pd.Series:
    preds = []
    history = list(y_history)
    start_trend = len(history)
    include_lag12 = "y_lag12" in feature_columns

    for step in range(1, horizon + 1):
        future_date = last_date + pd.DateOffset(months=step)
        if len(history) < 3 or (include_lag12 and len(history) < 12):
            raise RuntimeError("Insufficient history for baseline recursive forecast.")

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


def baseline_forecast_for_snapshot(
    actuals_series: pd.Series,
    snapshot_month: pd.Timestamp,
    horizon: int,
) -> Optional[pd.Series]:
    snapshot_ts = pd.Timestamp(snapshot_month)
    cutoff = snapshot_ts - pd.DateOffset(months=1)
    history = actuals_series[actuals_series.index <= cutoff].dropna()
    if history.empty or horizon <= 0:
        return None
    try:
        model, feature_cols, _ = train_baseline_model(history)
    except RuntimeError:
        return None
    try:
        forecast = forecast_baseline_recursive(
            model=model,
            y_history=list(history.values),
            last_date=history.index.max(),
            horizon=horizon,
            feature_columns=feature_cols,
        )
    except RuntimeError:
        return None
    return forecast


def train_models(df: pd.DataFrame) -> ModelBundle:
    adjustment_features = [
        "months_ahead",
        "target_month_num",
        "pipeline_primary",
        "pipeline_stage_weighted_primary",
        "delta_pipeline",
        "pct_delta_pipeline",
        "pipeline_coverage",
        "pipeline_slip_adjusted_primary",
        "pipeline_due_1m",
        "pipeline_due_2m",
        "pipeline_due_3m",
        "pipeline_stage_weighted_due_1m",
        "pipeline_stage_weighted_due_2m",
        "pipeline_stage_weighted_due_3m",
        "pipeline_slip_adjusted_due_1m",
        "pipeline_slip_adjusted_due_2m",
        "pipeline_slip_adjusted_due_3m",
        "slippage_months",
        "baseline_pred",
    ]
    summary_feature_prefix = "summary_"
    summary_cols = [col for col in df.columns if col.startswith(summary_feature_prefix)]
    candidate_summary = ["summary_primary", "summary_primary_coverage"] + summary_cols
    summary_available = [col for col in candidate_summary if df[col].notna().any()]
    adjustment_features.extend(summary_available)

    df = df.copy()
    df = df.dropna(subset=["actuals", "baseline_pred"])
    df["residual"] = df["actuals"] - df["baseline_pred"]

    adjustment_model = None
    if df[adjustment_features].notna().any().any():
        adjustment_model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        )
        adjustment_model.fit(df[adjustment_features], df["residual"])

    return ModelBundle(
        adjustment_model=adjustment_model,
        adjustment_features=adjustment_features,
    )


def predict_with_models(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    output = df.copy()

    if bundle.adjustment_model is not None:
        output["adjustment_pred"] = bundle.adjustment_model.predict(
            output[bundle.adjustment_features]
        )
    else:
        output["adjustment_pred"] = 0.0

    output["final_pred"] = output["baseline_pred"] + output["adjustment_pred"]
    output["final_pred"] = output["final_pred"].clip(lower=0.0)
    return output


def compute_rolling_cv_mae(training: pd.DataFrame, holdout_months: int) -> float:
    if training.empty:
        return np.nan
    target_months = sorted(training["target_month"].dropna().unique())
    if len(target_months) < holdout_months:
        return np.nan
    holdout_targets = target_months[-holdout_months:]
    fold_mae = []
    for cutoff in holdout_targets:
        train_df = training[training["target_month"] < cutoff].copy()
        test_df = training[training["target_month"] == cutoff].copy()
        if train_df.empty or test_df.empty:
            continue
        bundle = train_models(train_df)
        preds = predict_with_models(test_df, bundle)
        abs_err = (preds["final_pred"] - preds["actuals"]).abs()
        if not abs_err.empty:
            fold_mae.append(float(abs_err.mean()))
    return float(np.mean(fold_mae)) if fold_mae else np.nan


def _compute_test_window(history_months: int) -> int:
    """Match forecast script: smaller test windows for short histories."""
    return TEST_HORIZON


def _regressor_failure_metrics(model_name):
    """Consistent fallback metrics when a regressor model cannot be fit."""
    return {
        "Model": model_name,
        "Test_MAE": 1e9,
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
        with quiet_output(QUIET_MODEL_OUTPUT):
            return model.fit(disp=False)

    model = ExponentialSmoothing(
        y_train,
        trend=spec["trend"],
        damped_trend=spec["damped_trend"] if spec["trend"] is not None else False,
        seasonal=spec["seasonal"],
        seasonal_periods=spec["seasonal_periods"] if spec["seasonal"] is not None else None,
        initialization_method="estimated",
    )
    with quiet_output(QUIET_MODEL_OUTPUT):
        return model.fit(optimized=True, disp=False)


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
            sku_log(f"[WARN] ETSModel failed ({spec}){context}: {exc}")
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
            sku_log(f"[WARN] ExponentialSmoothing failed ({spec}){context}: {exc}")
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
    sku=None,
    bu=None,
    return_holdout: bool = False,
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
        return (metrics, pd.DataFrame()) if return_holdout else metrics

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
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    try:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(train[feature_cols], train["y"])
        preds = model.predict(test[feature_cols])
    except Exception as exc:
        metrics["Skip_Reason"] = f"ML fit failed: {exc}"
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    metrics["Test_MAE"] = safe_mae(test["y"].values, preds)
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
    if not return_holdout:
        return metrics
    holdout_df = _build_holdout_df(
        sku,
        bu,
        metrics["Model"],
        test.index,
        test["y"].values,
        preds,
    )
    return metrics, holdout_df


def evaluate_ml_pipeline_candidate(
    actuals_df: pd.DataFrame,
    actuals_series: pd.Series,
    pipeline_history: pd.DataFrame,
    feature_mode: str,
    summary_df: pd.DataFrame,
    holdout_months: int = TEST_HORIZON,
    sku=None,
    bu=None,
    return_holdout: bool = False,
):
    metrics = {
        "Model": PIPELINE_ML_MODEL_NAME,
        "Test_MAE": np.nan,
        "Test_RMSE": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
        "ROCV_MAE": np.nan,
        "Regressor_coef": np.nan,
        "Regressor_pvalue": np.nan,
        "Skip_Reason": "",
    }

    if pipeline_history.empty:
        metrics["Skip_Reason"] = "No pipeline history rows."
        return (metrics, pd.DataFrame()) if return_holdout else metrics
    if summary_df.empty:
        metrics["Skip_Reason"] = "No summary data for product/BU."
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    training = build_feature_frame(
        actuals_df,
        pipeline_history,
        holdout_months,
        feature_mode,
        summary_df=summary_df,
    )
    if training.empty:
        metrics["Skip_Reason"] = "No training rows after assembling pipeline features."
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    training["baseline_pred"] = np.nan
    for snapshot in sorted(training["snapshot_month"].dropna().unique()):
        snapshot_rows = training["snapshot_month"] == snapshot
        # Include the max months_ahead target month (months_ahead starts at 0 for snapshot month).
        horizon = int(training.loc[snapshot_rows, "months_ahead"].max()) + 1
        forecast = baseline_forecast_for_snapshot(actuals_series, snapshot, horizon)
        if forecast is None:
            continue
        training.loc[snapshot_rows, "baseline_pred"] = training.loc[
            snapshot_rows, "target_month"
        ].map(forecast)
    training = training.dropna(subset=["baseline_pred"])
    if training.empty:
        metrics["Skip_Reason"] = "No baseline predictions available for pipeline training."
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    metrics["ROCV_MAE"] = compute_rolling_cv_mae(training, holdout_months)

    holdout_start = actuals_series.index.max() - pd.DateOffset(months=holdout_months - 1)
    holdout_df = training[training["target_month"] >= holdout_start].copy()
    train_adj_df = training[training["target_month"] < holdout_start].copy()
    if holdout_df.empty or train_adj_df.empty:
        metrics["Skip_Reason"] = "Insufficient pipeline rows for 12-month holdout."
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    bundle = train_models(train_adj_df)
    holdout_pred = predict_with_models(holdout_df, bundle)
    abs_err = (holdout_pred["final_pred"] - holdout_pred["actuals"]).abs()
    metrics["Test_MAE"] = float(abs_err.mean()) if not abs_err.empty else np.nan
    metrics["Test_RMSE"] = float(
        np.sqrt(mean_squared_error(holdout_pred["actuals"], holdout_pred["final_pred"]))
    ) if not holdout_pred.empty else np.nan
    if not return_holdout:
        return metrics
    holdout_rows = _build_holdout_df(
        sku,
        bu,
        metrics["Model"],
        holdout_pred["target_month"],
        holdout_pred["actuals"],
        holdout_pred["final_pred"],
    )
    return metrics, holdout_rows


def fit_sarima_baseline(y_train, order, seasonal_order):
    """Fit baseline SARIMA model without exogenous regressors."""
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    with quiet_output(QUIET_MODEL_OUTPUT):
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
    with quiet_output(QUIET_MODEL_OUTPUT):
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
    exog_invalid=False,
    return_holdout: bool = False,
):
    """
    Fit model, compute test MAE/RMSE on last 'horizon' points,
    compute ROCV MAE, and return a dict of metrics.
    """
    y = pd.Series(y).astype(float)
    n_total = len(y)

    # If too short, bail with NaNs
    if n_total <= horizon + 5:  # need at least some training
        metrics = {
            "Model": model_name,
            "Test_MAE": np.nan,
            "Test_RMSE": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "ROCV_MAE": np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    if exog is not None:
        if exog_invalid:
            metrics = _regressor_failure_metrics(model_name)
            return (metrics, pd.DataFrame()) if return_holdout else metrics

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
            sku_log(f"[WARN] Low exog coverage ({coverage:.2%}) for {sku_txt}. Proceeding with cleaned exog.")

        # If still all NaN after cleaning, bail with large MAE
        if exog_train.notna().sum() == 0:
            metrics = _regressor_failure_metrics(model_name)
            return (metrics, pd.DataFrame()) if return_holdout else metrics

        # Fit SARIMAX
        try:
            res = fit_sarimax(y_train, exog_train, order=order, seasonal_order=seasonal_order)
        except Exception:
            metrics = _regressor_failure_metrics(model_name)
            return (metrics, pd.DataFrame()) if return_holdout else metrics

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

        metrics = {
            "Model": model_name,
            "Test_MAE": safe_mae(y_test.values, y_pred.values) if not y_pred.isna().all() else 1e9,
            "Test_RMSE": safe_rmse(y_test.values, y_pred.values) if not y_pred.isna().all() else 1e9,
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "ROCV_MAE": float(rocv_mae) if rocv_mae is not None else np.nan,
            "Regressor_coef": reg_coef,
            "Regressor_pvalue": reg_pval
        }
        if not return_holdout:
            return metrics
        holdout_df = _build_holdout_df(
            sku,
            bu,
            model_name,
            y_test.index,
            y_test.values,
            pd.Series(y_pred, index=y_test.index).values,
        )
        return metrics, holdout_df

    else:
        train_end = n_total - horizon
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:]

        # Fit baseline SARIMA
        try:
            res = fit_sarima_baseline(y_train, order=order, seasonal_order=seasonal_order)
        except Exception:
            metrics = {
                "Model": model_name,
                "Test_MAE": np.nan,
                "Test_RMSE": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "ROCV_MAE": np.nan,
                "Regressor_coef": np.nan,
                "Regressor_pvalue": np.nan
            }
            return (metrics, pd.DataFrame()) if return_holdout else metrics

        try:
            forecast_res = res.get_forecast(steps=horizon)
            y_pred = forecast_res.predicted_mean
        except Exception:
            y_pred = pd.Series(index=y_test.index, data=np.nan)

        rocv_mae = rolling_origin_cv_precomputed(rocv_spec, order=order, seasonal_order=seasonal_order)

        metrics = {
            "Model": model_name,
            "Test_MAE": safe_mae(y_test.values, y_pred.values),
            "Test_RMSE": safe_rmse(y_test.values, y_pred.values),
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "ROCV_MAE": float(rocv_mae) if rocv_mae is not None else np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }
        if not return_holdout:
            return metrics
        holdout_df = _build_holdout_df(
            sku,
            bu,
            model_name,
            y_test.index,
            y_test.values,
            pd.Series(y_pred, index=y_test.index).values,
        )
        return metrics, holdout_df


def evaluate_ets_model(
    y,
    model_name="ETS_baseline",
    horizon=TEST_HORIZON,
    sku=None,
    bu=None,
    rocv_spec=None,
    return_holdout: bool = False,
):
    """
    Fit ETS (best by AICc/AIC), compute test MAE/RMSE on last 'horizon' points,
    compute ROCV MAE, and return a dict of metrics.
    """
    y = pd.Series(y).astype(float)
    y_clean = y.dropna()
    n_total = len(y_clean)

    if n_total <= horizon:
        metrics = {
            "Model": model_name,
            "Test_MAE": np.nan,
            "Test_RMSE": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "ROCV_MAE": np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    train_end = n_total - horizon
    y_train = y_clean.iloc[:train_end]
    y_test = y_clean.iloc[train_end:]

    context = f" for {sku}/{bu}" if sku or bu else ""
    ets_res, ets_spec, ets_impl = _select_best_ets_model(y_train, context=context)
    if ets_res is None:
        metrics = {
            "Model": model_name,
            "Test_MAE": np.nan,
            "Test_RMSE": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "ROCV_MAE": np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    try:
        y_pred = _forecast_ets(ets_res, steps=horizon, index=y_test.index)
    except Exception:
        y_pred = pd.Series(index=y_test.index, data=np.nan)

    rocv_mae = rolling_origin_cv_ets_precomputed(rocv_spec, ets_spec, ets_impl)

    aic = getattr(ets_res, "aic", np.nan)
    bic = getattr(ets_res, "bic", np.nan)

    metrics = {
        "Model": model_name,
        "Test_MAE": safe_mae(y_test.values, y_pred.values),
        "Test_RMSE": safe_rmse(y_test.values, y_pred.values),
        "AIC": float(aic) if np.isfinite(aic) else np.nan,
        "BIC": float(bic) if np.isfinite(bic) else np.nan,
        "ROCV_MAE": float(rocv_mae) if rocv_mae is not None else np.nan,
        "Regressor_coef": np.nan,
        "Regressor_pvalue": np.nan
    }
    if not return_holdout:
        return metrics
    holdout_df = _build_holdout_df(
        sku,
        bu,
        model_name,
        y_test.index,
        y_test.values,
        pd.Series(y_pred, index=y_test.index).values,
    )
    return metrics, holdout_df


def evaluate_sba_model(
    y,
    model_name="SBA",
    horizon=TEST_HORIZON,
    sku=None,
    bu=None,
    alpha: float = SBA_ALPHA,
    return_holdout: bool = False,
):
    """
    Fit SBA model, compute test MAE/RMSE on last 'horizon' points,
    compute ROCV MAE, and return a dict of metrics.
    """
    y = pd.Series(y).astype(float)
    y_clean = y.dropna()
    n_total = len(y_clean)

    if n_total <= horizon:
        metrics = {
            "Model": model_name,
            "Test_MAE": np.nan,
            "Test_RMSE": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "ROCV_MAE": np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan,
        }
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    train_end = n_total - horizon
    y_train = y_clean.iloc[:train_end]
    y_test = y_clean.iloc[train_end:]

    forecast_value = _sba_forecast_value(y_train, alpha=alpha)
    y_pred = pd.Series(np.repeat(forecast_value, len(y_test)), index=y_test.index)

    rocv_mae = rolling_origin_cv_sba(y_clean, alpha=alpha, horizon=ROCV_HORIZON, min_obs=ROCV_MIN_OBS)

    metrics = {
        "Model": model_name,
        "Test_MAE": safe_mae(y_test.values, y_pred.values),
        "Test_RMSE": safe_rmse(y_test.values, y_pred.values),
        "AIC": np.nan,
        "BIC": np.nan,
        "ROCV_MAE": float(rocv_mae) if np.isfinite(rocv_mae) else np.nan,
        "Regressor_coef": np.nan,
        "Regressor_pvalue": np.nan,
    }
    if not return_holdout:
        return metrics
    holdout_df = _build_holdout_df(
        sku,
        bu,
        metrics["Model"],
        y_test.index,
        y_test.values,
        y_pred.values,
    )
    return metrics, holdout_df


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


def _evaluate_prophet_metrics(y_series, horizon=TEST_HORIZON, sku=None, bu=None, return_holdout: bool = False):
    metrics = {
        "Model": "PROPHET",
        "Test_MAE": np.nan,
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
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    y_clean = _coerce_month_start_series(y_series)
    if len(y_clean) < 24:
        metrics["Skip_Reason"] = "Insufficient history (<24 months)."
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    if len(y_clean) <= horizon + 5:
        metrics["Skip_Reason"] = "Insufficient history after holdout."
        return (metrics, pd.DataFrame()) if return_holdout else metrics

    train_end = len(y_clean) - horizon
    y_train = y_clean.iloc[:train_end]
    y_test = y_clean.iloc[train_end:]

    try:
        train_df = _prepare_prophet_df(y_train)
        if train_df.empty:
            metrics["Skip_Reason"] = "No usable Prophet training data."
            return (metrics, pd.DataFrame()) if return_holdout else metrics

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
        )
        with quiet_output(QUIET_MODEL_OUTPUT):
            model.fit(train_df)

        future_df = pd.DataFrame({"ds": y_test.index})
        future_df["ds"] = pd.to_datetime(future_df["ds"]).dt.tz_localize(None)
        future_df["ds"] = future_df["ds"].dt.to_period("M").dt.to_timestamp()
        forecast = model.predict(future_df)
        y_pred = pd.Series(forecast["yhat"].values, index=y_test.index)

        metrics["Test_MAE"] = safe_mae(y_test.values, y_pred.values)
        metrics["Test_RMSE"] = safe_rmse(y_test.values, y_pred.values)
        metrics["Forecast_valid"] = np.isfinite(y_pred.values).all() and len(y_pred) == len(y_test)
    except Exception as exc:
        sku_txt = f"{sku}/{bu}" if sku or bu else ""
        sku_log(f"Prophet failed for SKU {sku_txt}: {exc}")
        metrics["Skip_Reason"] = f"Prophet failed: {exc}"
        return (metrics, pd.DataFrame()) if return_holdout else metrics

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
            with quiet_output(QUIET_MODEL_OUTPUT):
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

    if not return_holdout:
        return metrics
    holdout_df = _build_holdout_df(
        sku,
        bu,
        metrics["Model"],
        y_test.index,
        y_test.values,
        pd.Series(y_pred, index=y_test.index).values,
    )
    return metrics, holdout_df


def _model_tiebreak_key(m):
    mae = m.get("Test_MAE", np.nan)
    rmse = m.get("Test_RMSE", np.nan)
    model_name = str(m.get("Model", ""))
    return (
        mae if np.isfinite(mae) else np.inf,
        rmse if np.isfinite(rmse) else np.inf,
        model_name,
    )


def _model_tiebreak_key_df(row: pd.Series) -> tuple:
    mae = row.get("Test_MAE", np.nan)
    rmse = row.get("Test_RMSE", np.nan)
    model_name = str(row.get("Model", ""))
    return (
        mae if np.isfinite(mae) else np.inf,
        rmse if np.isfinite(rmse) else np.inf,
        model_name,
    )


def compute_directional_accuracy(ts_df: pd.DataFrame, min_periods: int = DA_MIN_PERIODS) -> pd.DataFrame:
    if ts_df.empty:
        return pd.DataFrame(columns=[COL_PRODUCT, COL_DIVISION, "Model", "DA", "DA_Valid_Periods"])

    df = ts_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
    df["Forecast"] = pd.to_numeric(df["Forecast"], errors="coerce")
    df = df.dropna(subset=["Date", "Actual", "Forecast"])

    def _group_da(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values("Date")
        if len(group) < 2:
            return pd.Series({"DA": np.nan, "DA_Valid_Periods": 0})
        delta_actual = np.sign(np.diff(group["Actual"].values))
        delta_forecast = np.sign(np.diff(group["Forecast"].values))
        valid = np.isfinite(delta_actual) & np.isfinite(delta_forecast)
        valid_count = int(valid.sum())
        if valid_count < min_periods:
            return pd.Series({"DA": np.nan, "DA_Valid_Periods": valid_count})
        da = float(np.mean(delta_actual[valid] == delta_forecast[valid]))
        return pd.Series({"DA": da, "DA_Valid_Periods": valid_count})

    grouped = df.groupby([COL_PRODUCT, COL_DIVISION, "Model"], dropna=False)
    da_df = grouped.apply(_group_da)
    group_cols = [COL_PRODUCT, COL_DIVISION, "Model"]
    cols_to_drop = [c for c in group_cols if c in da_df.columns]
    if cols_to_drop:
        da_df = da_df.drop(columns=cols_to_drop)
    da_df = da_df.reset_index()
    return da_df


def compute_bias_metrics(ts_df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    if ts_df.empty:
        return pd.DataFrame(
            columns=[COL_PRODUCT, COL_DIVISION, "Model", "bias", "abs_bias", "bias_pct", "abs_bias_pct"]
        )

    df = ts_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
    df["Forecast"] = pd.to_numeric(df["Forecast"], errors="coerce")
    df = df.dropna(subset=["Date", "Actual", "Forecast"])
    if df.empty:
        return pd.DataFrame(
            columns=[COL_PRODUCT, COL_DIVISION, "Model", "bias", "abs_bias", "bias_pct", "abs_bias_pct"]
        )

    # Aggregate to unique target months so models with multiple snapshot rows
    # (e.g., pipeline ML) don't inflate bias from duplicated months.
    monthly = (
        df.groupby([COL_PRODUCT, COL_DIVISION, "Model", "Date"], dropna=False)
        .agg({"Actual": "mean", "Forecast": "mean"})
        .reset_index()
    )

    def _group_bias(group: pd.DataFrame) -> pd.Series:
        actual = group["Actual"].astype(float)
        forecast = group["Forecast"].astype(float)
        diff = forecast - actual
        bias = float(diff.sum())
        abs_bias = float(abs(bias))
        denom = float(actual.sum())
        denom = denom if abs(denom) > eps else eps
        bias_pct = bias / denom
        abs_bias_pct = abs(bias_pct)
        return pd.Series(
            {
                "bias": bias,
                "abs_bias": abs_bias,
                "bias_pct": bias_pct,
                "abs_bias_pct": abs_bias_pct,
            }
        )

    grouped = monthly.groupby([COL_PRODUCT, COL_DIVISION, "Model"], dropna=False)
    bias_df = grouped.apply(_group_bias)
    group_cols = [COL_PRODUCT, COL_DIVISION, "Model"]
    cols_to_drop = [c for c in group_cols if c in bias_df.columns]
    if cols_to_drop:
        bias_df = bias_df.drop(columns=cols_to_drop)
    bias_df = bias_df.reset_index()
    return bias_df


def compute_weighted_holdout_metrics(ts_df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """
    Compute weighted MAE/RMSE and bias on the holdout, emphasizing recent months.
    Weights: most recent 3 months = 55%, months 4-6 = 30%, months 7-12 = 15%.
    """
    if ts_df.empty:
        return pd.DataFrame(
            columns=[
                COL_PRODUCT, COL_DIVISION, "Model",
                "Weighted_MAE", "Weighted_RMSE",
                "bias", "abs_bias", "bias_pct", "abs_bias_pct",
            ]
        )

    df = ts_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
    df["Forecast"] = pd.to_numeric(df["Forecast"], errors="coerce")
    df = df.dropna(subset=["Date", "Actual", "Forecast"])
    if df.empty:
        return pd.DataFrame(
            columns=[
                COL_PRODUCT, COL_DIVISION, "Model",
                "Weighted_MAE", "Weighted_RMSE",
                "bias", "abs_bias", "bias_pct", "abs_bias_pct",
            ]
        )

    # Aggregate to unique target months so models with multiple snapshot rows
    # (e.g., pipeline ML) don't inflate metrics from duplicated months.
    monthly = (
        df.groupby([COL_PRODUCT, COL_DIVISION, "Model", "Date"], dropna=False)
        .agg({"Actual": "mean", "Forecast": "mean"})
        .reset_index()
    )

    def _assign_recency_weights(dates: pd.Series) -> pd.Series:
        max_date = dates.max()
        months_back = dates.apply(lambda d: month_diff(max_date, d))
        buckets = pd.Series(index=dates.index, dtype="object")
        buckets[months_back <= 2] = "recent"
        buckets[(months_back >= 3) & (months_back <= 5)] = "mid"
        buckets[months_back >= 6] = "old"

        base_weights = {"recent": 0.55, "mid": 0.30, "old": 0.15}
        counts = buckets.value_counts()
        present = {k: v for k, v in base_weights.items() if counts.get(k, 0) > 0}
        total = sum(present.values())
        if total <= 0:
            return pd.Series(np.nan, index=dates.index)

        normalized = {k: v / total for k, v in present.items()}
        weights = pd.Series(index=dates.index, dtype="float64")
        for bucket, weight in normalized.items():
            count = counts.get(bucket, 0)
            if count:
                weights[buckets == bucket] = weight / float(count)
        return weights

    def _group_metrics(group: pd.DataFrame) -> pd.Series:
        weights = _assign_recency_weights(group["Date"])
        actual = group["Actual"].astype(float)
        forecast = group["Forecast"].astype(float)
        err = forecast - actual
        abs_err = err.abs()
        weights = weights.fillna(0.0)

        w_mae = float((weights * abs_err).sum())
        w_rmse = float(np.sqrt((weights * (err ** 2)).sum()))
        bias = float((weights * err).sum())
        abs_bias = float(abs(bias))
        denom = float((weights * actual).sum())
        denom = denom if abs(denom) > eps else eps
        bias_pct = bias / denom
        abs_bias_pct = abs(bias_pct)

        return pd.Series(
            {
                "Weighted_MAE": w_mae,
                "Weighted_RMSE": w_rmse,
                "bias": bias,
                "abs_bias": abs_bias,
                "bias_pct": bias_pct,
                "abs_bias_pct": abs_bias_pct,
            }
        )

    grouped = monthly.groupby([COL_PRODUCT, COL_DIVISION, "Model"], dropna=False)
    metrics_df = grouped.apply(_group_metrics)
    group_cols = [COL_PRODUCT, COL_DIVISION, "Model"]
    cols_to_drop = [c for c in group_cols if c in metrics_df.columns]
    if cols_to_drop:
        metrics_df = metrics_df.drop(columns=cols_to_drop)
    metrics_df = metrics_df.reset_index()
    return metrics_df


def select_recommended_model(
    metrics_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    mae_tolerance: float = MAE_TOLERANCE,
    rmse_tolerance: float = RMSE_TOLERANCE,
    rocv_hard_multiplier: float = ROCV_HARD_MULTIPLIER,
    da_min_periods: int = DA_MIN_PERIODS,
    da_improvement_pp: float = DA_IMPROVEMENT_PP,
    da_close_mae_tol: float = DA_CLOSE_MAE_TOL,
    bias_use_pct: bool = True,
    bias_improvement_ratio: Optional[float] = 0.70,
    bias_improvement_abs_pp: Optional[float] = None,
    mae_close_tol: float = 0.10,
    rmse_close_tol: Optional[float] = None,
    return_rankings: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    if metrics_df.empty:
        summary = pd.DataFrame(columns=[
            COL_PRODUCT, COL_DIVISION, "Recommended_Model", "Baseline_Model", "Reason",
            "baseline_mae", "baseline_rmse", "baseline_rocv", "baseline_DA",
            "baseline_abs_bias", "baseline_abs_bias_pct",
            "mae_cutoff", "rmse_cutoff", "rocv_hard_max",
            "recommended_mae", "recommended_rmse", "recommended_rocv", "recommended_DA",
            "recommended_bias", "recommended_abs_bias", "recommended_bias_pct", "recommended_abs_bias_pct",
        ])
        return (summary, pd.DataFrame()) if return_rankings else summary

    metrics = metrics_df.copy()
    metrics["Test_MAE"] = pd.to_numeric(metrics["Test_MAE"], errors="coerce")
    metrics["Test_RMSE"] = pd.to_numeric(metrics["Test_RMSE"], errors="coerce")
    if "ROCV_MAE" in metrics.columns:
        metrics["ROCV_MAE"] = pd.to_numeric(metrics["ROCV_MAE"], errors="coerce")
    else:
        metrics["ROCV_MAE"] = np.nan
    metrics["Model"] = metrics["Model"].astype(str)

    da_df = compute_directional_accuracy(ts_df, min_periods=da_min_periods)
    weighted_df = compute_weighted_holdout_metrics(ts_df)
    metrics = metrics.merge(
        da_df, on=[COL_PRODUCT, COL_DIVISION, "Model"], how="left"
    ).merge(
        weighted_df, on=[COL_PRODUCT, COL_DIVISION, "Model"], how="left"
    )
    metrics["Raw_Test_MAE"] = metrics["Test_MAE"]
    metrics["Raw_Test_RMSE"] = metrics["Test_RMSE"]
    metrics["Test_MAE"] = metrics["Weighted_MAE"]
    metrics["Test_RMSE"] = metrics["Weighted_RMSE"]
    metrics["Test_MAE"] = pd.to_numeric(metrics["Test_MAE"], errors="coerce")
    metrics["Test_RMSE"] = pd.to_numeric(metrics["Test_RMSE"], errors="coerce")

    summary_rows = []
    ranking_rows = []

    for (prod, div), group in metrics.groupby([COL_PRODUCT, COL_DIVISION], dropna=False):
        group = group.copy()
        valid = group[np.isfinite(group["Test_MAE"])]
        if valid.empty:
            summary_rows.append({
                COL_PRODUCT: prod,
                COL_DIVISION: div,
                "Recommended_Model": None,
                "Baseline_Model": None,
                "Reason": "No valid model metrics available.",
                "baseline_mae": np.nan,
                "baseline_rmse": np.nan,
                "baseline_rocv": np.nan,
                "baseline_DA": np.nan,
                "mae_cutoff": np.nan,
                "rmse_cutoff": np.nan,
                "rocv_hard_max": np.nan,
                "recommended_mae": np.nan,
                "recommended_rmse": np.nan,
                "recommended_rocv": np.nan,
                "recommended_DA": np.nan,
            })
            continue

        baseline_idx = min(valid.index, key=lambda idx: _model_tiebreak_key_df(valid.loc[idx]))
        baseline = valid.loc[baseline_idx]
        baseline_model = baseline["Model"]
        baseline_mae = baseline["Test_MAE"]
        baseline_rmse = baseline["Test_RMSE"]
        baseline_rocv = baseline["ROCV_MAE"]
        baseline_da = baseline.get("DA", np.nan)
        baseline_abs_bias = baseline.get("abs_bias", np.nan)
        baseline_abs_bias_pct = baseline.get("abs_bias_pct", np.nan)

        mae_cutoff = baseline_mae * (1.0 + mae_tolerance)
        rmse_cutoff = baseline_rmse * (1.0 + rmse_tolerance)

        rocv_checks_enabled = np.isfinite(baseline_rocv) and baseline_rocv > 0
        rocv_hard_max = baseline_rocv * rocv_hard_multiplier if rocv_checks_enabled else np.nan

        def _passes_accuracy(row: pd.Series) -> bool:
            if row["Model"] == baseline_model:
                return True
            return bool(np.isfinite(row["Test_MAE"]) and np.isfinite(row["Test_RMSE"])
                        and row["Test_MAE"] <= mae_cutoff and row["Test_RMSE"] <= rmse_cutoff)

        def _passes_rocv(row: pd.Series) -> bool:
            if row["Model"] == baseline_model:
                return True
            if not rocv_checks_enabled:
                return True
            rocv = row.get("ROCV_MAE", np.nan)
            return bool(np.isfinite(rocv) and rocv <= rocv_hard_max)

        group["passes_accuracy"] = group.apply(_passes_accuracy, axis=1)
        group["passes_rocv_sanity"] = group.apply(_passes_rocv, axis=1)
        group["candidate"] = group["passes_accuracy"] & group["passes_rocv_sanity"]

        candidates = group[group["candidate"]]
        if candidates.empty:
            recommended = baseline
            reason = "Fallback to baseline: no model passed accuracy+ROCV sanity."
        else:
            tentative_idx = min(candidates.index, key=lambda idx: _model_tiebreak_key_df(candidates.loc[idx]))
            recommended = candidates.loc[tentative_idx]
            reason = "Selected best MAE among accuracy-qualified models passing ROCV sanity."

            challengers = candidates[candidates["Model"] != baseline_model].copy()
            if not challengers.empty and np.isfinite(baseline_da):
                challengers["da_valid"] = np.isfinite(challengers["DA"])
                challengers = challengers[challengers["da_valid"]]
                challengers = challengers[
                    challengers["DA"] >= baseline_da + da_improvement_pp
                ]
                challengers = challengers[
                    challengers["Test_MAE"] <= baseline_mae * (1.0 + da_close_mae_tol)
                ]
                if not challengers.empty:
                    best_idx = challengers.sort_values(
                        by=["DA", "Test_MAE", "Test_RMSE", "Model"],
                        ascending=[False, True, True, True],
                    ).index[0]
                    recommended = challengers.loc[best_idx]
                    delta_pp = (recommended["DA"] - baseline_da) * 100.0
                    reason = (
                        "Override: challenger improved directional accuracy by "
                        f"{delta_pp:.1f}pp (DA_challenger vs DA_baseline) "
                        "while within MAE+5% and passing accuracy+ROCV sanity."
                    )
                elif recommended["Model"] == baseline_model:
                    reason = "Baseline wins: best MAE; no challenger improved DA by >=10pp within MAE+5%."
            elif recommended["Model"] == baseline_model:
                reason = "Baseline wins: best MAE; no challenger improved DA by >=10pp within MAE+5%."

        used_bias_override = False
        bias_metric = "abs_bias_pct" if bias_use_pct else "abs_bias"
        bias_value = "bias_pct" if bias_use_pct else "bias"
        baseline_bias_metric = baseline.get(bias_metric, np.nan)

        group["qualifies_mae_close"] = False
        group["qualifies_bias"] = False
        group["qualifies_rmse_close"] = False

        if candidates.empty or bias_improvement_ratio is None:
            bias_challengers = pd.DataFrame()
        else:
            challengers = candidates[candidates["Model"] != baseline_model].copy()
            if challengers.empty:
                bias_challengers = pd.DataFrame()
            else:
                challengers["qualifies_mae_close"] = challengers["Test_MAE"] <= baseline_mae * (1.0 + mae_close_tol)
                if rmse_close_tol is not None and np.isfinite(baseline_rmse):
                    challengers["qualifies_rmse_close"] = (
                        challengers["Test_RMSE"] <= baseline_rmse * (1.0 + rmse_close_tol)
                    )
                else:
                    challengers["qualifies_rmse_close"] = True

                if not np.isfinite(baseline_bias_metric) or baseline_bias_metric <= 0:
                    challengers["qualifies_bias"] = False
                else:
                    challengers["qualifies_bias"] = challengers[bias_metric] <= (
                        baseline_bias_metric * bias_improvement_ratio
                    )
                    if bias_improvement_abs_pp is not None:
                        challengers["qualifies_bias"] &= (
                            (baseline_bias_metric - challengers[bias_metric]) >= bias_improvement_abs_pp
                        )

                bias_challengers = challengers[
                    challengers["qualifies_mae_close"]
                    & challengers["qualifies_bias"]
                    & challengers["qualifies_rmse_close"]
                ]

                group.loc[challengers.index, "qualifies_mae_close"] = challengers["qualifies_mae_close"]
                group.loc[challengers.index, "qualifies_bias"] = challengers["qualifies_bias"]
                group.loc[challengers.index, "qualifies_rmse_close"] = challengers["qualifies_rmse_close"]

        if not bias_challengers.empty:
            bias_idx = bias_challengers.sort_values(
                by=[bias_metric, "Test_MAE", "Test_RMSE", "Model"],
                ascending=[True, True, True, True],
            ).index[0]
            bias_selected = bias_challengers.loc[bias_idx]
            recommended = bias_selected
            used_bias_override = True
            reduction = 1.0 - (bias_selected[bias_metric] / baseline_bias_metric)
            reason = (
                "Bias override: |bias| improved by "
                f"{reduction * 100:.1f}% vs baseline while MAE within +10% "
                "and passing accuracy+ROCV sanity."
            )
            if bias_improvement_abs_pp is not None and bias_use_pct:
                reason = (
                    reason.rstrip(".")
                    + f" (>= {bias_improvement_abs_pp:.2%} abs bias pct improvement)."
                )
        else:
            if candidates.empty or bias_improvement_ratio is None:
                pass
            else:
                had_challengers = candidates[candidates["Model"] != baseline_model]
                if had_challengers.empty:
                    pass
                else:
                    any_mae_close = bool(had_challengers["Test_MAE"].le(baseline_mae * (1.0 + mae_close_tol)).any())
                    any_bias_ok = (
                        bool(
                            had_challengers[bias_metric]
                            .le(baseline_bias_metric * bias_improvement_ratio)
                            .any()
                        )
                        if np.isfinite(baseline_bias_metric) and baseline_bias_metric > 0
                        else False
                    )
                    if not any_mae_close:
                        reason = reason + " Bias override not applied: no challenger within MAE +10%."
                    elif not any_bias_ok:
                        reason = reason + " Bias override not applied: no challenger met bias-improvement threshold."

        group["baseline_model"] = baseline_model
        group["baseline_mae"] = baseline_mae
        group["baseline_rmse"] = baseline_rmse
        group["baseline_rocv"] = baseline_rocv
        group["baseline_DA"] = baseline_da
        group["baseline_abs_bias"] = baseline_abs_bias
        group["baseline_abs_bias_pct"] = baseline_abs_bias_pct
        group["mae_cutoff"] = mae_cutoff
        group["rmse_cutoff"] = rmse_cutoff
        group["rocv_hard_max"] = rocv_hard_max
        group["recommended_model"] = recommended["Model"]
        group["recommended_by_override"] = recommended["Model"] != baseline_model and reason.startswith("Override:")
        group["used_bias_override"] = used_bias_override

        ranking_rows.append(group)

        summary_rows.append({
            COL_PRODUCT: prod,
            COL_DIVISION: div,
            "Recommended_Model": recommended["Model"],
            "Baseline_Model": baseline_model,
            "Reason": reason,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_rocv": baseline_rocv,
            "baseline_DA": baseline_da,
            "baseline_abs_bias": baseline_abs_bias,
            "baseline_abs_bias_pct": baseline_abs_bias_pct,
            "mae_cutoff": mae_cutoff,
            "rmse_cutoff": rmse_cutoff,
            "rocv_hard_max": rocv_hard_max,
            "recommended_mae": recommended["Test_MAE"],
            "recommended_rmse": recommended["Test_RMSE"],
            "recommended_rocv": recommended.get("ROCV_MAE", np.nan),
            "recommended_DA": recommended.get("DA", np.nan),
            "recommended_bias": recommended.get("bias", np.nan),
            "recommended_abs_bias": recommended.get("abs_bias", np.nan),
            "recommended_bias_pct": recommended.get("bias_pct", np.nan),
            "recommended_abs_bias_pct": recommended.get("abs_bias_pct", np.nan),
            "used_bias_override": used_bias_override,
        })

    summary_df = pd.DataFrame(summary_rows)
    ranking_df = pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()

    return (summary_df, ranking_df) if return_rankings else summary_df


def choose_recommended_model(
    metrics_list,
    mae_tolerance: float = MAE_TOLERANCE,
    rmse_tolerance: float = RMSE_TOLERANCE,
    rocv_hard_multiplier: float = ROCV_HARD_MULTIPLIER,
    rocv_preferred_lower: float = ROCV_PREFERRED_LOWER,
    rocv_preferred_upper: float = ROCV_PREFERRED_UPPER,
    rocv_too_smooth: float = ROCV_TOO_SMOOTH,
    rocv_too_noisy: float = ROCV_TOO_NOISY,
):
    """
    We anchor model selection on the most accurate forecast, and only deviate from it
    when another model is similarly accurate and exhibits believable demand variability.
    """
    if not metrics_list:
        return None, None, "No model metrics available.", {}

    valid = [m for m in metrics_list if np.isfinite(m.get("Test_MAE", np.nan))]
    if not valid:
        return None, None, "No model metrics available.", {}

    baseline = min(valid, key=_model_tiebreak_key)
    baseline_mae = baseline.get("Test_MAE", np.nan)
    baseline_rmse = baseline.get("Test_RMSE", np.nan)
    baseline_rocv = baseline.get("ROCV_MAE", np.nan)

    mae_cutoff = baseline_mae * (1.0 + mae_tolerance)
    rmse_cutoff = baseline_rmse * (1.0 + rmse_tolerance)

    decision = {
        "baseline_model": baseline.get("Model"),
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_rocv": baseline_rocv,
        "mae_cutoff": mae_cutoff,
        "rmse_cutoff": rmse_cutoff,
        "rocv_hard_max": np.nan,
        "rocv_preferred_min": np.nan,
        "rocv_preferred_max": np.nan,
        "flags_by_model": {},
    }

    if np.isnan(baseline_rocv) or baseline_rocv <= 0:
        decision["reason"] = "Baseline ROCV unavailable or zero; selected lowest-MAE baseline."
        for m in valid:
            model_name = m.get("Model")
            decision["flags_by_model"][model_name] = {
                "passes_accuracy": model_name == baseline.get("Model"),
                "passes_rocv_hard": model_name == baseline.get("Model"),
                "in_preferred_band": False,
            }
        return baseline, baseline, decision["reason"], decision

    rocv_hard_max = baseline_rocv * rocv_hard_multiplier
    rocv_preferred_min = baseline_rocv * rocv_preferred_lower
    rocv_preferred_max = baseline_rocv * rocv_preferred_upper
    decision.update({
        "rocv_hard_max": rocv_hard_max,
        "rocv_preferred_min": rocv_preferred_min,
        "rocv_preferred_max": rocv_preferred_max,
    })

    candidates = []
    preferred_candidates = []
    for m in valid:
        model_name = m.get("Model")
        mae = m.get("Test_MAE", np.nan)
        rmse = m.get("Test_RMSE", np.nan)
        rocv = m.get("ROCV_MAE", np.nan)

        passes_accuracy = bool(np.isfinite(mae) and np.isfinite(rmse) and mae <= mae_cutoff and rmse <= rmse_cutoff)
        passes_rocv_hard = bool(np.isfinite(rocv) and rocv <= rocv_hard_max)
        if model_name == baseline.get("Model"):
            passes_accuracy = True
            passes_rocv_hard = True

        in_preferred_band = bool(np.isfinite(rocv) and rocv_preferred_min <= rocv <= rocv_preferred_max)

        decision["flags_by_model"][model_name] = {
            "passes_accuracy": passes_accuracy,
            "passes_rocv_hard": passes_rocv_hard,
            "in_preferred_band": in_preferred_band,
        }

        if passes_accuracy and passes_rocv_hard:
            candidates.append(m)
            if in_preferred_band:
                preferred_candidates.append(m)

    if not candidates:
        decision["reason"] = (
            "No alternative model met accuracy and variability sanity thresholds; "
            "selected lowest-MAE baseline."
        )
        return baseline, baseline, decision["reason"], decision

    if preferred_candidates:
        selected = min(preferred_candidates, key=_model_tiebreak_key)
        decision["reason"] = "Selected among accurate models with variability aligned to the baseline forecast."
    else:
        selected = min(candidates, key=_model_tiebreak_key)
        decision["reason"] = "Selected most accurate model among those passing variability sanity checks."

    if selected.get("Model") != baseline.get("Model"):
        selected_rocv = selected.get("ROCV_MAE", np.nan)
        too_smooth = np.isfinite(selected_rocv) and selected_rocv < baseline_rocv * rocv_too_smooth
        too_noisy = np.isfinite(selected_rocv) and selected_rocv > baseline_rocv * rocv_too_noisy
        if too_smooth or too_noisy:
            if too_smooth and too_noisy:
                decision["reason"] = (
                    "Selected model ROCV was outside acceptable smoothness/noise bounds; "
                    "reverted to lowest-MAE baseline."
                )
            elif too_smooth:
                decision["reason"] = (
                    "Selected model ROCV was too smooth relative to baseline; "
                    "reverted to lowest-MAE baseline."
                )
            else:
                decision["reason"] = (
                    "Selected model ROCV was too noisy relative to baseline; "
                    "reverted to lowest-MAE baseline."
                )
            return baseline, baseline, decision["reason"], decision

    return selected, baseline, decision["reason"], decision


def _model_type(model_name: str) -> str:
    if model_name == "ML_GBR":
        return "ML"
    if model_name == PIPELINE_ML_MODEL_NAME:
        return "ML"
    if model_name == "PROPHET":
        return "PROPHET"
    if model_name == "ETS_baseline":
        return "ETS"
    if model_name == "SBA":
        return "SBA"
    if model_name == BLEND_MODEL_NAME:
        return "BLEND"
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


def merge_summary_regressors(df_all: pd.DataFrame, summary_full: pd.DataFrame) -> pd.DataFrame:
    if summary_full.empty:
        return df_all

    summary = summary_full.copy()
    summary["Product_norm"] = summary["group_key"].apply(normalize_code)
    summary["Division_str"] = summary["BU"].astype(str).apply(normalize_code)
    summary = summary.rename(columns={"Month": COL_DATE})

    summary_cols = [
        col
        for col in summary.columns
        if col not in {"group_key", "BU", "Product_norm", "Division_str", COL_DATE}
    ]
    merge_cols = ["Product_norm", "Division_str", COL_DATE] + summary_cols
    summary = summary[merge_cols]

    df_all = df_all.copy()
    df_all["Product_norm"] = df_all[COL_PRODUCT].apply(normalize_code)
    df_all["Division_str"] = df_all[COL_DIVISION].astype(str).apply(normalize_code)
    df_all = df_all.merge(
        summary,
        on=["Product_norm", "Division_str", COL_DATE],
        how="left",
        suffixes=("", "_summary"),
    )

    for col in summary_cols:
        summary_col = f"{col}_summary"
        if summary_col in df_all.columns:
            df_all[col] = df_all[summary_col]
            df_all = df_all.drop(columns=[summary_col])

    df_all = df_all.drop(columns=["Product_norm", "Division_str"], errors="ignore")
    return df_all


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
        overrides[normalize_key(prod, div)] = (order, seas_order)
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
        orders_map[normalize_key(prod, div)] = (order, seas_order)

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
    cutoff = last_completed_month()
    df_all = filter_to_completed_months(df_all, cutoff=cutoff)
    print(f"Filtering actuals/bookings to completed months through {cutoff.date()}.")
    df_all = mark_prelaunch_actuals_as_missing(
        df_all,
        output_excel_path=REVISED_ACTUALS_FILE,
        output_sheet=REVISED_ACTUALS_SHEET,
    )
    summary_full = pd.DataFrame()
    if SUMMARY_PATH.exists():
        summary_full = load_summary_report(SUMMARY_PATH)
    else:
        print(f"[WARN] Summary file not found: {SUMMARY_PATH}")
    df_all = merge_summary_regressors(df_all, summary_full)
    print(f"Loading per-SKU SARIMA orders from {ORDER_FILE}...")
    orders_map = load_orders_map(ORDER_FILE, overrides_path=NOTES_FILE)

    catalog = pd.read_excel(PRODUCT_CATALOG_PATH)
    if "group_key" in catalog.columns:
        catalog["group_key_norm"] = catalog["group_key"].apply(normalize_code)
    if "business_unit_code" in catalog.columns:
        catalog["business_unit_code_str"] = catalog["business_unit_code"].astype(str).apply(normalize_code)

    pipeline_all = pd.DataFrame()
    if PIPELINE_FILES:
        pipeline_frames = []
        for path in PIPELINE_FILES:
            df = pd.read_excel(path)
            df["_source_file"] = path.name
            pipeline_frames.append(df)
        pipeline_all = pd.concat(pipeline_frames, ignore_index=True)
    if not pipeline_all.empty:
        if "Month" in pipeline_all.columns and "snapshot_month" not in pipeline_all.columns:
            pipeline_all["snapshot_month"] = parse_snapshot_month(pipeline_all["Month"])
        if "Close Date" in pipeline_all.columns and "target_month" not in pipeline_all.columns:
            pipeline_all["target_month"] = parse_close_date(pipeline_all["Close Date"])
        numeric_cols = [
            "Quantity",
            "Quantity W/ Decimal",
            "Probability",
            "Total Price",
            "Factored Quantity",
            "Factored Revenue",
        ]
        for col in numeric_cols:
            if col in pipeline_all.columns:
                pipeline_all[col] = pd.to_numeric(pipeline_all[col], errors="coerce")
        if "Product Code" in pipeline_all.columns:
            pipeline_all["Product_Code_num"] = pd.to_numeric(
                pipeline_all["Product Code"], errors="coerce"
            )
            pipeline_all["Product_Code_norm"] = pipeline_all["Product Code"].apply(
                normalize_code
            )
        if "Current OSC Product Name" in pipeline_all.columns:
            pipeline_all["Current_OSC_Product_Name_norm"] = pipeline_all[
                "Current OSC Product Name"
            ].apply(normalize_code)
    pipeline_preprocessed = (
        not pipeline_all.empty
        and "snapshot_month" in pipeline_all.columns
        and "target_month" in pipeline_all.columns
    )

    sf_reference = load_sf_product_reference()
    sf_code_map: Dict[Tuple[str, str], List[str]] = {}
    if not sf_reference.empty:
        sf_reference = sf_reference.copy()
        sf_reference["salesforce_product_name"] = sf_reference[
            "salesforce_product_name"
        ].astype(str)
        grouped = sf_reference.groupby(["group_key", "business_unit_code"])
        sf_code_map = {
            (normalize_code(group_key), normalize_code(div)): list(
                group["salesforce_product_name"].unique()
            )
            for (group_key, div), group in grouped
        }

    # Basic cleaning
    df_all[COL_DATE] = pd.to_datetime(df_all[COL_DATE])
    df_all = df_all.sort_values([COL_PRODUCT, COL_DIVISION, COL_DATE])

    results_rows = []
    recommended_rows = []
    ranking_rows_all = []
    holdout_rows_all = []
    blend_weight_rows_all = []
    skipped_no_order = []
    skipped_products = []

    # Loop over SKUs (by Product + Division)
    sku_groups = df_all.groupby([COL_PRODUCT, COL_DIVISION], dropna=False)
    total_skus = len(sku_groups)
    print(f"Found {total_skus} product+division combinations.")

    for idx, ((prod, div), df_sku) in enumerate(sku_groups, start=1):
        models_ran = []
        models_skipped = []
        sku_log(f"\n=== [{idx}/{total_skus}] Processing SKU: {prod}, Division: {div} ===")
        order_pair = orders_map.get(normalize_key(prod, div))
        if order_pair is None:
            sku_log("  -> Skipping: no order/seasonal_order found in sarimax_order_search_summary.")
            skipped_no_order.append({"Product": prod, "Division": div, "Reason": "No order in order search summary"})
            skipped_products.append(
                {
                    "Product": prod,
                    "Division": div,
                    "Model": "ALL",
                    "Reason": "No order in order search summary.",
                }
            )
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
            recommended_rows.append({
                "Product": prod,
                "Division": div,
                "Recommended_Model": "NO_ORDER_IN_FILE",
                "Reason": "No order in order search summary.",
                "baseline_model": None,
                "baseline_mae": np.nan,
                "baseline_rocv": np.nan,
                "baseline_abs_bias": np.nan,
                "baseline_abs_bias_pct": np.nan,
                "mae_cutoff": np.nan,
                "rmse_cutoff": np.nan,
                "rocv_hard_max": np.nan,
                "rocv_preferred_min": np.nan,
                "rocv_preferred_max": np.nan,
                "recommended_bias": np.nan,
                "recommended_abs_bias": np.nan,
                "recommended_bias_pct": np.nan,
                "recommended_abs_bias_pct": np.nan,
                "used_bias_override": False,
            })
            print(format_sku_summary(
                idx,
                total_skus,
                prod,
                div,
                ran=models_ran,
                skipped=["NO_ORDER_IN_FILE"],
                chosen="NO_ORDER_IN_FILE",
            ))
            continue

        order, seasonal_order = order_pair
        df_sku = df_sku.sort_values(COL_DATE).set_index(COL_DATE)

        summary_product = filter_summary_product(summary_full, prod, div)
        feature_mode = None
        pipeline_history = pd.DataFrame()
        pipeline_skip_reason = ""
        if enable_ml:
            try:
                feature_mode = load_feature_mode(prod, str(div), catalog=catalog)
            except RuntimeError as exc:
                pipeline_skip_reason = str(exc)
            sf_codes = sf_code_map.get((normalize_code(prod), normalize_code(div)), [])
            if not pipeline_all.empty:
                pipeline_history = load_pipeline_history(
                    prod,
                    str(div),
                    sf_codes=sf_codes,
                    pipeline_all=pipeline_all,
                    pipeline_preprocessed=pipeline_preprocessed,
                )
            else:
                pipeline_skip_reason = pipeline_skip_reason or "No pipeline files available."

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
            sku_log(f"  -> Skipping (only {n} observations; need at least {MIN_OBS_TOTAL}).")
            skipped_products.append(
                {
                    "Product": prod,
                    "Division": div,
                    "Model": "ALL",
                    "Reason": f"Insufficient history (<{MIN_OBS_TOTAL} observations).",
                }
            )
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
            recommended_rows.append({
                "Product": prod,
                "Division": div,
                "Recommended_Model": "INSUFFICIENT_HISTORY",
                "Reason": f"Insufficient history (<{MIN_OBS_TOTAL} observations).",
                "baseline_model": None,
                "baseline_mae": np.nan,
                "baseline_rocv": np.nan,
                "baseline_abs_bias": np.nan,
                "baseline_abs_bias_pct": np.nan,
                "mae_cutoff": np.nan,
                "rmse_cutoff": np.nan,
                "rocv_hard_max": np.nan,
                "rocv_preferred_min": np.nan,
                "rocv_preferred_max": np.nan,
                "recommended_bias": np.nan,
                "recommended_abs_bias": np.nan,
                "recommended_bias_pct": np.nan,
                "recommended_abs_bias_pct": np.nan,
                "used_bias_override": False,
            })
            print(format_sku_summary(
                idx,
                total_skus,
                prod,
                div,
                ran=models_ran,
                skipped=[f"INSUFFICIENT_HISTORY<{MIN_OBS_TOTAL}"],
                chosen="INSUFFICIENT_HISTORY",
            ))
            continue

        metrics_list = []
        ranking_rows = []
        holdout_rows = []

        # --- Baseline model ---
        if allow_sarima:
            models_ran.append("SARIMA")
            sku_log("  Fitting SARIMA_baseline...")
            baseline_metrics, holdout_df = evaluate_single_model(
                y=y,
                exog=None,
                model_name="SARIMA_baseline",
                horizon=test_horizon,
                order=order,
                seasonal_order=seasonal_order,
                sku=prod,
                bu=div,
                rocv_spec=baseline_rocv_spec,
                return_holdout=True,
            )
            if not holdout_df.empty:
                holdout_rows.append(holdout_df)
                holdout_rows_all.append(holdout_df)
            if (
                baseline_metrics is not None
                and not np.isnan(baseline_metrics.get("Test_MAE", np.nan))
                and abs(baseline_metrics["Test_MAE"]) <= BASELINE_MAE_ZERO_EPS
            ):
                sku_log(f"[INFO] Baseline MAE ~0 for {prod}/{div}; treating as perfect baseline.")
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
            models_skipped.append("SARIMA(gated)")
            sku_log("  Skipping SARIMA_baseline: history <12 months or first non-zero <12 months ago.")

        # --- ETS baseline ---
        models_ran.append("ETS")
        sku_log("  Fitting ETS_baseline...")
        ets_metrics, holdout_df = evaluate_ets_model(
            y=y,
            model_name="ETS_baseline",
            horizon=test_horizon,
            sku=prod,
            bu=div,
            rocv_spec=baseline_rocv_spec,
            return_holdout=True,
        )
        if not holdout_df.empty:
            holdout_rows.append(holdout_df)
            holdout_rows_all.append(holdout_df)
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

        # --- SBA challenger ---
        models_ran.append("SBA")
        sku_log("  Fitting SBA...")
        sba_metrics, holdout_df = evaluate_sba_model(
            y=y,
            model_name="SBA",
            horizon=test_horizon,
            sku=prod,
            bu=div,
            return_holdout=True,
        )
        if not holdout_df.empty:
            holdout_rows.append(holdout_df)
            holdout_rows_all.append(holdout_df)
        metrics_list.append(sba_metrics)
        ranking_rows.append({
            "Product": prod,
            "Division": div,
            "Model": sba_metrics["Model"],
            "Test_MAE": sba_metrics["Test_MAE"],
            "Test_RMSE": sba_metrics["Test_RMSE"],
            "ROCV_MAE": sba_metrics["ROCV_MAE"],
            "AIC": sba_metrics["AIC"],
            "BIC": sba_metrics["BIC"],
            "Regressor_Name": None,
            "Regressor_Lag": None,
            "Accepted_by_rules": False
        })

        # --- SARIMAX with New_Opportunities lags 1-3 ---
        sarimax_newopp_ran = False
        if allow_sarimax:
            for lag in [1, 2, 3]:
                col = f"New_Opportunities_l{lag}"
                if col in df_sku.columns:
                    sarimax_newopp_ran = True
                    sku_log(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_row, holdout_df = evaluate_single_model(
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
                        exog_invalid=exog_invalid,
                        return_holdout=True,
                    )
                    metrics_list.append(metrics_row)
                    if not holdout_df.empty:
                        holdout_rows.append(holdout_df)
                        holdout_rows_all.append(holdout_df)
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
            if sarimax_newopp_ran:
                models_ran.append("SARIMAX_NewOpp")
            else:
                models_skipped.append("SARIMAX_NewOpp(no_col)")
        else:
            models_skipped.append("SARIMAX_NewOpp(gated)")
            sku_log("  Skipping SARIMAX (New_Opportunities): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Open_Opportunities lags 1-3 ---
        sarimax_openopp_ran = False
        if allow_sarimax:
            for lag in [1, 2, 3]:
                col = f"Open_Opportunities_l{lag}"
                if col in df_sku.columns:
                    sarimax_openopp_ran = True
                    sku_log(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_row, holdout_df = evaluate_single_model(
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
                        exog_invalid=exog_invalid,
                        return_holdout=True,
                    )
                    metrics_list.append(metrics_row)
                    if not holdout_df.empty:
                        holdout_rows.append(holdout_df)
                        holdout_rows_all.append(holdout_df)
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
            if sarimax_openopp_ran:
                models_ran.append("SARIMAX_OpenOpp")
            else:
                models_skipped.append("SARIMAX_OpenOpp(no_col)")
        else:
            models_skipped.append("SARIMAX_OpenOpp(gated)")
            sku_log("  Skipping SARIMAX (Open_Opportunities): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Bookings lags 1-2 ---
        sarimax_bookings_ran = False
        if allow_sarimax:
            for lag in [1, 2]:
                col = f"Bookings_l{lag}"
                if col in df_sku.columns:
                    sarimax_bookings_ran = True
                    sku_log(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_row, holdout_df = evaluate_single_model(
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
                        exog_invalid=exog_invalid,
                        return_holdout=True,
                    )
                    metrics_list.append(metrics_row)
                    if not holdout_df.empty:
                        holdout_rows.append(holdout_df)
                        holdout_rows_all.append(holdout_df)
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
            if sarimax_bookings_ran:
                models_ran.append("SARIMAX_Bookings")
            else:
                models_skipped.append("SARIMAX_Bookings(no_col)")
        else:
            models_skipped.append("SARIMAX_Bookings(gated)")
            sku_log("  Skipping SARIMAX (Bookings): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Median_Months_Since_Last_Activity lags 1-2 ---
        sarimax_median_ran = False
        if allow_sarimax:
            for lag in [1, 2]:
                col = f"Median_Months_Since_Last_Activity_l{lag}"
                if col in df_sku.columns:
                    sarimax_median_ran = True
                    sku_log(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_row, holdout_df = evaluate_single_model(
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
                        exog_invalid=exog_invalid,
                        return_holdout=True,
                    )
                    metrics_list.append(metrics_row)
                    if not holdout_df.empty:
                        holdout_rows.append(holdout_df)
                        holdout_rows_all.append(holdout_df)
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
            if sarimax_median_ran:
                models_ran.append("SARIMAX_MedianMo")
            else:
                models_skipped.append("SARIMAX_MedianMo(no_col)")
        else:
            models_skipped.append("SARIMAX_MedianMo(gated)")
            sku_log("  Skipping SARIMAX (Median_Months_Since_Last_Activity): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Open_Not_Modified_90_Days lag 1 ---
        sarimax_open90_ran = False
        if allow_sarimax:
            for lag in [1]:
                col = f"Open_Not_Modified_90_Days_l{lag}"
                if col in df_sku.columns:
                    sarimax_open90_ran = True
                    sku_log(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_row, holdout_df = evaluate_single_model(
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
                        exog_invalid=exog_invalid,
                        return_holdout=True,
                    )
                    metrics_list.append(metrics_row)
                    if not holdout_df.empty:
                        holdout_rows.append(holdout_df)
                        holdout_rows_all.append(holdout_df)
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
            if sarimax_open90_ran:
                models_ran.append("SARIMAX_Open90")
            else:
                models_skipped.append("SARIMAX_Open90(no_col)")
        else:
            models_skipped.append("SARIMAX_Open90(gated)")
            sku_log("  Skipping SARIMAX (Open_Not_Modified_90_Days): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Pct_Open_Not_Modified_90_Days lags 1-2 ---
        sarimax_pctopen_ran = False
        if allow_sarimax:
            for lag in [1, 2]:
                col = f"Pct_Open_Not_Modified_90_Days_l{lag}"
                if col in df_sku.columns:
                    sarimax_pctopen_ran = True
                    sku_log(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_row, holdout_df = evaluate_single_model(
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
                        exog_invalid=exog_invalid,
                        return_holdout=True,
                    )
                    metrics_list.append(metrics_row)
                    if not holdout_df.empty:
                        holdout_rows.append(holdout_df)
                        holdout_rows_all.append(holdout_df)
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
            if sarimax_pctopen_ran:
                models_ran.append("SARIMAX_PctOpen90")
            else:
                models_skipped.append("SARIMAX_PctOpen90(no_col)")
        else:
            models_skipped.append("SARIMAX_PctOpen90(gated)")
            sku_log("  Skipping SARIMAX (Pct_Open_Not_Modified_90_Days): history <36 months or SARIMA gate not met.")

        # --- SARIMAX with Early_to_Late_Ratio lags 1-2 ---
        sarimax_earlylate_ran = False
        if allow_sarimax:
            for lag in [1, 2]:
                col = f"Early_to_Late_Ratio_l{lag}"
                if col in df_sku.columns:
                    sarimax_earlylate_ran = True
                    sku_log(f"  Fitting SARIMAX_{col}...")
                    exog_series = df_sku[col].astype(float)
                    exog_clean_full = exog_series.ffill().bfill()
                    non_nan = exog_clean_full.dropna()
                    exog_invalid = (len(non_nan) == 0) or (non_nan.nunique() <= 1)
                    rocv_spec = None if exog_invalid else build_rocv_spec(y, exog_series=exog_clean_full)
                    metrics_row, holdout_df = evaluate_single_model(
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
                        exog_invalid=exog_invalid,
                        return_holdout=True,
                    )
                    metrics_list.append(metrics_row)
                    if not holdout_df.empty:
                        holdout_rows.append(holdout_df)
                        holdout_rows_all.append(holdout_df)
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
            if sarimax_earlylate_ran:
                models_ran.append("SARIMAX_EarlyLate")
            else:
                models_skipped.append("SARIMAX_EarlyLate(no_col)")
        else:
            models_skipped.append("SARIMAX_EarlyLate(gated)")
            sku_log("  Skipping SARIMAX (Early_to_Late_Ratio): history <36 months or SARIMA gate not met.")

        # --- ML challenger (GradientBoostingRegressor) ---
        if enable_ml:
            use_pipeline_ml = False
            ml_metrics = None
            pipeline_reason = pipeline_skip_reason
            if summary_product.empty and not pipeline_reason:
                pipeline_reason = "No summary data for product/BU."
            if pipeline_history.empty and not pipeline_reason:
                pipeline_reason = "No pipeline history rows."

            if not pipeline_reason and feature_mode is not None:
                models_ran.append("ML_PIPE")
                sku_log(f"  Evaluating {PIPELINE_ML_MODEL_NAME}...")
                actuals_df = df_sku.reset_index()[[COL_DATE, COL_ACTUALS]].copy()
                actuals_df = actuals_df.rename(columns={COL_DATE: "Month", COL_ACTUALS: "Actuals"})
                actuals_df["Month"] = month_start(actuals_df["Month"])
                actuals_series = actuals_df.set_index("Month")["Actuals"].sort_index()
                ml_metrics, holdout_df = evaluate_ml_pipeline_candidate(
                    actuals_df=actuals_df,
                    actuals_series=actuals_series,
                    pipeline_history=pipeline_history,
                    feature_mode=feature_mode,
                    summary_df=summary_product,
                    holdout_months=test_horizon,
                    sku=prod,
                    bu=div,
                    return_holdout=True,
                )
                if not holdout_df.empty:
                    holdout_rows.append(holdout_df)
                    holdout_rows_all.append(holdout_df)
                if ml_metrics.get("Skip_Reason"):
                    pipeline_reason = ml_metrics["Skip_Reason"]
                else:
                    use_pipeline_ml = True

            if pipeline_reason:
                models_skipped.append("ML_PIPE(skip)")
                skipped_products.append(
                    {
                        "Product": prod,
                        "Division": div,
                        "Model": PIPELINE_ML_MODEL_NAME,
                        "Reason": pipeline_reason,
                    }
                )

            if not use_pipeline_ml:
                models_ran.append("ML_GBR")
                sku_log("  Evaluating ML_GBR...")
                ml_metrics, holdout_df = evaluate_ml_candidate(
                    df_sku,
                    horizon=test_horizon,
                    sku=prod,
                    bu=div,
                    return_holdout=True,
                )
                if not holdout_df.empty:
                    holdout_rows.append(holdout_df)
                    holdout_rows_all.append(holdout_df)

            metrics_list.append(ml_metrics)
            if ml_metrics.get("Skip_Reason"):
                models_skipped.append("ML_GBR(skip)")
                sku_log(f"  -> ML skipped: {ml_metrics['Skip_Reason']}")
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
            models_ran.append("PROPHET")
            sku_log("  Evaluating PROPHET...")
            prophet_metrics, holdout_df = _evaluate_prophet_metrics(
                y, horizon=test_horizon, sku=prod, bu=div, return_holdout=True
            )
            metrics_list.append(prophet_metrics)
            if not holdout_df.empty:
                holdout_rows.append(holdout_df)
                holdout_rows_all.append(holdout_df)
            if prophet_metrics.get("Skip_Reason"):
                models_skipped.append("PROPHET(skip)")
                sku_log(f"  -> Prophet skipped: {prophet_metrics['Skip_Reason']}")
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

        holdout_df = pd.concat(holdout_rows, ignore_index=True) if holdout_rows else pd.DataFrame()
        blend_metrics, blend_holdout, weight_rows = _build_softmax_blend(
            holdout_df,
            metrics_list,
            tau=BLEND_SOFTMAX_TAU,
        )
        if blend_metrics is not None and not blend_holdout.empty:
            metrics_list.append(blend_metrics)
            holdout_rows.append(blend_holdout)
            holdout_rows_all.append(blend_holdout)
            blend_weight_rows_all.extend(weight_rows)
            ranking_rows.append({
                "Product": prod,
                "Division": div,
                "Model": blend_metrics["Model"],
                "Model_Type": _model_type(blend_metrics["Model"]),
                "Test_MAE": blend_metrics["Test_MAE"],
                "Test_RMSE": blend_metrics["Test_RMSE"],
                "ROCV_MAE": blend_metrics["ROCV_MAE"],
                "AIC": blend_metrics["AIC"],
                "BIC": blend_metrics["BIC"],
                "Regressor_Name": None,
                "Regressor_Lag": None,
                "Accepted_by_rules": False,
                "Skip_Reason": "",
            })
            holdout_df = pd.concat(holdout_rows, ignore_index=True)

        if not holdout_df.empty:
            wape_by_model = (
                holdout_df.groupby("Model", dropna=False)
                .apply(lambda g: compute_wape(g["Actual"], g["Forecast"]))
                .to_dict()
            )
            for row in ranking_rows:
                row["WAPE"] = wape_by_model.get(row["Model"], np.nan)

        # --- Choose best model using recommended logic ---
        metrics_df = pd.DataFrame(ranking_rows)
        selection_summary, ranking_df = select_recommended_model(
            metrics_df,
            holdout_df,
            mae_tolerance=MAE_TOLERANCE,
            rmse_tolerance=RMSE_TOLERANCE,
            rocv_hard_multiplier=ROCV_HARD_MULTIPLIER,
            da_min_periods=DA_MIN_PERIODS,
            da_improvement_pp=DA_IMPROVEMENT_PP,
            da_close_mae_tol=DA_CLOSE_MAE_TOL,
            return_rankings=True,
        )
        summary_row = selection_summary.iloc[0] if not selection_summary.empty else {}
        selection_reason = summary_row.get("Reason", "No valid model metrics available.")
        recommended_model_name = summary_row.get("Recommended_Model")
        baseline_model_name = summary_row.get("Baseline_Model")
        best_model = next((m for m in metrics_list if m.get("Model") == recommended_model_name), None)
        baseline = next((m for m in metrics_list if m.get("Model") == baseline_model_name), None)

        if not ranking_df.empty:
            ranking_lookup = ranking_df.set_index("Model")
            for row in ranking_rows:
                if row["Model"] in ranking_lookup.index:
                    flags = ranking_lookup.loc[row["Model"]]
                    row["passes_accuracy"] = bool(flags.get("passes_accuracy"))
                    row["passes_rocv_sanity"] = bool(flags.get("passes_rocv_sanity"))
                    row["candidate"] = bool(flags.get("candidate"))
                    row["qualifies_mae_close"] = bool(flags.get("qualifies_mae_close"))
                    row["qualifies_bias"] = bool(flags.get("qualifies_bias"))
                    row["qualifies_rmse_close"] = bool(flags.get("qualifies_rmse_close"))
                    row["used_bias_override"] = bool(flags.get("used_bias_override"))
                    row["DA"] = flags.get("DA", np.nan)
                    row["DA_Valid_Periods"] = flags.get("DA_Valid_Periods", np.nan)
                    row["bias"] = flags.get("bias", np.nan)
                    row["abs_bias"] = flags.get("abs_bias", np.nan)
                    row["bias_pct"] = flags.get("bias_pct", np.nan)
                    row["abs_bias_pct"] = flags.get("abs_bias_pct", np.nan)
                    row["Accepted_by_rules"] = bool(flags.get("candidate"))

        # Identify best SARIMAX candidate by Test_MAE for regressor fallback.
        sarimax_candidates = [
            m for m in metrics_list
            if isinstance(m.get("Model"), str) and m["Model"].startswith("SARIMAX")
            and not np.isnan(m.get("Test_MAE", np.nan))
        ]
        def mae_rmse_key_local(m):
            mae = m.get("Test_MAE", np.nan)
            rmse = m.get("Test_RMSE", np.nan)
            model_name = str(m.get("Model", ""))
            return (
                mae if np.isfinite(mae) else np.inf,
                rmse if np.isfinite(rmse) else np.inf,
                model_name,
            )

        best_nonbaseline = None
        if sarimax_candidates:
            best_nonbaseline = min(sarimax_candidates, key=mae_rmse_key_local)

        if best_model is None:
            sku_log("  -> No valid model metrics; marking as FAILED.")
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
            recommended_rows.append({
                "Product": prod,
                "Division": div,
                "Recommended_Model": "FAILED",
                "Reason": "No valid model metrics available.",
                "baseline_model": None,
                "baseline_mae": np.nan,
                "baseline_rocv": np.nan,
                "baseline_abs_bias": np.nan,
                "baseline_abs_bias_pct": np.nan,
                "mae_cutoff": np.nan,
                "rmse_cutoff": np.nan,
                "rocv_hard_max": np.nan,
                "rocv_preferred_min": np.nan,
                "rocv_preferred_max": np.nan,
                "recommended_bias": np.nan,
                "recommended_abs_bias": np.nan,
                "recommended_bias_pct": np.nan,
                "recommended_abs_bias_pct": np.nan,
                "used_bias_override": False,
            })
            print(format_sku_summary(
                idx,
                total_skus,
                prod,
                div,
                ran=models_ran,
                skipped=models_skipped,
                chosen="FAILED",
                note="No valid model metrics",
            ))
            continue

        # Baseline info
        baseline_mae = baseline["Test_MAE"] if baseline is not None else np.nan
        baseline_rmse = baseline["Test_RMSE"] if baseline is not None else np.nan
        baseline_rocv = baseline["ROCV_MAE"] if baseline is not None else np.nan
        baseline_da = summary_row.get("baseline_DA", np.nan) if isinstance(summary_row, pd.Series) else np.nan

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

        accepted_by_rules = bool(baseline is not None and best_model["Model"] != baseline.get("Model"))

        sku_log(f"  -> Chosen model: {best_model['Model']}")
        sku_log(
            f"     Baseline MAE: {baseline_mae:.3f} | Chosen MAE: {chosen_mae:.3f} | "
            f"Improvement: {mae_improvement_pct * 100 if not np.isnan(mae_improvement_pct) else np.nan:.2f}%"
        )

        model_type = _model_type(best_model["Model"]) if (enable_ml or enable_prophet) else None
        recommended_rows.append({
            "Product": prod,
            "Division": div,
            "Recommended_Model": best_model["Model"],
            "Reason": selection_reason,
            "baseline_model": summary_row.get("Baseline_Model"),
            "baseline_mae": summary_row.get("baseline_mae"),
            "baseline_rmse": summary_row.get("baseline_rmse"),
            "baseline_rocv": summary_row.get("baseline_rocv"),
            "baseline_DA": summary_row.get("baseline_DA"),
            "baseline_abs_bias": summary_row.get("baseline_abs_bias"),
            "baseline_abs_bias_pct": summary_row.get("baseline_abs_bias_pct"),
            "mae_cutoff": summary_row.get("mae_cutoff"),
            "rmse_cutoff": summary_row.get("rmse_cutoff"),
            "rocv_hard_max": summary_row.get("rocv_hard_max"),
            "recommended_mae": summary_row.get("recommended_mae"),
            "recommended_rmse": summary_row.get("recommended_rmse"),
            "recommended_rocv": summary_row.get("recommended_rocv"),
            "recommended_DA": summary_row.get("recommended_DA"),
            "recommended_bias": summary_row.get("recommended_bias"),
            "recommended_abs_bias": summary_row.get("recommended_abs_bias"),
            "recommended_bias_pct": summary_row.get("recommended_bias_pct"),
            "recommended_abs_bias_pct": summary_row.get("recommended_abs_bias_pct"),
            "used_bias_override": summary_row.get("used_bias_override"),
        })
        regressor_coef = best_model.get("Regressor_coef", best_model.get("Regressor_Coef", np.nan))
        regressor_pvalue = best_model.get("Regressor_pvalue", best_model.get("Regressor_Pvalue", np.nan))
        results_rows.append({
            "Product": prod,
            "Division": div,
            "Chosen_Model": best_model["Model"],
            **({"Model_Type": model_type} if (enable_ml or enable_prophet) else {}),
            "Best_NonBaseline_Model": best_nonbaseline_model,
            "Best_NonBaseline_Regressor_Name": best_nonbaseline_reg_name,
            "Best_NonBaseline_Regressor_Lag": best_nonbaseline_reg_lag,
            "Baseline_MAE": baseline_mae,
            "Baseline_RMSE": baseline_rmse,
            "Baseline_ROCV_MAE": baseline_rocv,
            "Baseline_DA": baseline_da,
            "Chosen_MAE": chosen_mae,
            "Chosen_RMSE": chosen_rmse,
            "Chosen_ROCV_MAE": chosen_rocv,
            "Chosen_DA": summary_row.get("recommended_DA") if isinstance(summary_row, pd.Series) else np.nan,
            "MAE_Improvement_Pct": mae_improvement_pct,
            "Regressor_Name": regressor_name,
            "Regressor_Lag": regressor_lag,
            "Regressor_Coef": regressor_coef,
            "Regressor_pvalue": regressor_pvalue,
            "Baseline_AIC": baseline["AIC"] if baseline is not None else np.nan,
            "Baseline_BIC": baseline["BIC"] if baseline is not None else np.nan,
            "Chosen_AIC": chosen_aic,
            "Chosen_BIC": chosen_bic,
            "Accepted_by_rules": accepted_by_rules
        })

        ranking_rows_all.extend(ranking_rows)
        print(format_sku_summary(
            idx,
            total_skus,
            prod,
            div,
            ran=models_ran,
            skipped=models_skipped,
            chosen=best_model["Model"],
        ))

    # ===========================
    # 4. EXPORT RESULTS
    # ===========================
    print("\nWriting Excel summary...")
    df_results = pd.DataFrame(results_rows)
    df_recommended = pd.DataFrame(recommended_rows)

    # Nice ordering of columns
    col_order = [
        "Product", "Division",
        "Chosen_Model",
    ]
    if enable_ml or enable_prophet:
        col_order.append("Model_Type")
    col_order.extend([
        "Best_NonBaseline_Model", "Accepted_by_rules",
        "Baseline_MAE", "Baseline_RMSE", "Baseline_ROCV_MAE", "Baseline_DA",
        "Chosen_MAE", "Chosen_RMSE", "Chosen_ROCV_MAE", "Chosen_DA",
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

    holdout_ts = pd.concat(holdout_rows_all, ignore_index=True) if holdout_rows_all else pd.DataFrame()
    if not holdout_ts.empty:
        holdout_ts["Date"] = pd.to_datetime(holdout_ts["Date"], errors="coerce")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        holdout_ts.to_csv(HOLDOUT_TS_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        df_results.to_excel(writer, index=False, sheet_name="Model_Summary")
        if not df_recommended.empty:
            recommended_cols = [
                "Product",
                "Division",
                "Recommended_Model",
                "Reason",
                "baseline_model",
                "baseline_mae",
                "baseline_rmse",
                "baseline_rocv",
                "baseline_DA",
                "mae_cutoff",
                "rmse_cutoff",
                "rocv_hard_max",
                "recommended_mae",
                "recommended_rmse",
                "recommended_rocv",
                "recommended_DA",
            ]
            for col in recommended_cols:
                if col not in df_recommended.columns:
                    df_recommended[col] = np.nan
            df_recommended = df_recommended[recommended_cols]
            df_recommended.to_excel(writer, index=False, sheet_name="recommended_model_summary")
        if not df_rankings.empty:
            df_rankings.to_excel(writer, index=False, sheet_name="Model_Rankings")
        if not holdout_ts.empty:
            holdout_ts.to_excel(writer, index=False, sheet_name=HOLDOUT_TS_SHEET)
        if blend_weight_rows_all:
            pd.DataFrame(blend_weight_rows_all).to_excel(
                writer, index=False, sheet_name="Blended_Softmax_Weights"
            )
        if skipped_no_order:
            pd.DataFrame(skipped_no_order).to_excel(
                writer, index=False, sheet_name="Skipped_No_Order"
            )

    if skipped_products:
        skipped_path = BASE_DIR / "skipped_products.csv"
        pd.DataFrame(skipped_products).to_csv(skipped_path, index=False)

    if skipped_no_order:
        print("\nSkipped SKUs (no order found):")
        for row in skipped_no_order:
            print(f"  - Product: {row['Product']}, Division: {row['Division']}")

    if not holdout_ts.empty:
        print(f"Holdout forecast-vs-actuals written to: {HOLDOUT_TS_CSV}")

    if args.compare_model_selection:
        try:
            import compare_model_selection
            compare_model_selection.main()
        except Exception as exc:
            print(f"[WARN] Model selection comparison failed: {exc}")

    print(f"\nDone. Summary written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    elapsed = time.perf_counter() - t0
    minutes = elapsed / 60.0
    print(f"\nTotal runtime: {elapsed:,.2f} seconds ({minutes:,.2f} minutes)")
