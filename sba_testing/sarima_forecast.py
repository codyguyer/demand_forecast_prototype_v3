import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import ast
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False


# ===========================
# CONFIG
# ===========================

BASE_DIR = Path(__file__).resolve().parent
SALESFORCE_DATA_DIR = BASE_DIR / "salesforce_data"
if not SALESFORCE_DATA_DIR.exists():
    SALESFORCE_DATA_DIR = BASE_DIR.parent / "salesforce_data"

INPUT_FILE = "all_products_actuals_and_bookings.xlsx"
REVISED_ACTUALS_FILE = "all_products_actuals_and_bookings_revised.xlsx"
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
SBA_ALPHA = 0.1
BLEND_SOFTMAX_TAU = 0.10
BLEND_EPS = 1e-9
BLEND_MODEL_NAME = "blended_softmax"
HOLDOUT_TS_CSV = BASE_DIR / "data_storage" / "holdout_forecast_actuals.csv"

# Labels for outward-facing model descriptions
MODEL_LABELS = {
    "baseline_sarima": "Seasonal Baseline",
    "baseline_ets": "ETS Baseline",
    "sba": "SBA",
    "ml_gbr": "ML GBR",
    "ml_gbr_pipeline": "ML GBR Pipeline",
    "prophet": "PROPHET",
    BLEND_MODEL_NAME: "Blended (Softmax)",
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


def _blend_family_from_holdout(model_name: str) -> Optional[str]:
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


def _load_holdout_ts() -> pd.DataFrame:
    if HOLDOUT_TS_CSV.exists():
        return pd.read_csv(HOLDOUT_TS_CSV)
    return pd.DataFrame()


def _compute_holdout_weights(
    holdout_df: pd.DataFrame,
    product_id: str,
    bu_id: str,
    tau: float = BLEND_SOFTMAX_TAU,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float], List[dict]]:
    if holdout_df.empty:
        return {}, {}, []

    prod_norm, div_norm = normalize_key(product_id, bu_id)
    sku_df = holdout_df.copy()
    sku_df["_prod_norm"] = sku_df[COL_PRODUCT].apply(normalize_code)
    sku_df["_div_norm"] = sku_df[COL_DIVISION].apply(normalize_code)
    sku_df = sku_df[(sku_df["_prod_norm"] == prod_norm) & (sku_df["_div_norm"] == div_norm)].copy()
    if sku_df.empty:
        return {}, {}, []

    sku_df = _add_holdout_horizon(sku_df)
    sku_df["Family"] = sku_df["Model"].apply(_blend_family_from_holdout)
    sku_df = sku_df[sku_df["Family"].notna()]
    if sku_df.empty:
        return {}, {}, []

    # Overall WAPE by family (best model within family).
    overall_wape = (
        sku_df.groupby(["Family", "Model"], dropna=False)
        .apply(lambda g: compute_wape(g["Actual"], g["Forecast"]))
        .reset_index(name="WAPE")
    )
    overall_wape = overall_wape[np.isfinite(overall_wape["WAPE"])]
    if overall_wape.empty:
        return {}, {}, []

    best_by_family = (
        overall_wape.sort_values("WAPE")
        .groupby("Family", as_index=False)
        .first()
    )
    selected_models = set(best_by_family["Model"].tolist())
    family_to_model = dict(zip(best_by_family["Family"], best_by_family["Model"]))
    overall_wape_map = {
        row["Family"]: row["WAPE"] for _, row in best_by_family.iterrows()
    }
    overall_weights = _softmax_weights_from_wape(overall_wape_map, tau)

    weights_by_horizon = {}
    weight_rows = []
    sku_selected = sku_df[sku_df["Model"].isin(selected_models)]
    for horizon, h_df in sku_selected.groupby("Horizon"):
        wape_by_family = {}
        for family, g_family in h_df.groupby("Family"):
            best_wape = compute_wape(g_family["Actual"], g_family["Forecast"])
            if np.isfinite(best_wape):
                wape_by_family[family] = best_wape
        weights = _softmax_weights_from_wape(wape_by_family, tau)
        if not weights:
            continue
        weights_by_horizon[int(horizon)] = weights
        for family, weight in weights.items():
            weight_rows.append({
                COL_PRODUCT: product_id,
                COL_DIVISION: bu_id,
                "Horizon": int(horizon),
                "Model_Family": family,
                "Model": family_to_model.get(family),
                "WAPE": wape_by_family.get(family, np.nan),
                "Weight": weight,
                "Tau": tau,
                "Weight_Source": "holdout_horizon",
            })

    return weights_by_horizon, overall_weights, weight_rows


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


def build_future_frame(
    actuals: pd.DataFrame,
    pipeline_history: pd.DataFrame,
    feature_mode: str,
    summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    actuals_series = actuals.set_index("Month")["Actuals"].sort_index()
    last_actual_month = actuals_series.index.max()
    if pipeline_history.empty:
        return pd.DataFrame()

    snapshot_month = pd.Timestamp(pipeline_history["snapshot_month"].max())
    current_pipeline = pipeline_history[
        pipeline_history["snapshot_month"] == snapshot_month
    ].copy()
    if current_pipeline.empty:
        return pd.DataFrame()
    snapshot_for_lags = min(snapshot_month - pd.DateOffset(months=1), last_actual_month)

    summary_row = {}
    summary_cols = []
    if summary_df is not None and not summary_df.empty:
        summary_cols = [c for c in summary_df.columns if c not in {"Month", "group_key", "BU"}]
        summary_row = (
            summary_df[summary_df["Month"] == snapshot_month].iloc[0].to_dict()
            if (summary_df["Month"] == snapshot_month).any()
            else {}
        )
    summary_payload = {f"summary_{col}": summary_row.get(col, np.nan) for col in summary_cols}

    if feature_mode == "quantity":
        current_pipeline["pipeline_primary"] = current_pipeline["pipeline_factored_qty"]
        current_pipeline["pipeline_stage_weighted_primary"] = current_pipeline[
            "pipeline_stage_weighted_qty"
        ]
    else:
        current_pipeline["pipeline_primary"] = current_pipeline["pipeline_factored_revenue"]
        current_pipeline["pipeline_stage_weighted_primary"] = current_pipeline[
            "pipeline_stage_weighted_revenue"
        ]

    slip_months = current_pipeline["slippage_months"].median()
    slip_months = int(round(slip_months)) if pd.notna(slip_months) else 0
    if slip_months:
        current_pipeline["slip_target_month"] = current_pipeline["target_month"] + pd.DateOffset(
            months=slip_months
        )
    else:
        current_pipeline["slip_target_month"] = current_pipeline["target_month"]

    pipeline_keyed = current_pipeline.set_index(["snapshot_month", "target_month"])
    pipeline_keyed_full = pipeline_history.set_index(["snapshot_month", "target_month"])

    slip_target_sum = (
        current_pipeline.groupby("slip_target_month")["pipeline_primary"].sum().to_dict()
    )

    snapshot_metrics = {}
    for win in [1, 2, 3]:
        cutoff = snapshot_month + pd.DateOffset(months=win)
        due_mask = current_pipeline["target_month"] <= cutoff
        slip_mask = current_pipeline["slip_target_month"] <= cutoff
        snapshot_metrics[f"pipeline_due_{win}m"] = float(
            current_pipeline.loc[due_mask, "pipeline_primary"].sum()
        )
        snapshot_metrics[f"pipeline_stage_weighted_due_{win}m"] = float(
            current_pipeline.loc[due_mask, "pipeline_stage_weighted_primary"].sum()
        )
        snapshot_metrics[f"pipeline_slip_adjusted_due_{win}m"] = float(
            current_pipeline.loc[slip_mask, "pipeline_primary"].sum()
        )

    lag_months = [snapshot_for_lags - pd.DateOffset(months=i) for i in range(0, 12)]
    lag_values = [
        np.nan if actuals_series.get(m) is None else actuals_series.get(m) for m in lag_months
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
    summary_primary_coverage = (
        summary_primary / avg_actuals_12
        if avg_actuals_12 and not np.isnan(avg_actuals_12)
        else np.nan
    )

    rows = []
    for i in range(1, FORECAST_HORIZON + 1):
        target_month = last_actual_month + pd.DateOffset(months=i)
        months_ahead = month_diff(target_month, snapshot_month)

        key = (snapshot_month, target_month)
        pipeline_row = pipeline_keyed.loc[key] if key in pipeline_keyed.index else None
        pipeline_qty = float(pipeline_row["pipeline_qty"]) if pipeline_row is not None else 0.0
        pipeline_factored_qty = (
            float(pipeline_row["pipeline_factored_qty"]) if pipeline_row is not None else 0.0
        )
        pipeline_factored_revenue = (
            float(pipeline_row["pipeline_factored_revenue"]) if pipeline_row is not None else 0.0
        )
        pipeline_stage_weighted_qty = (
            float(pipeline_row["pipeline_stage_weighted_qty"])
            if pipeline_row is not None
            else 0.0
        )
        pipeline_stage_weighted_revenue = (
            float(pipeline_row["pipeline_stage_weighted_revenue"])
            if pipeline_row is not None
            else 0.0
        )

        pipeline_primary = (
            pipeline_factored_qty if feature_mode == "quantity" else pipeline_factored_revenue
        )
        pipeline_stage_weighted_primary = (
            pipeline_stage_weighted_qty
            if feature_mode == "quantity"
            else pipeline_stage_weighted_revenue
        )
        prior_snapshot = snapshot_month - pd.DateOffset(months=1)
        prior_row = (
            pipeline_keyed_full.loc[(prior_snapshot, target_month)]
            if (prior_snapshot, target_month) in pipeline_keyed_full.index
            else None
        )
        prior_primary = (
            float(
                prior_row["pipeline_factored_qty"]
                if feature_mode == "quantity"
                else prior_row["pipeline_factored_revenue"]
            )
            if prior_row is not None
            else 0.0
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

        trend_idx = month_diff(snapshot_for_lags, actuals_series.index.min())

        rows.append(
            {
                "snapshot_month": snapshot_month,
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
                "pipeline_slip_adjusted_primary": float(slip_target_sum.get(target_month, 0.0)),
                "pipeline_due_1m": snapshot_metrics.get("pipeline_due_1m", np.nan),
                "pipeline_due_2m": snapshot_metrics.get("pipeline_due_2m", np.nan),
                "pipeline_due_3m": snapshot_metrics.get("pipeline_due_3m", np.nan),
                "pipeline_stage_weighted_due_1m": snapshot_metrics.get(
                    "pipeline_stage_weighted_due_1m", np.nan
                ),
                "pipeline_stage_weighted_due_2m": snapshot_metrics.get(
                    "pipeline_stage_weighted_due_2m", np.nan
                ),
                "pipeline_stage_weighted_due_3m": snapshot_metrics.get(
                    "pipeline_stage_weighted_due_3m", np.nan
                ),
                "pipeline_slip_adjusted_due_1m": snapshot_metrics.get(
                    "pipeline_slip_adjusted_due_1m", np.nan
                ),
                "pipeline_slip_adjusted_due_2m": snapshot_metrics.get(
                    "pipeline_slip_adjusted_due_2m", np.nan
                ),
                "pipeline_slip_adjusted_due_3m": snapshot_metrics.get(
                    "pipeline_slip_adjusted_due_3m", np.nan
                ),
                "slippage_months": slip_months,
                "summary_primary": summary_primary,
                "summary_primary_coverage": summary_primary_coverage,
                **summary_payload,
            }
        )

    return pd.DataFrame(rows)


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
def _friendly_regressor_name(reg_name: Optional[str]) -> str:
    """Return a natural-language regressor name."""
    if reg_name is None:
        return "Regressor"
    mapping = {
        "New_Opportunities": "Salesforce",
        "Open_Opportunities": "Salesforce",
        "Bookings": "Bookings",
        "Median_Months_Since_Last_Activity": "Median Months Since Last Activity",
        "Open_Not_Modified_90_Days": "Open Not Modified 90 Days",
        "Pct_Open_Not_Modified_90_Days": "Pct Open Not Modified 90 Days",
        "Early_to_Late_Ratio": "Early to Late Ratio",
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
    if model_name == "SBA":
        return "sba"
    if model_name.startswith("SARIMAX"):
        return "with_regressor"
    if model_name == "ML_GBR":
        return "ml_gbr"
    if model_name == PIPELINE_ML_MODEL_NAME:
        return "ml_gbr_pipeline"
    if model_name == "PROPHET":
        return "prophet"
    if model_name == "BLENDED_SOFTMAX":
        return BLEND_MODEL_NAME
    return None


def _apply_recommended_model_flags(
    fc_df: pd.DataFrame,
    preferred_group: Optional[str],
) -> pd.DataFrame:
    """
    Mark recommended_model only when the model_status is ok.
    If the preferred group failed, fall back to the best available ok group.
    """
    fc_df = fc_df.copy()
    fc_df["recommended_model"] = False

    priority = []
    if preferred_group:
        priority.append(preferred_group)
    priority.extend(
        g
        for g in [
            "baseline_sarima",
            "with_regressor",
            "baseline_ets",
            "sba",
            "ml_gbr_pipeline",
            "ml_gbr",
            "prophet",
        ]
        if g not in priority
    )

    chosen_group = None
    for model_group in priority:
        ok_rows = fc_df[
            (fc_df["model_group"] == model_group) & (fc_df["model_status"] == "ok")
        ]
        if not ok_rows.empty:
            chosen_group = model_group
            break

    if chosen_group is None:
        ok_any = fc_df[fc_df["model_status"] == "ok"]
        if not ok_any.empty:
            chosen_group = ok_any.iloc[0]["model_group"]

    if chosen_group is not None:
        idx = fc_df[
            (fc_df["model_group"] == chosen_group) & (fc_df["model_status"] == "ok")
        ].index
        fc_df.loc[idx, "recommended_model"] = True

    return fc_df


def _apply_recommended_flags_all(
    all_forecasts: pd.DataFrame,
    model_choices: Dict[Tuple[str, str], dict],
) -> pd.DataFrame:
    all_forecasts = all_forecasts.copy()
    all_forecasts["recommended_model"] = False
    for (prod, div), idx in all_forecasts.groupby(["product_id", "bu_id"]).groups.items():
        choice = model_choices.get(normalize_key(prod, div))
        preferred_group = _model_group_from_choice(choice)
        flagged = _apply_recommended_model_flags(all_forecasts.loc[idx], preferred_group)
        all_forecasts.loc[idx, "recommended_model"] = flagged["recommended_model"].values
    return all_forecasts


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
        overrides[normalize_key(prod, div)] = (order, seas_order)
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
        orders_map[normalize_key(prod, div)] = (order, seas_order)

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
    xls = pd.ExcelFile(path)
    if "Model_Summary" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Model_Summary")
    else:
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    recommended_map = {}
    if "recommended_model_summary" in xls.sheet_names:
        df_rec = pd.read_excel(xls, sheet_name="recommended_model_summary")
        rec_cols = {COL_PRODUCT, COL_DIVISION, "Recommended_Model"}
        if rec_cols.issubset(df_rec.columns):
            recommended_map = {
                normalize_key(row[COL_PRODUCT], row[COL_DIVISION]): row["Recommended_Model"]
                for _, row in df_rec.iterrows()
            }
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
        if "Median_Months_Since_Last_Activity" in model_name:
            lag = int(model_name.split("_l")[-1]) if "_l" in model_name else 1
            return "Median_Months_Since_Last_Activity", lag
        if "Open_Not_Modified_90_Days" in model_name:
            lag = int(model_name.split("_l")[-1]) if "_l" in model_name else 1
            return "Open_Not_Modified_90_Days", lag
        if "Pct_Open_Not_Modified_90_Days" in model_name:
            lag = int(model_name.split("_l")[-1]) if "_l" in model_name else 1
            return "Pct_Open_Not_Modified_90_Days", lag
        if "Early_to_Late_Ratio" in model_name:
            lag = int(model_name.split("_l")[-1]) if "_l" in model_name else 1
            return "Early_to_Late_Ratio", lag
        return None, None

    choices: Dict[Tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        prod = row[COL_PRODUCT]
        div = row[COL_DIVISION]
        model_name = row["Chosen_Model"]
        key = normalize_key(prod, div)
        recommended_name = recommended_map.get(key)
        if recommended_name is not None and not pd.isna(recommended_name):
            model_name = recommended_name
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

        choices[key] = {
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
        ("Median_Months_Since_Last_Activity", 1): "Median_Months_Since_Last_Activity_l1",
        ("Median_Months_Since_Last_Activity", 2): "Median_Months_Since_Last_Activity_l2",
        ("Open_Not_Modified_90_Days", 1): "Open_Not_Modified_90_Days_l1",
        ("Pct_Open_Not_Modified_90_Days", 1): "Pct_Open_Not_Modified_90_Days_l1",
        ("Pct_Open_Not_Modified_90_Days", 2): "Pct_Open_Not_Modified_90_Days_l2",
        ("Early_to_Late_Ratio", 1): "Early_to_Late_Ratio_l1",
        ("Early_to_Late_Ratio", 2): "Early_to_Late_Ratio_l2",
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
        ("Median_Months_Since_Last_Activity", 1, "Median_Months_Since_Last_Activity_l1"),
        ("Median_Months_Since_Last_Activity", 2, "Median_Months_Since_Last_Activity_l2"),
        ("Open_Not_Modified_90_Days", 1, "Open_Not_Modified_90_Days_l1"),
        ("Pct_Open_Not_Modified_90_Days", 1, "Pct_Open_Not_Modified_90_Days_l1"),
        ("Pct_Open_Not_Modified_90_Days", 2, "Pct_Open_Not_Modified_90_Days_l2"),
        ("Early_to_Late_Ratio", 1, "Early_to_Late_Ratio_l1"),
        ("Early_to_Late_Ratio", 2, "Early_to_Late_Ratio_l2"),
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


def _sba_forecast_value(y_train: pd.Series, alpha: float = SBA_ALPHA) -> float:
    """Compute SBA forecast level for intermittent demand."""
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


def _compute_test_window(history_months: int) -> int:
    """Compute test window length based on history length."""
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


def _evaluate_sba_metrics(
    y: pd.Series,
    test_window: int,
    alpha: float = SBA_ALPHA,
    rocv_horizon: int = ROCV_HORIZON,
    rocv_max_origins: int = ROCV_MAX_ORIGINS,
) -> dict:
    """Fit SBA, compute test MAE/RMSE and ROCV MAE."""
    y_clean = y.dropna().astype(float)
    if len(y_clean) <= test_window:
        return {"Test_MAE": np.nan, "Test_RMSE": np.nan, "ROCV_MAE": np.nan}

    y_train = y_clean.iloc[:-test_window]
    y_test = y_clean.iloc[-test_window:]

    forecast_value = _sba_forecast_value(y_train, alpha=alpha)
    y_pred = pd.Series(np.repeat(forecast_value, len(y_test)), index=y_test.index)

    test_mae = _safe_mae(y_test.values, y_pred.values)
    test_rmse = _safe_rmse(y_test.values, y_pred.values)

    rocv_maes = []
    min_train = max(12, rocv_horizon)
    origins = _rocv_origins(len(y_clean), rocv_horizon, rocv_max_origins, min_train)
    for origin in origins:
        y_train_rocv = y_clean.iloc[:origin]
        y_test_rocv = y_clean.iloc[origin:origin + rocv_horizon]
        forecast_value = _sba_forecast_value(y_train_rocv, alpha=alpha)
        y_pred_rocv = np.repeat(forecast_value, len(y_test_rocv))
        mae = _safe_mae(y_test_rocv.values, y_pred_rocv)
        if np.isfinite(mae):
            rocv_maes.append(mae)

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
    allow_sba: bool,
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

    if allow_sba:
        sba_metrics = _evaluate_sba_metrics(
            y_clean, test_window, alpha=SBA_ALPHA
        )
        sba_metrics["model_group"] = "sba"
        metrics.append(sba_metrics)

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
    pipeline_history: Optional[pd.DataFrame] = None,
    feature_mode: Optional[str] = None,
    summary_df: Optional[pd.DataFrame] = None,
    use_pipeline_ml: bool = False,
    allow_sarima: bool = True,
    allow_sarimax: bool = True,
    allow_ets: bool = True,
    allow_ets_seasonal: bool = True,
    allow_sba: bool = True,
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

    # SBA (Syntetos–Boylan Approximation)
    if not allow_sba:
        reason = status_reasons.get("sba", "N/A: SBA disabled.")
        dfs.append(_build_na_rows("sba", "SBA", _natural_model_label("sba"), reason))
    else:
        try:
            y_train = y.dropna()
            if y_train.empty:
                raise ValueError("No non-null history for SBA.")
            forecast_value = _sba_forecast_value(y_train, alpha=SBA_ALPHA)
            sba_df = pd.DataFrame({
                "forecast_month": forecast_index,
                "forecast_value": [forecast_value] * len(forecast_index),
                "lower_ci": [np.nan] * len(forecast_index),
                "upper_ci": [np.nan] * len(forecast_index),
                "model_group": "sba",
                "model_type": "SBA",
                "model_label": _natural_model_label("sba"),
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
            dfs.append(_attach_status(_apply_forecast_floor(sba_df), "ok"))
        except Exception as exc:
            print(f"  Failed SBA: {exc}")
            dfs.append(_build_na_rows(
                "sba",
                "SBA",
                _natural_model_label("sba"),
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
        if use_pipeline_ml:
            try:
                actuals_df = y.reset_index()
                actuals_df = actuals_df.rename(columns={COL_DATE: "Month", COL_ACTUALS: "Actuals"})
                actuals_df["Month"] = month_start(actuals_df["Month"])
                actuals_series = actuals_df.set_index("Month")["Actuals"].sort_index()

                training = build_feature_frame(
                    actuals_df,
                    pipeline_history if pipeline_history is not None else pd.DataFrame(),
                    FORECAST_HORIZON,
                    feature_mode or "quantity",
                    summary_df=summary_df,
                )
                if training.empty:
                    raise ValueError("No training rows after assembling pipeline features.")

                training["baseline_pred"] = np.nan
                for snapshot in sorted(training["snapshot_month"].dropna().unique()):
                    snapshot_rows = training["snapshot_month"] == snapshot
                    horizon = int(training.loc[snapshot_rows, "months_ahead"].max())
                    forecast = baseline_forecast_for_snapshot(actuals_series, snapshot, horizon)
                    if forecast is None:
                        continue
                    training.loc[snapshot_rows, "baseline_pred"] = training.loc[
                        snapshot_rows, "target_month"
                    ].map(forecast)
                training = training.dropna(subset=["baseline_pred"])
                if training.empty:
                    raise ValueError("No baseline predictions available for pipeline training.")

                future_frame = build_future_frame(
                    actuals_df,
                    pipeline_history if pipeline_history is not None else pd.DataFrame(),
                    feature_mode or "quantity",
                    summary_df=summary_df,
                )
                if future_frame.empty:
                    raise ValueError("No future rows after assembling pipeline features.")

                snapshot_month = pd.Timestamp(
                    pipeline_history["snapshot_month"].max()
                ) if pipeline_history is not None and not pipeline_history.empty else None
                if snapshot_month is None:
                    raise ValueError("No snapshot month available for pipeline forecast.")

                baseline_future = baseline_forecast_for_snapshot(
                    actuals_series=actuals_series,
                    snapshot_month=snapshot_month,
                    horizon=FORECAST_HORIZON,
                )
                if baseline_future is None:
                    raise ValueError("Baseline forecast unavailable for pipeline ML.")

                future_frame["baseline_pred"] = future_frame["target_month"].map(baseline_future)
                future_frame = future_frame.dropna(subset=["baseline_pred"])
                if future_frame.empty:
                    raise ValueError("No baseline predictions for future pipeline frame.")

                bundle = train_models(training)
                future_pred = predict_with_models(future_frame, bundle)
                future_pred = future_pred.set_index("target_month").sort_index()
                mean = future_pred.reindex(forecast_index)["final_pred"]

                ml_df = pd.DataFrame({
                    "forecast_month": forecast_index,
                    "forecast_value": mean.values,
                    "lower_ci": [np.nan] * len(forecast_index),
                    "upper_ci": [np.nan] * len(forecast_index),
                    "model_group": "ml_gbr_pipeline",
                    "model_type": "ML",
                    "model_label": _natural_model_label("ml_gbr_pipeline"),
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
                reason = status_reasons.get("ml_gbr_pipeline", f"Failed: {exc}")
                dfs.append(_build_na_rows(
                    "ml_gbr_pipeline",
                    "ML",
                    _natural_model_label("ml_gbr_pipeline"),
                    reason,
                    status="Failed",
                ))
        else:
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


def _build_blended_softmax_forecast(
    fc_df: pd.DataFrame,
    product_id: str,
    bu_id: str,
    weights_by_horizon: Dict[int, Dict[str, float]],
    overall_weights: Dict[str, float],
    weight_rows_holdout: List[dict],
) -> Tuple[pd.DataFrame, List[dict]]:
    if fc_df.empty:
        return pd.DataFrame(), []

    family_by_group = {
        "baseline_sarima": "SARIMA",
        "with_regressor": "SARIMAX",
        "baseline_ets": "ETS",
        "sba": "SBA",
        "prophet": "PROPHET",
        "ml_gbr": "ML",
        "ml_gbr_pipeline": "ML",
    }
    fc_ok = fc_df[fc_df["model_status"] == "ok"].copy()
    fc_ok = fc_ok[fc_ok["model_group"].isin(family_by_group)]
    if fc_ok.empty:
        return pd.DataFrame(), []

    fc_ok["family"] = fc_ok["model_group"].map(family_by_group)
    fc_ok["horizon"] = fc_ok["horizon_months_ahead"].astype(int)

    forecast_map = {}
    for (family, horizon), group in fc_ok.groupby(["family", "horizon"]):
        forecast_map[(family, horizon)] = float(group["forecast_value"].iloc[0])

    weight_rows = list(weight_rows_holdout)
    wape_lookup = {
        (int(row["Horizon"]), row["Model_Family"]): row.get("WAPE", np.nan)
        for row in weight_rows_holdout
        if "Horizon" in row and "Model_Family" in row
    }
    existing_keys = set(wape_lookup.keys())

    blended_rows = []
    horizons = sorted(fc_ok["horizon"].unique())
    families = sorted(fc_ok["family"].unique())
    for horizon in horizons:
        weights = weights_by_horizon.get(horizon)
        source = "holdout_horizon"
        if not weights:
            weights = overall_weights
            source = "holdout_overall"
        if not weights:
            equal = 1.0 / len(families) if families else 0.0
            weights = {family: equal for family in families}
            source = "equal_fallback"

        total = 0.0
        weight_sum = 0.0
        for family, weight in weights.items():
            fc_val = forecast_map.get((family, horizon))
            if fc_val is None or not np.isfinite(fc_val):
                continue
            total += weight * fc_val
            weight_sum += weight
            key = (horizon, family)
            if source != "holdout_horizon" or key not in existing_keys:
                weight_rows.append({
                    COL_PRODUCT: product_id,
                    COL_DIVISION: bu_id,
                    "Horizon": horizon,
                    "Model_Family": family,
                    "WAPE": wape_lookup.get(key, np.nan),
                    "Weight": weight,
                    "Tau": BLEND_SOFTMAX_TAU,
                    "Weight_Source": source,
                })
        if weight_sum <= BLEND_EPS:
            continue
        blended_rows.append({
            "forecast_month": fc_ok.loc[fc_ok["horizon"] == horizon, "forecast_month"].iloc[0],
            "forecast_value": total / weight_sum,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "model_group": BLEND_MODEL_NAME,
            "model_type": "BLEND",
            "model_label": _natural_model_label(BLEND_MODEL_NAME),
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

    blended_df = pd.DataFrame(blended_rows)
    if blended_df.empty:
        return pd.DataFrame(), weight_rows

    blended_df["product_id"] = product_id
    blended_df["bu_id"] = bu_id
    blended_df["run_id"] = fc_df["run_id"].iloc[0]
    blended_df["training_start"] = fc_df["training_start"].iloc[0]
    blended_df["training_end"] = fc_df["training_end"].iloc[0]
    blended_df["horizon_months_ahead"] = (
        blended_df.groupby(["product_id", "model_group"]).cumcount() + 1
    )
    blended_df = _attach_status(_apply_forecast_floor(blended_df), "ok")
    return blended_df, weight_rows


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
    cutoff = last_completed_month()
    df_all = filter_to_completed_months(df_all, cutoff=cutoff)
    print(f"Filtering actuals/bookings to completed months through {cutoff.date()}.")
    df_all = mark_prelaunch_actuals_as_missing(
        df_all,
        output_excel_path=REVISED_ACTUALS_FILE,
        output_sheet=REVISED_ACTUALS_SHEET,
    )
    holdout_ts_all = _load_holdout_ts()
    blend_weight_rows_all = []
    summary_full = pd.DataFrame()
    if SUMMARY_PATH.exists():
        summary_full = load_summary_report(SUMMARY_PATH)
    else:
        print(f"[WARN] Summary file not found: {SUMMARY_PATH}")
    df_all = merge_summary_regressors(df_all, summary_full)
    df_all[COL_DATE] = pd.to_datetime(df_all[COL_DATE])
    df_all = df_all.sort_values([COL_PRODUCT, COL_DIVISION, COL_DATE])

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

    print("Loading chosen models...")
    model_choices = load_model_choices(SUMMARY_FILE)

    print("Loading per-SKU orders...")
    orders_map = load_orders_map(ORDER_FILE, overrides_path=NOTES_FILE)

    results_rows = []
    sku_groups = df_all.groupby([COL_PRODUCT, COL_DIVISION], dropna=False)
    print(f"Found {len(sku_groups)} product+division combinations.")

    for (prod, div), df_sku in sku_groups:
        print(f"\n=== {prod} / {div} ===")
        key = normalize_key(prod, div)
        choice = model_choices.get(key)
        order_pair = orders_map.get(key)

        if choice is None:
            print("  Skipping: no chosen model found in summary file.")
            continue
        if order_pair is None:
            print("  Skipping: no SARIMA order found in order file.")
            continue

        order, seasonal_order = order_pair
        df_sku = df_sku.set_index(COL_DATE)
        summary_product = filter_summary_product(summary_full, prod, div)
        feature_mode = None
        pipeline_history = pd.DataFrame()
        pipeline_reason = ""
        if ENABLE_ML_CHALLENGER:
            try:
                feature_mode = load_feature_mode(prod, str(div), catalog=catalog)
            except RuntimeError as exc:
                pipeline_reason = str(exc)
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
                pipeline_reason = pipeline_reason or "No pipeline files available."
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
        use_pipeline_ml = False
        if not allow_ml:
            status_reasons["ml_gbr"] = "N/A: ML challenger disabled."
            status_reasons["ml_gbr_pipeline"] = "N/A: ML challenger disabled."
        else:
            if summary_product.empty and not pipeline_reason:
                pipeline_reason = "No summary data for product/BU."
            if pipeline_history.empty and not pipeline_reason:
                pipeline_reason = "No pipeline history rows."
            if pipeline_reason:
                status_reasons["ml_gbr_pipeline"] = pipeline_reason
            if feature_mode is not None and not pipeline_reason:
                use_pipeline_ml = True
            else:
                status_reasons.setdefault("ml_gbr", "Using legacy ML (no pipeline data).")

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
            pipeline_history=pipeline_history,
            feature_mode=feature_mode,
            summary_df=summary_product,
            use_pipeline_ml=use_pipeline_ml,
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

        weights_by_horizon, overall_weights, weight_rows = _compute_holdout_weights(
            holdout_ts_all,
            prod,
            div,
            tau=BLEND_SOFTMAX_TAU,
        )
        if weights_by_horizon or overall_weights:
            blended_df, weight_rows_used = _build_blended_softmax_forecast(
                fc_df,
                prod,
                div,
                weights_by_horizon,
                overall_weights,
                weight_rows,
            )
            if not blended_df.empty:
                fc_df = pd.concat([fc_df, blended_df], ignore_index=True)
                blend_weight_rows_all.extend(weight_rows_used)

        recommended_group = _model_group_from_choice(choice)
        fc_df = _apply_recommended_model_flags(fc_df, recommended_group)

        results_rows.append(fc_df)
        print(f"  Generated {fc_df['model_group'].nunique()} model variants.")

    if not results_rows:
        print("No forecasts generated. Check inputs.")
        return

    all_forecasts = pd.concat(results_rows, axis=0, ignore_index=True)
    all_forecasts = _apply_recommended_flags_all(all_forecasts, model_choices)
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
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        all_forecasts.to_excel(writer, index=False, sheet_name="Forecast_Library")
        if blend_weight_rows_all:
            weights_df = pd.DataFrame(blend_weight_rows_all)
            weights_df.to_excel(writer, index=False, sheet_name="Blended_Softmax_Weights")
    print(f"Done. Saved to {output_file}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    _print_total_runtime(start_time)
