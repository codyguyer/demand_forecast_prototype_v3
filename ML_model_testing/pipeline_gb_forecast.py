from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None


BASE_DIR = Path(__file__).resolve().parent

ACTUALS_PATH = BASE_DIR / "all_products_with_sf_and_bookings.xlsx"
PRODUCT_CATALOG_PATH = BASE_DIR / "product_catalog_master.xlsx"
PIPELINE_FILES = sorted(BASE_DIR.glob("Merged Salesforce Pipeline *.xlsx"))
SF_PRODUCT_REFERENCE_PATH = BASE_DIR / "sf_product_reference_key.csv"
OUTPUT_PATH = BASE_DIR / "ml_pipeline_forecast.xlsx"
SUMMARY_REPORT_NAME = "Salesforce Pipeline Monthly Summary.xlsx"
SUMMARY_FALLBACK_PATH = Path(
    r"C:\Users\cguyer\OneDrive - Midmark Corporation\Documents\Sales Ops\Reporting\Salesforce Archive"
) / SUMMARY_REPORT_NAME

PRODUCT_IDS: list[str] | None = None
DIVISION: str | None = None

FORECAST_HORIZON = 12
HOLDOUT_MONTHS = 12


@dataclass
class ModelBundle:
    adjustment_model: Pipeline | None
    adjustment_features: list[str]


def month_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def parse_snapshot_month(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    dt = pd.to_datetime(raw, format="%Y-%m", errors="coerce")
    if dt.isna().any():
        dt2 = pd.to_datetime(raw, errors="coerce")
        dt = dt.fillna(dt2)
    return dt.dt.to_period("M").dt.to_timestamp()


def parse_close_date(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    dt = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
    if dt.isna().any():
        dt2 = pd.to_datetime(raw, errors="coerce")
        dt = dt.fillna(dt2)
    return dt.dt.to_period("M").dt.to_timestamp()


def month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def safe_nanmean(values: list[float | int | None]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def aggregate_monthly_actuals(
    df: pd.DataFrame,
    product_col: str = "Product",
    division_col: str = "Division",
    date_col: str = "Month",
    sum_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = month_start(df[date_col])
    if "Actuals" in df.columns:
        df["Actuals"] = pd.to_numeric(df["Actuals"], errors="coerce")

    sum_cols = sum_cols or ["Actuals"]
    present_sum_cols = [c for c in sum_cols if c in df.columns]
    group_cols = [c for c in [product_col, division_col, date_col] if c in df.columns]
    if not group_cols:
        group_cols = [date_col]

    agg_dict = {col: "sum" for col in present_sum_cols}
    for col in df.columns:
        if col in group_cols or col in agg_dict:
            continue
        agg_dict[col] = "first"

    return (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(agg_dict)
        .sort_values(group_cols)
    )


def mark_prelaunch_actuals_as_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Month" in df.columns:
        df["Month"] = month_start(df["Month"])
    if "Actuals" in df.columns:
        df["Actuals"] = pd.to_numeric(df["Actuals"], errors="coerce")
    first_active = df.loc[df["Actuals"] > 0, "Month"].min()
    if pd.isna(first_active):
        df["Actuals"] = np.nan
        return df
    prelaunch_mask = (df["Month"] < first_active) & (df["Actuals"] == 0)
    df.loc[prelaunch_mask, "Actuals"] = np.nan
    return df


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
    return raw


def load_actuals(product_id: str, division: str) -> pd.DataFrame:
    df = pd.read_excel(ACTUALS_PATH)
    df = aggregate_monthly_actuals(df)
    prod_str = df["Product"].apply(normalize_code)
    div_str = df["Division"].astype(str)
    df = df[(prod_str == normalize_code(product_id)) & (div_str == str(division))].copy()
    df = mark_prelaunch_actuals_as_missing(df)
    return df


def load_feature_mode(product_id: str, division: str) -> str:
    catalog = pd.read_excel(PRODUCT_CATALOG_PATH)
    match = catalog[
        (catalog["group_key"].astype(str) == normalize_code(product_id))
        & (catalog["business_unit_code"].astype(str) == division)
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
    if not SF_PRODUCT_REFERENCE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SF_PRODUCT_REFERENCE_PATH)
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


def resolve_summary_path() -> Path | None:
    candidate = BASE_DIR / SUMMARY_REPORT_NAME
    if candidate.exists():
        return candidate
    if SUMMARY_FALLBACK_PATH.exists():
        return SUMMARY_FALLBACK_PATH
    return None


def load_summary_report(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"group_key": str, "BU": str})
    if "Month" in df.columns:
        df["Month"] = month_start(df["Month"])
    df["group_key"] = df["group_key"].apply(normalize_code)
    df["BU"] = df["BU"].astype(str)
    return df


def filter_summary_product(
    df: pd.DataFrame, product_id: str, division: str
) -> pd.DataFrame:
    if df.empty:
        return df
    product_key = normalize_code(product_id)
    division_key = str(division)
    product_mask = df["group_key"].apply(normalize_code) == product_key
    division_mask = df["BU"].astype(str) == division_key
    return df[product_mask & division_mask].copy()


def filter_pipeline_product(
    df: pd.DataFrame, product_id: str, sf_codes: list[str] | None = None
) -> pd.DataFrame:
    product_mask = pd.Series([False] * len(df), index=df.index)
    codes = {normalize_code(product_id)}
    if sf_codes:
        codes.update(normalize_code(code) for code in sf_codes if normalize_code(code))

    if "Product Code" in df.columns:
        product_codes = pd.to_numeric(df["Product Code"], errors="coerce")
        numeric_codes = []
        for code in codes:
            if code.replace(".", "", 1).isdigit():
                numeric_codes.append(float(code))
        if numeric_codes:
            product_mask |= product_codes.isin(numeric_codes)
        product_mask |= df["Product Code"].apply(normalize_code).isin(codes)
    if "Current OSC Product Name" in df.columns:
        product_mask |= df["Current OSC Product Name"].apply(normalize_code).isin(codes)
    return df[product_mask].copy()


def load_pipeline_history(
    product_id: str, division: str, sf_codes: list[str] | None = None
) -> pd.DataFrame:
    frames = []
    slip_frames = []
    for path in PIPELINE_FILES:
        df = pd.read_excel(path)
        if "Business Unit" in df.columns:
            df = df[df["Business Unit"] == division]
        df = filter_pipeline_product(df, product_id, sf_codes=sf_codes)
        df["snapshot_month"] = parse_snapshot_month(df["Month"])
        df["target_month"] = parse_close_date(df["Close Date"])
        df = df.dropna(subset=["target_month"])

        quantity_col = "Quantity" if "Quantity" in df.columns else "Quantity W/ Decimal"
        df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")
        if "Probability" in df.columns:
            df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
        if "Total Price" in df.columns:
            df["Total Price"] = pd.to_numeric(df["Total Price"], errors="coerce")
        if "Factored Quantity" in df.columns:
            df["Factored Quantity"] = pd.to_numeric(df["Factored Quantity"], errors="coerce")
        if "Factored Revenue" in df.columns:
            df["Factored Revenue"] = pd.to_numeric(df["Factored Revenue"], errors="coerce")

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
        slip_all["slip_months"] = slip_all.apply(
            lambda x: month_diff(x["target_month"], x["prior_target_month"])
            if pd.notna(x["prior_target_month"])
            else np.nan,
            axis=1,
        )
        slip_stats = slip_all.groupby("snapshot_month")["slip_months"].median()
        slippage_months_by_snapshot = slip_stats.to_dict()

    if slippage_months_by_snapshot:
        pipeline_history["slippage_months"] = pipeline_history["snapshot_month"].map(
            slippage_months_by_snapshot
        )
    else:
        pipeline_history["slippage_months"] = np.nan

    return pipeline_history


def select_latest_snapshot(pipeline_history: pd.DataFrame) -> pd.DataFrame:
    if pipeline_history.empty:
        return pipeline_history
    latest_snapshot = pipeline_history["snapshot_month"].max()
    return pipeline_history[pipeline_history["snapshot_month"] == latest_snapshot].copy()


def build_feature_frame(
    actuals: pd.DataFrame,
    pipeline: pd.DataFrame,
    max_horizon: int,
    feature_mode: str,
    summary_df: pd.DataFrame | None = None,
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

    summary_features: dict[pd.Timestamp, dict[str, float]] = {}
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
            pipeline_factored_qty = (
                float(snap_factored_qty.get(target_month, 0.0))
            )
            pipeline_factored_revenue = (
                float(snap_factored_revenue.get(target_month, 0.0))
            )
            pipeline_stage_weighted_qty = (
                float(snap_stage_weighted_qty.get(target_month, 0.0))
            )
            pipeline_stage_weighted_revenue = (
                float(snap_stage_weighted_revenue.get(target_month, 0.0))
            )

            pipeline_primary = (
                pipeline_factored_qty if feature_mode == "quantity" else pipeline_factored_revenue
            )
            pipeline_stage_weighted_primary = (
                pipeline_stage_weighted_qty
                if feature_mode == "quantity"
                else pipeline_stage_weighted_revenue
            )
            prior_primary = (
                float(
                    prior_factored_qty.get(target_month, 0.0)
                    if feature_mode == "quantity"
                    else prior_factored_revenue.get(target_month, 0.0)
                )
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
                    **{f"summary_{key}": summary_row.get(key, np.nan) for key in summary_cols},
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


def train_baseline_model(
    actuals_series: pd.Series,
) -> tuple[GradientBoostingRegressor, list[str], pd.Series]:
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
    y_history: list[float],
    last_date: pd.Timestamp,
    horizon: int,
    feature_columns: list[str],
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


def build_future_frame(
    actuals: pd.DataFrame,
    pipeline_history: pd.DataFrame,
    feature_mode: str,
    summary_df: pd.DataFrame | None = None,
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

        lag_months = [snapshot_for_lags - pd.DateOffset(months=i) for i in range(0, 12)]
        lag_values = [
            np.nan if actuals_series.get(m) is None else actuals_series.get(m)
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
                **{
                    f"summary_{col}": summary_row.get(col, np.nan)
                    for col in summary_cols
                    if col not in {"Month", "group_key", "BU"}
                },
            }
        )

    return pd.DataFrame(rows)


def baseline_forecast_for_snapshot(
    actuals_series: pd.Series,
    snapshot_month: pd.Timestamp,
    horizon: int,
) -> pd.Series | None:
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


def main() -> None:
    start_time = time.perf_counter()
    print("Starting pipeline GB forecast run...")
    all_future = []
    all_sf_pipeline_check = []
    skipped_products: list[dict[str, object]] = []
    holdout_results: list[dict[str, object]] = []
    summary_path = resolve_summary_path()
    summary_full = load_summary_report(summary_path) if summary_path else pd.DataFrame()
    sf_reference = load_sf_product_reference()
    sf_code_map: dict[tuple[str, str], list[str]] = {}
    if not sf_reference.empty:
        sf_reference = sf_reference.copy()
        sf_reference["salesforce_product_name"] = sf_reference[
            "salesforce_product_name"
        ].astype(str)
        grouped = sf_reference.groupby(["group_key", "business_unit_code"])
        sf_code_map = {
            (normalize_code(group_key), str(div)): list(
                group["salesforce_product_name"].unique()
            )
            for (group_key, div), group in grouped
        }
    product_pairs: list[tuple[str, str]] = []
    if PRODUCT_IDS:
        divisions = [DIVISION] if DIVISION else []
        if not divisions:
            raise RuntimeError("DIVISION must be set when PRODUCT_IDS is provided.")
        product_pairs = [
            (normalize_code(product_id), div) for product_id in PRODUCT_IDS for div in divisions
        ]
    else:
        actuals_all = pd.read_excel(ACTUALS_PATH)
        pairs = actuals_all[["Product", "Division"]].dropna(subset=["Product", "Division"])
        pairs["Product"] = pairs["Product"].apply(normalize_code)
        pairs["Division"] = pairs["Division"].astype(str)
        pairs = pairs[pairs["Product"] != ""]
        if DIVISION:
            pairs = pairs[pairs["Division"] == str(DIVISION)]
        product_pairs = (
            pairs.drop_duplicates()
            .sort_values(["Division", "Product"])
            .apply(lambda r: (str(r["Product"]), str(r["Division"])), axis=1)
            .tolist()
        )

    for product_id, division in product_pairs:
        print(f"Processing product {product_id} / {division}...")
        try:
            sf_codes = sf_code_map.get((product_id, division), [])
            has_sf_reference = bool(sf_codes)
            actuals = load_actuals(product_id, division)
            actuals_series = actuals.set_index("Month")["Actuals"].sort_index()
            actuals_clean = actuals_series.dropna()
            if actuals_clean.empty:
                raise RuntimeError(
                    f"No actuals available after prelaunch adjustment for {product_id}/{division}."
                )

            feature_mode = load_feature_mode(product_id, division)
            pipeline_history = load_pipeline_history(product_id, division, sf_codes=sf_codes)
            summary_product = filter_summary_product(summary_full, product_id, division)
            has_pipeline = not pipeline_history.empty
            all_sf_pipeline_check.append(
                {
                    "product_id": product_id,
                    "division": division,
                    "has_sf_reference": has_sf_reference,
                    "sf_code_count": len(sf_codes),
                    "pipeline_rows": len(pipeline_history),
                    "has_pipeline_data": has_pipeline,
                }
            )
            if pipeline_history.empty:
                skipped_products.append(
                    {
                        "product_id": product_id,
                        "division": division,
                        "reason": "no_pipeline_history",
                    }
                )
                print(
                    f"Skipping product {product_id} / {division} (no pipeline history rows)."
                )
                continue
        except RuntimeError as exc:
            skipped_products.append(
                {"product_id": product_id, "division": division, "reason": str(exc)}
            )
            print(f"Skipping product {product_id} / {division}: {exc}")
            continue

        training = build_feature_frame(
            actuals,
            pipeline_history,
            FORECAST_HORIZON,
            feature_mode,
            summary_df=summary_product,
        )
        if training.empty:
            raise RuntimeError(
                f"No training rows available after assembling features for {product_id}/{division}."
            )

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

        rocv_mae = compute_rolling_cv_mae(training, HOLDOUT_MONTHS)

        holdout_start = actuals_series.index.max() - pd.DateOffset(months=HOLDOUT_MONTHS - 1)
        holdout_df = training[training["target_month"] >= holdout_start].copy()
        train_adj_df = training[training["target_month"] < holdout_start].copy()
        if not holdout_df.empty and not train_adj_df.empty:
            bundle = train_models(train_adj_df)
            holdout_pred = predict_with_models(holdout_df, bundle)
            abs_err = (holdout_pred["final_pred"] - holdout_pred["actuals"]).abs()
            holdout_mae = float(abs_err.mean()) if not abs_err.empty else np.nan
            holdout_rmse = float(
                np.sqrt(mean_squared_error(holdout_pred["actuals"], holdout_pred["final_pred"]))
            ) if not holdout_pred.empty else np.nan

            aic = np.nan
            bic = np.nan
            is_target_product = normalize_code(product_id) == "224" and str(division) == "D100"
            if is_target_product and SARIMAX is not None:
                train_series = actuals_series[actuals_series.index < holdout_start].dropna()
                if not train_series.empty:
                    try:
                        sarima = SARIMAX(
                            train_series,
                            order=(2, 1, 0),
                            seasonal_order=(0, 1, 0, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        sarima_fit = sarima.fit(disp=False)
                        aic = float(sarima_fit.aic)
                        bic = float(sarima_fit.bic)
                    except Exception:
                        aic = np.nan
                        bic = np.nan

            holdout_results.append(
                {
                    "product_id": product_id,
                    "division": division,
                    "holdout_start": holdout_start,
                    "holdout_end": actuals_series.index.max(),
                    "holdout_mae": holdout_mae,
                    "holdout_rmse": holdout_rmse,
                    "rocv_mae": rocv_mae,
                    "sarima_aic": aic,
                    "sarima_bic": bic,
                    "holdout_rows": int(len(holdout_pred)),
                }
            )

        future_frame = build_future_frame(
            actuals,
            pipeline_history,
            feature_mode,
            summary_df=summary_product,
        )
        future_pred = None
        if not future_frame.empty:
            snapshot_month = pd.Timestamp(pipeline_history["snapshot_month"].max())
            baseline_future = baseline_forecast_for_snapshot(
                actuals_series=actuals_series,
                snapshot_month=snapshot_month,
                horizon=FORECAST_HORIZON,
            )
            if baseline_future is not None:
                future_frame["baseline_pred"] = future_frame["target_month"].map(baseline_future)
            future_frame = future_frame.dropna(subset=["baseline_pred"])
            if not future_frame.empty:
                bundle = train_models(training)
                future_pred = predict_with_models(future_frame, bundle)
                future_pred["product_id"] = product_id
                future_pred["division"] = division
                all_future.append(future_pred)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        if all_future:
            future_out = pd.concat(all_future, ignore_index=True)
            keep_cols = ["product_id", "division", "target_month", "months_ahead", "final_pred"]
            future_out = future_out[[col for col in keep_cols if col in future_out.columns]]
            future_out.to_excel(
                writer, sheet_name="forecast_next_12", index=False
            )
        if holdout_results:
            holdout_df = pd.DataFrame(holdout_results)
            holdout_df.to_excel(writer, sheet_name="holdout_mae", index=False)
            overall_mae = holdout_df["holdout_mae"].mean()
            overall_rmse = holdout_df["holdout_rmse"].mean()
            overall_rocv = holdout_df["rocv_mae"].mean()
            summary_df = pd.DataFrame(
                [
                    {
                        "overall_holdout_mae": overall_mae,
                        "overall_holdout_rmse": overall_rmse,
                        "overall_rocv_mae": overall_rocv,
                    }
                ]
            )
            summary_df.to_excel(writer, sheet_name="holdout_summary", index=False)
    if all_sf_pipeline_check:
        missing = [
            row
            for row in all_sf_pipeline_check
            if row["has_sf_reference"] and not row["has_pipeline_data"]
        ]
        if missing:
            print(
                f"Warning: {len(missing)} product/BU pairs have SF reference but no pipeline data."
            )
    if skipped_products:
        pd.DataFrame(skipped_products).to_csv(
            BASE_DIR / "skipped_products.csv", index=False
        )
    elapsed = time.perf_counter() - start_time
    print(f"Completed. Output saved to {OUTPUT_PATH}")
    print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()
