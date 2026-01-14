from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, average_precision_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "xgboost is required for this script. Install with: pip install xgboost"
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent

ACTUALS_PATH = BASE_DIR / "all_products_with_sf_and_bookings.xlsx"
PRODUCT_CATALOG_PATH = BASE_DIR / "product_catalog_master.xlsx"
PIPELINE_FILES = sorted(BASE_DIR.glob("Merged Salesforce Pipeline *.xlsx"))
SF_PRODUCT_REFERENCE_PATH = BASE_DIR / "sf_product_reference_key.csv"

OUTPUT_PATH = BASE_DIR / "refactored_ml_forecast_output.xlsx"

PRODUCT_ID = "M11-050"
DIVISION = "D200"

FORECAST_HORIZON = 12
HOLDOUT_MONTHS = 12
SPIKE_LABEL_METHOD = "quantile"  # "hybrid" or "quantile"
SPIKE_K_CANDIDATES = [1.25, 1.5, 1.75]
SPIKE_QUANTILE_CANDIDATES = [0.8, 0.75]
SPIKE_ABS_FLOOR = 100.0
PIPELINE_LAGS = [1, 2, 3]  # months before target month


@dataclass
class ModelArtifacts:
    baseline_model: GradientBoostingRegressor
    spike_classifier: XGBClassifier
    spike_regressor: XGBRegressor | None
    baseline_feature_cols: list[str]
    spike_feature_cols: list[str]
    spike_k: float
    risk_scale_min: float
    risk_scale_max: float
    base_mae: float
    spike_mae: float


def month_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def end_of_month(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts) + pd.offsets.MonthEnd(0)


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


def aggregate_monthly_actuals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Month" in df.columns:
        df["Month"] = month_start(df["Month"])
    if "Actuals" in df.columns:
        df["Actuals"] = pd.to_numeric(df["Actuals"], errors="coerce")

    group_cols = [c for c in ["Product", "Division", "Month"] if c in df.columns]
    agg_dict = {"Actuals": "sum"} if "Actuals" in df.columns else {}
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

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def compute_spike_baseline(y: pd.Series) -> pd.Series:
    y = y.copy().sort_index()
    month_counts = y.groupby(y.index.month).count()
    use_seasonal = (month_counts >= 3).all()
    if use_seasonal:
        baseline = (
            y.groupby(y.index.month)
            .expanding()
            .median()
            .shift(1)
            .droplevel(0)
        )
        return baseline
    return y.rolling(window=12, min_periods=6).median().shift(1)


def compute_spike_labels(y: pd.Series, method: str, param: float) -> pd.Series:
    baseline = compute_spike_baseline(y)
    resid = y - baseline

    if method == "quantile":
        rolling_q = (
            y.rolling(window=12, min_periods=6)
            .quantile(param)
            .shift(1)
        )
        fallback = y.shift(1).dropna().quantile(param)
        rolling_q = rolling_q.fillna(fallback)
        labels = (y > rolling_q).astype(int)
        return labels

    if method != "hybrid":
        raise ValueError(f"Unsupported spike label method: {method}")

    mad = (
        resid.abs().rolling(window=12, min_periods=6).median().fillna(resid.abs().median())
    )
    threshold = np.maximum(param * mad, SPIKE_ABS_FLOOR)
    labels = (resid > threshold).astype(int)
    return labels


def build_features(
    actuals: pd.DataFrame,
    pipeline_history: pd.DataFrame,
    feature_mode: str,
) -> pd.DataFrame:
    actuals = actuals.copy()
    actuals["Month"] = month_start(actuals["Month"])
    actuals["Actuals"] = pd.to_numeric(actuals["Actuals"], errors="coerce")
    actuals = actuals.sort_values("Month")

    df = actuals.set_index("Month")[["Actuals"]].copy()
    df["month_of_year"] = df.index.month
    df["quarter"] = df.index.quarter
    df["trend_index"] = np.arange(len(df))

    df["y_lag1"] = df["Actuals"].shift(1)
    df["y_lag2"] = df["Actuals"].shift(2)
    df["y_lag3"] = df["Actuals"].shift(3)
    if df["Actuals"].dropna().shape[0] >= 24:
        df["y_lag12"] = df["Actuals"].shift(12)

    if pipeline_history.empty:
        df["pipeline_missing_flag"] = 1
        return df

    pipeline_history = pipeline_history.copy()
    pipeline_history["snapshot_month"] = month_start(pipeline_history["snapshot_month"])
    pipeline_history["target_month"] = month_start(pipeline_history["target_month"])

    if feature_mode == "quantity":
        pipeline_history["pipeline_primary"] = pipeline_history["pipeline_factored_qty"]
        pipeline_history["pipeline_stage_weighted_primary"] = pipeline_history[
            "pipeline_stage_weighted_qty"
        ]
    else:
        pipeline_history["pipeline_primary"] = pipeline_history["pipeline_factored_revenue"]
        pipeline_history["pipeline_stage_weighted_primary"] = pipeline_history[
            "pipeline_stage_weighted_revenue"
        ]

    snapshot_groups = pipeline_history.groupby("snapshot_month")
    snapshot_metrics = {}
    for snap_month, group in snapshot_groups:
        metrics = {}
        for win in [1, 2, 3]:
            cutoff = snap_month + pd.DateOffset(months=win)
            due_mask = group["target_month"] <= cutoff
            metrics[f"pipeline_due_{win}m"] = float(
                group.loc[due_mask, "pipeline_primary"].sum()
            )
            metrics[f"pipeline_stage_weighted_due_{win}m"] = float(
                group.loc[due_mask, "pipeline_stage_weighted_primary"].sum()
            )
        snapshot_metrics[pd.Timestamp(snap_month)] = metrics

    def _latest_snapshot(cutoff: pd.Timestamp) -> pd.DataFrame:
        eligible = pipeline_history[pipeline_history["snapshot_month"] <= cutoff]
        if eligible.empty:
            return pd.DataFrame()
        latest = eligible["snapshot_month"].max()
        return eligible[eligible["snapshot_month"] == latest].copy()

    for lag in PIPELINE_LAGS:
        feature_cols = [
            "pipeline_primary",
            "pipeline_stage_weighted_primary",
            "pipeline_qty",
            "pipeline_factored_qty",
            "pipeline_factored_revenue",
            "pipeline_stage_weighted_qty",
            "pipeline_stage_weighted_revenue",
        ]
        for col in feature_cols:
            df[f"{col}_lag{lag}"] = np.nan
        for win in [1, 2, 3]:
            df[f"pipeline_due_{win}m_lag{lag}"] = np.nan
            df[f"pipeline_stage_weighted_due_{win}m_lag{lag}"] = np.nan
        df[f"snapshot_age_lag{lag}"] = np.nan

        for target_month in df.index:
            cutoff = end_of_month(target_month - pd.DateOffset(months=lag))
            snap_df = _latest_snapshot(cutoff)
            if snap_df.empty:
                continue

            snap_month = snap_df["snapshot_month"].iloc[0]
            row = snap_df[snap_df["target_month"] == target_month]
            if not row.empty:
                row = row.iloc[0]
                for col in feature_cols:
                    if col in row:
                        df.loc[target_month, f"{col}_lag{lag}"] = row[col]

            metrics = snapshot_metrics.get(pd.Timestamp(snap_month), {})
            for win in [1, 2, 3]:
                df.loc[target_month, f"pipeline_due_{win}m_lag{lag}"] = metrics.get(
                    f"pipeline_due_{win}m", np.nan
                )
                df.loc[
                    target_month, f"pipeline_stage_weighted_due_{win}m_lag{lag}"
                ] = metrics.get(f"pipeline_stage_weighted_due_{win}m", np.nan)

            df.loc[target_month, f"snapshot_age_lag{lag}"] = month_diff(
                target_month, snap_month
            )

    pipeline_cols = [
        col
        for col in df.columns
        if col.startswith("pipeline_") or col.startswith("snapshot_age")
    ]
    if pipeline_cols:
        df["pipeline_missing_flag"] = df[pipeline_cols].isna().all(axis=1).astype(int)
    else:
        df["pipeline_missing_flag"] = 1

    return df


def _baseline_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith("y_lag")
        or col in {"month_of_year", "trend_index"}
    ]


def _spike_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if "pipeline_" in col
        or col.startswith("snapshot_")
        or col in {"month_of_year", "quarter", "pipeline_missing_flag"}
    ]


def train_baseline(X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model


def _fit_spike_classifier(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    pos = y.sum()
    neg = len(y) - pos
    scale = (neg / pos) if pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _fit_spike_regressor(
    X: pd.DataFrame, y: pd.Series
) -> XGBRegressor | None:
    if X.empty or y.empty:
        return None
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_spike_models(
    X_spike: pd.DataFrame, y_spike_label: pd.Series, y_spike_mag: pd.Series
) -> tuple[XGBClassifier, XGBRegressor | None]:
    clf = _fit_spike_classifier(X_spike, y_spike_label)
    spike_mask = (y_spike_label == 1) & y_spike_mag.notna() & np.isfinite(y_spike_mag)
    reg = _fit_spike_regressor(X_spike[spike_mask], y_spike_mag[spike_mask])
    return clf, reg


def forecast_ml_recursive(
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
        month_of_year = future_date.month
        trend_index = start_trend + (step - 1)

        if len(history) < 3 or (include_lag12 and len(history) < 12):
            raise RuntimeError("Insufficient history for recursive ML baseline forecast.")

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


def _rolling_cv_splits(index: pd.DatetimeIndex, test_size: int = 6, n_splits: int = 5):
    max_splits = (len(index) // test_size) - 1
    if max_splits < 1:
        return
    use_splits = min(n_splits, max_splits)
    splitter = TimeSeriesSplit(n_splits=use_splits, test_size=test_size)
    for train_idx, test_idx in splitter.split(index):
        yield train_idx, test_idx


def _evaluate_spike_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def _impute_spike_features(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
    spike_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    non_pipeline = {"month_of_year", "quarter", "pipeline_missing_flag"}
    pipeline_cols = [c for c in spike_cols if c not in non_pipeline]
    if not pipeline_cols:
        return out

    ref = reference_df.copy() if reference_df is not None else out.copy()
    ref = ref.sort_index()
    out = out.sort_index()

    if reference_df is not None:
        combined = pd.concat([ref[pipeline_cols], out[pipeline_cols]]).ffill()
        out[pipeline_cols] = combined.iloc[len(ref):].values
    else:
        out[pipeline_cols] = out[pipeline_cols].ffill()

    by_month = ref.groupby("month_of_year")[pipeline_cols].median()
    overall = ref[pipeline_cols].median()
    for col in pipeline_cols:
        out[col] = out[col].fillna(out["month_of_year"].map(by_month[col]))
        out[col] = out[col].fillna(overall[col])
    return out


def predict(
    df: pd.DataFrame,
    baseline_model: GradientBoostingRegressor,
    spike_clf: XGBClassifier,
    spike_reg: XGBRegressor | None,
    baseline_feature_cols: list[str],
    spike_feature_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    X_base = out[baseline_feature_cols]
    out["baseline_pred"] = baseline_model.predict(X_base)

    X_spike = out[spike_feature_cols]
    out["spike_prob"] = spike_clf.predict_proba(X_spike)[:, 1]
    if spike_reg is None:
        out["spike_mag_pred"] = 0.0
    else:
        out["spike_mag_pred"] = spike_reg.predict(X_spike)
        out["spike_mag_pred"] = out["spike_mag_pred"].clip(lower=0.0)

    out["final_forecast"] = out["baseline_pred"] + out["spike_prob"] * out[
        "spike_mag_pred"
    ]
    return out


def main() -> None:
    print("Loading data...")
    actuals = load_actuals(PRODUCT_ID, DIVISION)
    if actuals.empty:
        raise RuntimeError("No actuals found for specified product/division.")

    sf_reference = load_sf_product_reference()
    sf_codes = []
    if not sf_reference.empty:
        match = sf_reference[
            (sf_reference["group_key"] == normalize_code(PRODUCT_ID))
            & (sf_reference["business_unit_code"] == DIVISION)
        ]
        sf_codes = list(match["salesforce_product_name"].unique())

    feature_mode = load_feature_mode(PRODUCT_ID, DIVISION)
    pipeline_history = load_pipeline_history(PRODUCT_ID, DIVISION, sf_codes=sf_codes)

    df = build_features(actuals, pipeline_history, feature_mode)
    df = df.dropna(subset=["Actuals"])

    baseline_cols = _baseline_feature_cols(df)
    spike_cols = _spike_feature_cols(df)

    df = df.dropna(subset=baseline_cols)
    if df.empty:
        raise RuntimeError("Insufficient history after baseline feature creation.")
    if pipeline_history.empty:
        print("Warning: no pipeline history found; spike model will use calendar features only.")

    df = df.sort_index()
    holdout = df.iloc[-HOLDOUT_MONTHS:] if len(df) > HOLDOUT_MONTHS else df.iloc[0:0]
    train_df = df.iloc[:-HOLDOUT_MONTHS] if len(df) > HOLDOUT_MONTHS else df.copy()

    if train_df.empty:
        raise RuntimeError("Not enough history for train/holdout split.")

    print("Tuning spike threshold via rolling CV...")
    cv_results = []
    spike_params = SPIKE_K_CANDIDATES if SPIKE_LABEL_METHOD == "hybrid" else SPIKE_QUANTILE_CANDIDATES
    fold_spike_counts = []
    for param in spike_params:
        fold_mae = []
        fold_pr_auc = []
        fold_spike_mae = []

        idx = train_df.index
        fold_id = 0
        for train_idx, test_idx in _rolling_cv_splits(idx):
            fold_id += 1
            train_split = train_df.iloc[train_idx]
            test_split = train_df.iloc[test_idx]

            y_train = train_split["Actuals"]
            y_test = test_split["Actuals"]

            baseline_model = train_baseline(train_split[baseline_cols], y_train)
            baseline_pred = baseline_model.predict(test_split[baseline_cols])

            spike_labels = compute_spike_labels(y_train, SPIKE_LABEL_METHOD, param)
            baseline_series = compute_spike_baseline(y_train)
            spike_mag = (y_train - baseline_series).clip(lower=0.0)

            X_spike_train = _impute_spike_features(train_split[spike_cols], None, spike_cols)
            X_spike_test = _impute_spike_features(test_split[spike_cols], X_spike_train, spike_cols)

            spike_clf, spike_reg = train_spike_models(
                X_spike_train, spike_labels, spike_mag
            )

            spike_prob = spike_clf.predict_proba(X_spike_test)[:, 1]
            if spike_reg is None:
                spike_mag_pred = np.zeros(len(X_spike_test))
            else:
                spike_mag_pred = spike_reg.predict(X_spike_test)
                spike_mag_pred = np.clip(spike_mag_pred, 0.0, None)

            final_pred = baseline_pred + spike_prob * spike_mag_pred
            fold_mae.append(mean_absolute_error(y_test, final_pred))

            label_series = compute_spike_labels(
                pd.concat([y_train, y_test]).sort_index(), SPIKE_LABEL_METHOD, param
            )
            pr_auc = _evaluate_spike_pr_auc(label_series.loc[y_test.index], spike_prob)
            fold_pr_auc.append(pr_auc)

            spike_months = spike_labels == 1
            spike_eval_mask = spike_months & spike_mag.notna() & np.isfinite(spike_mag)
            if spike_eval_mask.any() and spike_reg is not None:
                spike_train_mae = mean_absolute_error(
                    spike_mag[spike_eval_mask],
                    spike_reg.predict(X_spike_train[spike_eval_mask]),
                )
                fold_spike_mae.append(spike_train_mae)

            fold_spike_counts.append(
                {
                    "param": param,
                    "fold": fold_id,
                    "train_spike_count": int(spike_labels.sum()),
                    "test_spike_count": int(label_series.loc[y_test.index].sum()),
                }
            )

        cv_results.append(
            {
                "param": param,
                "cv_mae": float(np.mean(fold_mae)) if fold_mae else np.nan,
                "cv_pr_auc": float(np.nanmean(fold_pr_auc)) if fold_pr_auc else np.nan,
                "cv_spike_mae": float(np.mean(fold_spike_mae)) if fold_spike_mae else np.nan,
            }
        )

    cv_df = pd.DataFrame(cv_results).sort_values("cv_mae")
    best_param = float(cv_df.iloc[0]["param"])
    print(f"Selected spike param={best_param} based on CV MAE.")
    best_fold_counts = [row for row in fold_spike_counts if row["param"] == best_param]
    if best_fold_counts:
        counts_str = "; ".join(
            f"fold {row['fold']}: train={row['train_spike_count']}, test={row['test_spike_count']}"
            for row in best_fold_counts
        )
        print(f"Spike counts by fold (best param): {counts_str}")

    print("Training final models and evaluating holdout...")
    baseline_model = train_baseline(train_df[baseline_cols], train_df["Actuals"])

    spike_labels = compute_spike_labels(train_df["Actuals"], SPIKE_LABEL_METHOD, best_param)
    baseline_series = compute_spike_baseline(train_df["Actuals"])
    spike_mag = (train_df["Actuals"] - baseline_series).clip(lower=0.0)

    X_spike_train = _impute_spike_features(train_df[spike_cols], None, spike_cols)
    spike_clf, spike_reg = train_spike_models(X_spike_train, spike_labels, spike_mag)

    holdout_metrics = {}
    if not holdout.empty:
        holdout_base = holdout.copy()
        holdout_spike = _impute_spike_features(holdout[spike_cols], X_spike_train, spike_cols)
        holdout_base[spike_cols] = holdout_spike[spike_cols]
        holdout_pred = predict(
            holdout_base,
            baseline_model,
            spike_clf,
            spike_reg,
            baseline_cols,
            spike_cols,
        )
        holdout_mae = mean_absolute_error(holdout_pred["Actuals"], holdout_pred["final_forecast"])
        label_series = compute_spike_labels(
            pd.concat([train_df["Actuals"], holdout["Actuals"]]).sort_index(),
            SPIKE_LABEL_METHOD,
            best_param,
        )
        holdout_pr_auc = _evaluate_spike_pr_auc(
            label_series.loc[holdout.index],
            holdout_pred["spike_prob"].values,
        )
        holdout_metrics = {
            "holdout_mae": holdout_mae,
            "holdout_pr_auc": holdout_pr_auc,
        }
        print(f"Holdout MAE: {holdout_mae:.2f}")
        if not np.isnan(holdout_pr_auc):
            print(f"Holdout PR-AUC: {holdout_pr_auc:.3f}")

    # Retrain on full history
    baseline_model = train_baseline(df[baseline_cols], df["Actuals"])
    spike_labels_full = compute_spike_labels(df["Actuals"], SPIKE_LABEL_METHOD, best_param)
    baseline_series_full = compute_spike_baseline(df["Actuals"])
    spike_mag_full = (df["Actuals"] - baseline_series_full).clip(lower=0.0)
    X_spike_full = _impute_spike_features(df[spike_cols], None, spike_cols)
    spike_clf, spike_reg = train_spike_models(X_spike_full, spike_labels_full, spike_mag_full)
    total_spikes = int(spike_labels_full.sum())
    spike_month_list = spike_labels_full[spike_labels_full == 1].index.to_list()
    print(f"Spike months (overall): {total_spikes}")
    if spike_month_list:
        month_str = ", ".join(dt.strftime("%Y-%m") for dt in spike_month_list)
        print(f"Spike months list: {month_str}")

    # Forecast next 12 months
    last_month = df.index.max()
    future_index = pd.date_range(
        start=last_month + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq="MS"
    )

    history_series = df["Actuals"].dropna()
    baseline_future = forecast_ml_recursive(
        baseline_model,
        list(history_series.values),
        history_series.index.max(),
        FORECAST_HORIZON,
        baseline_cols,
    )

    future_stub = pd.DataFrame(index=future_index)
    future_stub["Actuals"] = np.nan
    future_full = pd.concat([df[["Actuals"]], future_stub], axis=0)
    future_features = build_features(
        future_full.reset_index().rename(columns={"index": "Month"}),
        pipeline_history,
        feature_mode,
    )
    future_features = future_features.loc[future_index]
    future_features["baseline_pred"] = baseline_future

    X_spike_future = _impute_spike_features(
        future_features[spike_cols], X_spike_full, spike_cols
    )
    spike_prob_future = spike_clf.predict_proba(X_spike_future)[:, 1]
    if spike_reg is None:
        spike_mag_future = np.zeros(len(X_spike_future))
    else:
        spike_mag_future = spike_reg.predict(X_spike_future)
        spike_mag_future = np.clip(spike_mag_future, 0.0, None)

    final_future = baseline_future + spike_prob_future * spike_mag_future

    # Risk bands
    pipeline_risk = X_spike_full.filter(regex="pipeline_primary_lag1").max(axis=1)
    if pipeline_risk.empty:
        risk_min, risk_max = 0.0, 1.0
    else:
        risk_min = float(np.nanmin(pipeline_risk))
        risk_max = float(np.nanmax(pipeline_risk)) if np.nanmax(pipeline_risk) != risk_min else risk_min + 1.0

    base_mae = holdout_metrics.get("holdout_mae", cv_df.iloc[0]["cv_mae"])
    spike_mae = cv_df.iloc[0]["cv_spike_mae"] if not cv_df.empty else base_mae
    if pd.isna(spike_mae):
        spike_mae = base_mae

    future_risk = X_spike_future.filter(regex="pipeline_primary_lag1").max(axis=1)
    risk_scale = (future_risk - risk_min) / (risk_max - risk_min)
    risk_scale = risk_scale.fillna(0.0).clip(lower=0.0, upper=1.0)

    band_width = base_mae + risk_scale * spike_mae
    lower = (final_future - band_width).clip(lower=0.0)
    upper = final_future + band_width

    forecast_df = pd.DataFrame(
        {
            "product_id": PRODUCT_ID,
            "division": DIVISION,
            "target_month": future_index,
            "baseline_pred": baseline_future.values,
            "spike_prob": spike_prob_future,
            "spike_mag_pred": spike_mag_future,
            "final_forecast": final_future.values,
            "risk_lower": lower.values,
            "risk_upper": upper.values,
        }
    )

    reg_importance = (
        spike_reg.feature_importances_
        if spike_reg is not None
        else np.zeros(len(spike_cols))
    )
    spike_importance = pd.DataFrame(
        {
            "feature": spike_cols,
            "classifier_importance": spike_clf.feature_importances_,
            "regressor_importance": reg_importance,
        }
    ).sort_values("classifier_importance", ascending=False)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="forecast", index=False)
        cv_df.to_excel(writer, sheet_name="cv_summary", index=False)
        pd.DataFrame([holdout_metrics]).to_excel(writer, sheet_name="holdout_metrics", index=False)
        spike_importance.to_excel(writer, sheet_name="spike_feature_importance", index=False)
        pd.DataFrame(fold_spike_counts).to_excel(
            writer, sheet_name="cv_spike_counts", index=False
        )
        spike_months = pd.DataFrame(
            {
                "month": spike_labels_full[spike_labels_full == 1].index,
                "actuals": df.loc[spike_labels_full == 1, "Actuals"].values,
            }
        )
        spike_months.to_excel(writer, sheet_name="spike_months", index=False)

    print(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
