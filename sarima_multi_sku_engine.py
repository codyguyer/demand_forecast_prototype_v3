import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import ast
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===========================
# 1. CONFIG
# ===========================

INPUT_FILE = "all_products_with_sf_and_bookings.xlsx"  # <--- change to your file
OUTPUT_FILE = "sarima_multi_sku_summary.xlsx"
ORDER_FILE = "sarimax_order_search_summary.xlsx"       # per-SKU SARIMA orders

# Column names – adjust if your master file uses different names
COL_PRODUCT = "Product"
COL_DIVISION = "Division"
COL_DATE = "Month"
COL_ACTUALS = "Actuals"
COL_NEW_QUOTES = "New_Quotes"
COL_OPEN_OPPS = "Open_Opportunities"
COL_BOOKINGS = "Bookings"

# Evaluation config
TEST_HORIZON = 12       # last 12 months as holdout for Test MAE/RMSE
MIN_OBS_TOTAL = 30      # minimum total points to attempt modeling
ROCV_MIN_OBS = 24       # minimum obs for rolling-origin CV
ROCV_HORIZON = 1        # 1-step ahead in ROCV

# Acceptance thresholds
EPSILON_IMPROVEMENT = 0.02  # model must improve Test MAE by at least 2%
DELTA_ROCV_TOLERANCE = 0.05 # ROCV_MAE can be up to 5% worse than baseline


# ===========================
# 2. HELPER FUNCTIONS
# ===========================

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
    mse = mean_squared_error(y_true[mask], y_pred[mask])  # no squared arg
    return np.sqrt(mse)


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
            "Test_RMSE": safe_rmse(y_test.values, y_pred.values),
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "ROCV_MAE": float(rocv_mae) if rocv_mae is not None else np.nan,
            "Regressor_coef": np.nan,
            "Regressor_pvalue": np.nan
        }


def choose_best_model(metrics_list, epsilon=EPSILON_IMPROVEMENT, delta=DELTA_ROCV_TOLERANCE):
    """
    Given a list of metrics dicts (all for same SKU),
    apply acceptance rules and pick a best model.
    Returns chosen dict + baseline dict + some derived fields.
    """
    if not metrics_list:
        return None, None

    # Identify baseline and copy it out
    baseline = None
    for m in metrics_list:
        if m["Model"] == "SARIMA_baseline":
            baseline = m
            break

    if baseline is None:
        # Fallback: choose minimum Test_MAE
        valid = [m for m in metrics_list if not np.isnan(m["Test_MAE"])]
        if not valid:
            return None, None
        best = min(valid, key=lambda x: x["Test_MAE"])
        return best, None

    baseline_mae = baseline["Test_MAE"]
    baseline_rocv = baseline["ROCV_MAE"]

    # If baseline itself is NaN on Test_MAE, just choose lowest Test_MAE overall
    if np.isnan(baseline_mae):
        valid = [m for m in metrics_list if not np.isnan(m["Test_MAE"])]
        if not valid:
            return None, baseline
        best = min(valid, key=lambda x: x["Test_MAE"])
        return best, baseline

    accepted = []
    for m in metrics_list:
        if m["Model"] == "SARIMA_baseline":
            continue  # baseline is always a fallback, not an "accepted regressor" candidate

        mae = m["Test_MAE"]
        rocv = m["ROCV_MAE"]

        if np.isnan(mae):
            continue

        # Rule 1: must improve MAE by at least epsilon
        if mae > baseline_mae * (1.0 - epsilon):
            continue

        # Rule 2: ROCV stability (if ROCV_MAE available for both)
        if (not np.isnan(baseline_rocv)) and (not np.isnan(rocv)):
            if rocv > baseline_rocv * (1.0 + delta):
                continue

        # If ROCV is NaN, we don't auto-accept (for now, you can relax this if you want)
        if np.isnan(rocv) and not np.isnan(baseline_rocv):
            continue

        accepted.append(m)

    if not accepted:
        # No regressor model passes the rules → choose baseline
        return baseline, baseline

    # If multiple accepted, choose the one with lowest Test_MAE
    best = min(accepted, key=lambda x: x["Test_MAE"])
    return best, baseline


def engineer_regressors(df_sku):
    """
    Given a single-SKU dataframe (sorted by Month),
    engineer lagged exogenous regressors we want to test.
    NOTE: We intentionally exclude lag-0 regressors; only lags 1-3 are kept.
    """
    df = df_sku.copy()
    # Ensure numeric
    for col in [COL_NEW_QUOTES, COL_OPEN_OPPS, COL_BOOKINGS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Lagged features we are willing to use (1-3 where applicable)
    if COL_NEW_QUOTES in df.columns:
        df["New_Quotes_l1"] = df[COL_NEW_QUOTES].shift(1)
        df["New_Quotes_l2"] = df[COL_NEW_QUOTES].shift(2)
        df["New_Quotes_l3"] = df[COL_NEW_QUOTES].shift(3)

    if COL_OPEN_OPPS in df.columns:
        df["Open_Opportunities_l1"] = df[COL_OPEN_OPPS].shift(1)
        df["Open_Opportunities_l2"] = df[COL_OPEN_OPPS].shift(2)
        df["Open_Opportunities_l3"] = df[COL_OPEN_OPPS].shift(3)

    if COL_BOOKINGS in df.columns:
        df["Bookings_l1"] = df[COL_BOOKINGS].shift(1)
        df["Bookings_l2"] = df[COL_BOOKINGS].shift(2)

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


def load_orders_map(path=ORDER_FILE):
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
    return orders_map


def main():
    print("Loading data...")
    df_all = pd.read_excel(INPUT_FILE)
    print(f"Loading per-SKU SARIMA orders from {ORDER_FILE}...")
    orders_map = load_orders_map(ORDER_FILE)

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
        baseline_rocv_spec = build_rocv_spec(y, exog_series=None)

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
        print("  Fitting SARIMA_baseline...")
        baseline_metrics = evaluate_single_model(
            y=y,
            exog=None,
            model_name="SARIMA_baseline",
            horizon=TEST_HORIZON,
            order=order,
            seasonal_order=seasonal_order,
            sku=prod,
            bu=div,
            rocv_spec=baseline_rocv_spec
        )
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

        # --- SARIMAX with New_Quotes lags 1-3 ---
        for lag in [1, 2, 3]:
            col = f"New_Quotes_l{lag}"
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
                        horizon=TEST_HORIZON,
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
                    "Regressor_Name": "New_Quotes",
                    "Regressor_Lag": lag,
                    "Accepted_by_rules": False  # updated later once chosen
                })

        # --- SARIMAX with Open_Opportunities lags 1-3 ---
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
                        horizon=TEST_HORIZON,
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

        # --- SARIMAX with Bookings lags 1-2 ---
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
                        horizon=TEST_HORIZON,
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

        # --- Choose best model according to rules ---
        best_model, baseline = choose_best_model(metrics_list)

        # Identify best non-baseline candidate by Test_MAE (forcing regressor if available)
        non_baseline_candidates = [m for m in metrics_list if m["Model"] != "SARIMA_baseline" and not np.isnan(m["Test_MAE"])]
        best_nonbaseline = min(non_baseline_candidates, key=lambda x: x["Test_MAE"]) if non_baseline_candidates else None

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
            mae_improvement_pct = (baseline_mae - chosen_mae) / baseline_mae
        else:
            mae_improvement_pct = np.nan

        # Determine regressor name + lag
        regressor_name = None
        regressor_lag = None
        if best_model["Model"].startswith("SARIMAX"):
            if "New_Quotes" in best_model["Model"]:
                regressor_name = "New_Quotes"
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

        # Best non-baseline info
        best_nonbaseline_model = best_nonbaseline["Model"] if best_nonbaseline is not None else None
        best_nonbaseline_reg_name = None
        best_nonbaseline_reg_lag = None
        if best_nonbaseline_model is not None and best_nonbaseline_model.startswith("SARIMAX"):
            if "New_Quotes" in best_nonbaseline_model:
                best_nonbaseline_reg_name = "New_Quotes"
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

        accepted_by_rules = (best_model["Model"] != "SARIMA_baseline")

        print(f"  -> Chosen model: {best_model['Model']}")
        print(f"     Baseline MAE: {baseline_mae:.3f} | Chosen MAE: {chosen_mae:.3f} | Improvement: {mae_improvement_pct * 100 if not np.isnan(mae_improvement_pct) else np.nan:.2f}%")

        results_rows.append({
            "Product": prod,
            "Division": div,
            "Chosen_Model": best_model["Model"],
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
        "Chosen_Model", "Best_NonBaseline_Model", "Accepted_by_rules",
        "Baseline_MAE", "Baseline_ROCV_MAE",
        "Chosen_MAE", "Chosen_RMSE", "Chosen_ROCV_MAE",
        "MAE_Improvement_Pct",
        "Regressor_Name", "Regressor_Lag",
        "Best_NonBaseline_Regressor_Name", "Best_NonBaseline_Regressor_Lag",
        "Regressor_Coef", "Regressor_pvalue",
        "Baseline_AIC", "Baseline_BIC",
        "Chosen_AIC", "Chosen_BIC"
    ]
    df_results = df_results[col_order]

    # Build ranking table (one row per model candidate) if available
    df_rankings = pd.DataFrame(ranking_rows_all)
    if not df_rankings.empty:
        # rank within SKU by Test_MAE (ascending)
        df_rankings["Rank_by_Test_MAE"] = (
            df_rankings.groupby(["Product", "Division"])["Test_MAE"]
            .rank(method="first")
        )

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
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

    print(f"\nDone. Summary written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:,.2f} seconds")
