import os
import warnings
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress noisy convergence messages while grid searching
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEASONAL_PERIOD = 12  # monthly data with annual seasonality
DEFAULT_D = 1
DEFAULT_D_SEASONAL = 1
MAX_P = 2
MAX_Q = 2
MAX_P_SEASONAL = 1
MAX_Q_SEASONAL = 1

# Rolling-origin CV settings
ROCV_HORIZON = 6   # forecast horizon per split (months)
ROCV_SPLITS = 3    # number of rolling splits

# Holdout evaluation
TOP_N_CANDIDATES = 5
HOLDOUT_HORIZON = 12
MIN_TRAIN_FOR_HOLDOUT = 24
MIN_SERIES_LENGTH = 30  # guardrail to skip very short series


def _output_path(filename: str) -> str:
    if not filename:
        return filename
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    return os.path.join(OUTPUT_DIR, filename)


# --------------------------------------------------
# DATA PREP
# --------------------------------------------------
def aggregate_monthly_duplicates(
    df: pd.DataFrame,
    product_col: str = "Product",
    division_col: str = "Division",
    date_col: str = "Month",
    sum_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Collapse duplicate Product/Division/Month rows by summing key numeric fields.
    Also preserves any extra columns by taking the first value within each group.
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    # Only sum true additive series; keep others (e.g., opportunities) as first.
    sum_cols = sum_cols or ["Actuals", "Bookings"]
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


# --------------------------------------------------
# 1. ORDER GRID
# --------------------------------------------------
def build_order_grid(
    max_p: int = MAX_P,
    max_q: int = MAX_Q,
    max_P: int = MAX_P_SEASONAL,
    max_Q: int = MAX_Q_SEASONAL,
    d: int = DEFAULT_D,
    D: int = DEFAULT_D_SEASONAL,
    m: int = SEASONAL_PERIOD,
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]:
    """Enumerate (p,d,q)(P,D,Q,m) combinations."""
    orders = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(max_P + 1):
                for Q in range(max_Q + 1):
                    if (p, d, q) == (0, 0, 0) and (P, D, Q, m) == (0, 0, 0, m):
                        continue
                    orders.append(((p, d, q), (P, D, Q, m)))
    return orders


# --------------------------------------------------
# 2. UTILS: ROCV SPLITS
# --------------------------------------------------
def generate_rocv_splits(n_obs: int, horizon: int = ROCV_HORIZON, n_splits: int = ROCV_SPLITS):
    """Rolling-origin splits; returns a list of (train_end, test_end) indices."""
    splits = []
    total_needed = horizon * n_splits + horizon
    if n_obs < total_needed:
        n_splits = max(1, (n_obs // horizon) - 1)

    for i in range(n_splits):
        test_end = n_obs - (n_splits - (i + 1)) * horizon
        train_end = test_end - horizon
        if train_end <= horizon:
            continue
        splits.append((train_end, test_end))
    return splits


def _safe_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute MAE while ignoring NaNs/Infs."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    mask = np.isfinite(y_true.values) & np.isfinite(y_pred.values)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(y_true.values[mask] - y_pred.values[mask])))


def _clean_series_and_exog(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """Align and drop rows with missing values in y or exog."""
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    if exog is None:
        return y.dropna(), None

    exog = pd.DataFrame(exog).apply(pd.to_numeric, errors="coerce")
    df = pd.concat([y.rename("y"), exog], axis=1)
    df = df.dropna()
    if df.empty:
        return pd.Series([], dtype=float), exog.iloc[0:0]
    y_clean = df["y"]
    exog_clean = df.drop(columns=["y"])
    return y_clean, exog_clean


# --------------------------------------------------
# 3. FIT + EVALUATE A SINGLE ORDER
# --------------------------------------------------
def evaluate_sarimax_order(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> dict:
    results = {
        "order": order,
        "seasonal_order": seasonal_order,
        "converged": False,
        "in_sample_MAE": np.nan,
        "rocv_MAE": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
    }

    y_clean, exog_clean = _clean_series_and_exog(y, exog)
    if len(y_clean) < 3:
        return results

    try:
        model = SARIMAX(
            y_clean,
            exog=exog_clean,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit_res = model.fit(disp=False, maxiter=200)
    except Exception:
        return results

    # In-sample fit quality
    fitted = fit_res.fittedvalues
    y_aligned, fitted_aligned = y_clean.align(fitted, join="inner")
    in_sample_mae = _safe_mae(y_aligned, fitted_aligned)

    # Information criteria
    aic = fit_res.aic
    bic = fit_res.bic

    # Rolling-origin CV
    n_obs = len(y_clean)
    splits = generate_rocv_splits(n_obs)
    rocv_maes = []
    for train_end, test_end in splits:
        y_train = y_clean.iloc[:train_end]
        y_test = y_clean.iloc[train_end:test_end]
        exog_train = exog_clean.iloc[:train_end, :] if exog_clean is not None else None
        exog_test = exog_clean.iloc[train_end:test_end, :] if exog_clean is not None else None

        try:
            model_cv = SARIMAX(
                y_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit_cv = model_cv.fit(disp=False, maxiter=200)
            forecast = fit_cv.forecast(steps=len(y_test), exog=exog_test)
            rocv_maes.append(_safe_mae(y_test, forecast))
        except Exception:
            rocv_maes = []
            break

    rocv_mae = np.nanmean(rocv_maes) if rocv_maes else np.nan

    results.update(
        {
            "converged": True,
            "in_sample_MAE": in_sample_mae,
            "rocv_MAE": rocv_mae,
            "AIC": aic,
            "BIC": bic,
        }
    )
    return results


# --------------------------------------------------
# 4. CHOOSE BEST MODEL FOR ONE SKU
# --------------------------------------------------
def evaluate_holdout_horizon(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    horizon: int = HOLDOUT_HORIZON,
    min_train: int = MIN_TRAIN_FOR_HOLDOUT,
) -> float:
    """Evaluate a single order on a final holdout block."""
    y_clean, exog_clean = _clean_series_and_exog(y, exog)
    n_obs = len(y_clean)
    if n_obs <= horizon + min_train:
        return np.nan

    train_end = n_obs - horizon
    y_train = y_clean.iloc[:train_end]
    y_test = y_clean.iloc[train_end:]
    exog_train = exog_clean.iloc[:train_end, :] if exog_clean is not None else None
    exog_test = exog_clean.iloc[train_end:, :] if exog_clean is not None else None

    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit_res = model.fit(disp=False, maxiter=200)
        forecast = fit_res.forecast(steps=horizon, exog=exog_test)
        holdout_mae = _safe_mae(y_test, forecast)
        return holdout_mae
    except Exception:
        return np.nan


def select_best_model_for_sku(y: pd.Series, exog: Optional[pd.DataFrame] = None):
    """
    Grid-search SARIMA/SARIMAX orders for a single SKU.
    Returns (df_results, best_row_dict or None).
    """
    order_grid = build_order_grid()
    all_results = []
    for order, seasonal_order in order_grid:
        metrics = evaluate_sarimax_order(y, exog, order, seasonal_order)
        all_results.append(metrics)
    df_results = pd.DataFrame(all_results)

    df_valid = df_results[df_results["converged"] & df_results["rocv_MAE"].notna()].copy()
    if df_valid.empty:
        df_valid = df_results[df_results["converged"]].copy()
    if df_valid.empty:
        return df_results, None

    df_valid = df_valid.sort_values(["rocv_MAE", "in_sample_MAE"], ascending=[True, True])
    top_candidates = df_valid.head(TOP_N_CANDIDATES).copy()

    holdout_maes = []
    for _, row in top_candidates.iterrows():
        order = tuple(row["order"])
        seasonal_order = tuple(row["seasonal_order"])
        holdout_mae = evaluate_holdout_horizon(
            y,
            exog,
            order,
            seasonal_order,
            horizon=HOLDOUT_HORIZON,
            min_train=MIN_TRAIN_FOR_HOLDOUT,
        )
        holdout_maes.append(holdout_mae)
    top_candidates["holdout_MAE"] = holdout_maes

    mask_valid_holdout = np.isfinite(top_candidates["holdout_MAE"])
    if mask_valid_holdout.any():
        df_h = top_candidates[mask_valid_holdout].copy().sort_values("holdout_MAE")
        best_row = df_h.iloc[0].to_dict()
    else:
        best_row = top_candidates.iloc[0].to_dict()

    return df_results, best_row


# --------------------------------------------------
# 5. MASTER LOOP OVER PRODUCTS
# --------------------------------------------------
def run_order_search_for_all_products(
    df: pd.DataFrame,
    product_col: str = "Product",
    division_col: str = "Division",
    date_col: str = "Month",
    target_col: str = "Actuals",
    exog_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run order search for every (product, division) pair."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([product_col, division_col, date_col])

    results_summary = []
    for (prod, div), grp in df.groupby([product_col, division_col]):
        grp = grp.set_index(date_col)
        y_raw = grp[target_col]
        exog_raw = grp[exog_cols] if exog_cols is not None else None
        y, exog = _clean_series_and_exog(y_raw, exog_raw)

        if len(y) < MIN_SERIES_LENGTH:
            results_summary.append(
                {
                    "Product": prod,
                    "Division": div,
                    "Status": f"Too short (<{MIN_SERIES_LENGTH})",
                    "Chosen_Order": None,
                    "Chosen_Seasonal_Order": None,
                    "Chosen_ROCV_MAE": np.nan,
                    "Chosen_InSample_MAE": np.nan,
                    "Chosen_Holdout_MAE": np.nan,
                    "Chosen_AIC": np.nan,
                    "Chosen_BIC": np.nan,
                }
            )
            continue

        print(f"Processing Product {prod}, Division {div} (n={len(y)})")
        all_results, best = select_best_model_for_sku(y, exog)
        if best is None:
            results_summary.append(
                {
                    "Product": prod,
                    "Division": div,
                    "Status": "No converged models",
                    "Chosen_Order": None,
                    "Chosen_Seasonal_Order": None,
                    "Chosen_ROCV_MAE": np.nan,
                    "Chosen_InSample_MAE": np.nan,
                    "Chosen_Holdout_MAE": np.nan,
                    "Chosen_AIC": np.nan,
                    "Chosen_BIC": np.nan,
                }
            )
            continue

        results_summary.append(
            {
                "Product": prod,
                "Division": div,
                "Status": "OK",
                "Chosen_Order": best["order"],
                "Chosen_Seasonal_Order": best["seasonal_order"],
                "Chosen_ROCV_MAE": best.get("rocv_MAE", np.nan),
                "Chosen_InSample_MAE": best.get("in_sample_MAE", np.nan),
                "Chosen_Holdout_MAE": best.get("holdout_MAE", np.nan),
                "Chosen_AIC": best.get("AIC", np.nan),
                "Chosen_BIC": best.get("BIC", np.nan),
            }
        )

    return pd.DataFrame(results_summary)


# --------------------------------------------------
# 6. EXAMPLE USAGE
# --------------------------------------------------
if __name__ == "__main__":
    t0 = time.perf_counter()
    combined_file = "all_products_with_sf_and_bookings.xlsx"
    df_all = pd.read_excel(combined_file)
    df_all = aggregate_monthly_duplicates(
        df_all,
        product_col="Product",
        division_col="Division",
        date_col="Month",
    )

    exog_columns = [
        "Open_Opportunities",
        "New_Opportunities",
        "Bookings",
        "Median_Months_Since_Last_Activity",
        "Open_Not_Modified_90_Days",
        "Pct_Open_Not_Modified_90_Days",
        "Early_to_Late_Ratio",
    ]

    summary = run_order_search_for_all_products(
        df_all,
        product_col="Product",
        division_col="Division",
        date_col="Month",
        target_col="Actuals",
        exog_cols=exog_columns,
    )
    output_path = _output_path("sarimax_order_search_summary.xlsx")
    summary.to_excel(output_path, index=False)
    print(f"Saved {output_path}")
    elapsed = time.perf_counter() - t0
    minutes = elapsed / 60.0
    print(f"Total runtime: {elapsed:,.2f} seconds ({minutes:,.2f} minutes)")
