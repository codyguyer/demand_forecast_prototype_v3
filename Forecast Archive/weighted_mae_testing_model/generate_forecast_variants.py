import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _output_path(filename: str) -> str:
    if not filename:
        return filename
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    return os.path.join(OUTPUT_DIR, filename)


# Lock in the Forecast Table Shape (in Pandas)
def generate_forecast_variants(
    y,
    product_id,
    bu_id,
    sarima_order=(2,1,0),
    seasonal_order=(1,1,0,12),
    horizon=12,
    X_train=None,
    X_future=None
):
    """
    Returns a long-form DataFrame with:
    SARIMA baseline, ETS baseline, and SARIMAX-with-regressor (if exog provided).
    """

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    # ---------------------------
    # 1) SARIMA baseline
    # ---------------------------
    sarima_model = SARIMAX(
        y,
        order=sarima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()

    sarima_forecast = sarima_model.get_forecast(steps=horizon)
    sarima_mean = sarima_forecast.predicted_mean
    sarima_df = sarima_mean.to_frame(name="forecast_value")
    sarima_df["model_group"] = "baseline_sarima"
    sarima_df["model_type"]  = "SARIMA"
    sarima_df["p"], sarima_df["d"], sarima_df["q"] = sarima_order
    sarima_df["P"], sarima_df["D"], sarima_df["Q"], sarima_df["s"] = seasonal_order
    sarima_df["regressor_names"]   = [None] * len(sarima_df)
    sarima_df["regressor_details"] = [None] * len(sarima_df)

    # ---------------------------
    # 2) ETS baseline (AICc/AIC selection)
    # ---------------------------
    def _series_is_nonnegative(y_series):
        y_clean = pd.Series(y_series).dropna()
        return (not y_clean.empty) and (y_clean >= 0).all()

    def _ets_candidate_specs(y_series):
        allow_mul = _series_is_nonnegative(y_series)
        error_opts = ["add"] + (["mul"] if allow_mul else [])
        trend_opts = [None, "add"]
        seasonal_opts = [None, "add"] + (["mul"] if allow_mul else [])
        specs = []
        for error in error_opts:
            for trend in trend_opts:
                damped_opts = [False] if trend is None else [False, True]
                for damped in damped_opts:
                    for seasonal in seasonal_opts:
                        specs.append({
                            "error": error,
                            "trend": trend,
                            "damped_trend": damped,
                            "seasonal": seasonal,
                            "seasonal_periods": 12 if seasonal is not None else None,
                        })
        return specs

    def _info_criterion(res):
        aicc = getattr(res, "aicc", None)
        if aicc is not None and np.isfinite(aicc):
            return float(aicc)
        aic = getattr(res, "aic", None)
        if aic is not None and np.isfinite(aic):
            return float(aic)
        return np.inf

    def _fit_ets_candidate(y_series, spec, use_state_space):
        if use_state_space:
            model = ETSModel(
                y_series,
                error=spec["error"],
                trend=spec["trend"],
                damped_trend=spec["damped_trend"],
                seasonal=spec["seasonal"],
                seasonal_periods=spec["seasonal_periods"],
                initialization_method="estimated",
            )
            return model.fit()
        model = ExponentialSmoothing(
            y_series,
            trend=spec["trend"],
            damped_trend=spec["damped_trend"] if spec["trend"] is not None else False,
            seasonal=spec["seasonal"],
            seasonal_periods=spec["seasonal_periods"] if spec["seasonal"] is not None else None,
            initialization_method="estimated",
        )
        return model.fit(optimized=True)

    def _select_best_ets_model(y_series):
        best_res = None
        best_spec = None
        best_score = np.inf

        for spec in _ets_candidate_specs(y_series):
            try:
                res = _fit_ets_candidate(y_series, spec, use_state_space=True)
                score = _info_criterion(res)
            except Exception:
                continue
            if score < best_score:
                best_res = res
                best_spec = spec
                best_score = score

        if best_res is not None:
            return best_res, best_spec

        for spec in _ets_candidate_specs(y_series):
            if spec["error"] != "add":
                continue
            try:
                res = _fit_ets_candidate(y_series, spec, use_state_space=False)
                score = _info_criterion(res)
            except Exception:
                continue
            if score < best_score:
                best_res = res
                best_spec = spec
                best_score = score
        return best_res, best_spec

    ets_df = None
    try:
        ets_res, _ets_spec = _select_best_ets_model(y)
        if ets_res is None:
            raise ValueError("ETS baseline could not be fit.")
        try:
            ets_forecast = ets_res.get_forecast(steps=horizon)
            ets_mean = ets_forecast.predicted_mean
        except Exception:
            ets_mean = ets_res.forecast(steps=horizon)

        ets_df = ets_mean.to_frame(name="forecast_value")
        ets_df["model_group"] = "baseline_ets"
        ets_df["model_type"]  = "ETS"
        ets_df["p"], ets_df["d"], ets_df["q"] = (np.nan, np.nan, np.nan)
        ets_df["P"], ets_df["D"], ets_df["Q"], ets_df["s"] = (np.nan, np.nan, np.nan, np.nan)
        ets_df["regressor_names"]   = [None] * len(ets_df)
        ets_df["regressor_details"] = [None] * len(ets_df)
    except Exception as exc:
        print(f"[WARN] ETS baseline failed: {exc}")

    dfs = [sarima_df] + ([ets_df] if ets_df is not None else [])

    # ---------------------------
    # 3) With regressor(s), if provided
    # ---------------------------
    if (X_train is not None) and (X_future is not None):
        regressor_list = list(X_train.columns)

        sarimax_model = SARIMAX(
            y,
            exog=X_train,
            order=sarima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

        sarimax_forecast = sarimax_model.get_forecast(steps=horizon, exog=X_future)
        sarimax_mean = sarimax_forecast.predicted_mean
        sarimax_df = sarimax_mean.to_frame(name="forecast_value")
        sarimax_df["model_group"] = "with_regressor"
        sarimax_df["model_type"]  = "SARIMAX"
        sarimax_df["p"], sarimax_df["d"], sarimax_df["q"] = sarima_order
        sarimax_df["P"], sarimax_df["D"], sarimax_df["Q"], sarimax_df["s"] = seasonal_order
        sarimax_df["regressor_names"]   = [regressor_list] * len(sarimax_df)
        sarimax_df["regressor_details"] = [None] * len(sarimax_df)  # can upgrade later

        dfs.append(sarimax_df)

    # ---------------------------
    # Combine and add common metadata
    # ---------------------------
    all_df = pd.concat(dfs)
    all_df.index.name = "forecast_month"
    all_df.reset_index(inplace=True)

    # horizon (1..horizon)
    all_df["horizon_months_ahead"] = (
        all_df.groupby(["model_group"]).cumcount() + 1
    )

    all_df["product_id"]     = product_id
    all_df["bu_id"]          = bu_id
    all_df["run_id"]         = str(run_id)
    all_df["training_start"] = y.index.min()
    all_df["training_end"]   = y.index.max()

    # ensure forecast_month is a date (month start)
    all_df["forecast_month"] = pd.to_datetime(all_df["forecast_month"]).dt.to_period("M").dt.to_timestamp()

    return all_df


# Add a “Batch Runner” That Writes to CSV (Temporary Stand-in for Postgres)
def run_all_products(product_bu_list, data_loader, horizon=12):
    """
    product_bu_list: list of (product_id, bu_id)
    data_loader: function that returns (y, X_train, X_future) for a product/bu
    """

    all_results = []

    for product_id, bu_id in product_bu_list:
        print(f"Processing product {product_id}, BU {bu_id}...")
        y, X_train, X_future = data_loader(product_id, bu_id)

        df_forecasts = generate_forecast_variants(
            y=y,
            product_id=product_id,
            bu_id=bu_id,
            sarima_order=(2,1,0),          # example
            seasonal_order=(1,1,0,12),     # example
            horizon=horizon,
            X_train=X_train,
            X_future=X_future
        )
        all_results.append(df_forecasts)

    final_df = pd.concat(all_results, ignore_index=True)

    # TEMP: write to CSV
    final_df.to_csv(_output_path("forecast_library_temp.csv"), index=False)

    return final_df
