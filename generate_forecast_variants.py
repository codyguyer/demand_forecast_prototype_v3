import uuid
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    SARIMA baseline, ARIMA baseline, and SARIMAX-with-regressor (if exog provided).
    """

    run_id = uuid.uuid4()

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
    # 2) ARIMA baseline (seasonal orders set to 0)
    # ---------------------------
    arima_model = SARIMAX(
        y,
        order=sarima_order,
        seasonal_order=(0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()

    arima_forecast = arima_model.get_forecast(steps=horizon)
    arima_mean = arima_forecast.predicted_mean
    arima_df = arima_mean.to_frame(name="forecast_value")
    arima_df["model_group"] = "baseline_arima"
    arima_df["model_type"]  = "ARIMA"
    arima_df["p"], arima_df["d"], arima_df["q"] = sarima_order
    arima_df["P"], arima_df["D"], arima_df["Q"], arima_df["s"] = (0,0,0,0)
    arima_df["regressor_names"]   = [None] * len(arima_df)
    arima_df["regressor_details"] = [None] * len(arima_df)

    dfs = [sarima_df, arima_df]

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
    final_df.to_csv("forecast_library_temp.csv", index=False)

    return final_df
