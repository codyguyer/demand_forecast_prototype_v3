# ML Model Testing: Pipeline GB Forecast

This README documents the end-to-end logic used by `pipeline_gb_forecast.py`.

## Overview
The script builds a baseline ML forecast from actuals only, then learns how Salesforce
pipeline data shifts that baseline. It runs a snapshot-based backtest (walk-forward by
snapshot month), reports MAE by horizon, and produces a forward 12-month forecast for
the latest snapshot.

## Inputs
- Actuals: `all_products_actuals_and_bookings.xlsx`
  - Required columns: `Product`, `Division`, `Month`, `Actuals`
- Pipeline snapshots: `Merged Salesforce Pipeline *.xlsx`
  - Required columns: `Month` (snapshot month), `Close Date`, `Business Unit`,
    `Factored Quantity`, `Factored Revenue`, plus `Quantity` or `Quantity W/ Decimal`
- Product catalog: `product_catalog_master.xlsx`
  - Required columns: `group_key`, `business_unit_code`, `salesforce_feature_mode`

## Configuration (top of script)
- `PRODUCT_IDS`: list of product IDs to evaluate.
- `DIVISION`: business unit code (e.g., `D100`).
- `FORECAST_HORIZON`: months ahead to model (default 12).
- `APPLY_LIFT_TO_FINAL`: if True, `final_pred` is replaced by `final_pred_lift`.

## Preprocessing
### Actuals
1. Filter by `Product` and `Division`.
2. Aggregate to month start and sum `Actuals` for duplicate months.
3. Prelaunch handling: all zeros before the first positive actual are set to `NaN`.

### Pipeline
1. Load all snapshot files and filter by `Division` and `Product`.
2. Parse:
   - `snapshot_month` from `Month`.
   - `target_month` from `Close Date`.
3. Aggregate by `(snapshot_month, target_month)` and sum:
   - `pipeline_qty`
   - `pipeline_factored_qty`
   - `pipeline_factored_revenue`

### Feature mode
`product_catalog_master.xlsx` defines whether pipeline uses `quantity` or `revenue`.
- `pipeline_primary` = `pipeline_factored_qty` if mode is `quantity`, else `pipeline_factored_revenue`.
- `pipeline_coverage` = `pipeline_primary / avg_actuals_12` (if available).

## Feature Engineering
### Snapshot-anchored baseline features (no leakage)
All lag and rolling features are computed **as of snapshot**:
- Latest allowed actual is `snapshot_month - 1`.
- Lags: 1, 2, 3 months (`lag_1`, `lag_2`, `lag_3`).
- Rolling means: 3, 6, 12 months (`roll_mean_3`, `roll_mean_6`, `roll_mean_12`).
- `trend_idx`: months from first actual to snapshot-1.

### Forecast row eligibility
- Only include rows where a target month has an actual value.
- Only include rows with `months_ahead <= FORECAST_HORIZON`.

## Baseline Model (actuals-only)
Baseline ML mirrors the SARIMA ML structure:
- Features: `y_lag1`, `y_lag2`, `y_lag3`, optional `y_lag12` (if >= 24 months),
  `month_of_year`, `trend_index`.
- Model: `GradientBoostingRegressor(random_state=42)`
- Training requirement: >5 non-null training rows.
- Forecasting: recursive multi-step using predicted values as new lags.

Baseline predictions for training are computed per snapshot using only actuals
available up to `snapshot_month - 1`.

## Adjustment Model (residual model)
Learns how pipeline shifts baseline:
- Target: `residual = actuals - baseline_pred`
- Features: `months_ahead`, `target_month_num`, `pipeline_primary`,
  `pipeline_coverage`, `baseline_pred`
- Model: `GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3)`
- Missing values are median-imputed.

Final prediction:
- `final_pred = baseline_pred + adjustment_pred` (clipped at 0)

## Lift Model (optional)
Predicts multiplicative lift in log space:
- Target: `log(actuals+1) - log(baseline_pred+1)`
- Features: `months_ahead`, `target_month_num`, `pipeline_primary`, `pipeline_coverage`
- Model: `GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3)`
- Output: `final_pred_lift = exp(log(baseline_pred+1) + log_lift_pred) - 1`

If `APPLY_LIFT_TO_FINAL = True`, `final_pred` is overwritten with `final_pred_lift`.

## Backtest: Snapshot Walk-Forward
- For each snapshot month S:
  - Train on all snapshots < S.
  - Predict on snapshot S.
- Outputs are concatenated across snapshots.

## Metrics
- MAE per horizon: `months_ahead` vs `baseline_pred`, `final_pred`, and `final_pred_lift`.
- Metrics are written as a table in the output Excel file.

## Outputs
`ml_pipeline_forecast.xlsx` contains 3 consolidated tabs for all products:
- `snapshot_backtest_preds`: backtest rows for all products.
- `snapshot_backtest_mae`: MAE by horizon for all products.
- `forecast_next_12`: forward 12-month forecast for the latest snapshot.

## Runtime Output
The script prints:
- Start message
- Per-product progress
- Output file location
- Total runtime in seconds and minutes

## Notes / Tolerances
- Minimum baseline training data: >5 rows after lag feature creation.
- Forecast horizon: 12 months.
- Missing pipeline rows default to 0 for pipeline quantities/revenues.
- `pipeline_coverage` is `NaN` when the 12-month average is unavailable.
- All forecasts are clipped at 0.
