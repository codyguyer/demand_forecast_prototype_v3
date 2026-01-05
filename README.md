# Demand Forecast Prototype v3

Python scripts for selecting SARIMA/SARIMAX orders by SKU, evaluating exogenous regressors, and generating a library of forecast variants from monthly data.

## Repository contents
- `sarima_order_selector.py` - grid-searches SARIMA/SARIMAX orders per (Product, Division) with rolling-origin CV and a holdout block; writes `sarimax_order_search_summary.xlsx`.
- `sarima_multi_sku_engine.py` - uses the chosen orders to compare baseline SARIMA/ETS vs. SARIMAX models with lagged regressors; writes `sarima_multi_sku_summary.xlsx`.
- `sarima_forecast.py` - builds forecast variants (baseline SARIMA/ETS and SARIMAX if allowed) for each SKU; writes `stats_model_forecasts_YYYY-Mon.xlsx`.
- `generate_forecast_variants.py` - small utility showing how to assemble forecast variants for a given SKU (used as a reference/helper).
- `.gitignore` - excludes data files (`*.xlsx`, `*.csv`) and Python build artifacts.

## Expected inputs (not committed)
Place these Excel files in the project root before running:
- `all_products_with_sf_and_bookings.xlsx` - historical monthly actuals and exogenous signals per Product/Division.
- `sarimax_order_search_summary.xlsx` - produced by `sarima_order_selector.py`.
- `sarima_multi_sku_summary.xlsx` - produced by `sarima_multi_sku_engine.py`.

Minimum required columns in `all_products_with_sf_and_bookings.xlsx`:
- `Product` - SKU or product identifier.
- `Division` - business unit or division key.
- `Month` - monthly date; treated as month-start timestamps.
- `Actuals` - historical demand (target).
- Optional exogenous signals (used when present):
  - `New_Opportunities`
  - `Open_Opportunities`
  - `Bookings`
  - `New_Quotes` (only used by the ML challenger).

## Setup
1) Install Python 3.10+
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Typical workflow
1) Search SARIMA/SARIMAX orders per SKU:
```bash
python sarima_order_selector.py
```
   - Reads `all_products_with_sf_and_bookings.xlsx`
   - Outputs `sarimax_order_search_summary.xlsx`

2) Evaluate regressors and pick a model per SKU:
```bash
python sarima_multi_sku_engine.py
```
   - Reads `all_products_with_sf_and_bookings.xlsx` and `sarimax_order_search_summary.xlsx`
   - Outputs `sarima_multi_sku_summary.xlsx`

3) Generate forecast variants:
```bash
python sarima_forecast.py
```
   - Reads `all_products_with_sf_and_bookings.xlsx`, `sarimax_order_search_summary.xlsx`, and `sarima_multi_sku_summary.xlsx`
   - Outputs `stats_model_forecasts_YYYY-Mon.xlsx`

## Model logic and assumptions (new analyst guide)

This section explains how each script makes decisions, along with thresholds, assumptions, and fallbacks.

### Shared data handling
- Duplicate Product/Division/Month rows are collapsed by summing additive series (Actuals, Bookings) and keeping the first value for other fields.
- Month fields are coerced to datetime and treated as month-start dates.
- Pre-launch zeros in `Actuals` are treated as missing: for each Product/Division, any zero before the first positive actual is replaced with NaN. A revised file `all_products_with_sf_and_bookings_revised.xlsx` is written by the engine/forecast scripts.
- If a series has no positive actuals, all actuals are set to NaN and the SKU is effectively skipped downstream.

### 1) SARIMA/SARIMAX order search (`sarima_order_selector.py`)
Purpose: choose a stable (p,d,q)(P,D,Q,s) order per SKU using SARIMAX with optional exogenous regressors.

Order grid and seasonality:
- Seasonal period `s = 12` (monthly data).
- Non-seasonal orders: `p = 0..2`, `d = 1`, `q = 0..2`.
- Seasonal orders: `P = 0..1`, `D = 1`, `Q = 0..1`.
- The all-zero order is skipped.

Evaluation logic:
- Rolling-origin CV uses `ROCV_HORIZON = 6` months and `ROCV_SPLITS = 3`. If history is short, splits are reduced to fit available data.
- Candidate orders are ranked by `rocv_MAE` then `in_sample_MAE`.
- The top `TOP_N_CANDIDATES = 5` are evaluated on a holdout block (`HOLDOUT_HORIZON = 12`), with a minimum training length of 24 months.
- If at least one candidate has a valid holdout MAE, the best holdout MAE wins; otherwise the best ROCV MAE wins.
 - NaNs are dropped before fitting or scoring, and MAE is computed with a NaN-safe helper.

Guardrails:
- SKUs with fewer than `MIN_SERIES_LENGTH = 30` observations are skipped.
- Any order that fails to converge is treated as invalid.

Exogenous regressors in order search:
- If present, `Open_Opportunities`, `New_Opportunities`, and `Bookings` are passed into SARIMAX.
- Regressor lagging is not handled here; raw columns are used.

### 2) Multi-SKU model evaluation (`sarima_multi_sku_engine.py`)
Purpose: compare baseline SARIMA and ETS against SARIMAX regressor candidates, plus optional ML and Prophet challengers, then choose a model per SKU.

Data requirements and skips:
- Minimum total observations: `MIN_OBS_TOTAL = 30`.
- If a SKU is missing an order in `sarimax_order_search_summary.xlsx`, it is skipped.
 - SARIMA baseline is only evaluated when there are at least 12 non-null months and at least 12 months since first non-zero actual.
 - SARIMAX regressors are only evaluated when there are at least 36 non-null months and SARIMA prerequisites are met.

Baseline models:
- SARIMA baseline uses the chosen order from the order search and no exogenous regressors.
- ETS baseline uses a small grid of ETS candidates:
  - Error: additive or multiplicative (multiplicative only if series is nonnegative).
  - Trend: none or additive, with optional damping.
  - Seasonal: none or additive (multiplicative only if series is nonnegative).
  - Seasonal period is 12 when seasonal is enabled.
  - Best ETS candidate is chosen by AICc (falls back to AIC).
  - If ETSModel cannot fit any candidate, it falls back to `ExponentialSmoothing` with additive error.
  - ETS fitting and scoring use non-null history only.

Regressor candidates (SARIMAX):
- Only lagged regressors are considered; lag 0 is intentionally excluded.
- Lags tested:
  - `New_Opportunities` lag 1-3
  - `Open_Opportunities` lag 1-3
  - `Bookings` lag 1-2
- Regressor series are forward/back filled for ROCV, and a regressor is considered invalid if it is all-NaN or constant.

Metrics and evaluation windows:
- Test metrics use a final holdout block of `TEST_HORIZON = 12` months, but short histories use a reduced holdout (25% of history, minimum 6).
- Rolling-origin CV (ROCV) uses `ROCV_HORIZON = 1` and `ROCV_MIN_OBS = 24`.
- Baseline MAE values at or below `BASELINE_MAE_ZERO_EPS = 1e-9` are treated as effectively perfect.

ML challenger (optional):
- GradientBoostingRegressor using lagged target features and calendar/trend:
  - Target lags: 1, 2, 3; lag 12 if 24+ observations.
  - Calendar: month-of-year; trend: index position.
  - Optional exogenous features if present: `New_Quotes`, `Open_Opportunities`, `Bookings` (with lags).
- Requires enough rows after dropping NaNs; otherwise the model is skipped.
- ROCV for ML rebuilds features at each origin and averages MAE.

Prophet challenger (optional):
- Requires `prophet` to be installed and at least 24 months of usable history.
- ROCV uses 1-step horizons with a minimum training length of 24.

Model acceptance rules:
- Baseline is the better of SARIMA vs ETS by Test MAE (ties broken by RMSE).
- A non-baseline candidate must satisfy BOTH:
  - Test MAE improves baseline by at least 2% (`EPSILON_IMPROVEMENT = 0.02`).
  - ROCV MAE is not more than 5% worse than baseline (`DELTA_ROCV_TOLERANCE = 0.05`).
- If candidate ROCV is NaN but baseline ROCV is valid, the candidate is rejected.
- If no candidate passes, the baseline is selected.

Outputs:
- `sarima_multi_sku_summary.xlsx` contains the chosen model, baseline metrics, improvement %, and selected regressor name/lag.
- `Model_Rankings` sheet includes all candidates and whether each passed the acceptance rules.

### 3) Forecast generation (`sarima_forecast.py`)
Purpose: generate forecast variants for each SKU and flag the recommended model from the summary file.

History thresholds:
- SARIMA baseline is allowed if:
  - `history_months >= 12`, and
  - the first nonzero actual is at least 12 months in the past (>= 11 months difference).
- SARIMAX (with regressor) is allowed if:
  - `history_months >= 36`, and
  - SARIMA prerequisites are met.
- ETS baseline is always allowed; seasonal ETS requires 24+ months.
- ML and Prophet are controlled by `ENABLE_ML_CHALLENGER` / `ENABLE_PROPHET_CHALLENGER` and package availability.

Regressor selection for forecasting:
- Uses the regressor name/lag in `sarima_multi_sku_summary.xlsx` if present.
- If the chosen model was not SARIMAX and a regressor is missing, the engine selects a default regressor with the most usable history among:
  - `New_Opportunities` lag 1-3
  - `Open_Opportunities` lag 1-3
  - `Bookings` lag 1-2
- Any lag-0 regressor is dropped and the SARIMAX variant is skipped.

Future exogenous handling:
- The regressor series is reindexed to the forecast horizon.
- Missing future values are forward-filled.
- If `FILL_MISSING_FUTURE_EXOG_WITH_LAST = True`, remaining gaps are filled with the last observed value.
- If future exog is still missing after these steps, the SARIMAX variant is skipped and the model falls back to baselines/others.

Forecast variants:
- Baseline SARIMA (seasonal) using the selected order.
- ETS baseline using the same AICc/AIC selection logic as the engine.
- SARIMAX with the chosen regressor if allowed and future exog is available.
- ML and Prophet challengers (if enabled).

Stability and post-processing:
- Forecast horizon is 12 months (`FORECAST_HORIZON = 12`).
- Forecast values and lower bounds are clipped at `FORECAST_FLOOR = 0.0`.
- ETS forecasts are validated for stability to avoid extreme or negative values for nonnegative series.
- The output adds a `recommended_model` boolean based on the chosen model in `sarima_multi_sku_summary.xlsx`.

### Manual SARIMA order overrides (`Notes.xlsx`)
If `Notes.xlsx` is present, the scripts will override SARIMA orders per SKU.
Required columns:
- `Product`, `Division` (or `group_key`, `BU`)
- `Chosen_Order` (tuple-like, length 3)
- `Chosen_Seasonal_Order` (tuple-like, length 4)

### Configuration summary (quick reference)
Edit these at the top of each script to change behavior:
- `sarima_order_selector.py`: order grid ranges, ROCV horizon/splits, holdout horizon, minimum series length.
- `sarima_multi_sku_engine.py`: test horizon, ROCV horizon/min obs, acceptance thresholds, ETS seasonal period, ML/Prophet toggles.
- `sarima_forecast.py`: forecast horizon, history thresholds for SARIMA/SARIMAX, ETS stability limits, exogenous filling behavior, ML/Prophet toggles.

## Notes
- Exogenous candidates include lagged Salesforce signals (`New_Opportunities`, `Open_Opportunities`) and `Bookings`; lag logic is explicitly defined in each script.
- Rolling-origin CV and holdout windows are configured at the top of the scripts; adjust there for different horizons or minimum history.
- Data files are ignored by Git; share them separately if collaborators need to reproduce runs.
