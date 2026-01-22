# Demand Forecast Prototype v3

Python scripts for selecting SARIMA/SARIMAX orders by SKU, evaluating exogenous regressors, and generating a library of forecast variants from monthly data.

## Repository contents
- `sarima_order_selector.py` - grid-searches SARIMA/SARIMAX orders per (Product, Division) with rolling-origin CV and a holdout block; writes `sarimax_order_search_summary.xlsx`.
- `sarima_multi_sku_engine.py` - uses the chosen orders to compare baseline SARIMA/ETS vs. SARIMAX, plus SBA and optional ML/Prophet challengers; writes `sarima_multi_sku_summary.xlsx`.
- `sarima_forecast.py` - builds forecast variants per SKU, then keeps only the recommended model and the blended softmax forecast; writes `stats_model_forecasts_YYYY-Mon.xlsx`.
- `generate_forecast_variants.py` - small utility showing how to assemble forecast variants for a given SKU (used as a reference/helper).
- `compare_model_selection.py` - optional report comparing legacy vs. current selection logic (bias override impacts).
- `.gitignore` - excludes data files (`*.xlsx`, `*.csv`) and Python build artifacts.

## Expected inputs (not committed)
Place these Excel files in the project root before running:
- `all_products_actuals_and_bookings.xlsx` - historical monthly actuals and exogenous signals per Product/Division.
- `sarimax_order_search_summary.xlsx` - produced by `sarima_order_selector.py`.
- `sarima_multi_sku_summary.xlsx` - produced by `sarima_multi_sku_engine.py`.
If you are sourcing from the raw Essbase exports, generate the combined actuals/bookings file with:
```bash
python build_actuals_bookings_from_raw.py
```
Additional Salesforce inputs (stored in `salesforce_data/`):
- `Salesforce Pipeline Monthly Summary.xlsx` - monthly Salesforce summary metrics by `group_key` and `BU` (now the source for SARIMAX regressors and pipeline ML features).
- `Merged Salesforce Pipeline *.xlsx` - historical pipeline snapshots used by the pipeline GB model.

Minimum required columns in `all_products_actuals_and_bookings.xlsx`:
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
   - Reads `all_products_actuals_and_bookings.xlsx`
   - Outputs `sarimax_order_search_summary.xlsx`

2) Evaluate regressors and pick a model per SKU:
```bash
python sarima_multi_sku_engine.py
```
   - Reads `all_products_actuals_and_bookings.xlsx` and `sarimax_order_search_summary.xlsx`
   - Outputs `sarima_multi_sku_summary.xlsx`
   - Optional: add `--compare-model-selection` to generate old vs. new selection comparison CSVs in `data_storage/model_selection_eval/`

3) Generate forecasts:
```bash
python sarima_forecast.py
```
   - Reads `all_products_actuals_and_bookings.xlsx`, `sarimax_order_search_summary.xlsx`, and `sarima_multi_sku_summary.xlsx`
   - Outputs `stats_model_forecasts_YYYY-Mon.xlsx`

## Model logic and assumptions (new analyst guide)

This section explains how each script makes decisions, along with thresholds, assumptions, and fallbacks.

### Shared data handling
- Duplicate Product/Division/Month rows are collapsed by summing additive series (Actuals, Bookings) and keeping the first value for other fields.
- Month fields are coerced to datetime and treated as month-start dates.
- Pre-launch zeros in `Actuals` are treated as missing: for each Product/Division, any zero before the first positive actual is replaced with NaN. A revised file `all_products_actuals_and_bookings_revised.xlsx` is written by the engine/forecast scripts.
- If a series has no positive actuals, all actuals are set to NaN and the SKU is effectively skipped downstream.
- SARIMAX regressors are sourced from `salesforce_data/Salesforce Pipeline Monthly Summary.xlsx` (by `group_key`, `BU`, and Month) instead of the actuals file.

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
- A `skipped_products.csv` file is written with reasons for skips (no order, insufficient history, or pipeline ML prerequisites).

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

SBA challenger:
- SBA (Syntetos-Boylan Approximation) is evaluated for intermittent demand.
- SBA metrics use the same 12-month holdout and ROCV scheme as other models.

Regressor candidates (SARIMAX):
- Only lagged regressors are considered; lag 0 is intentionally excluded.
- Lags tested:
  - `New_Opportunities` lag 1-3
  - `Open_Opportunities` lag 1-3
  - `Bookings` lag 1-2
- Regressor series are forward/back filled for ROCV, and a regressor is considered invalid if it is all-NaN or constant.

Metrics and evaluation windows:
- Test metrics always use a final holdout block of `TEST_HORIZON = 12` months for all models.
- Rolling-origin CV (ROCV) uses `ROCV_HORIZON = 1` and `ROCV_MIN_OBS = 24`.
- Baseline MAE values at or below `BASELINE_MAE_ZERO_EPS = 1e-9` are treated as effectively perfect.
 - Holdout accuracy metrics (MAE/RMSE/Bias) are weighted toward recent months:
   - Most recent 3 months = 55%
   - Months 4-6 = 30%
   - Months 7-12 = 15%

ML challenger (optional):
- `ML_GBR_PIPELINE` is evaluated when Salesforce pipeline history and summary data are available for the SKU:
  - Baseline GBR on lagged actuals + calendar/trend.
  - Adjustment GBR on residuals using pipeline + summary features.
  - Metrics use the same 12-month holdout as other models.
- If pipeline data is missing or incomplete, the engine falls back to legacy `ML_GBR`:
  - GradientBoostingRegressor using lagged target features and calendar/trend.
  - Target lags: 1, 2, 3; lag 12 if 24+ observations.
  - Calendar: month-of-year; trend: index position.
  - Optional exogenous features if present: `New_Quotes`, `Open_Opportunities`, `Bookings` (with lags).
  - Requires enough rows after dropping NaNs; otherwise the model is skipped.
  - ROCV for ML rebuilds features at each origin and averages MAE.

Prophet challenger (optional):
- Requires `prophet` to be installed and at least 24 months of usable history.
- ROCV uses 1-step horizons with a minimum training length of 24.

Recommended model selection (updated logic):
We anchor on accuracy, keep ROCV as a sanity check only, and allow a directional-accuracy override when it materially improves decision usefulness without sacrificing accuracy.

Inputs per candidate row:
- `Product`, `Division`, `Model`
- `Test_MAE`, `Test_RMSE`
- `ROCV_MAE` (sanity-only; higher = more volatile)
- Holdout forecast-vs-actual table to compute directional accuracy (DA)

Step 0: Baseline model (most accurate)
- `baseline_model` = model with lowest `Test_MAE`
- Capture `baseline_mae`, `baseline_rmse`, `baseline_rocv`, and `baseline_DA`
- Baseline always passes gates

Step 1: Accuracy gate (relative to baseline)
- `mae_cutoff = baseline_mae * (1 + mae_tolerance)`
- `rmse_cutoff = baseline_rmse * (1 + rmse_tolerance)`
- Defaults: `mae_tolerance = 0.20`, `rmse_tolerance = 0.20`
- `passes_accuracy = (Test_MAE <= mae_cutoff) AND (Test_RMSE <= rmse_cutoff)`

Step 2: ROCV sanity check (non-preferential)
- `rocv_hard_max = baseline_rocv * rocv_hard_multiplier`
- Default: `rocv_hard_multiplier = 1.50`
- If baseline ROCV is missing or <=0, ROCV checks are disabled for the group
- Otherwise, `passes_rocv_sanity = (ROCV_MAE <= rocv_hard_max)`; missing ROCV fails for non-baselines

Step 3: Candidate set and primary selection
- `candidates = passes_accuracy AND passes_rocv_sanity`
- `tentative_winner` = lowest `Test_MAE` in candidates
- Tie-breakers: lower `Test_RMSE`, then alphabetical `Model`

Step 4: Directional accuracy override
- DA = mean of sign(Actual[t] - Actual[t-1]) == sign(Forecast[t] - Forecast[t-1])
- Require at least `DA_MIN_PERIODS = 6` valid deltas
- Override challenger must satisfy:
  - DA improves by >= `DA_IMPROVEMENT_PP = 0.10` (10pp)
  - `Test_MAE <= baseline_mae * (1 + DA_CLOSE_MAE_TOL)` with default 0.05
  - passes accuracy + ROCV sanity
- If any challenger qualifies: pick highest DA; tie-break by MAE, RMSE, Model

Step 5: Bias override (new)
- Bias is computed on the same holdout block as MAE/RMSE:
  - `bias = sum(Forecast - Actual)`
  - `abs_bias = abs(bias)`
  - `bias_pct = bias / sum(Actual)` with epsilon guard
  - `abs_bias_pct = abs(bias_pct)`
- Challenger must already be accuracy-qualified and also meet:
  - `Test_MAE <= baseline_mae * (1 + mae_close_tol)` (default 0.10)
  - `abs_bias_pct <= baseline_abs_bias_pct * bias_improvement_ratio` (default 0.70)
  - optional RMSE-close check if enabled
- If any bias challenger qualifies: pick lowest abs bias (pct), tie-break by MAE, RMSE, Model.
- Bias override never special-cases ML; it can only select challengers already passing accuracy + ROCV sanity.

Outputs (summary and debug):
- `Product`, `Division`, `Recommended_Model`, `Reason`
- `baseline_model`, `baseline_mae`, `baseline_rmse`, `baseline_rocv`, `baseline_DA`
- `baseline_abs_bias`, `baseline_abs_bias_pct`
- `mae_cutoff`, `rmse_cutoff`, `rocv_hard_max`
- `recommended_mae`, `recommended_rmse`, `recommended_rocv`, `recommended_DA`
- `recommended_bias`, `recommended_abs_bias`, `recommended_bias_pct`, `recommended_abs_bias_pct`
- Per-model flags: `passes_accuracy`, `passes_rocv_sanity`, `candidate`, `DA`, `DA_Valid_Periods`,
  `qualifies_mae_close`, `qualifies_bias`, `qualifies_rmse_close`, `used_bias_override`

Outputs:
- `sarima_multi_sku_summary.xlsx` contains the chosen model, baseline metrics, improvement %, and selected regressor name/lag.
- `Model_Rankings` sheet includes all candidates and whether each passed the accuracy and ROCV gates.
- `recommended_model_summary` sheet includes the recommended model, reason, and threshold fields used by the new selection logic.
- If `--compare-model-selection` is used, CSVs are written to `data_storage/model_selection_eval/`:
  - `model_selection_comparison_summary.csv`
  - `model_selection_flips_bias_best.csv`
  - `model_selection_flips_mae_worst.csv`
  - `model_selection_detail_all.csv`

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
 - ML forecasting uses `ML_GBR_PIPELINE` when pipeline data is available; otherwise it falls back to the legacy `ML_GBR` model.

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
- SBA (intermittent-demand baseline).
- ML and Prophet challengers (if enabled).
- Blended softmax forecast (if holdout history is available), which blends the best-performing model family per horizon.

Stability and post-processing:
- Forecast horizon is 12 months (`FORECAST_HORIZON = 12`).
- Forecast values and lower bounds are clipped at `FORECAST_FLOOR = 0.0`.
- ETS forecasts are validated for stability to avoid extreme or negative values for nonnegative series.
- The output adds a `recommended_model` boolean based on the chosen model in `sarima_multi_sku_summary.xlsx`.
- The final output keeps only the recommended model and the blended softmax forecast.
- Columns `lower_ci`, `upper_ci`, `model_group`, and `model_label` are removed.
- A `forecast_description` column is appended for business-facing context.
- The `regressor_names` column remains for traceability.
- The blended softmax weights sheet is not written in the forecast output.

### Manual SARIMA order overrides (`Notes.xlsx`)
If `Notes.xlsx` is present, the scripts will override SARIMA orders per SKU.
Required columns:
- `Product`, `Division` (or `group_key`, `BU`)
- `Chosen_Order` (tuple-like, length 3)
- `Chosen_Seasonal_Order` (tuple-like, length 4)

### Configuration summary (quick reference)
Edit these at the top of each script to change behavior:
- `sarima_order_selector.py`: order grid ranges, ROCV horizon/splits, holdout horizon, minimum series length.
- `sarima_multi_sku_engine.py`: test horizon, ROCV horizon/min obs, acceptance thresholds, ETS seasonal period, ML/Prophet toggles, quiet output flags.
- `sarima_forecast.py`: forecast horizon, history thresholds for SARIMA/SARIMAX, ETS stability limits, exogenous filling behavior, ML/Prophet toggles, quiet output flags, output filtering.

## Notes
- Exogenous candidates include lagged Salesforce signals (`New_Opportunities`, `Open_Opportunities`) and `Bookings`; lag logic is explicitly defined in each script.
- Rolling-origin CV and holdout windows are configured at the top of the scripts; adjust there for different horizons or minimum history.
- Data files are ignored by Git; share them separately if collaborators need to reproduce runs.
 - Model-selection unit tests are in `tests/test_model_selection.py` and can be run with:
```bash
python -m unittest tests/test_model_selection.py
```
