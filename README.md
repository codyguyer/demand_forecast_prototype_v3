# Demand Forecast Prototype v3

Python scripts for selecting SARIMA/SARIMAX orders by SKU, evaluating exogenous regressors, and generating a library of forecast variants from monthly data.

## Repository contents
- `sarima_order_selector.py` — grid-searches SARIMA/SARIMAX orders per (Product, Division) with rolling-origin CV and a holdout block; writes `sarimax_order_search_summary.xlsx`.
- `sarima_multi_sku_engine.py` — uses the chosen orders to compare baseline SARIMA vs. SARIMAX models with lagged regressors; writes `sarima_multi_sku_summary.xlsx`.
- `sarima_forecast.py` — builds forecast variants (baseline SARIMA/ARIMA and SARIMAX if allowed) for each SKU; writes `stats_model_forecasts_YYYY-Mon.xlsx`.
- `generate_forecast_variants.py` — small utility showing how to assemble forecast variants for a given SKU (used as a reference/helper).
- `.gitignore` — excludes data files (`*.xlsx`, `*.csv`) and Python build artifacts.

## Expected inputs (not committed)
Place these Excel files in the project root before running:
- `all_products_with_sf_and_bookings.xlsx` — historical monthly actuals and exogenous signals per Product/Division.
- `sarimax_order_search_summary.xlsx` — produced by `sarima_order_selector.py`.
- `sarima_multi_sku_summary.xlsx` — produced by `sarima_multi_sku_engine.py`.

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

## Notes
- Exogenous candidates include lagged Salesforce signals (`New_Opportunities`, `Open_Opportunities`) and `Bookings`; lag logic lives in each script.
- Rolling-origin CV and holdout windows are configured at the top of the scripts; adjust there for different horizons or minimum history.
- Data files are ignored by Git; share them separately if collaborators need to reproduce runs.
