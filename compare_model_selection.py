import pandas as pd
import numpy as np

from sarima_multi_sku_engine import (
    choose_recommended_model,
    compute_weighted_holdout_metrics,
    compute_directional_accuracy,
    select_recommended_model,
    DA_CLOSE_MAE_TOL,
    DA_IMPROVEMENT_PP,
    DA_MIN_PERIODS,
    HOLDOUT_TS_CSV,
    MAE_TOLERANCE,
    RMSE_TOLERANCE,
    ROCV_HARD_MULTIPLIER,
    OUTPUT_DIR,
)


def _old_selection(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (prod, div), group in metrics_df.groupby(["Product", "Division"], dropna=False):
        metrics_list = group.to_dict("records")
        best, baseline, reason, _ = choose_recommended_model(metrics_list)
        rows.append({
            "Product": prod,
            "Division": div,
            "Recommended_Model": best.get("Model") if best else None,
            "Baseline_Model": baseline.get("Model") if baseline else None,
            "Reason": reason,
        })
    return pd.DataFrame(rows)


def _lookup_metric_map(metrics_df: pd.DataFrame, metric: str) -> dict:
    key_cols = ["Product", "Division", "Model"]
    lookup = metrics_df[key_cols + [metric]].drop_duplicates(key_cols)
    return {
        (row["Product"], row["Division"], row["Model"]): row[metric]
        for _, row in lookup.iterrows()
    }


def main():
    rankings_path = "sarima_multi_sku_summary.xlsx"
    rankings_df = pd.read_excel(rankings_path, sheet_name="Model_Rankings")
    ts_df = pd.read_csv(HOLDOUT_TS_CSV)

    new_summary = select_recommended_model(
        rankings_df,
        ts_df,
        mae_tolerance=MAE_TOLERANCE,
        rmse_tolerance=RMSE_TOLERANCE,
        rocv_hard_multiplier=ROCV_HARD_MULTIPLIER,
        da_min_periods=DA_MIN_PERIODS,
        da_improvement_pp=DA_IMPROVEMENT_PP,
        da_close_mae_tol=DA_CLOSE_MAE_TOL,
        return_rankings=False,
    )
    old_summary = _old_selection(rankings_df)

    compare = old_summary.merge(
        new_summary,
        on=["Product", "Division"],
        suffixes=("_old", "_new"),
        how="outer",
    )

    weighted_df = compute_weighted_holdout_metrics(ts_df)
    metrics_enriched = rankings_df.merge(
        weighted_df, on=["Product", "Division", "Model"], how="left"
    )
    mae_map = _lookup_metric_map(metrics_enriched, "Weighted_MAE")
    rmse_map = _lookup_metric_map(metrics_enriched, "Weighted_RMSE")
    bias_map = _lookup_metric_map(metrics_enriched, "bias")
    abs_bias_map = _lookup_metric_map(metrics_enriched, "abs_bias")
    abs_bias_pct_map = _lookup_metric_map(metrics_enriched, "abs_bias_pct")

    def _map_metric(row, model_col, lookup):
        key = (row["Product"], row["Division"], row[model_col])
        return lookup.get(key, np.nan)

    compare["old_mae"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_old", mae_map), axis=1)
    compare["new_mae"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_new", mae_map), axis=1)
    compare["old_rmse"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_old", rmse_map), axis=1)
    compare["new_rmse"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_new", rmse_map), axis=1)
    compare["old_bias"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_old", bias_map), axis=1)
    compare["new_bias"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_new", bias_map), axis=1)
    compare["old_abs_bias"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_old", abs_bias_map), axis=1)
    compare["new_abs_bias"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_new", abs_bias_map), axis=1)
    compare["old_abs_bias_pct"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_old", abs_bias_pct_map), axis=1)
    compare["new_abs_bias_pct"] = compare.apply(lambda r: _map_metric(r, "Recommended_Model_new", abs_bias_pct_map), axis=1)
    compare["mae_delta"] = compare["new_mae"] - compare["old_mae"]
    compare["rmse_delta"] = compare["new_rmse"] - compare["old_rmse"]
    compare["abs_bias_delta"] = compare["new_abs_bias"] - compare["old_abs_bias"]
    compare["abs_bias_pct_delta"] = compare["new_abs_bias_pct"] - compare["old_abs_bias_pct"]
    compare["winner_changed"] = compare["Recommended_Model_old"] != compare["Recommended_Model_new"]

    flips = compare[compare["winner_changed"]].copy()
    top_bias_improved = flips.sort_values("abs_bias_delta", ascending=True).head(20)
    top_mae_worsened = flips.sort_values("mae_delta", ascending=False).head(20)

    summary_stats = pd.DataFrame([{
        "total_skus": int(compare.shape[0]),
        "winner_changed_count": int(flips.shape[0]),
        "winner_changed_pct": float(flips.shape[0]) / compare.shape[0] if compare.shape[0] else np.nan,
        "mae_delta_mean": compare["mae_delta"].mean(),
        "mae_delta_median": compare["mae_delta"].median(),
        "abs_bias_delta_mean": compare["abs_bias_delta"].mean(),
        "abs_bias_delta_median": compare["abs_bias_delta"].median(),
    }])

    out_dir = OUTPUT_DIR / "model_selection_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "model_selection_comparison_summary.csv"
    flips_bias_path = out_dir / "model_selection_flips_bias_best.csv"
    flips_mae_path = out_dir / "model_selection_flips_mae_worst.csv"
    detail_path = out_dir / "model_selection_detail_all.csv"

    summary_stats.to_csv(summary_path, index=False)
    flips.to_csv(detail_path, index=False)
    top_bias_improved.to_csv(flips_bias_path, index=False)
    top_mae_worsened.to_csv(flips_mae_path, index=False)

    print(f"Saved summary CSV to {summary_path}")
    print(f"Saved flips (all) CSV to {detail_path}")
    print(f"Saved top bias-improved flips CSV to {flips_bias_path}")
    print(f"Saved top MAE-worst flips CSV to {flips_mae_path}")


if __name__ == "__main__":
    main()
