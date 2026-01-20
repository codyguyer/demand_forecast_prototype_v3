import unittest
import pandas as pd

from sarima_multi_sku_engine import select_recommended_model


def _dates(n=8):
    return pd.date_range("2024-01-01", periods=n, freq="MS")


def _ts_df(product, division, model, actuals, forecasts):
    return pd.DataFrame({
        "Product": product,
        "Division": division,
        "Model": model,
        "Date": _dates(len(actuals)),
        "Actual": actuals,
        "Forecast": forecasts,
    })


class TestModelSelection(unittest.TestCase):
    def test_da_override_wins(self):
        metrics_df = pd.DataFrame([
            {"Product": "A", "Division": "D1", "Model": "BASE", "Test_MAE": 10.0, "Test_RMSE": 12.0, "ROCV_MAE": 5.0},
            {"Product": "A", "Division": "D1", "Model": "ML", "Test_MAE": 10.4, "Test_RMSE": 12.5, "ROCV_MAE": 6.0},
        ])
        actuals = [100, 90, 100, 90, 100, 90, 100, 90, 100, 90, 100, 90]
        base_forecast = [100] * len(actuals)
        ml_forecast = [94.4 if i % 2 == 0 else 84.4 for i in range(len(actuals))]
        ts_base = _ts_df("A", "D1", "BASE", actuals, base_forecast)
        ts_ml = _ts_df("A", "D1", "ML", actuals, ml_forecast)
        ts_df = pd.concat([ts_base, ts_ml], ignore_index=True)

        summary = select_recommended_model(metrics_df, ts_df, bias_improvement_ratio=None)
        row = summary.iloc[0]
        self.assertEqual(row["Recommended_Model"], "ML")
        self.assertTrue(row["Reason"].startswith("Override:"))

    def test_mae_close_required_for_override(self):
        metrics_df = pd.DataFrame([
            {"Product": "B", "Division": "D1", "Model": "BASE", "Test_MAE": 10.0, "Test_RMSE": 12.0, "ROCV_MAE": 5.0},
            {"Product": "B", "Division": "D1", "Model": "ML", "Test_MAE": 11.0, "Test_RMSE": 12.5, "ROCV_MAE": 6.0},
        ])
        actuals = [100, 90, 100, 90, 100, 90, 100, 90, 100, 90, 100, 90]
        base_forecast = [100] * len(actuals)
        ml_forecast = [90 if i % 2 == 0 else 80 for i in range(len(actuals))]
        ts_base = _ts_df("B", "D1", "BASE", actuals, base_forecast)
        ts_ml = _ts_df("B", "D1", "ML", actuals, ml_forecast)
        ts_df = pd.concat([ts_base, ts_ml], ignore_index=True)

        summary = select_recommended_model(metrics_df, ts_df, bias_improvement_ratio=None)
        row = summary.iloc[0]
        self.assertEqual(row["Recommended_Model"], "BASE")
        self.assertEqual(
            row["Reason"],
            "Baseline wins: best MAE; no challenger improved DA by >=10pp within MAE+5%.",
        )

    def test_da_nan_blocks_override(self):
        metrics_df = pd.DataFrame([
            {"Product": "C", "Division": "D1", "Model": "BASE", "Test_MAE": 9.0, "Test_RMSE": 10.0, "ROCV_MAE": 4.0},
            {"Product": "C", "Division": "D1", "Model": "ML", "Test_MAE": 9.2, "Test_RMSE": 10.1, "ROCV_MAE": 4.5},
        ])
        actuals = [5, 6, 7, 8, 9, 10]
        ts_base = _ts_df("C", "D1", "BASE", actuals, actuals)
        ts_ml = _ts_df("C", "D1", "ML", actuals, actuals)
        ts_df = pd.concat([ts_base, ts_ml], ignore_index=True)

        summary = select_recommended_model(metrics_df, ts_df, da_min_periods=6, bias_improvement_ratio=None)
        row = summary.iloc[0]
        self.assertEqual(row["Recommended_Model"], "BASE")
        self.assertEqual(
            row["Reason"],
            "Baseline wins: best MAE; no challenger improved DA by >=10pp within MAE+5%.",
        )

    def test_rocv_missing_fails_sanity(self):
        metrics_df = pd.DataFrame([
            {"Product": "D", "Division": "D1", "Model": "BASE", "Test_MAE": 8.0, "Test_RMSE": 9.0, "ROCV_MAE": 4.0},
            {"Product": "D", "Division": "D1", "Model": "ML", "Test_MAE": 8.5, "Test_RMSE": 9.2, "ROCV_MAE": None},
        ])
        actuals = [2, 3, 4, 3, 4, 3, 4, 3]
        ts_base = _ts_df("D", "D1", "BASE", actuals, actuals)
        ts_ml = _ts_df("D", "D1", "ML", actuals, actuals)
        ts_df = pd.concat([ts_base, ts_ml], ignore_index=True)

        summary = select_recommended_model(metrics_df, ts_df, bias_improvement_ratio=None)
        row = summary.iloc[0]
        self.assertEqual(row["Recommended_Model"], "BASE")

    def test_bias_override_wins(self):
        metrics_df = pd.DataFrame([
            {"Product": "E", "Division": "D1", "Model": "BASE", "Test_MAE": 10.0, "Test_RMSE": 12.0, "ROCV_MAE": 5.0},
            {"Product": "E", "Division": "D1", "Model": "ALT", "Test_MAE": 10.5, "Test_RMSE": 12.4, "ROCV_MAE": 5.5},
        ])
        actuals = [10] * 12
        base_forecast = [12] * len(actuals)
        alt_forecast = [13.2 if i % 2 == 0 else 9.0 for i in range(len(actuals))]
        ts_base = _ts_df("E", "D1", "BASE", actuals, base_forecast)
        ts_alt = _ts_df("E", "D1", "ALT", actuals, alt_forecast)
        ts_df = pd.concat([ts_base, ts_alt], ignore_index=True)

        summary = select_recommended_model(metrics_df, ts_df)
        row = summary.iloc[0]
        self.assertEqual(row["Recommended_Model"], "ALT")
        self.assertTrue(row["Reason"].startswith("Bias override:"))

    def test_bias_override_blocked_by_mae(self):
        metrics_df = pd.DataFrame([
            {"Product": "F", "Division": "D1", "Model": "BASE", "Test_MAE": 10.0, "Test_RMSE": 12.0, "ROCV_MAE": 5.0},
            {"Product": "F", "Division": "D1", "Model": "ALT", "Test_MAE": 12.5, "Test_RMSE": 13.0, "ROCV_MAE": 5.5},
        ])
        actuals = [10] * 12
        base_forecast = [12] * len(actuals)
        alt_forecast = [13.0 if i % 2 == 0 else 8.0 for i in range(len(actuals))]
        ts_base = _ts_df("F", "D1", "BASE", actuals, base_forecast)
        ts_alt = _ts_df("F", "D1", "ALT", actuals, alt_forecast)
        ts_df = pd.concat([ts_base, ts_alt], ignore_index=True)

        summary = select_recommended_model(metrics_df, ts_df)
        row = summary.iloc[0]
        self.assertEqual(row["Recommended_Model"], "BASE")

    def test_bias_override_blocked_by_bias(self):
        metrics_df = pd.DataFrame([
            {"Product": "G", "Division": "D1", "Model": "BASE", "Test_MAE": 10.0, "Test_RMSE": 12.0, "ROCV_MAE": 5.0},
            {"Product": "G", "Division": "D1", "Model": "ALT", "Test_MAE": 10.5, "Test_RMSE": 12.4, "ROCV_MAE": 5.5},
        ])
        actuals = [10] * 12
        base_forecast = [12] * len(actuals)
        alt_forecast = [12.2 if i % 2 == 0 else 11.9 for i in range(len(actuals))]
        ts_base = _ts_df("G", "D1", "BASE", actuals, base_forecast)
        ts_alt = _ts_df("G", "D1", "ALT", actuals, alt_forecast)
        ts_df = pd.concat([ts_base, ts_alt], ignore_index=True)

        summary = select_recommended_model(metrics_df, ts_df)
        row = summary.iloc[0]
        self.assertEqual(row["Recommended_Model"], "BASE")

    def test_bias_override_missing_bias(self):
        metrics_df = pd.DataFrame([
            {"Product": "H", "Division": "D1", "Model": "BASE", "Test_MAE": 10.0, "Test_RMSE": 12.0, "ROCV_MAE": 5.0},
            {"Product": "H", "Division": "D1", "Model": "ALT", "Test_MAE": 10.5, "Test_RMSE": 12.4, "ROCV_MAE": 5.5},
        ])
        actuals = [10] * 12
        ts_base = _ts_df("H", "D1", "BASE", actuals, [12] * len(actuals))
        ts_alt = _ts_df("H", "D1", "ALT", actuals, [None] * len(actuals))
        ts_df = pd.concat([ts_base, ts_alt], ignore_index=True)

        summary = select_recommended_model(metrics_df, ts_df)
        row = summary.iloc[0]
        self.assertEqual(row["Recommended_Model"], "BASE")


if __name__ == "__main__":
    unittest.main()
