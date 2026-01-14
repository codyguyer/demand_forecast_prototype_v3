import argparse
import os
import re
from datetime import datetime

import pandas as pd


INPUT_PATTERN = re.compile(r"^stats_model_forecasts_(\d{4})-([A-Za-z]{3})\.xlsx$")
FORECAST_SHEET = "Forecast_Library"
ARCHIVE_DIR = "Forecast Archive"
OUTPUT_PREFIX = "compiled_stats_model_forecasts_"
USE_REGRESSOR_IN_KEY = False


def _parse_run_label(filename: str) -> datetime:
    match = INPUT_PATTERN.match(os.path.basename(filename))
    if not match:
        raise ValueError(f"Unrecognized forecast filename: {filename}")
    year = int(match.group(1))
    month_str = match.group(2)
    dt = datetime.strptime(f"{year}-{month_str}-01", "%Y-%b-%d")
    return dt


def _find_latest_run_file(root_dir: str) -> str:
    candidates = []
    for name in os.listdir(root_dir):
        if INPUT_PATTERN.match(name):
            candidates.append(os.path.join(root_dir, name))
    if not candidates:
        raise FileNotFoundError("No stats_model_forecasts_YYYY-Mon.xlsx files found.")
    candidates.sort(key=_parse_run_label)
    return candidates[-1]


def _build_archive_filename(month_dt: datetime) -> str:
    return f"stats_model_forecasts_{month_dt.strftime('%Y-%b')}.xlsx"


def _normalize_forecast_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _build_key(df: pd.DataFrame, use_regressor: bool) -> pd.Series:
    cols = ["product_id", "bu_id", "forecast_month", "model_type"]
    if use_regressor and "regressor_names" in df.columns:
        cols.append("regressor_names")
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")
    return df[cols].astype(str).agg("|".join, axis=1)


def _load_forecast_sheet(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=FORECAST_SHEET)
    if "forecast_month" not in df.columns:
        raise ValueError(f"forecast_month column missing in {path}")
    df["forecast_month"] = _normalize_forecast_month(df["forecast_month"])
    return df


def _closed_months(current_start: datetime) -> list:
    year_start = datetime(current_start.year, 1, 1)
    months = []
    cursor = year_start
    while cursor < current_start:
        months.append(cursor)
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)
    return months


def compile_forecasts(
    root_dir: str,
    current_file: str,
    archive_dir: str,
    output_file: str,
    use_regressor_key: bool,
) -> None:
    current_path = os.path.join(root_dir, current_file)
    archive_path = os.path.join(root_dir, archive_dir)

    current_df = _load_forecast_sheet(current_path)
    current_start = current_df["forecast_month"].min()
    if pd.isna(current_start):
        raise ValueError("No forecast_month values found in current run.")

    closed_months = _closed_months(current_start.to_pydatetime())
    if not closed_months:
        print("No closed months to backfill; writing current file as compiled output.")
        current_df.to_excel(output_file, index=False, sheet_name=FORECAST_SHEET)
        return

    base_columns = list(current_df.columns)
    current_key = _build_key(current_df, use_regressor_key)
    current_key_set = set(current_key.tolist())

    backfill_rows = []
    for month_dt in closed_months:
        archive_file = _build_archive_filename(month_dt)
        archive_full = os.path.join(archive_path, archive_file)
        if not os.path.exists(archive_full):
            print(f"Archive missing for {month_dt.strftime('%Y-%b')}: {archive_full}")
            continue
        archive_df = _load_forecast_sheet(archive_full)
        month_mask = archive_df["forecast_month"] == month_dt
        month_df = archive_df.loc[month_mask].copy()
        if month_df.empty:
            print(f"No rows for {month_dt.strftime('%Y-%b')} in {archive_file}")
            continue

        # Align columns to current run schema.
        for col in base_columns:
            if col not in month_df.columns:
                month_df[col] = pd.NA
        month_df = month_df[base_columns]

        month_key = _build_key(month_df, use_regressor_key)
        keep_mask = ~month_key.isin(current_key_set)
        month_df = month_df.loc[keep_mask]
        if month_df.empty:
            continue
        backfill_rows.append(month_df)

    if backfill_rows:
        backfill_df = pd.concat(backfill_rows, ignore_index=True)
        compiled = pd.concat([current_df, backfill_df], ignore_index=True)
    else:
        compiled = current_df

    compiled.to_excel(output_file, index=False, sheet_name=FORECAST_SHEET)
    print(f"Compiled forecast written to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile stats model forecasts with frozen closed months."
    )
    parser.add_argument(
        "--root-dir",
        default=os.getcwd(),
        help="Root directory containing current forecast file and Forecast Archive.",
    )
    parser.add_argument(
        "--current-file",
        default=None,
        help="Current run filename (defaults to latest stats_model_forecasts_YYYY-Mon.xlsx).",
    )
    parser.add_argument(
        "--archive-dir",
        default=ARCHIVE_DIR,
        help="Folder name containing historical forecast files.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output filename (default: compiled_stats_model_forecasts_YYYY-Mon.xlsx).",
    )
    parser.add_argument(
        "--use-regressor-key",
        action="store_true",
        help="Include regressor_names in the matching key.",
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    current_file = args.current_file
    if current_file is None:
        current_path = _find_latest_run_file(root_dir)
        current_file = os.path.basename(current_path)
    current_label = _parse_run_label(current_file).strftime("%Y-%b")
    output_file = args.output_file or f"{OUTPUT_PREFIX}{current_label}.xlsx"
    output_path = os.path.join(root_dir, output_file)

    compile_forecasts(
        root_dir=root_dir,
        current_file=current_file,
        archive_dir=args.archive_dir,
        output_file=output_path,
        use_regressor_key=args.use_regressor_key or USE_REGRESSOR_IN_KEY,
    )


if __name__ == "__main__":
    main()
