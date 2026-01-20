import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_storage"

ACTUALS_RAW_PATH = DATA_DIR / "Actuals_raw_data.xlsx"
BOOKINGS_RAW_PATH = DATA_DIR / "Bookings_raw_data.xlsx"
PRODUCT_CATALOG_PATH = BASE_DIR / "product_catalog_master.xlsx"
OUTPUT_PATH = BASE_DIR / "all_products_actuals_and_bookings.xlsx"

ACTUALS_SHEETS = (
    ("Essbase_raw_data_qty", 5),
    ("Essbase_raw_data_qty (Division)", 5),
    ("Essbase_raw_data_dollars", 5),
)
BOOKINGS_SHEETS = (
    ("Essbase_raw_data", 3),
    ("Essbase_raw_data (division)", 3),
)

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def normalize_code(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value).strip().upper()
    raw = str(value).strip()
    if raw.endswith(".0") and raw.replace(".", "", 1).isdigit():
        raw = raw[:-2]
    return raw.upper()


def strip_prefix(value: str) -> str:
    return re.sub(r"^(PROD|ITEM)[_-]?", "", value, flags=re.IGNORECASE)


def parse_fiscal_year(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    year_num = int(digits)
    if year_num < 100:
        return 2000 + year_num
    return year_num


def parse_month_year(month_value: object, fy_value: object) -> pd.Timestamp:
    month_raw = normalize_code(month_value)
    year = parse_fiscal_year(fy_value)
    if not month_raw or year is None:
        return pd.NaT
    month_num = MONTH_MAP.get(month_raw[:3].upper())
    if not month_num:
        return pd.NaT
    return pd.Timestamp(year=year, month=month_num, day=1)


@dataclass
class CatalogMaps:
    by_sku_bu: Dict[Tuple[str, str], str]
    by_sku: Dict[str, str]
    by_group: Dict[str, str]


def build_catalog_maps(path: Path) -> CatalogMaps:
    catalog = pd.read_excel(path)
    required = {"group_key", "business_unit_code", "sku_list"}
    missing = required - set(catalog.columns)
    if missing:
        raise RuntimeError(f"product_catalog_master.xlsx missing columns: {sorted(missing)}")

    by_sku_bu: Dict[Tuple[str, str], str] = {}
    by_sku: Dict[str, str] = {}
    by_group: Dict[str, str] = {}
    for _, row in catalog.iterrows():
        bu = normalize_code(row.get("business_unit_code"))
        group = normalize_code(row.get("group_key"))
        raw_sku_list = row.get("sku_list")
        if isinstance(raw_sku_list, str) and raw_sku_list.strip():
            sku_items = [normalize_code(item) for item in raw_sku_list.split("|")]
        else:
            sku_items = [normalize_code(raw_sku_list)]

        for sku in sku_items:
            if sku and bu and group:
                by_sku_bu[(sku, bu)] = group
            if sku and group and sku not in by_sku:
                by_sku[sku] = group
        if group:
            by_group[group] = group

    return CatalogMaps(by_sku_bu=by_sku_bu, by_sku=by_sku, by_group=by_group)


def map_group_key(
    raw_code: object, raw_bu: object, catalog_maps: CatalogMaps
) -> Optional[str]:
    code_norm = normalize_code(raw_code)
    bu_norm = normalize_code(raw_bu)
    if (code_norm, bu_norm) in catalog_maps.by_sku_bu:
        return catalog_maps.by_sku_bu[(code_norm, bu_norm)]
    if code_norm in catalog_maps.by_sku:
        return catalog_maps.by_sku[code_norm]
    if code_norm in catalog_maps.by_group:
        return catalog_maps.by_group[code_norm]

    stripped = strip_prefix(code_norm)
    if (stripped, bu_norm) in catalog_maps.by_sku_bu:
        return catalog_maps.by_sku_bu[(stripped, bu_norm)]
    if stripped in catalog_maps.by_sku:
        return catalog_maps.by_sku[stripped]
    if stripped in catalog_maps.by_group:
        return catalog_maps.by_group[stripped]
    return None


def load_raw_sheet(
    path: Path,
    sheet_name: str,
    skiprows: int,
    value_name: str,
    catalog_maps: CatalogMaps,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        skiprows=skiprows,
        usecols=[0, 1, 2, 3, 4],
    )
    df.columns = ["raw_product", "Division", "raw_month", "raw_fy", value_name]
    df = df.dropna(subset=["raw_product", "Division", "raw_month", "raw_fy"], how="any")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df["Month"] = [
        parse_month_year(m, fy) for m, fy in zip(df["raw_month"], df["raw_fy"])
    ]
    df["Product"] = [
        map_group_key(code, bu, catalog_maps) for code, bu in zip(df["raw_product"], df["Division"])
    ]

    invalid = df[df["Product"].isna()].copy()
    df = df.dropna(subset=["Product", "Month"])
    df = df[["Product", "Division", "Month", value_name]]
    return df, invalid


def load_raw_sheets(
    path: Path,
    sheets: Iterable[Tuple[str, int]],
    value_name: str,
    catalog_maps: CatalogMaps,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    invalid_frames = []
    for sheet, skiprows in sheets:
        df, invalid = load_raw_sheet(
            path,
            sheet_name=sheet,
            skiprows=skiprows,
            value_name=value_name,
            catalog_maps=catalog_maps,
        )
        if not df.empty:
            frames.append(df)
        if not invalid.empty:
            invalid["source_sheet"] = sheet
            invalid_frames.append(invalid)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    invalid_combined = (
        pd.concat(invalid_frames, ignore_index=True) if invalid_frames else pd.DataFrame()
    )
    return combined, invalid_combined


def main() -> None:
    if not ACTUALS_RAW_PATH.exists():
        raise FileNotFoundError(f"Missing actuals raw file: {ACTUALS_RAW_PATH}")
    if not BOOKINGS_RAW_PATH.exists():
        raise FileNotFoundError(f"Missing bookings raw file: {BOOKINGS_RAW_PATH}")
    if not PRODUCT_CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing product catalog: {PRODUCT_CATALOG_PATH}")

    catalog_maps = build_catalog_maps(PRODUCT_CATALOG_PATH)

    actuals_df, actuals_invalid = load_raw_sheets(
        ACTUALS_RAW_PATH, ACTUALS_SHEETS, "Actuals", catalog_maps
    )
    bookings_df, bookings_invalid = load_raw_sheets(
        BOOKINGS_RAW_PATH, BOOKINGS_SHEETS, "Bookings", catalog_maps
    )

    if not actuals_df.empty:
        actuals_df = (
            actuals_df.groupby(["Product", "Division", "Month"], as_index=False)["Actuals"]
            .sum()
        )
    if not bookings_df.empty:
        bookings_df = (
            bookings_df.groupby(["Product", "Division", "Month"], as_index=False)["Bookings"]
            .sum()
        )

    combined = pd.merge(
        actuals_df, bookings_df, on=["Product", "Division", "Month"], how="outer"
    )
    if not combined.empty:
        combined["Month"] = pd.to_datetime(combined["Month"], errors="coerce")
        combined = combined.dropna(subset=["Product", "Division", "Month"], how="any")
        combined = (
            combined.groupby(["Product", "Division", "Month"], as_index=False)
            .agg(
                {
                    "Actuals": lambda s: s.sum(min_count=1),
                    "Bookings": lambda s: s.sum(min_count=1),
                }
            )
        )
        combined = combined.sort_values(["Product", "Division", "Month"])

    combined.to_excel(OUTPUT_PATH, index=False)

    if not actuals_invalid.empty or not bookings_invalid.empty:
        invalid_output = DATA_DIR / "raw_mapping_issues.xlsx"
        with pd.ExcelWriter(invalid_output, engine="openpyxl") as writer:
            if not actuals_invalid.empty:
                actuals_invalid.to_excel(writer, index=False, sheet_name="Actuals_Unmapped")
            if not bookings_invalid.empty:
                bookings_invalid.to_excel(writer, index=False, sheet_name="Bookings_Unmapped")
        print(f"Wrote unmapped rows to {invalid_output}.")

    if not combined.empty:
        duplicate_count = combined.duplicated(subset=["Product", "Division", "Month"]).sum()
        print(f"Duplicate Product/Division/Month rows: {duplicate_count}")
    if not actuals_invalid.empty or not bookings_invalid.empty:
        print(
            "Unmapped rows: "
            f"Actuals={len(actuals_invalid)} | Bookings={len(bookings_invalid)}"
        )

    print(f"Wrote combined actuals/bookings to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
