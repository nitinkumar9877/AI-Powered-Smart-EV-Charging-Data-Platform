import numpy as np
import pandas as pd

from config import BRONZE_INGESTED_FILE, SILVER_FILE


REQUIRED_COLUMNS = [
    "User ID",
    "Charging Station ID",
    "Charging Station Location",
    "Charging Start Time",
    "Charging End Time",
    "Energy Consumed (kWh)",
    "Charging Duration (hours)",
    "Charging Rate (kW)",
    "traffic_score",
    "temperature_c",
]

NUMERIC_COLUMNS = [
    "Battery Capacity (kWh)",
    "Energy Consumed (kWh)",
    "Charging Duration (hours)",
    "Charging Rate (kW)",
    "Charging Cost (USD)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Distance Driven (since last charge) (km)",
    "Temperature (°C)",
    "Vehicle Age (years)",
    "latitude",
    "longitude",
    "rating",
    "review_count",
    "num_connectors",
    "total_kw",
    "temperature_c",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_kph",
    "traffic_score",
    "road_congestion_index",
    "nearby_events_count",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Charging Start Time"] = pd.to_datetime(df["Charging Start Time"], errors="coerce")
    df["Charging End Time"] = pd.to_datetime(df["Charging End Time"], errors="coerce")
    return df


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])]
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def _handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df["Charging Station Location"] = df["Charging Station Location"].fillna("Unknown")
    df["traffic_level"] = df["traffic_level"].fillna("Unknown")
    df["weather_condition"] = df["weather_condition"].fillna("Unknown")
    df["Charger Type"] = df["Charger Type"].fillna("Unknown")
    df["User Type"] = df["User Type"].fillna("Unknown")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median_value) else median_value)
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["quality_is_valid"] = True
    df["quality_issue"] = ""

    rules = [
        (df["Charging Start Time"].isna(), "missing_start_time"),
        (df["Charging End Time"].isna(), "missing_end_time"),
        (df["Charging End Time"] < df["Charging Start Time"], "end_before_start"),
        (df["Energy Consumed (kWh)"] < 0, "negative_energy"),
        (df["Charging Duration (hours)"] <= 0, "non_positive_duration"),
        (~df["traffic_score"].between(0, 100), "traffic_score_out_of_range"),
        (~df["humidity_percent"].between(0, 100), "humidity_out_of_range"),
    ]

    for condition, issue in rules:
        df.loc[condition, "quality_is_valid"] = False
        df.loc[condition, "quality_issue"] = df.loc[condition, "quality_issue"].mask(
            df.loc[condition, "quality_issue"].eq(""),
            issue,
        ).fillna(issue)

    return df[df["quality_is_valid"]].copy()


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df["charging_date"] = df["Charging Start Time"].dt.date.astype("string")
    df["charging_hour"] = df["Charging Start Time"].dt.hour
    df["charging_month"] = df["Charging Start Time"].dt.month
    df["day_name"] = df["Charging Start Time"].dt.day_name()
    df["is_weekend"] = df["Charging Start Time"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_morning_peak"] = df["charging_hour"].between(7, 10).astype(int)
    df["is_evening_peak"] = df["charging_hour"].between(17, 21).astype(int)
    df["is_peak_hour"] = ((df["is_morning_peak"] == 1) | (df["is_evening_peak"] == 1)).astype(int)
    df["energy_per_hour_kwh"] = df["Energy Consumed (kWh)"] / df["Charging Duration (hours)"].replace(0, np.nan)
    df["soc_delta_percent"] = df["State of Charge (End %)"] - df["State of Charge (Start %)"]
    df["connector_capacity_kw"] = df["total_kw"] / df["num_connectors"].replace(0, np.nan)
    df["station_key"] = df["Charging Station ID"].astype(str) + "_" + df["Charging Station Location"].astype(str)
    df["station_city_key"] = df["Charging Station Location"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    df["charging_session_count"] = 1
    df["energy_per_hour_kwh"] = df["energy_per_hour_kwh"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["connector_capacity_kw"] = df["connector_capacity_kw"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def run_silver_cleaning() -> pd.DataFrame:
    df = pd.read_csv(BRONZE_INGESTED_FILE)
    before = len(df)
    df = _clean_strings(df)
    df = _normalize_columns(df)
    df = _handle_nulls(df)
    df = df.drop_duplicates()
    df = _validate(df)
    df = _feature_engineer(df)

    SILVER_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SILVER_FILE, index=False)
    print(f"Silver cleaning complete: {before:,} input rows -> {len(df):,} valid rows -> {SILVER_FILE}")
    return df


if __name__ == "__main__":
    run_silver_cleaning()
