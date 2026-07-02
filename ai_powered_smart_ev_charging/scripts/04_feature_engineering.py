import numpy as np
import pandas as pd

from config import FEATURE_FILE, SILVER_FILE, TARGET_OCCUPANCY, TARGET_OVERLOAD, TARGET_WAITING_TIME


def _add_model_targets(df: pd.DataFrame) -> pd.DataFrame:
    station_hour = df.groupby(["station_key", "charging_date", "charging_hour"])["charging_session_count"].transform("sum")
    city_hour = df.groupby(["Charging Station Location", "charging_date", "charging_hour"])["charging_session_count"].transform("sum")
    max_station_hour = max(station_hour.max(), 1)

    duration_pressure = df["Charging Duration (hours)"].rank(pct=True)
    traffic_pressure = df["traffic_score"] / 100.0
    demand_pressure = station_hour / max_station_hour
    peak_pressure = df["is_peak_hour"] * 0.12
    weather_pressure = np.where(df["weather_condition"].isin(["Heavy Rain", "Snow"]), 0.08, 0)

    occupancy = 100 * (0.45 * demand_pressure + 0.25 * traffic_pressure + 0.20 * duration_pressure + peak_pressure + weather_pressure)
    df[TARGET_OCCUPANCY] = np.clip(occupancy, 5, 100).round(2)

    waiting_time = (
        2
        + 0.32 * df[TARGET_OCCUPANCY]
        + 0.11 * df["traffic_score"]
        + 1.6 * station_hour
        + np.where(df["is_peak_hour"] == 1, 8, 0)
        + np.where(df["is_weekend"] == 1, -3, 0)
        + np.where(df["weather_condition"].isin(["Heavy Rain", "Snow"]), 6, 0)
        + 0.5 * city_hour
    )
    df[TARGET_WAITING_TIME] = np.clip(waiting_time, 0, 90).round(2)
    df[TARGET_OVERLOAD] = ((df[TARGET_OCCUPANCY] >= 78) | (df[TARGET_WAITING_TIME] >= 35)).astype(int)
    return df


def build_ml_features() -> pd.DataFrame:
    df = pd.read_csv(SILVER_FILE)
    df = _add_model_targets(df)
    selected = [
        "station_key",
        "Charging Station ID",
        "Charging Station Location",
        "Vehicle Model",
        "Charger Type",
        "User Type",
        "weather_condition",
        "traffic_level",
        "charging_hour",
        "charging_month",
        "is_weekend",
        "is_peak_hour",
        "Energy Consumed (kWh)",
        "Charging Duration (hours)",
        "Charging Rate (kW)",
        "Battery Capacity (kWh)",
        "State of Charge (Start %)",
        "State of Charge (End %)",
        "Distance Driven (since last charge) (km)",
        "temperature_c",
        "humidity_percent",
        "precipitation_mm",
        "wind_speed_kph",
        "traffic_score",
        "road_congestion_index",
        "nearby_events_count",
        TARGET_OCCUPANCY,
        TARGET_WAITING_TIME,
        TARGET_OVERLOAD,
    ]
    features = df[selected].copy()
    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_FILE, index=False)
    print(f"ML feature dataset written: {FEATURE_FILE} ({len(features):,} rows)")
    return features


if __name__ == "__main__":
    build_ml_features()

