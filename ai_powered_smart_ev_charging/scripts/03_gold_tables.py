import pandas as pd

from config import GOLD_DIR, SILVER_FILE


def _write(df: pd.DataFrame, name: str) -> None:
    path = GOLD_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Gold table written: {path} ({len(df):,} rows)")


def build_station_performance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["station_key", "Charging Station ID", "Charging Station Location"], dropna=False)
        .agg(
            total_sessions=("charging_session_count", "sum"),
            total_energy_kwh=("Energy Consumed (kWh)", "sum"),
            avg_charging_duration_hours=("Charging Duration (hours)", "mean"),
            avg_charging_rate_kw=("Charging Rate (kW)", "mean"),
            avg_cost_usd=("Charging Cost (USD)", "mean"),
            avg_traffic_score=("traffic_score", "mean"),
            peak_hour_sessions=("is_peak_hour", "sum"),
            avg_temperature_c=("temperature_c", "mean"),
            avg_precipitation_mm=("precipitation_mm", "mean"),
            matched_station_metadata_rows=("station_source_identifier", lambda s: s.notna().sum()),
        )
        .reset_index()
        .sort_values(["total_sessions", "total_energy_kwh"], ascending=False)
    )


def build_traffic_analytics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Charging Station Location", "traffic_level", "charging_hour", "is_weekend"], dropna=False)
        .agg(
            sessions=("charging_session_count", "sum"),
            avg_traffic_score=("traffic_score", "mean"),
            avg_congestion_index=("road_congestion_index", "mean"),
            avg_duration_hours=("Charging Duration (hours)", "mean"),
            avg_energy_kwh=("Energy Consumed (kWh)", "mean"),
            nearby_events=("nearby_events_count", "sum"),
        )
        .reset_index()
    )


def build_weather_impact(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Charging Station Location", "weather_condition", "charging_month"], dropna=False)
        .agg(
            sessions=("charging_session_count", "sum"),
            avg_temperature_c=("temperature_c", "mean"),
            avg_humidity_percent=("humidity_percent", "mean"),
            total_precipitation_mm=("precipitation_mm", "sum"),
            avg_wind_speed_kph=("wind_speed_kph", "mean"),
            avg_energy_kwh=("Energy Consumed (kWh)", "mean"),
            avg_charging_duration_hours=("Charging Duration (hours)", "mean"),
        )
        .reset_index()
    )


def build_charging_demand(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Charging Station Location", "charging_date", "charging_hour", "Charger Type"], dropna=False)
        .agg(
            sessions=("charging_session_count", "sum"),
            unique_users=("User ID", "nunique"),
            total_energy_kwh=("Energy Consumed (kWh)", "sum"),
            avg_soc_start=("State of Charge (Start %)", "mean"),
            avg_soc_end=("State of Charge (End %)", "mean"),
            avg_distance_since_last_charge_km=("Distance Driven (since last charge) (km)", "mean"),
            peak_hour_flag=("is_peak_hour", "max"),
        )
        .reset_index()
    )


def run_gold_tables() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SILVER_FILE, parse_dates=["Charging Start Time", "Charging End Time"])
    _write(build_station_performance(df), "station_performance")
    _write(build_traffic_analytics(df), "traffic_analytics")
    _write(build_weather_impact(df), "weather_impact_analytics")
    _write(build_charging_demand(df), "charging_demand_analytics")


if __name__ == "__main__":
    run_gold_tables()

