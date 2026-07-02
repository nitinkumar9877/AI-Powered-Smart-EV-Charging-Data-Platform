CREATE SCHEMA IF NOT EXISTS ev_charging_gold;

CREATE OR REPLACE TABLE ev_charging_gold.station_performance
USING DELTA
AS
SELECT
    station_key,
    charging_station_id,
    charging_station_location,
    COUNT(*) AS total_sessions,
    SUM(energy_consumed_kwh) AS total_energy_kwh,
    AVG(charging_duration_hours) AS avg_charging_duration_hours,
    AVG(charging_rate_kw) AS avg_charging_rate_kw,
    AVG(charging_cost_usd) AS avg_cost_usd,
    AVG(traffic_score) AS avg_traffic_score,
    SUM(is_peak_hour) AS peak_hour_sessions,
    AVG(temperature_c) AS avg_temperature_c,
    AVG(precipitation_mm) AS avg_precipitation_mm
FROM ev_charging_silver.charging_events_cleaned
GROUP BY station_key, charging_station_id, charging_station_location;

CREATE OR REPLACE TABLE ev_charging_gold.traffic_analytics
USING DELTA
AS
SELECT
    charging_station_location,
    traffic_level,
    charging_hour,
    is_weekend,
    COUNT(*) AS sessions,
    AVG(traffic_score) AS avg_traffic_score,
    AVG(road_congestion_index) AS avg_congestion_index,
    AVG(charging_duration_hours) AS avg_duration_hours,
    SUM(nearby_events_count) AS nearby_events
FROM ev_charging_silver.charging_events_cleaned
GROUP BY charging_station_location, traffic_level, charging_hour, is_weekend;

CREATE OR REPLACE TABLE ev_charging_gold.weather_impact_analytics
USING DELTA
AS
SELECT
    charging_station_location,
    weather_condition,
    charging_month,
    COUNT(*) AS sessions,
    AVG(temperature_c) AS avg_temperature_c,
    AVG(humidity_percent) AS avg_humidity_percent,
    SUM(precipitation_mm) AS total_precipitation_mm,
    AVG(wind_speed_kph) AS avg_wind_speed_kph,
    AVG(energy_consumed_kwh) AS avg_energy_kwh
FROM ev_charging_silver.charging_events_cleaned
GROUP BY charging_station_location, weather_condition, charging_month;

CREATE OR REPLACE TABLE ev_charging_gold.charging_demand_analytics
USING DELTA
AS
SELECT
    charging_station_location,
    charging_date,
    charging_hour,
    charger_type,
    COUNT(*) AS sessions,
    COUNT(DISTINCT user_id) AS unique_users,
    SUM(energy_consumed_kwh) AS total_energy_kwh,
    AVG(soc_start_percent) AS avg_soc_start,
    AVG(soc_end_percent) AS avg_soc_end,
    MAX(is_peak_hour) AS peak_hour_flag
FROM ev_charging_silver.charging_events_cleaned
GROUP BY charging_station_location, charging_date, charging_hour, charger_type;

