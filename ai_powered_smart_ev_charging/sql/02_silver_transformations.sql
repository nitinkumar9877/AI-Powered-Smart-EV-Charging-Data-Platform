CREATE SCHEMA IF NOT EXISTS ev_charging_silver;

CREATE OR REPLACE TABLE ev_charging_silver.charging_events_cleaned
USING DELTA
AS
SELECT DISTINCT
    *,
    DATE(charging_start_time) AS charging_date,
    HOUR(charging_start_time) AS charging_hour,
    MONTH(charging_start_time) AS charging_month,
    CASE WHEN DAYOFWEEK(charging_start_time) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
    CASE
        WHEN HOUR(charging_start_time) BETWEEN 7 AND 10
          OR HOUR(charging_start_time) BETWEEN 17 AND 21
        THEN 1 ELSE 0
    END AS is_peak_hour,
    energy_consumed_kwh / NULLIF(charging_duration_hours, 0) AS energy_per_hour_kwh,
    soc_end_percent - soc_start_percent AS soc_delta_percent,
    CONCAT(charging_station_id, '_', charging_station_location) AS station_key
FROM ev_charging_bronze.raw_charging_events
WHERE charging_station_id IS NOT NULL
  AND charging_start_time IS NOT NULL
  AND charging_end_time >= charging_start_time
  AND energy_consumed_kwh >= 0
  AND charging_duration_hours > 0
  AND traffic_score BETWEEN 0 AND 100
  AND humidity_percent BETWEEN 0 AND 100;

