# Data Flow

1. Raw ingestion reads `data/bronze/ev_charging_raw_data.csv`.
2. Bronze output adds ingestion metadata and stores `bronze_ev_charging_events.csv`.
3. Silver processing cleans strings, casts numeric/date columns, fills useful defaults, removes duplicates, and validates operational rules.
4. Silver feature engineering creates time, traffic, weather, and station keys.
5. Gold processing creates business analytics tables for station performance, traffic, weather, and demand.
6. ML feature engineering creates target variables for occupancy, waiting time, and overload risk.
7. Model training builds three models and stores metrics and model artifacts under `models/`.
8. Power BI consumes gold CSV tables and ML output for dashboarding.

## Validation Rules
- Required station, location, and timestamp fields must exist.
- Charging end time cannot be earlier than charging start time.
- Energy consumed must be non-negative.
- Charging duration must be positive.
- Traffic score must be between 0 and 100.
- Humidity must be between 0 and 100.

## Feature Engineering
- Time features: hour, month, date, day name, weekend flag.
- Peak indicators: morning peak, evening peak, combined peak.
- Charging behavior: energy per hour, SOC delta.
- Station features: station key, city key, connector capacity.
- External context: weather condition, precipitation, traffic score, congestion, nearby events.

