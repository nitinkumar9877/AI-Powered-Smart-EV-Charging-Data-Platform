# Power BI Dashboard Requirements

## Dashboard Name
AI-Powered Smart EV Charging Network Optimization Platform

## Data Sources
- `data/gold/station_performance.csv`
- `data/gold/traffic_analytics.csv`
- `data/gold/weather_impact_analytics.csv`
- `data/gold/charging_demand_analytics.csv`
- `data/silver/ml_features.csv`
- `models/model_metrics.json`

## Pages

### 1. Executive Overview
- KPI cards: total sessions, total energy kWh, average charging duration, average waiting time, overload risk rate.
- Map or station table by `Charging Station Location`.
- Top 10 busy stations by total sessions.
- Slicer: city, charger type, day type, weather condition.

### 2. Station Utilization
- Station utilization ranking by sessions and energy.
- Peak-hour session share.
- Average charging rate and duration by station.
- Conditional formatting for overloaded stations.

### 3. Traffic Impact
- Traffic score by hour heatmap.
- Charging demand by traffic level.
- Average waiting time by traffic level.
- Nearby events vs demand trend.

### 4. Weather Impact
- Sessions by weather condition.
- Energy and duration by temperature band.
- Precipitation impact on waiting time.
- Wind and humidity distribution cards.

### 5. Peak Hours and Demand
- Hourly demand line chart.
- Weekday vs weekend demand comparison.
- Charger type demand matrix.
- City and station drill-through.

### 6. Occupancy Forecast
- Predicted occupancy by station and hour.
- Predicted waiting time by station and hour.
- Overload risk indicator.
- Recommended less crowded stations table sorted by low predicted occupancy and waiting time.

## Suggested Measures
```DAX
Total Sessions = SUM(station_performance[total_sessions])
Total Energy kWh = SUM(station_performance[total_energy_kwh])
Average Traffic Score = AVERAGE(traffic_analytics[avg_traffic_score])
Peak Hour Sessions = SUM(station_performance[peak_hour_sessions])
Overload Risk Rate = AVERAGE(ml_features[overload_risk])
Average Waiting Time = AVERAGE(ml_features[estimated_waiting_time_minutes])
Average Occupancy = AVERAGE(ml_features[station_occupancy_percent])
```

## Design Guidance
- Use a dark text on light neutral canvas for readability.
- Use red or amber only for high-risk overload alerts.
- Keep station recommendation visuals at the top of the forecast page.
- Add slicers for city, hour, charger type, weekday/weekend, and weather condition.

