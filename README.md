# AI-Powered Smart EV Charging Network Optimization Platform

An end-to-end Data Engineering, Analytics, and Machine Learning portfolio project for optimizing EV charging station operations. The platform ingests raw EV charging sessions, builds Bronze/Silver/Gold data layers, creates business analytics tables, trains ML models, and defines a Power BI dashboard for decision makers and EV users.

## Project Goals
- Predict charging station occupancy.
- Predict estimated waiting time.
- Identify overloaded stations.
- Help EV users find less crowded charging stations.

## Tech Stack
- Python
- Pandas
- SQL
- scikit-learn
- PySpark reference implementation
- SQLite local serving layer
- Databricks-style medallion architecture
- Delta Lake concepts
- Power BI dashboard specification

## Folder Structure
```text
project/
├── data/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── notebooks/
├── scripts/
├── sql/
├── dashboard/
├── models/
├── docs/
└── README.md
```

## Pipeline Layers

### Bronze
Stores raw charging data with ingestion metadata. This layer preserves source columns and provides traceability.

### Silver
Cleans and validates the dataset:
- Standardizes data types.
- Handles nulls.
- Removes duplicates.
- Applies data quality rules.
- Engineers station, time, traffic, and weather features.

### Gold
Creates business-ready analytics tables:
- `station_performance.csv`
- `traffic_analytics.csv`
- `weather_impact_analytics.csv`
- `charging_demand_analytics.csv`

## Machine Learning
The project trains three models:
- Occupancy prediction: regression.
- Waiting time prediction: regression.
- Overload risk prediction: classification.

Model artifacts and evaluation metrics are stored under `models/`.

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
python app.py
```

`python app.py` starts the Streamlit EV Charging Intelligence Platform on localhost and opens it in the browser.

## Key Outputs
- Cleaned Silver data: `data/silver/silver_ev_charging_events.csv`
- Gold analytics tables: `data/gold/`
- ML feature dataset: `data/silver/ml_features.csv`
- Trained models: `models/*.pkl`
- Metrics: `models/model_metrics.json`
- Dashboard requirements: `dashboard/power_bi_dashboard_requirements.md`

## Documentation
- Architecture: `docs/architecture.md`
- Data flow: `docs/data_flow.md`
- Execution guide: `docs/execution_guide.md`

## Resume Talking Points
- Built a medallion-style EV charging data platform with Bronze, Silver, and Gold layers.
- Implemented data validation, null handling, deduplication, and operational feature engineering.
- Designed curated analytics tables for station performance, demand, traffic impact, and weather impact.
- Developed ML pipelines for occupancy, waiting time, and overload risk prediction.
- Created Power BI dashboard requirements for operational decision support.
