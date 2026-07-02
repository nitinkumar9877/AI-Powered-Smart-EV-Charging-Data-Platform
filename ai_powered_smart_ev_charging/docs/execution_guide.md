# Step-by-Step Execution Guide

## 1. Create a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Verify Raw Input
Confirm this file exists:
```text
data/bronze/ev_charging_raw_data.csv
```

## 3. Run the Full Pipeline
```bash
python scripts/run_pipeline.py
```

## 4. Run Individual Steps
```bash
python scripts/01_bronze_ingestion.py
python scripts/02_silver_clean_validate.py
python scripts/03_gold_tables.py
python scripts/04_feature_engineering.py
python scripts/05_train_models.py
```

## 5. Review Outputs
- Bronze: `data/bronze/bronze_ev_charging_events.csv`
- Silver: `data/silver/silver_ev_charging_events.csv`
- ML features: `data/silver/ml_features.csv`
- Gold: `data/gold/*.csv`
- Models: `models/*.pkl`
- Metrics: `models/model_metrics.json`

## 6. Power BI
Open Power BI Desktop and import the CSV files from `data/gold` and `data/silver/ml_features.csv`. Build pages using `dashboard/power_bi_dashboard_requirements.md`.

## 7. Run the Web Application
```bash
python app.py
```

The launcher starts the EV Charging Intelligence Platform on localhost, selects an available port starting at `8501`, and opens the browser automatically.
