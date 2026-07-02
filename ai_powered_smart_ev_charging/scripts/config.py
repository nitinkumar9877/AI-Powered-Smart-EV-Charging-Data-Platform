from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILE = BRONZE_DIR / "ev_charging_raw_data.csv"
BRONZE_INGESTED_FILE = BRONZE_DIR / "bronze_ev_charging_events.csv"
SILVER_FILE = SILVER_DIR / "silver_ev_charging_events.csv"
FEATURE_FILE = SILVER_DIR / "ml_features.csv"

TARGET_OCCUPANCY = "station_occupancy_percent"
TARGET_WAITING_TIME = "estimated_waiting_time_minutes"
TARGET_OVERLOAD = "overload_risk"

