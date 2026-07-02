import pandas as pd

from config import BRONZE_INGESTED_FILE, RAW_FILE


def run_bronze_ingestion() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw input file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    df["ingestion_source_file"] = RAW_FILE.name
    df["ingestion_timestamp_utc"] = pd.Timestamp.now("UTC").isoformat()
    df["bronze_record_id"] = range(1, len(df) + 1)

    BRONZE_INGESTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BRONZE_INGESTED_FILE, index=False)
    print(f"Bronze ingestion complete: {len(df):,} rows -> {BRONZE_INGESTED_FILE}")
    return df


if __name__ == "__main__":
    run_bronze_ingestion()
