import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "01_bronze_ingestion.py",
    "02_silver_clean_validate.py",
    "03_gold_tables.py",
    "04_feature_engineering.py",
    "05_train_models.py",
]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        subprocess.run([sys.executable, str(script_dir / script)], check=True)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()

