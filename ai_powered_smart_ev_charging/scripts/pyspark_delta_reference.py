from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, month, to_date, when


def create_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("smart-ev-charging-network")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def run_delta_style_pipeline(project_root: str) -> None:
    spark = create_spark()
    raw_path = f"{project_root}/data/bronze/ev_charging_raw_data.csv"
    bronze_delta = f"{project_root}/data/bronze/delta/ev_charging_events"
    silver_delta = f"{project_root}/data/silver/delta/ev_charging_events"

    bronze = spark.read.option("header", True).option("inferSchema", True).csv(raw_path)
    bronze.write.format("delta").mode("overwrite").save(bronze_delta)

    silver = (
        spark.read.format("delta").load(bronze_delta)
        .dropDuplicates()
        .withColumn("charging_date", to_date(col("Charging Start Time")))
        .withColumn("charging_hour", hour(col("Charging Start Time")))
        .withColumn("charging_month", month(col("Charging Start Time")))
        .withColumn("is_peak_hour", when(col("charging_hour").between(7, 10) | col("charging_hour").between(17, 21), 1).otherwise(0))
        .filter(col("Charging Station ID").isNotNull())
        .filter(col("Energy Consumed (kWh)") >= 0)
    )
    silver.write.format("delta").mode("overwrite").save(silver_delta)


if __name__ == "__main__":
    run_delta_style_pipeline(".")
