#!/usr/bin/env python3

# ---------------------------------------------------------
# CICIDS2017 PySpark ETL Script
# Cleans raw CSV data and writes Parquet + ETL logs to S3
# ---------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim
import datetime
import json
import boto3

# ------------------------------------
# S3 Paths
# ------------------------------------
raw_path = "s3a://ai-cloudsec-raw-gallada-2025/cicids2017/raw/"
processed_path = "s3a://ai-cloudsec-raw-gallada-2025/cicids2017/processed/"

# ------------------------------------
# Spark Session
# ------------------------------------
spark = SparkSession.builder.appName("CICIDS2017-ETL").getOrCreate()

# ------------------------------------
# 1. Load Raw CSV Data
# ------------------------------------
df = spark.read.option("header", True).option("inferSchema", True).csv(raw_path)
print("Initial record count:", df.count())

# ------------------------------------
# 2. Trim Whitespace
# ------------------------------------
df = df.select([trim(col(c)).alias(c) for c in df.columns])

# ------------------------------------
# 3. Drop Duplicates and Nulls
# ------------------------------------
df = df.dropDuplicates().na.drop()

# ------------------------------------
# 4. Convert Columns to Proper Data Types
# ------------------------------------
numeric_cols = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow IAT Mean'
]

for c in numeric_cols:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast("double"))

# ------------------------------------
# 5. Rename Columns (Spaces → Underscores)
# ------------------------------------
for old in df.columns:
    df = df.withColumnRenamed(old, old.replace(" ", "_"))

# ------------------------------------
# 6. Normalize Label Column
# ------------------------------------
if "Label" in df.columns:
    df = df.withColumn("Label", trim(col("Label")))

# ------------------------------------
# 7. Write Cleaned Data to S3 (Parquet)
# ------------------------------------
df.write.mode("overwrite").parquet(processed_path)
print("Cleaned data written to:", processed_path)

# Validation
spark.read.parquet(processed_path).show(5)

# ------------------------------------
# 8. Write ETL Log to S3
# ------------------------------------
log = {
    "dataset": "CICIDS2017",
    "records": df.count(),
    "columns": df.columns,
    "processed_date": str(datetime.datetime.utcnow()),
    "status": "success"
}

s3 = boto3.client("s3")

s3.put_object(
    Bucket="ai-cloudsec-raw-gallada-2025",
    Key="cicids2017/logs/etl_log.json",
    Body=json.dumps(log)
)

print("ETL log successfully uploaded to S3.")
