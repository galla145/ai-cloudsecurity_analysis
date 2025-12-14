# preprocess.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType

import argparse

def create_spark():
    return SparkSession.builder \
        .appName("CICIDS2017 Preprocessing") \
        .getOrCreate()

def load_data(spark, input_path):
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    print(f"Loaded dataset with {df.count()} rows and {len(df.columns)} columns.")
    return df

def clean_df(df):
    # Drop columns with >40% missing
    threshold = int(df.count() * 0.40)
    df = df.dropna(thresh=threshold)

    # Replace infinity and non-numeric values
    for c in df.columns:
        df = df.withColumn(c, when(col(c) == "Infinity", None).otherwise(col(c)))

    # Drop rows with any remaining nulls
    df = df.dropna()

    return df

def encode_labels(df):
    # Convert attack label to binary "label"
    df = df.withColumn(
        "label",
        when(col("Label") == "BENIGN", 0).otherwise(1)
    )
    return df.drop("Label")

def save_df(df, output_path):
    df.write.mode("overwrite").parquet(output_path)
    print(f"Saved processed dataset to {output_path}")

def main(args):
    spark = create_spark()
    df = load_data(spark, args.input)
    df = clean_df(df)
    df = encode_labels(df)
    save_df(df, args.output)
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
