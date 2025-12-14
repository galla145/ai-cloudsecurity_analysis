# features.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
import argparse

def create_spark():
    return SparkSession.builder.appName("Feature Engineering").getOrCreate()

def load_df(spark, input_path):
    return spark.read.parquet(input_path)

def build_features(df):
    feature_cols = [c for c in df.columns if c != "label"]
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )
    df = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    return df.select("features", "label")

def save_df(df, output_path):
    df.write.mode("overwrite").parquet(output_path)
    print(f"Saved feature dataset to {output_path}")

def main(args):
    spark = create_spark()
    df = load_df(spark, args.input)
    df = build_features(df)
    save_df(df, args.output)
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
