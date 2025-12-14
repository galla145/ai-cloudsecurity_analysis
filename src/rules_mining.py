# rules_mining.py
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import col
import argparse
import os

def create_spark():
    return SparkSession.builder.appName("Association Rules Mining").getOrCreate()

def load_df(spark, input_path):
    return spark.read.parquet(input_path)

def binarize(df, threshold=0.5):
    """
    Convert continuous features → binary for FP-growth.
    """
    bin_cols = []

    for c in df.columns:
        if c == "label":
            continue

        df = df.withColumn(c, (col(c) > threshold).cast("int"))
        bin_cols.append(c)

    # Convert to array of items
    df = df.withColumn("items", col("*"))
    return df.select("items")

def mine_rules(df, out_dir):
    fp = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.3)
    model = fp.fit(df)

    model.freqItemsets.write.mode("overwrite").json(os.path.join(out_dir, "freq_itemsets"))
    model.associationRules.write.mode("overwrite").json(os.path.join(out_dir, "rules"))

def main(args):
    spark = create_spark()
    df = load_df(spark, args.input)
    bdf = binarize(df)
    mine_rules(bdf, args.out_dir)
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)
