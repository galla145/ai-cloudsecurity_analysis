# cluster_anomalies.py
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
import argparse
import os

def create_spark():
    return SparkSession.builder.appName("Clustering Anomaly Detection").getOrCreate()

def load_df(spark, input_path):
    return spark.read.parquet(input_path)

def cluster(df, k=5):
    model = KMeans(k=k, seed=42, featuresCol="features")
    kmodel = model.fit(df)
    preds = kmodel.transform(df)

    # Distance UDF
    def distance(vec1, vec2):
        return float(Vectors.squared_distance(vec1, vec2))

    dist_udf = udf(lambda v: float(Vectors.norm(v, 2)), DoubleType())

    # Compute anomaly score = distance from cluster center
    centers = kmodel.clusterCenters()
    centers_bc = df.sparkSession.sparkContext.broadcast(centers)

    @udf("double")
    def anomaly_score(cluster, features):
        center = centers_bc.value[int(cluster)]
        return float(Vectors.squared_distance(features, center))

    preds = preds.withColumn("anomaly_score", anomaly_score(col("prediction"), col("features")))

    return preds

def save_results(df, out_dir):
    df.write.mode("overwrite").parquet(os.path.join(out_dir, "cluster_results"))

def main(args):
    spark = create_spark()
    df = load_df(spark, args.input)
    result = cluster(df)
    save_results(result, args.out_dir)
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)
