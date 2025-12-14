# train_classifier.py
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import argparse
import json
import os

def create_spark():
    return SparkSession.builder.appName("Classification Models").getOrCreate()

def load_df(spark, input_path):
    return spark.read.parquet(input_path)

def evaluate(model_name, predictions):
    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    acc = evaluator.evaluate(predictions)
    return {"model": model_name, "accuracy": acc}

def train_and_save(df, out_dir):
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    results = []

    models = [
        ("RandomForest", RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=40)),
        ("LogisticRegression", LogisticRegression(labelCol="label", featuresCol="features", maxIter=20)),
        ("LinearSVC", LinearSVC(labelCol="label", featuresCol="features"))
    ]

    for name, model in models:
        print(f"Training {name}...")
        m = model.fit(train)
        preds = m.transform(test)
        metrics = evaluate(name, preds)

        model_path = os.path.join(out_dir, f"{name}_model")
        m.write().overwrite().save(model_path)
        print(f"{name} saved to {model_path}")

        results.append(metrics)

    # save metrics JSON
    with open(os.path.join(out_dir, "classification_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

def main(args):
    spark = create_spark()
    df = load_df(spark, args.input)
    train_and_save(df, args.out_dir)
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)
